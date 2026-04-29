from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.harmonizer_input import build_harmonizer_input
from src.data.manifest import read_jsonl


def _load_rgb_uint8(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"expected RGB image at {path}")
    return arr


def _load_mask_uint8(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.ndim != 2:
        raise RuntimeError(f"expected grayscale mask at {path}")
    return arr


class MaterializedStripDataset(Dataset):
    """Dataset backed by materialized strip triplets.

    Manifest rows must contain:
    - input_path
    - target_path
    - mask_path
    - seam_x
    - split
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        boundary_band_px: int = 24,
        preload: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.boundary_band_px = boundary_band_px
        self.rows = [row for row in read_jsonl(self.manifest_path) if not split or row.get("split") == split]
        self.preload = preload
        self._cache: list[dict[str, np.ndarray]] | None = None
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}
        if self.preload:
            self._cache = [self._load_arrays(row) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def _load_arrays(self, row: dict) -> dict[str, np.ndarray]:
        if "shard_path" in row:
            shard = self._load_shard(row["shard_path"])
            shard_index = int(row["shard_index"])
            return {
                "input": shard["inputs"][shard_index],
                "target": shard["targets"][shard_index],
                "mask": shard["masks"][shard_index],
            }
        return {
            "input": _load_rgb_uint8(self._resolve(row["input_path"])),
            "target": _load_rgb_uint8(self._resolve(row["target_path"])),
            "mask": _load_mask_uint8(self._resolve(row["mask_path"])),
        }

    def _load_shard(self, path: str) -> dict[str, np.ndarray]:
        cache_key = path
        cached = self._shard_cache.get(cache_key)
        if cached is not None:
            return cached
        with np.load(self._resolve(path), allow_pickle=False) as data:
            loaded = {
                "inputs": data["inputs"],
                "targets": data["targets"],
                "masks": data["masks"],
            }
        self._shard_cache[cache_key] = loaded
        return loaded

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 3:
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        if arr.ndim == 2:
            return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        raise RuntimeError(f"unexpected array rank: {arr.ndim}")

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        arrays = self._cache[idx] if self._cache is not None else self._load_arrays(row)
        input_rgb = self._to_tensor(arrays["input"])
        target = self._to_tensor(arrays["target"])
        mask = self._to_tensor(arrays["mask"])
        seam_x = int(row.get("seam_x", 128))
        outer_width = int(row.get("outer_width", 128))
        built = build_harmonizer_input(
            input_rgb,
            outer_width=outer_width,
            boundary_band_px=self.boundary_band_px,
            seam_x=seam_x,
        )
        # Preserve materialized mask to avoid any silent mismatch between saved sample and rebuilt features.
        built["mask"] = mask
        built["inner_region_mask"] = mask
        meta = dict(row)
        meta["materialized_manifest"] = str(self.manifest_path)
        return {
            **built,
            "target": target,
            "meta": meta,
        }
