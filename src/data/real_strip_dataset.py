from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.harmonizer_input import build_harmonizer_input
from src.data.manifest import read_jsonl
from src.data.strip_geometry import StripSpec
from src.data.structural_filter import keep_structurally_matched_strip


class RealPairedStripDataset(Dataset):
    """Dataset for production-pipeline finetune strips.

    Manifest rows must contain `input_strip_path` and `target_strip_path`, both
    already canonicalized as outer-left / inner-right RGB strips.

    Rows that fail the structural filter are removed at init time with a warning,
    so __getitem__ never raises and DataLoader workers never crash.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        spec: StripSpec | None = None,
        boundary_band_px: int = 24,
        structural_threshold: float = 0.6,
    ) -> None:
        raw_rows = [row for row in read_jsonl(manifest_path) if not split or row.get("split") == split]
        self.spec = spec or StripSpec(seam_jitter_px=0)
        self.boundary_band_px = boundary_band_px
        self.structural_threshold = structural_threshold
        self.rows = self._prefilter(raw_rows)
        n_dropped = len(raw_rows) - len(self.rows)
        if n_dropped > 0:
            warnings.warn(
                f"RealPairedStripDataset: dropped {n_dropped}/{len(raw_rows)} rows that failed the structural filter.",
                UserWarning,
                stacklevel=2,
            )

    def _prefilter(self, rows: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for row in rows:
            try:
                input_rgb = self._load_rgb(row["input_strip_path"])
                target = self._load_rgb(row["target_strip_path"])
            except Exception:
                continue
            if keep_structurally_matched_strip(
                input_rgb,
                target,
                outer_width=self.spec.outer_width,
                band_px=int(row.get("structural_band_px", 32)),
                threshold=float(row.get("structural_threshold", self.structural_threshold)),
            ):
                kept.append(row)
        return kept

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, path: str) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if tensor.shape[-2:] != (self.spec.strip_height, self.spec.width):
            raise RuntimeError(f"strip must be {self.spec.strip_height}x{self.spec.width}, got {tuple(tensor.shape[-2:])}")
        return tensor

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        input_rgb = self._load_rgb(row["input_strip_path"])
        target = self._load_rgb(row["target_strip_path"])
        built = build_harmonizer_input(input_rgb, outer_width=self.spec.outer_width, boundary_band_px=self.boundary_band_px)
        return {
            **built,
            "target": target,
            "meta": row,
        }
