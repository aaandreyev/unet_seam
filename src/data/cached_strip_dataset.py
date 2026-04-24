from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.manifest import read_jsonl
from src.data.strip_geometry import build_decay_mask, make_boundary_band_mask


def _load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_gray(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class CachedStripDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cache_root: str | Path) -> None:
        self.rows = read_jsonl(manifest_path)
        self.cache_root = Path(cache_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        sample_dir = self.cache_root / row["split"] / row["id"]
        input_rgb = _load_rgb(sample_dir / "input.png")
        target = _load_rgb(sample_dir / "target.png")
        mask = _load_gray(sample_dir / "mask.png")
        distance = _load_gray(sample_dir / "distance.png")
        seam_x = 128 + int(row["strip"]["seam_jitter_px"])
        height, width = input_rgb.shape[-2:]
        boundary = make_boundary_band_mask(height, width, seam_x, 24).squeeze(0)
        decay = build_decay_mask(height, width, seam_x, int(row["strip"]["inner_width"])).squeeze(0)
        return {
            "input": torch.cat([input_rgb, mask, distance], dim=0),
            "target": target,
            "input_rgb": input_rgb,
            "mask": mask,
            "distance": distance,
            "inner_region_mask": mask,
            "boundary_band_mask": boundary,
            "decay_mask": decay,
            "meta": {
                "image_id": row["id"],
                "axis": row["strip"]["axis"],
                "side": row["strip"]["original_side"],
                "rotation_k": row["strip"]["rotation_k"],
                "flip_h": row["strip"]["flip_h"],
                "seam_jitter_px": row["strip"]["seam_jitter_px"],
                "inner_width": row["strip"]["inner_width"],
                "edge_padded_pixels": row["strip"]["edge_padded_pixels"],
                "ops": row["corruption"]["ops"],
                "scene_tags": row.get("scene_tags", []),
                "split": row["split"],
                "cluster_id": row.get("cluster_id"),
                "seam_x_frac_in_source": row["strip"]["seam_x_frac_in_source"],
            },
        }
