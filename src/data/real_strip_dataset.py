from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.manifest import read_jsonl
from src.data.strip_geometry import StripSpec, make_boundary_band_mask, make_distance_to_seam, make_inner_mask
from src.data.structural_filter import keep_structurally_matched_strip


class RealPairedStripDataset(Dataset):
    """Dataset for production-pipeline finetune strips.

    Manifest rows must contain `input_strip_path` and `target_strip_path`, both
    already canonicalized as outer-left / inner-right RGB strips.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        spec: StripSpec | None = None,
        boundary_band_px: int = 24,
        structural_threshold: float = 0.6,
    ) -> None:
        self.rows = [row for row in read_jsonl(manifest_path) if not split or row.get("split") == split]
        self.spec = spec or StripSpec(seam_jitter_px=0)
        self.boundary_band_px = boundary_band_px
        self.structural_threshold = structural_threshold

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
        if not keep_structurally_matched_strip(
            input_rgb,
            target,
            outer_width=self.spec.outer_width,
            band_px=int(row.get("structural_band_px", 32)),
            threshold=float(row.get("structural_threshold", self.structural_threshold)),
        ):
            raise RuntimeError(f"structural filter failed for row {idx}: {row.get('id', idx)}")
        mask = make_inner_mask(self.spec.strip_height, self.spec.width, self.spec.outer_width)
        distance = make_distance_to_seam(self.spec.strip_height, self.spec.width, self.spec.outer_width)
        boundary = make_boundary_band_mask(self.spec.strip_height, self.spec.width, self.spec.outer_width, self.boundary_band_px)
        return {
            "input": torch.cat([input_rgb, mask.squeeze(0), distance.squeeze(0)], dim=0),
            "target": target,
            "input_rgb": input_rgb,
            "mask": mask.squeeze(0),
            "distance": distance.squeeze(0),
            "inner_region_mask": mask.squeeze(0),
            "boundary_band_mask": boundary.squeeze(0),
            "meta": row,
        }
