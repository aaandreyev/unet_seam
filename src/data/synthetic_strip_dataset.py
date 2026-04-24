from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.data.corruptions import apply_random_corruptions
from src.data.manifest import read_jsonl
from src.data.strip_geometry import (
    StripSpec,
    build_decay_mask,
    canonicalize_strip,
    make_boundary_band_mask,
    make_distance_to_seam,
    make_inner_mask,
)


@dataclass(frozen=True)
class SampleConfig:
    axis: str
    side: str
    seam_x_frac: float
    flip_h: bool
    rotation_k: int
    seam_jitter_px: int
    inner_width: int


class SyntheticStripDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        strips_per_image: int = 25,
        split: str | None = None,
        seed: int = 42,
        spec: StripSpec | None = None,
        boundary_band_px: int = 24,
    ) -> None:
        self.rows = [row for row in read_jsonl(manifest_path) if not split or row.get("split") == split]
        self.strips_per_image = strips_per_image
        self.seed = seed
        self.spec = spec or StripSpec()
        self.boundary_band_px = boundary_band_px
        self.inner_widths = [96, 128, 160, 192]
        self.base_variants = self._build_base_variants()

    def _build_base_variants(self) -> list[SampleConfig]:
        variants: list[SampleConfig] = []
        seam_positions = np.linspace(0.25, 0.75, 8)
        for axis in ("vertical", "horizontal"):
            sides = ("left", "right") if axis == "vertical" else ("top", "bottom")
            for frac in seam_positions:
                for flip_h in (False, True):
                    for rotation_k in range(4):
                        side = sides[(rotation_k + int(flip_h)) % len(sides)]
                        variants.append(
                            SampleConfig(
                                axis=axis,
                                side=side,
                                seam_x_frac=float(frac),
                                flip_h=flip_h,
                                rotation_k=rotation_k,
                                seam_jitter_px=0,
                                inner_width=128,
                            )
                        )
        return variants

    def __len__(self) -> int:
        return len(self.rows) * self.strips_per_image

    def _config_for_index(self, idx: int) -> SampleConfig:
        rng = random.Random(self.seed + idx)
        image_slot = idx // max(self.strips_per_image, 1)
        epoch_variants = self.base_variants.copy()
        rng.shuffle(epoch_variants)
        base = epoch_variants[idx % min(len(epoch_variants), self.strips_per_image) if self.strips_per_image <= len(epoch_variants) else idx % len(epoch_variants)]
        del image_slot
        return SampleConfig(
            axis=base.axis,
            side=base.side,
            seam_x_frac=base.seam_x_frac,
            flip_h=base.flip_h,
            rotation_k=base.rotation_k,
            seam_jitter_px=rng.randint(-self.spec.seam_jitter_px, self.spec.seam_jitter_px),
            inner_width=rng.choice(self.inner_widths),
        )

    def _load_image(self, row: dict) -> torch.Tensor:
        arr = np.asarray(Image.open(row["source_path"]).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _augment_source_image(self, image: torch.Tensor, cfg: SampleConfig) -> torch.Tensor:
        if cfg.flip_h:
            image = torch.flip(image, dims=(-1,))
        if cfg.rotation_k:
            image = torch.rot90(image, k=cfg.rotation_k, dims=(-2, -1))
        return image

    def _extract_clean_strip(self, image: torch.Tensor, cfg: SampleConfig) -> torch.Tensor:
        h, w = image.shape[-2:]
        spec = StripSpec(strip_height=self.spec.strip_height, outer_width=self.spec.outer_width, inner_width=cfg.inner_width, seam_jitter_px=self.spec.seam_jitter_px)
        if cfg.axis == "vertical":
            seam_x = int(round(cfg.seam_x_frac * w))
            seam_x = min(max(spec.outer_width, seam_x), w - spec.inner_width)
            y_start = max(0, min(h - spec.strip_height, h // 2 - spec.strip_height // 2))
            outer = image[:, y_start : y_start + spec.strip_height, seam_x - spec.outer_width : seam_x]
            inner = image[:, y_start : y_start + spec.strip_height, seam_x : seam_x + cfg.inner_width]
            strip = torch.cat([outer, inner], dim=-1)
            if cfg.side == "right":
                strip = canonicalize_strip(torch.flip(strip, dims=(-1,)), "right")
        else:
            seam_y = int(round(cfg.seam_x_frac * h))
            seam_y = min(max(spec.outer_width, seam_y), h - cfg.inner_width)
            x_start = max(0, min(w - spec.strip_height, w // 2 - spec.strip_height // 2))
            outer = image[:, seam_y - spec.outer_width : seam_y, x_start : x_start + spec.strip_height]
            inner = image[:, seam_y : seam_y + cfg.inner_width, x_start : x_start + spec.strip_height]
            strip = torch.cat([outer, inner], dim=-2)
            strip = canonicalize_strip(strip, cfg.side)
        return strip

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx // self.strips_per_image]
        cfg = self._config_for_index(idx)
        image = self._augment_source_image(self._load_image(row), cfg)
        clean_strip = self._extract_clean_strip(image, cfg)
        if clean_strip.shape[-2:] != (self.spec.strip_height, self.spec.outer_width + cfg.inner_width):
            raise RuntimeError("unexpected strip shape")
        seam_x = self.spec.outer_width + cfg.seam_jitter_px
        pad_left = max(0, -cfg.seam_jitter_px)
        pad_right = max(0, cfg.seam_jitter_px)
        clean_strip = F.pad(clean_strip, (pad_left, pad_right, 0, 0), mode="replicate")
        clean_strip = clean_strip[..., : self.spec.strip_height, : self.spec.outer_width + cfg.inner_width]
        target = clean_strip.clone()
        input_rgb = clean_strip.unsqueeze(0)
        inner = input_rgb[..., self.spec.outer_width :]
        corrupted = apply_random_corruptions(inner, torch.Generator().manual_seed(self.seed + idx))
        input_rgb[..., self.spec.outer_width :] = corrupted.image
        mask = make_inner_mask(self.spec.strip_height, self.spec.outer_width + cfg.inner_width, seam_x)
        distance = make_distance_to_seam(self.spec.strip_height, self.spec.outer_width + cfg.inner_width, seam_x)
        boundary = make_boundary_band_mask(self.spec.strip_height, self.spec.outer_width + cfg.inner_width, seam_x, self.boundary_band_px)
        decay = build_decay_mask(self.spec.strip_height, self.spec.outer_width + cfg.inner_width, seam_x, cfg.inner_width)
        sample = {
            "input": torch.cat([input_rgb.squeeze(0), mask.squeeze(0), distance.squeeze(0)], dim=0),
            "target": target,
            "input_rgb": input_rgb.squeeze(0),
            "mask": mask.squeeze(0),
            "distance": distance.squeeze(0),
            "inner_region_mask": mask.squeeze(0),
            "boundary_band_mask": boundary.squeeze(0),
            "decay_mask": decay.squeeze(0),
            "meta": {
                "image_id": row["id"],
                "axis": cfg.axis,
                "side": cfg.side,
                "rotation_k": cfg.rotation_k,
                "flip_h": cfg.flip_h,
                "seam_jitter_px": cfg.seam_jitter_px,
                "inner_width": cfg.inner_width,
                "edge_padded_pixels": 0,
                "ops": corrupted.ops,
                "scene_tags": row.get("scene_tags", []),
                "split": row.get("split"),
                "cluster_id": row.get("cluster_id"),
                "seam_x_frac_in_source": cfg.seam_x_frac,
            },
        }
        return sample


def collate_strip_batch(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("empty batch")
    tensor_keys = [key for key, value in samples[0].items() if isinstance(value, torch.Tensor)]
    batch: dict = {}
    max_w = max(sample["input"].shape[-1] for sample in samples)
    for key in tensor_keys:
        items = []
        for sample in samples:
            tensor = sample[key]
            pad_w = max_w - tensor.shape[-1]
            if pad_w > 0:
                mode = "replicate" if tensor.shape[0] in {3, 5} else "constant"
                if tensor.ndim == 3:
                    if mode == "constant":
                        tensor = F.pad(tensor, (0, pad_w, 0, 0), mode=mode, value=0.0)
                    else:
                        tensor = F.pad(tensor, (0, pad_w, 0, 0), mode=mode)
                else:
                    raise RuntimeError(f"unexpected tensor rank for key={key}")
            items.append(tensor)
        batch[key] = torch.stack(items, dim=0)
    batch["meta"] = [sample["meta"] for sample in samples]
    return batch
