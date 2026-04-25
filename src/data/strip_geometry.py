from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Side = Literal["left", "right", "top", "bottom"]


@dataclass(frozen=True)
class StripSpec:
    strip_height: int = 1024
    outer_width: int = 128
    inner_width: int = 128
    seam_jitter_px: int = 16

    @property
    def width(self) -> int:
        return self.outer_width + self.inner_width


def canonicalize_strip(strip: torch.Tensor, side: Side) -> torch.Tensor:
    if side == "left":
        return strip
    if side == "right":
        return torch.flip(strip, dims=(-1,))
    if side == "top":
        return torch.rot90(strip, k=1, dims=(-2, -1))
    if side == "bottom":
        return torch.rot90(strip, k=3, dims=(-2, -1))
    raise ValueError(f"unsupported side: {side}")


def decanonicalize_strip(strip: torch.Tensor, side: Side) -> torch.Tensor:
    if side == "left":
        return strip
    if side == "right":
        return torch.flip(strip, dims=(-1,))
    if side == "top":
        return torch.rot90(strip, k=3, dims=(-2, -1))
    if side == "bottom":
        return torch.rot90(strip, k=1, dims=(-2, -1))
    raise ValueError(f"unsupported side: {side}")


def make_inner_mask(height: int, width: int, seam_x: int) -> torch.Tensor:
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    return (xs >= seam_x).to(torch.float32).expand(1, 1, height, width)


def make_distance_to_seam(height: int, width: int, seam_x: int) -> torch.Tensor:
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    inner_width = max(width - seam_x, 1)
    distance = ((xs - float(seam_x)).clamp(min=0.0) / float(max(inner_width - 1, 1))).clamp(0.0, 1.0)
    return distance.expand(1, 1, height, width)


def make_boundary_band_mask(height: int, width: int, seam_x: int, band_px: int = 24) -> torch.Tensor:
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    return (xs.sub(float(seam_x)).abs() <= band_px).to(torch.float32).expand(1, 1, height, width)


def build_decay_mask(height: int, width: int, seam_x: int, inner_width: int) -> torch.Tensor:
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    t = ((xs - seam_x) / max(inner_width, 1)).clamp(0.0, 1.0)
    decay = 0.5 * (1.0 + torch.cos(math.pi * t))
    decay = torch.where(xs < seam_x, torch.zeros_like(decay), decay)
    return decay.expand(1, 1, height, width)


def _replicate_pad_strip(strip: torch.Tensor, target_width: int) -> tuple[torch.Tensor, int]:
    edge_padded = max(target_width - strip.shape[-1], 0)
    if edge_padded == 0:
        return strip, 0
    padded = F.pad(strip, (0, edge_padded, 0, 0), mode="replicate")
    return padded, edge_padded


def extract_side_strip(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    spec: StripSpec,
) -> tuple[torch.Tensor, dict]:
    if image.ndim != 3:
        raise ValueError("expected CHW image tensor")
    _, h, w = image.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        y_center = (y0 + y1) // 2
        y_start = max(0, min(h - spec.strip_height, y_center - spec.strip_height // 2))
        outer = image[:, y_start : y_start + spec.strip_height, max(0, x0 - spec.outer_width) : x0]
        inner = image[:, y_start : y_start + spec.strip_height, x0 : min(w, x0 + spec.inner_width)]
        strip = torch.cat([outer, inner], dim=-1)
        strip, edge_padded = _replicate_pad_strip(strip, spec.width)
    elif side == "right":
        y_center = (y0 + y1) // 2
        y_start = max(0, min(h - spec.strip_height, y_center - spec.strip_height // 2))
        inner = image[:, y_start : y_start + spec.strip_height, max(0, x1 - spec.inner_width) : x1]
        outer = image[:, y_start : y_start + spec.strip_height, x1 : min(w, x1 + spec.outer_width)]
        strip = torch.cat([inner, outer], dim=-1)
        strip, edge_padded = _replicate_pad_strip(strip, spec.width)
        strip = canonicalize_strip(strip, "right")
    elif side == "top":
        x_center = (x0 + x1) // 2
        x_start = max(0, min(w - spec.strip_height, x_center - spec.strip_height // 2))
        outer = image[:, max(0, y0 - spec.outer_width) : y0, x_start : x_start + spec.strip_height]
        inner = image[:, y0 : min(h, y0 + spec.inner_width), x_start : x_start + spec.strip_height]
        strip = torch.cat([outer, inner], dim=-2)
        strip = F.pad(strip, (0, 0, 0, max(spec.width - strip.shape[-2], 0)), mode="replicate")
        edge_padded = max(spec.width - strip.shape[-2], 0)
        strip = canonicalize_strip(strip, "top")
    elif side == "bottom":
        x_center = (x0 + x1) // 2
        x_start = max(0, min(w - spec.strip_height, x_center - spec.strip_height // 2))
        inner = image[:, max(0, y1 - spec.inner_width) : y1, x_start : x_start + spec.strip_height]
        outer = image[:, y1 : min(h, y1 + spec.outer_width), x_start : x_start + spec.strip_height]
        strip = torch.cat([inner, outer], dim=-2)
        strip = F.pad(strip, (0, 0, 0, max(spec.width - strip.shape[-2], 0)), mode="replicate")
        edge_padded = max(spec.width - strip.shape[-2], 0)
        strip = canonicalize_strip(strip, "bottom")
    else:
        raise ValueError(f"unsupported side: {side}")
    meta = {"edge_padded_pixels": int(edge_padded)}
    if side in {"left", "right"}:
        meta["y_start"] = int(y_start)
    else:
        meta["x_start"] = int(x_start)
    return strip, meta


def validate_roundtrip(strip: torch.Tensor, side: Side) -> bool:
    return torch.equal(decanonicalize_strip(canonicalize_strip(strip, side), side), strip)
