from __future__ import annotations

import math

import torch


def build_side_weight_map(mask: torch.Tensor, side: str) -> torch.Tensor:
    _, _, h, w = mask.shape
    yy = torch.linspace(0.0, 1.0, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    if side == "left":
        t = 1.0 - xx
    elif side == "right":
        t = xx
    elif side == "top":
        t = 1.0 - yy
    elif side == "bottom":
        t = yy
    else:
        raise ValueError(f"unsupported side: {side}")
    weight = 0.5 * (1.0 - torch.cos(math.pi * t))
    return weight * mask


def merge_side_residuals(side_residuals: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not side_residuals:
        zeros = torch.zeros(mask.shape[0], 3, mask.shape[-2], mask.shape[-1], device=mask.device, dtype=mask.dtype)
        return zeros, {}
    if len(side_residuals) == 1:
        side, residual = next(iter(side_residuals.items()))
        ones = torch.ones_like(mask)
        return residual * mask, {side: ones}
    weights = {side: build_side_weight_map(mask, side) for side in side_residuals}
    total_w = sum(weights.values()) + 1e-8
    merged = sum((weights[side] / total_w) * side_residuals[side] for side in side_residuals)
    merged = merged * mask
    return merged, weights
