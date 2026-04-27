from __future__ import annotations

import math

import torch


def build_side_weight_map(mask: torch.Tensor, side: str, power: float = 1.5) -> torch.Tensor:
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
    base = 0.5 * (1.0 - torch.cos(math.pi * t))
    if power != 1.0:
        base = base.clamp(0.0, 1.0).pow(power)
    return base * mask


def merge_side_deltas(
    side_deltas: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    side_confidences: dict[str, torch.Tensor] | None = None,
    min_confidence_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not side_deltas:
        zeros = torch.zeros(mask.shape[0], 3, mask.shape[-2], mask.shape[-1], device=mask.device, dtype=mask.dtype)
        return zeros, {}
    if len(side_deltas) == 1:
        side, delta = next(iter(side_deltas.items()))
        return delta * mask, {side: torch.ones_like(mask)}
    weights: dict[str, torch.Tensor] = {}
    for side, delta in side_deltas.items():
        support = (delta.abs().mean(dim=1, keepdim=True) > 1e-8).to(mask.dtype)
        weight = build_side_weight_map(mask, side) * support
        if side_confidences and side in side_confidences:
            confidence = side_confidences[side].to(device=mask.device, dtype=mask.dtype)
            confidence = min_confidence_weight + (1.0 - min_confidence_weight) * confidence.clamp(0.0, 1.0)
            weight = weight * confidence
        weights[side] = weight
    total_w = sum(weights.values()) + 1e-8
    merged = sum((weights[side] / total_w) * side_deltas[side] for side in side_deltas)
    return merged * mask, weights
