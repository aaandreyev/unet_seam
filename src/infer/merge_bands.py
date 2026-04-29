from __future__ import annotations

import math

import torch

__all__ = [
    "build_side_weight_map",
    "build_seam_local_weight_map",
    "merge_side_deltas",
]


def build_side_weight_map(mask: torch.Tensor, side: str, power: float = 1.5) -> torch.Tensor:
    """Image-global edge weights (legacy; use :func:`build_seam_local_weight_map` in production)."""
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
    base = _hann_taper_from_t(t)
    if power != 1.0:
        base = base.clamp(0.0, 1.0).pow(power)
    return base * mask


def _hann_taper_from_t(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 - torch.cos(math.pi * t.clamp(0.0, 1.0)))


def build_seam_local_weight_map(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    inner_width: int,
    power: float = 1.5,
) -> torch.Tensor:
    """
    Cosine falloff from the *seam* into the inner band, in bbox coordinates.
    Corner zones where two perpendicular seam bands overlap are tapered to zero
    so each side owns its stretch of the seam and corners are blended cleanly
    without the double-application that causes color blowup at high strength.
    """
    _, _, h, w = mask.shape
    x0, y0, x1, y1 = [int(x) for x in bbox]
    device, dtype = mask.device, mask.dtype
    yy = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    iw = float(max(1, min(int(inner_width), bw)))
    ih = float(max(1, min(int(inner_width), bh)))
    # Corner taper distance: keep small so corners don't develop a visible "L-notch"
    # where neither side owns the seam.  ~1/16 of inner_width, clamped to [4, 10].
    cpx_h = float(max(4, min(10, int(inner_width) // 16)))
    cpx_w = float(max(4, min(10, int(inner_width) // 16)))
    if side == "left":
        d = (xx - float(x0)) / iw
        t = (1.0 - d).clamp(0.0, 1.0)
        t_near_y0 = ((yy - float(y0)) / cpx_h).clamp(0.0, 1.0)
        t_near_y1 = ((float(y1) - yy) / cpx_h).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_y0), _hann_taper_from_t(t_near_y1))
    elif side == "right":
        d = (float(x1) - xx) / iw
        t = (1.0 - d).clamp(0.0, 1.0)
        t_near_y0 = ((yy - float(y0)) / cpx_h).clamp(0.0, 1.0)
        t_near_y1 = ((float(y1) - yy) / cpx_h).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_y0), _hann_taper_from_t(t_near_y1))
    elif side == "top":
        d = (yy - float(y0)) / ih
        t = (1.0 - d).clamp(0.0, 1.0)
        t_near_x0 = ((xx - float(x0)) / cpx_w).clamp(0.0, 1.0)
        t_near_x1 = ((float(x1) - xx) / cpx_w).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_x0), _hann_taper_from_t(t_near_x1))
    elif side == "bottom":
        d = (float(y1) - yy) / ih
        t = (1.0 - d).clamp(0.0, 1.0)
        t_near_x0 = ((xx - float(x0)) / cpx_w).clamp(0.0, 1.0)
        t_near_x1 = ((float(x1) - xx) / cpx_w).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_x0), _hann_taper_from_t(t_near_x1))
    else:
        raise ValueError(f"unsupported side: {side}")
    base = _hann_taper_from_t(t)
    if power != 1.0:
        base = base.clamp(0.0, 1.0).pow(power)
    return base * corner * mask


def merge_side_deltas(
    side_deltas: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    side_confidences: dict[str, torch.Tensor] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    inner_width: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not side_deltas:
        zeros = torch.zeros(mask.shape[0], 3, mask.shape[-2], mask.shape[-1], device=mask.device, dtype=mask.dtype)
        return zeros, {}
    if len(side_deltas) == 1:
        side, delta = next(iter(side_deltas.items()))
        if side_confidences and side in side_confidences:
            confidence = side_confidences[side].to(device=mask.device, dtype=mask.dtype)
            return delta * confidence * mask, {side: confidence * mask}
        return delta * mask, {side: torch.ones_like(mask)}

    use_seam = bbox is not None and inner_width is not None and inner_width > 0

    weights: dict[str, torch.Tensor] = {}
    for side, delta in side_deltas.items():
        support = (delta.abs().mean(dim=1, keepdim=True) > 1e-8).to(mask.dtype)
        if use_seam:
            bmap = build_seam_local_weight_map(mask, bbox, side, int(inner_width)) * support
        else:
            bmap = build_side_weight_map(mask, side) * support
        if side_confidences and side in side_confidences:
            confidence = side_confidences[side].to(device=mask.device, dtype=mask.dtype)
            bmap = bmap * confidence
        weights[side] = bmap

    sides_order = list(side_deltas.keys())
    w_stack = torch.stack([weights[s] for s in sides_order], dim=0)
    d_stack = torch.stack([side_deltas[s] for s in sides_order], dim=0)
    total_w = w_stack.sum(dim=0) + 1e-8
    merged0 = (w_stack * d_stack).sum(dim=0) / total_w
    return merged0 * mask, weights
