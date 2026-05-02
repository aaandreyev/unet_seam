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
    blend_falloff_px: int | None = None,
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
    fw = float(max(1, min(int(blend_falloff_px or inner_width), bw)))
    fh = float(max(1, min(int(blend_falloff_px or inner_width), bh)))
    # Corner taper distance: wide enough to avoid L-shaped artifacts at bbox corners
    # where two perpendicular seam bands overlap.  1/8 of inner_width, clamped to [12, 32].
    # Old value (inner_width//16, max 10px) was too tight — created visible angular seams
    # at corners. 16px minimum for inner_width=128 gives ~12% of the correction band.
    cpx_h = float(max(12, min(32, int(inner_width) // 8)))
    cpx_w = float(max(12, min(32, int(inner_width) // 8)))
    if side == "left":
        d_px = (xx - float(x0)).clamp_min(0.0)
        # Always fade from seam to interior. blend_falloff_px controls how
        # many pixels this fade occupies; beyond that, weight is zero.
        t = (1.0 - (d_px / fw)).clamp(0.0, 1.0)
        t_near_y0 = ((yy - float(y0)) / cpx_h).clamp(0.0, 1.0)
        t_near_y1 = ((float(y1) - yy) / cpx_h).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_y0), _hann_taper_from_t(t_near_y1))
    elif side == "right":
        d_px = (float(x1) - xx).clamp_min(0.0)
        t = (1.0 - (d_px / fw)).clamp(0.0, 1.0)
        t_near_y0 = ((yy - float(y0)) / cpx_h).clamp(0.0, 1.0)
        t_near_y1 = ((float(y1) - yy) / cpx_h).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_y0), _hann_taper_from_t(t_near_y1))
    elif side == "top":
        d_px = (yy - float(y0)).clamp_min(0.0)
        t = (1.0 - (d_px / fh)).clamp(0.0, 1.0)
        t_near_x0 = ((xx - float(x0)) / cpx_w).clamp(0.0, 1.0)
        t_near_x1 = ((float(x1) - xx) / cpx_w).clamp(0.0, 1.0)
        corner = torch.minimum(_hann_taper_from_t(t_near_x0), _hann_taper_from_t(t_near_x1))
    elif side == "bottom":
        d_px = (float(y1) - yy).clamp_min(0.0)
        t = (1.0 - (d_px / fh)).clamp(0.0, 1.0)
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
    blend_falloff_px: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not side_deltas:
        zeros = torch.zeros(mask.shape[0], 3, mask.shape[-2], mask.shape[-1], device=mask.device, dtype=mask.dtype)
        return zeros, {}
    use_seam = bbox is not None and inner_width is not None and inner_width > 0
    if len(side_deltas) == 1:
        side, delta = next(iter(side_deltas.items()))
        support = (delta.abs().mean(dim=1, keepdim=True) > 1e-8).to(mask.dtype)
        if use_seam:
            weight = build_seam_local_weight_map(
                mask,
                bbox,
                side,
                int(inner_width),
                blend_falloff_px=blend_falloff_px,
            ) * support
        else:
            weight = mask * support
        if side_confidences and side in side_confidences:
            confidence = side_confidences[side].to(device=mask.device, dtype=mask.dtype)
            # Confidence attenuates the delta amplitude; spatial weight controls taper.
            d_eff = delta * confidence
        else:
            d_eff = delta
        return d_eff * weight, {side: weight}

    # Multi-side: confidence is applied to each delta BEFORE spatial blending, and the
    # spatial weights alone normalise the blend. This ensures confidence consistently
    # attenuates correction amplitude whether one or many sides are active.
    #
    # Previous behaviour put confidence INTO the spatial weight, causing it to cancel
    # out when all sides had equal confidence (e.g. two sides at conf=0.5 produced the
    # same merged result as two sides at conf=1.0 — correction not attenuated at all).
    weights: dict[str, torch.Tensor] = {}
    d_effs: dict[str, torch.Tensor] = {}
    for side, delta in side_deltas.items():
        support = (delta.abs().mean(dim=1, keepdim=True) > 1e-8).to(mask.dtype)
        if use_seam:
            bmap = build_seam_local_weight_map(
                mask,
                bbox,
                side,
                int(inner_width),
                blend_falloff_px=blend_falloff_px,
            ) * support
        else:
            bmap = build_side_weight_map(mask, side) * support
        weights[side] = bmap
        if side_confidences and side in side_confidences:
            confidence = side_confidences[side].to(device=mask.device, dtype=mask.dtype)
            d_effs[side] = delta * confidence
        else:
            d_effs[side] = delta

    sides_order = list(side_deltas.keys())
    w_stack = torch.stack([weights[s] for s in sides_order], dim=0)
    d_stack = torch.stack([d_effs[s] for s in sides_order], dim=0)
    total_w = w_stack.sum(dim=0) + 1e-8
    merged0 = (w_stack * d_stack).sum(dim=0) / total_w
    return merged0 * mask, weights
