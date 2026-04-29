from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.harmonizer_input import build_harmonizer_input
from src.data.structural_filter import gradient_cosine_similarity, sobel_magnitude
from src.infer.extract_strips import extract_active_strips
from src.infer.merge_bands import merge_side_deltas


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _canonical_model_input(strip_batch: torch.Tensor, outer_width: int, boundary_band_px: int = 24) -> torch.Tensor:
    return build_harmonizer_input(strip_batch, outer_width=outer_width, boundary_band_px=boundary_band_px, seam_x=outer_width)["input"]


def _inner_taper(height: int, inner_width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if inner_width <= 1:
        return torch.ones(1, 1, height, inner_width, device=device, dtype=dtype)
    u = torch.arange(inner_width, device=device, dtype=dtype).view(1, 1, 1, inner_width)
    taper = 0.5 * (1.0 + torch.cos(torch.pi * u / float(inner_width - 1)))
    taper = 0.15 + 0.85 * taper
    return taper.expand(1, 1, height, inner_width)


def _structural_strength_scale(strip: torch.Tensor, outer_width: int = 128, band_px: int = 16) -> tuple[float, float]:
    band = min(band_px, outer_width, strip.shape[-1] - outer_width)
    if band <= 0:
        return 1.0, 1.0
    outer_band = torch.flip(strip[..., outer_width - band : outer_width], dims=(-1,)).unsqueeze(0)
    inner_band = strip[..., outer_width : outer_width + band].unsqueeze(0)
    score = float(gradient_cosine_similarity(outer_band, inner_band).item())
    if score <= 0.15:
        return 0.2, score
    if score >= 0.75:
        return 1.0, score
    scale = 0.2 + 0.8 * ((score - 0.15) / 0.60)
    return float(scale), score


def _place_inner_map(
    canonical_inner: torch.Tensor,
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    inner_width: int,
    meta: dict,
) -> torch.Tensor:
    placed = torch.zeros(image.shape[0], canonical_inner.shape[1], image.shape[-2], image.shape[-1], device=image.device, dtype=canonical_inner.dtype)
    x0, y0, x1, y1 = bbox
    if side == "left":
        width = min(inner_width, x1 - x0)
        y_start = int(meta["y_start"])
        y_end = min(y_start + canonical_inner.shape[-2], image.shape[-2])
        placed[:, :, y_start:y_end, x0 : x0 + width] = canonical_inner[:, :, : y_end - y_start, :width]
    elif side == "right":
        width = min(inner_width, x1 - x0)
        y_start = int(meta["y_start"])
        y_end = min(y_start + canonical_inner.shape[-2], image.shape[-2])
        placed[:, :, y_start:y_end, x1 - width : x1] = torch.flip(canonical_inner[:, :, : y_end - y_start, :width], dims=(-1,))
    elif side == "top":
        height = min(inner_width, y1 - y0)
        top_map = torch.rot90(canonical_inner[:, :, :, :height], k=3, dims=(-2, -1))
        x_start = int(meta["x_start"])
        x_end = min(x_start + top_map.shape[-1], image.shape[-1])
        placed[:, :, y0 : y0 + height, x_start:x_end] = top_map[:, :, :height, : x_end - x_start]
    elif side == "bottom":
        height = min(inner_width, y1 - y0)
        bottom_map = torch.rot90(canonical_inner[:, :, :, :height], k=1, dims=(-2, -1))
        x_start = int(meta["x_start"])
        x_end = min(x_start + bottom_map.shape[-1], image.shape[-1])
        placed[:, :, y1 - height : y1, x_start:x_end] = bottom_map[:, :, :height, : x_end - x_start]
    else:
        raise ValueError(f"unsupported side: {side}")
    return placed


def _expand_band_map(x: torch.Tensor, inner_width: int) -> torch.Tensor:
    if x.shape[-1] >= inner_width:
        return x[..., :inner_width]
    pad = inner_width - x.shape[-1]
    return torch.nn.functional.pad(x, (0, pad, 0, 0), mode="replicate")


def _seam_error_components(outer_ref: torch.Tensor, inner_band: torch.Tensor) -> torch.Tensor:
    rgb_err = (inner_band - outer_ref).abs().mean(dim=1, keepdim=True)
    outer_luma = 0.2126 * outer_ref[:, 0:1] + 0.7152 * outer_ref[:, 1:2] + 0.0722 * outer_ref[:, 2:3]
    inner_luma = 0.2126 * inner_band[:, 0:1] + 0.7152 * inner_band[:, 1:2] + 0.0722 * inner_band[:, 2:3]
    luma_err = (inner_luma - outer_luma).abs()
    grad_err = (sobel_magnitude(inner_band) - sobel_magnitude(outer_ref)).abs()
    return rgb_err + 0.35 * luma_err + 0.20 * grad_err


def _build_safety_gate(
    canonical_strip: torch.Tensor,
    corrected_strip: torch.Tensor,
    outer_width: int,
    inner_width: int,
    *,
    band_px: int = 24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    band = min(band_px, outer_width, inner_width)
    if band <= 0:
        ones = torch.ones(1, 1, canonical_strip.shape[-2], inner_width, device=canonical_strip.device, dtype=canonical_strip.dtype)
        zeros = torch.zeros(1, 1, canonical_strip.shape[-2], inner_width, device=canonical_strip.device, dtype=canonical_strip.dtype)
        return ones, zeros, zeros
    outer_ref = torch.flip(canonical_strip[..., outer_width - band : outer_width], dims=(-1,))
    inner_before = canonical_strip[..., outer_width : outer_width + band]
    inner_after = corrected_strip[..., outer_width : outer_width + band]
    before_err = _seam_error_components(outer_ref, inner_before)
    after_err = _seam_error_components(outer_ref, inner_after)
    worse_ratio = ((after_err - before_err).clamp_min(0.0) / (before_err + 0.01)).clamp(0.0, 1.0)
    band_safety = 1.0 - 0.85 * worse_ratio

    row_worse = worse_ratio.mean(dim=-1, keepdim=True)
    xs = torch.arange(inner_width, device=canonical_strip.device, dtype=canonical_strip.dtype).view(1, 1, 1, inner_width)
    # Lift the row-level safety damp away from the seam so the seam itself stays
    # correctable when the row-aggregated worse_ratio is non-zero.  The first `band`
    # pixels are still gated by the per-pixel `band_safety` below, which is the
    # accurate local check for "did the model worsen the seam here?".
    decay = 1.0 - torch.exp(-xs / max(float(band), 1.0))
    row_safety = 1.0 - 0.75 * row_worse * decay
    full_safety = row_safety.expand(1, 1, canonical_strip.shape[-2], inner_width).clone()
    full_safety[..., :band] = torch.minimum(full_safety[..., :band], band_safety)
    return full_safety.clamp(0.15, 1.0), before_err, after_err


def _smooth_profile(profile: torch.Tensor, kernel: int) -> torch.Tensor:
    if profile.shape[-1] <= 1:
        return profile
    kernel = max(3, int(kernel) | 1)
    kernel = min(kernel, profile.shape[-1] if profile.shape[-1] % 2 == 1 else max(profile.shape[-1] - 1, 1))
    if kernel <= 1:
        return profile
    pad = kernel // 2
    return F.avg_pool1d(F.pad(profile, (pad, pad), mode="replicate"), kernel_size=kernel, stride=1)


def _row_guard_from_profile(profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if profile.shape[-1] <= 1:
        ones = torch.ones_like(profile)
        return ones, profile
    smoothed = _smooth_profile(profile, kernel=33)
    dy = (smoothed[..., 1:] - smoothed[..., :-1]).abs()
    dy = F.pad(dy, (1, 0), mode="replicate")
    flatness = 1.0 - (dy / (smoothed + 0.01) * 3.0).clamp(0.0, 1.0)
    # Higher trigger threshold so weak profile signals (~0.05) don't suppress
    # 30%+ of the correction along the whole side.  Now requires a clearly
    # large flat-profile delta before the global tonal-shift damp kicks in.
    strength = ((smoothed - 0.030) / 0.060).clamp(0.0, 1.0)
    guard = 1.0 - 0.55 * flatness * strength
    return guard.clamp(0.35, 1.0), smoothed


def _expand_row_guard(
    row_guard: torch.Tensor,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    # Suppression weakest at the seam (xs=0), strongest deep inside the inner band.
    # Lets the seam itself receive full correction while still damping flat-profile
    # global tonal shifts further from the seam.
    decay = 0.35 + 0.65 * (1.0 - torch.exp(-xs / max(width * 0.45, 1.0)))
    row_guard = row_guard.unsqueeze(-1)
    return (1.0 - (1.0 - row_guard) * decay).clamp(0.35, 1.0)


def _expand_col_guard(
    col_guard: torch.Tensor,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Same as row expansion along the seam→inner axis for top/bottom patches (H×W layout)."""
    ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    decay = 0.35 + 0.65 * (1.0 - torch.exp(-ys / max(height * 0.45, 1.0)))
    return (1.0 - (1.0 - col_guard.unsqueeze(-2)) * decay).clamp(0.35, 1.0)


def _build_profile_guard(
    merged: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    *,
    band_px: int = 24,
    inner_width: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    """Per-side profile from ``band_px`` near the seam; guard maps span ``inner_width`` into the bbox."""
    x0, y0, x1, y1 = bbox
    delta_mag = merged.abs().mean(dim=1, keepdim=True) * mask
    bw, bh = max(x1 - x0, 0), max(y1 - y0, 0)
    # Depth of the harmonizer band along each side (same cap as strip placement).
    iw = int(inner_width) if inner_width is not None else max(bw, bh)
    left_span = min(iw, bw)
    top_span = min(iw, bh)
    sample_w = min(int(band_px), left_span) if left_span > 0 else 0
    sample_h = min(int(band_px), top_span) if top_span > 0 else 0
    # Running sum/count so perimeter overlaps (e.g. corners) use a mean, not a harsh minimum
    # that over-suppresses the merged delta in corners and looks like a “gap” before the fix.
    acc_guard = torch.zeros_like(mask)
    acc_count = torch.zeros_like(mask)
    side_guards: dict[str, torch.Tensor] = {}
    stats: dict[str, dict[str, float]] = {}

    def merge_side_guard(side: str, guard_patch: torch.Tensor, signal_patch: torch.Tensor, region: tuple[int, int, int, int]) -> None:
        rx0, ry0, rx1, ry1 = region
        patch = torch.ones_like(mask)
        patch[:, :, ry0:ry1, rx0:rx1] = guard_patch
        side_guards[side] = patch
        acc_guard[:, :, ry0:ry1, rx0:rx1] = acc_guard[:, :, ry0:ry1, rx0:rx1] + guard_patch
        acc_count[:, :, ry0:ry1, rx0:rx1] = acc_count[:, :, ry0:ry1, rx0:rx1] + 1.0
        stats[side] = {
            "profile_guard_mean": float(guard_patch.mean().item()),
            "profile_signal_mean": float(signal_patch.mean().item()),
        }

    left_band = max(left_span, 0)
    if left_band > 0 and sample_w > 0:
        patch = delta_mag[:, :, y0:y1, x0 : x0 + sample_w]
        profile = patch.mean(dim=-1)
        row_guard, signal = _row_guard_from_profile(profile)
        guard_patch = _expand_row_guard(row_guard, left_band, merged.device, merged.dtype)
        merge_side_guard("left", guard_patch, signal, (x0, y0, x0 + left_band, y1))

        patch = delta_mag[:, :, y0:y1, x1 - sample_w : x1]
        profile = patch.mean(dim=-1)
        row_guard, signal = _row_guard_from_profile(profile)
        guard_patch = torch.flip(_expand_row_guard(row_guard, left_band, merged.device, merged.dtype), dims=(-1,))
        merge_side_guard("right", guard_patch, signal, (x1 - left_band, y0, x1, y1))

    top_band = max(top_span, 0)
    if top_band > 0 and sample_h > 0:
        patch = delta_mag[:, :, y0 : y0 + sample_h, x0:x1]
        profile = patch.mean(dim=-2)
        col_guard, signal = _row_guard_from_profile(profile)
        guard_patch = _expand_col_guard(col_guard, top_band, merged.device, merged.dtype)
        merge_side_guard("top", guard_patch, signal, (x0, y0, x1, y0 + top_band))

        patch = delta_mag[:, :, y1 - sample_h : y1, x0:x1]
        profile = patch.mean(dim=-2)
        col_guard, signal = _row_guard_from_profile(profile)
        guard_patch = torch.flip(_expand_col_guard(col_guard, top_band, merged.device, merged.dtype), dims=(-2,))
        merge_side_guard("bottom", guard_patch, signal, (x0, y1 - top_band, x1, y1))

    full_guard = torch.where(
        acc_count > 0.0,
        (acc_guard / acc_count.clamp_min(1.0)).clamp(0.35, 1.0),
        torch.ones_like(mask),
    )
    return full_guard * mask + (1.0 - mask), side_guards, stats


def apply_corrector_to_full_frame(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: list[str],
    inner_width: int,
    strength: float = 1.0,
    structural_gate: bool = True,
    corner_disagreement_threshold: float = 0.03,
) -> tuple[torch.Tensor, dict]:
    if strength < 0.0 or strength > 10.0:
        raise RuntimeError("strength must be in [0, 10]")
    outputs = extract_active_strips(image[0], bbox, sides, inner_width)
    side_deltas: dict[str, torch.Tensor] = {}
    side_confidences: dict[str, torch.Tensor] = {}
    debug = {"per_side": {}, "corner_disagreement_threshold": float(corner_disagreement_threshold)}
    side_order = list(outputs.keys())
    if not side_order:
        return image, {"per_side": {}, "weights": {}, "side_deltas": {}, "merged_delta": torch.zeros_like(image)}
    outer_width = int(getattr(model, "outer_width", 128))
    strip_batch = torch.stack([outputs[side]["strip"] for side in side_order], dim=0)
    boundary_band_px = int(getattr(model, "boundary_band_px", 24))
    model_in = _canonical_model_input(strip_batch, outer_width, boundary_band_px=boundary_band_px).to(_model_device(model))
    with torch.inference_mode():
        model_out = model(model_in)
    if isinstance(model_out, dict):
        inner_delta = (model_out["corrected_inner"] - model_in[:, :3, :, outer_width:]).cpu()
        inner_confidence = model_out["confidence"].detach().cpu()
        corrected_strip_batch = model_out["corrected_strip"].detach().cpu()
        taper = _inner_taper(inner_delta.shape[-2], inner_delta.shape[-1], inner_delta.device, inner_delta.dtype)
        inner_delta = inner_delta * taper
        debug["architecture"] = "seam_harmonizer_v3"
        for key in ("gain_lowres", "gamma_lowres", "bias_lowres", "detail_lowres", "gate_lowres"):
            debug[key] = model_out[key].detach().cpu()
    else:
        raise RuntimeError("SeamHarmonizerV3 inference requires dict outputs with corrected_inner")
    for i, side in enumerate(side_order):
        delta_inner = inner_delta[i : i + 1]
        confidence_inner = inner_confidence[i : i + 1]
        canonical_strip = strip_batch[i : i + 1]
        corrected_strip = corrected_strip_batch[i : i + 1]
        side_deltas[side] = torch.zeros_like(image)
        x0, y0, x1, y1 = bbox
        meta = outputs[side]["meta"]
        side_scale = 1.0
        structural_score = None
        if structural_gate:
            side_scale, structural_score = _structural_strength_scale(outputs[side]["strip"], outer_width=outer_width)
            delta_inner = delta_inner * side_scale
        safety_gate, before_err, after_err = _build_safety_gate(
            canonical_strip,
            corrected_strip,
            outer_width,
            delta_inner.shape[-1],
            band_px=min(boundary_band_px, delta_inner.shape[-1]),
        )
        delta_inner = delta_inner * safety_gate
        confidence_inner = confidence_inner * safety_gate
        side_deltas[side] = _place_inner_map(delta_inner.to(image.device), image, bbox, side, inner_width, meta)
        side_confidences[side] = _place_inner_map(confidence_inner.to(image.device), image, bbox, side, inner_width, meta)
        debug["per_side"][side] = {
            "edge_padded_pixels": outputs[side]["meta"]["edge_padded_pixels"],
            "structural_scale": side_scale,
            "structural_grad_cosine": structural_score,
            "confidence_mean": float(confidence_inner.mean().item()),
            "delta_abs_mean": float(delta_inner.abs().mean().item()),
            "before_error_mean": float(before_err.mean().item()),
            "after_error_mean": float(after_err.mean().item()),
            "safety_gate_mean": float(safety_gate.mean().item()),
        }
        debug.setdefault("side_safety_gates", {})[side] = _place_inner_map(safety_gate.to(image.device), image, bbox, side, inner_width, meta)
        debug.setdefault("side_before_error", {})[side] = _place_inner_map(_expand_band_map(before_err, inner_width).to(image.device), image, bbox, side, inner_width, meta)
        debug.setdefault("side_after_error", {})[side] = _place_inner_map(_expand_band_map(after_err, inner_width).to(image.device), image, bbox, side, inner_width, meta)
    merged, weights = merge_side_deltas(
        side_deltas,
        mask,
        side_confidences=side_confidences,
        bbox=bbox,
        inner_width=inner_width,
        corner_disagreement_threshold=corner_disagreement_threshold,
    )
    profile_guard, side_profile_guards, profile_stats = _build_profile_guard(
        merged,
        mask,
        bbox,
        band_px=min(boundary_band_px, inner_width),
        inner_width=inner_width,
    )
    merged = merged * profile_guard
    for side, side_stats in profile_stats.items():
        debug["per_side"].setdefault(side, {}).update(side_stats)
    corrected = (image + merged * strength).clamp(0.0, 1.0)
    corrected = corrected * mask + image * (1.0 - mask)
    max_diff = ((corrected * (1.0 - mask)) - (image * (1.0 - mask))).abs().max().item()
    if max_diff >= 1e-6:
        raise AssertionError(f"outer hard-copy violated: max_diff={max_diff}")
    debug["weights"] = weights
    debug["side_deltas"] = side_deltas
    debug["side_confidences"] = side_confidences
    debug["profile_guard"] = profile_guard
    debug["side_profile_guards"] = side_profile_guards
    debug["merged_delta"] = merged
    return corrected, debug
