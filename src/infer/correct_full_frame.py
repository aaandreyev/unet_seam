from __future__ import annotations

import torch

from src.data.structural_filter import gradient_cosine_similarity
from src.infer.extract_strips import extract_active_strips
from src.infer.merge_bands import merge_side_deltas


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _canonical_model_input(strip_batch: torch.Tensor, outer_width: int) -> torch.Tensor:
    _, _, height, width = strip_batch.shape
    inner_width = width - outer_width
    xs = torch.arange(width, device=strip_batch.device, dtype=strip_batch.dtype).view(1, 1, 1, width)
    mask = (xs >= outer_width).to(strip_batch.dtype).expand(strip_batch.shape[0], 1, height, width)
    inner_u = (xs - outer_width).clamp(min=0.0)
    distance = (inner_u / float(max(inner_width - 1, 1))).clamp(0.0, 1.0)
    distance = distance.expand(strip_batch.shape[0], 1, height, width)
    return torch.cat([strip_batch, mask, distance], dim=1)


def _inner_taper(height: int, inner_width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if inner_width <= 1:
        return torch.ones(1, 1, height, inner_width, device=device, dtype=dtype)
    u = torch.arange(inner_width, device=device, dtype=dtype).view(1, 1, 1, inner_width)
    taper = 0.5 * (1.0 + torch.cos(torch.pi * u / float(inner_width - 1)))
    return taper.expand(1, 1, height, inner_width)


def _structural_strength_scale(strip: torch.Tensor, outer_width: int = 128, band_px: int = 16) -> tuple[float, float]:
    band = min(band_px, outer_width, strip.shape[-1] - outer_width)
    if band <= 0:
        return 1.0, 1.0
    outer_band = torch.flip(strip[..., outer_width - band : outer_width], dims=(-1,)).unsqueeze(0)
    inner_band = strip[..., outer_width : outer_width + band].unsqueeze(0)
    score = float(gradient_cosine_similarity(outer_band, inner_band).item())
    if score < 0.35:
        return 0.0, score
    if score < 0.5:
        return 0.5, score
    return 1.0, score


def apply_corrector_to_full_frame(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: list[str],
    inner_width: int,
    strength: float = 1.0,
    structural_gate: bool = True,
) -> tuple[torch.Tensor, dict]:
    if strength < 0.0 or strength > 1.0:
        raise RuntimeError("strength must be in [0, 1]")
    outputs = extract_active_strips(image[0], bbox, sides, inner_width)
    side_deltas: dict[str, torch.Tensor] = {}
    debug = {"per_side": {}}
    side_order = list(outputs.keys())
    if not side_order:
        return image, {"per_side": {}, "weights": {}, "side_deltas": {}, "merged_delta": torch.zeros_like(image)}
    outer_width = 128
    strip_batch = torch.stack([outputs[side]["strip"] for side in side_order], dim=0)
    model_in = _canonical_model_input(strip_batch, outer_width).to(_model_device(model))
    with torch.inference_mode():
        model_out = model(model_in)
    if isinstance(model_out, dict):
        inner_delta = (model_out["corrected_inner"] - model_in[:, :3, :, outer_width:]).cpu()
        taper = _inner_taper(inner_delta.shape[-2], inner_delta.shape[-1], inner_delta.device, inner_delta.dtype)
        inner_delta = inner_delta * taper
        debug["architecture"] = "seam_harmonizer_v1"
        debug["curves"] = model_out["curves"].detach().cpu()
        debug["shading_lowres"] = model_out["shading_lowres"].detach().cpu()
    else:
        raise RuntimeError("SeamHarmonizerV1 inference requires dict outputs with corrected_inner")
    for i, side in enumerate(side_order):
        delta_inner = inner_delta[i : i + 1]
        side_deltas[side] = torch.zeros_like(image)
        x0, y0, x1, y1 = bbox
        meta = outputs[side]["meta"]
        side_scale = 1.0
        structural_score = None
        if structural_gate:
            side_scale, structural_score = _structural_strength_scale(outputs[side]["strip"], outer_width=outer_width)
            delta_inner = delta_inner * side_scale
        if side == "left":
            width = min(inner_width, x1 - x0)
            y_start = int(meta["y_start"])
            y_end = min(y_start + delta_inner.shape[-2], image.shape[-2])
            side_deltas[side][:, :, y_start:y_end, x0 : x0 + width] = delta_inner[:, :, : y_end - y_start, :width]
        elif side == "right":
            width = min(inner_width, x1 - x0)
            y_start = int(meta["y_start"])
            y_end = min(y_start + delta_inner.shape[-2], image.shape[-2])
            side_deltas[side][:, :, y_start:y_end, x1 - width : x1] = torch.flip(delta_inner[:, :, : y_end - y_start, :width], dims=(-1,))
        elif side == "top":
            height = min(inner_width, y1 - y0)
            top_delta = torch.rot90(delta_inner[:, :, :, :height], k=3, dims=(-2, -1))
            x_start = int(meta["x_start"])
            x_end = min(x_start + top_delta.shape[-1], image.shape[-1])
            side_deltas[side][:, :, y0 : y0 + height, x_start:x_end] = top_delta[:, :, :height, : x_end - x_start]
        elif side == "bottom":
            height = min(inner_width, y1 - y0)
            bottom_delta = torch.rot90(delta_inner[:, :, :, :height], k=1, dims=(-2, -1))
            x_start = int(meta["x_start"])
            x_end = min(x_start + bottom_delta.shape[-1], image.shape[-1])
            side_deltas[side][:, :, y1 - height : y1, x_start:x_end] = bottom_delta[:, :, :height, : x_end - x_start]
        debug["per_side"][side] = {
            "edge_padded_pixels": outputs[side]["meta"]["edge_padded_pixels"],
            "structural_scale": side_scale,
            "structural_grad_cosine": structural_score,
        }
    merged, weights = merge_side_deltas(side_deltas, mask)
    corrected = (image + merged * strength).clamp(0.0, 1.0)
    corrected = corrected * mask + image * (1.0 - mask)
    max_diff = ((corrected * (1.0 - mask)) - (image * (1.0 - mask))).abs().max().item()
    if max_diff >= 1e-6:
        raise AssertionError(f"outer hard-copy violated: max_diff={max_diff}")
    debug["weights"] = weights
    debug["side_deltas"] = side_deltas
    debug["merged_delta"] = merged
    return corrected, debug
