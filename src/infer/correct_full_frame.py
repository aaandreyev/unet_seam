from __future__ import annotations

import torch

from src.infer.extract_strips import extract_active_strips
from src.infer.merge_residuals import merge_side_residuals


def apply_corrector_to_full_frame(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: list[str],
    inner_width: int,
    strength: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if strength < 0.0 or strength > 1.0:
        raise RuntimeError("strength must be in [0, 1]")
    outputs = extract_active_strips(image[0], bbox, sides, inner_width)
    side_residuals: dict[str, torch.Tensor] = {}
    debug = {"per_side": {}}
    for side, payload in outputs.items():
        strip = payload["strip"].unsqueeze(0)
        seam_x = 128
        mask_ch = (torch.arange(strip.shape[-1], device=strip.device).view(1, 1, 1, -1) >= seam_x).float().expand(1, 1, strip.shape[-2], strip.shape[-1])
        dist = (torch.arange(strip.shape[-1], device=strip.device).view(1, 1, 1, -1) - seam_x).abs() / float(max(seam_x, strip.shape[-1] - seam_x))
        dist = dist.expand(1, 1, strip.shape[-2], strip.shape[-1])
        model_in = torch.cat([strip, mask_ch, dist], dim=1).to(next(model.parameters()).device)
        residual = model(model_in).cpu()
        side_residuals[side] = torch.zeros_like(image)
        x0, y0, x1, y1 = bbox
        if side == "left":
            width = min(inner_width, x1 - x0)
            side_residuals[side][:, :, y0:y1, x0 : x0 + width] = residual[:, :, :, 128 : 128 + width]
        elif side == "right":
            width = min(inner_width, x1 - x0)
            side_residuals[side][:, :, y0:y1, x1 - width : x1] = torch.flip(residual[:, :, :, 128 : 128 + width], dims=(-1,))
        elif side == "top":
            height = min(inner_width, y1 - y0)
            top_residual = torch.rot90(residual[:, :, :, 128 : 128 + height], k=3, dims=(-2, -1))
            side_residuals[side][:, :, y0 : y0 + height, x0:x1] = top_residual[:, :, :height, : x1 - x0]
        elif side == "bottom":
            height = min(inner_width, y1 - y0)
            bottom_residual = torch.rot90(residual[:, :, :, 128 : 128 + height], k=1, dims=(-2, -1))
            side_residuals[side][:, :, y1 - height : y1, x0:x1] = bottom_residual[:, :, :height, : x1 - x0]
        debug["per_side"][side] = {"edge_padded_pixels": payload["meta"]["edge_padded_pixels"]}
    merged, weights = merge_side_residuals(side_residuals, mask)
    corrected = (image + merged * strength).clamp(0.0, 1.0)
    corrected = corrected * mask + image * (1.0 - mask)
    max_diff = ((corrected * (1.0 - mask)) - (image * (1.0 - mask))).abs().max().item()
    if max_diff >= 1e-6:
        raise AssertionError(f"outer hard-copy violated: max_diff={max_diff}")
    debug["weights"] = weights
    debug["side_residuals"] = side_residuals
    debug["merged_residual"] = merged
    return corrected, debug
