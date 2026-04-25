from __future__ import annotations

import numpy as np
import torch

from src.metrics.deltae import boundary_ciede2000
from src.models.blocks import gaussian_blur_tensor


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(0, 2, 3, 1).numpy()


def _inner(x: torch.Tensor, outer_width: int = 128) -> torch.Tensor:
    return x[..., outer_width:]


def _mae_band(pred: torch.Tensor, target: torch.Tensor, width: int) -> float:
    return float((pred[..., :width] - target[..., :width]).abs().mean().item())


def _grad_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    dxp = pred[..., :, 1:] - pred[..., :, :-1]
    dyp = pred[..., 1:, :] - pred[..., :-1, :]
    dxt = target[..., :, 1:] - target[..., :, :-1]
    dyt = target[..., 1:, :] - target[..., :-1, :]
    return float((dxp - dxt).abs().mean().item() + (dyp - dyt).abs().mean().item())


def evaluate_harmonizer_batch(
    corrected_strip: torch.Tensor,
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    curves: torch.Tensor,
    shading: torch.Tensor,
    outer_width: int = 128,
) -> dict[str, float]:
    pred_inner = _inner(corrected_strip, outer_width)
    target_inner = _inner(target, outer_width)
    input_inner = _inner(input_rgb, outer_width)
    pred_np = _to_numpy(pred_inner)
    target_np = _to_numpy(target_inner)
    input_np = _to_numpy(input_inner)
    de16 = []
    de32 = []
    base_de16 = []
    base_de32 = []
    for i in range(pred_np.shape[0]):
        mask16 = np.zeros((*pred_np.shape[1:3], 1), dtype=np.float32)
        mask32 = np.zeros_like(mask16)
        mask16[:, :16, :] = 1.0
        mask32[:, :32, :] = 1.0
        de16.append(boundary_ciede2000(pred_np[i], target_np[i], mask16))
        de32.append(boundary_ciede2000(pred_np[i], target_np[i], mask32))
        base_de16.append(boundary_ciede2000(input_np[i], target_np[i], mask16))
        base_de32.append(boundary_ciede2000(input_np[i], target_np[i], mask32))
    low_pred = gaussian_blur_tensor(pred_inner, 5.0)
    low_target = gaussian_blur_tensor(target_inner, 5.0)
    slopes = curves[..., 1:] - curves[..., :-1]
    return {
        "boundary_mae_8": _mae_band(pred_inner, target_inner, 8),
        "boundary_mae_16": _mae_band(pred_inner, target_inner, 16),
        "boundary_mae_32": _mae_band(pred_inner, target_inner, 32),
        "baseline_boundary_mae_16": _mae_band(input_inner, target_inner, 16),
        "boundary_ciede2000_16": float(np.mean(de16)),
        "boundary_ciede2000_32": float(np.mean(de32)),
        "baseline_boundary_ciede2000_16": float(np.mean(base_de16)),
        "baseline_boundary_ciede2000_32": float(np.mean(base_de32)),
        "lowfreq_mae": float((low_pred - low_target).abs().mean().item()),
        "gradient_mae": _grad_mae(pred_inner, target_inner),
        "curve_max_slope": float(slopes.max().item()),
        "curve_min_slope": float(slopes.min().item()),
        "shading_abs_mean": float(shading.abs().mean().item()),
    }
