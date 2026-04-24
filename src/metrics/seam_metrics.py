from __future__ import annotations

import numpy as np
import torch

from src.metrics.deltae import boundary_ciede2000
from src.metrics.lowfreq_metrics import lowfreq_mae
from src.models.blocks import gaussian_blur_tensor


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(0, 2, 3, 1).numpy()


def match_lowfreq(input_rgb: torch.Tensor, target: torch.Tensor, inner_mask: torch.Tensor, sigma: float = 16.0) -> torch.Tensor:
    correction = gaussian_blur_tensor(target - input_rgb, sigma=sigma)
    return (input_rgb + correction * inner_mask).clamp(0.0, 1.0)


def _boundary_mae(pred: torch.Tensor, target: torch.Tensor, boundary_band: torch.Tensor) -> float:
    return float(((pred - target).abs() * boundary_band).mean().item())


def _residual_abs_p99(residual: torch.Tensor, max_elements: int = 4_000_000) -> float:
    """p99 of |residual|; subsample when flat size exceeds torch.quantile limits (large batch × image)."""
    v = residual.abs().detach().float().reshape(-1)
    n = v.numel()
    if n == 0:
        return 0.0
    if n > max_elements:
        idx = torch.randint(0, n, (max_elements,), device=v.device, dtype=torch.long)
        v = v[idx]
    return float(torch.quantile(v, 0.99).item())


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, input_rgb: torch.Tensor, inner_mask: torch.Tensor, boundary_band: torch.Tensor, residual: torch.Tensor) -> dict[str, float]:
    pred_np = _to_numpy(pred)[0]
    target_np = _to_numpy(target)[0]
    input_np = _to_numpy(input_rgb)[0]
    boundary_np = _to_numpy(boundary_band)[0][..., :1]
    baseline_boundary = boundary_ciede2000(input_np, target_np, boundary_np)
    oracle = match_lowfreq(input_rgb, target, inner_mask, sigma=16.0)
    oracle_np = _to_numpy(oracle)[0]
    oracle_boundary = boundary_ciede2000(oracle_np, target_np, boundary_np)
    pred_boundary = boundary_ciede2000(pred_np, target_np, boundary_np)
    metrics = {
        "boundary_ciede2000": pred_boundary,
        "boundary_mae": _boundary_mae(pred, target, boundary_band),
        "inner_mae": float(((pred - target).abs() * inner_mask).mean().item()),
        "outer_identity_error": float(((pred - input_rgb).abs() * (1.0 - inner_mask)).max().item()),
        "lowfreq_mae_sigma16": float(lowfreq_mae(pred, target, sigma=16.0).item()),
        "residual_magnitude_p99": _residual_abs_p99(residual),
        "baseline_boundary_ciede2000": baseline_boundary,
        "baseline_boundary_mae": _boundary_mae(input_rgb, target, boundary_band),
        "oracle_boundary_ciede2000": oracle_boundary,
        "relative_improvement": float((baseline_boundary - pred_boundary) / (baseline_boundary - oracle_boundary + 1e-8)),
    }
    return metrics
