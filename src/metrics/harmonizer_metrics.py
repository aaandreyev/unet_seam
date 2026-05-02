from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.deltae import boundary_ciede2000
from src.models.blocks import gaussian_blur_tensor


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(0, 2, 3, 1).numpy()


def _inner(x: torch.Tensor, outer_width: int = 128) -> torch.Tensor:
    return x[..., outer_width:]


def _mae_band_mean(pred: torch.Tensor, target: torch.Tensor, width: int) -> float:
    return (pred[..., :width] - target[..., :width]).abs().mean().item()


def _grad_mae_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    dxp = pred[..., :, 1:] - pred[..., :, :-1]
    dyp = pred[..., 1:, :] - pred[..., :-1, :]
    dxt = target[..., :, 1:] - target[..., :, :-1]
    dyt = target[..., 1:, :] - target[..., :-1, :]
    return (dxp - dxt).abs().mean().item() + (dyp - dyt).abs().mean().item()


def _luma(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def _row_profile_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    row_pred = pred.mean(dim=-1)
    row_target = target.mean(dim=-1)
    return (row_pred - row_target).abs().mean().item()


def _extract_aux(outputs_or_curves, shading: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(outputs_or_curves, dict):
        confidence = outputs_or_curves.get("confidence")
        detail = outputs_or_curves.get("detail")
        gain = outputs_or_curves.get("gain")
        return confidence, detail, gain
    return None, None, shading


def _harmonizer_metrics_torch(
    pred_inner: torch.Tensor,
    target_inner: torch.Tensor,
    input_inner: torch.Tensor,
    outputs_or_curves: dict[str, torch.Tensor] | torch.Tensor,
    shading: torch.Tensor | None,
) -> dict[str, float]:
    low_pred = gaussian_blur_tensor(pred_inner, 5.0)
    low_target = gaussian_blur_tensor(target_inner, 5.0)
    delta_pred = pred_inner - input_inner
    delta_target = target_inner - input_inner
    # Signed blur for profile metrics (captures low-frequency drift direction).
    low_delta_pred = gaussian_blur_tensor(delta_pred, 7.0)
    low_delta_target = gaussian_blur_tensor(delta_target, 7.0)
    # abs-then-blur for overcorrection (matches loss computation; no sign cancellation across channels).
    # Uses sigma=5.0 to match HarmonizerLossComputer.low_sigma default.
    overcorr_pred_mag = gaussian_blur_tensor(delta_pred.abs(), 5.0).mean(dim=1, keepdim=True)
    overcorr_target_mag = gaussian_blur_tensor(delta_target.abs(), 5.0).mean(dim=1, keepdim=True)
    confidence, detail, gain = _extract_aux(outputs_or_curves, shading)
    out = {
        "boundary_mae_8": _mae_band_mean(pred_inner, target_inner, 8),
        "boundary_mae_16": _mae_band_mean(pred_inner, target_inner, 16),
        "boundary_mae_32": _mae_band_mean(pred_inner, target_inner, 32),
        "baseline_boundary_mae_16": _mae_band_mean(input_inner, target_inner, 16),
        "lowfreq_mae": (low_pred - low_target).abs().mean().item(),
        "gradient_mae": _grad_mae_mean(pred_inner, target_inner),
        "delta_luma_profile_mae": _row_profile_mae(_luma(low_delta_pred), _luma(low_delta_target)),
        "delta_chroma_profile_mae": _row_profile_mae(low_delta_pred[:, 0:1] - low_delta_pred[:, 1:2], low_delta_target[:, 0:1] - low_delta_target[:, 1:2]),
        "overcorrection_mae": (overcorr_pred_mag - overcorr_target_mag).clamp_min(0.0).mean().item(),
    }
    if confidence is not None:
        out["confidence_mean"] = confidence.mean().item()
        # NOTE: confidence_alignment_mae compares the gate (amplitude head) against a fixed-0.08
        # normalization. The attn loss trains attention_lowres (separate head) against 85th-percentile
        # normalization. These are unrelated: this metric grows monotonically as the gate becomes more
        # spatially modulated. It is kept for backward-compat with dashboards but should not be used
        # for checkpoint selection or as a progress indicator.
        target_conf = ((target_inner - input_inner).abs().mean(dim=1, keepdim=True) / 0.08).clamp(0.0, 1.0)
        out["confidence_alignment_mae"] = (confidence - target_conf).abs().mean().item()
    if detail is not None:
        out["detail_abs_mean"] = detail.abs().mean().item()
    if gain is not None:
        out["gain_abs_log_mean"] = gain.clamp_min(1e-6).log().abs().mean().item()
    # Attention alignment: compares attention_lowres against the same percentile-normalized target
    # used by the attn loss — the only reliable "where-to-act" quality signal.
    if isinstance(outputs_or_curves, dict):
        attn_lowres = outputs_or_curves.get("attention_lowres")
        if attn_lowres is not None:
            attn_full = F.interpolate(attn_lowres, size=pred_inner.shape[-2:], mode="bilinear", align_corners=False)
            delta_abs = gaussian_blur_tensor((target_inner - input_inner).abs(), 5.0).mean(dim=1, keepdim=True)
            delta_abs_f = delta_abs.float() if delta_abs.dtype not in (torch.float32, torch.float64) else delta_abs
            norm = torch.quantile(delta_abs_f.flatten(start_dim=1), 0.85, dim=1).view(-1, 1, 1, 1).clamp_min(0.03)
            norm = norm.to(device=delta_abs.device, dtype=delta_abs.dtype)
            target_attn = (delta_abs / norm).clamp(0.0, 1.0)
            out["attention_alignment_mae"] = (attn_full - target_attn).abs().mean().item()
    return out


def evaluate_harmonizer_batch_fast(
    corrected_strip: torch.Tensor,
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    outputs_or_curves: dict[str, torch.Tensor] | torch.Tensor,
    shading: torch.Tensor | None = None,
    outer_width: int = 128,
) -> dict[str, float]:
    pred_inner = _inner(corrected_strip, outer_width)
    target_inner = _inner(target, outer_width)
    input_inner = _inner(input_rgb, outer_width)
    return _harmonizer_metrics_torch(pred_inner, target_inner, input_inner, outputs_or_curves, shading)


def evaluate_harmonizer_batch(
    corrected_strip: torch.Tensor,
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    outputs_or_curves: dict[str, torch.Tensor] | torch.Tensor,
    shading: torch.Tensor | None = None,
    outer_width: int = 128,
) -> dict[str, float]:
    pred_inner = _inner(corrected_strip, outer_width)
    target_inner = _inner(target, outer_width)
    input_inner = _inner(input_rgb, outer_width)
    out = _harmonizer_metrics_torch(pred_inner, target_inner, input_inner, outputs_or_curves, shading)
    pred_np = _to_numpy(pred_inner)
    target_np = _to_numpy(target_inner)
    input_np = _to_numpy(input_inner)
    de16: list[float] = []
    de32: list[float] = []
    base_de16: list[float] = []
    base_de32: list[float] = []
    for i in range(pred_np.shape[0]):
        mask16 = np.zeros((*pred_np.shape[1:3], 1), dtype=np.float32)
        mask32 = np.zeros_like(mask16)
        mask16[:, :16, :] = 1.0
        mask32[:, :32, :] = 1.0
        de16.append(boundary_ciede2000(pred_np[i], target_np[i], mask16))
        de32.append(boundary_ciede2000(pred_np[i], target_np[i], mask32))
        base_de16.append(boundary_ciede2000(input_np[i], target_np[i], mask16))
        base_de32.append(boundary_ciede2000(input_np[i], target_np[i], mask32))
    out["boundary_ciede2000_16"] = float(np.mean(de16))
    out["boundary_ciede2000_32"] = float(np.mean(de32))
    out["baseline_boundary_ciede2000_16"] = float(np.mean(base_de16))
    out["baseline_boundary_ciede2000_32"] = float(np.mean(base_de32))
    return out
