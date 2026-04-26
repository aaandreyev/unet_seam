from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.blocks import gaussian_blur_tensor
from src.models.harmonizer_blocks import tv_loss


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff.square() + eps * eps)


_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = x.shape[1]
    kx = _SOBEL_X.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
    ky = _SOBEL_Y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kx, groups=c)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), ky, groups=c)
    return gx, gy


def _inner(x: torch.Tensor, outer_width: int) -> torch.Tensor:
    return x[..., outer_width:]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def _masked_mean_or_zero(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if float(mask.sum().detach().item()) <= eps:
        return x.new_zeros(())
    return _masked_mean(x, mask, eps=eps)


def _color_opponent(x: torch.Tensor) -> torch.Tensor:
    rg = x[:, 0:1] - x[:, 1:2]
    yb = 0.5 * (x[:, 0:1] + x[:, 1:2]) - x[:, 2:3]
    return torch.cat([rg, yb], dim=1)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(1e-6).pow(2.4))


def _rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    x = _srgb_to_linear(x.clamp(0.0, 1.0))
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    xx = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    yy = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    zz = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, t.clamp_min(1e-6).pow(1.0 / 3.0), t / (3.0 * delta**2) + 4.0 / 29.0)

    fx = f(xx / xn)
    fy = f(yy / yn)
    fz = f(zz / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    # Normalize channels so L/a/b contribute on comparable scales.
    return torch.cat([L / 100.0, a / 110.0, b / 110.0], dim=1)


class HarmonizerLossComputer:
    def __init__(
        self,
        outer_width: int = 128,
        low_sigma: float = 5.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.outer_width = outer_width
        self.low_sigma = low_sigma
        default_weights = {
            "rec": 0.8,
            "seam": 1.1,
            "low": 1.2,
            "grad": 0.25,
            "chroma": 0.6,
            "stats": 0.35,
            "gate": 0.04,
            "field": 0.10,
            "detail": 0.10,
            "matrix": 0.10,
            "lab": 0.40,
        }
        self.weights = default_weights | (weights or {})

    def __call__(self, outputs: dict[str, torch.Tensor], batch_or_target: dict | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(batch_or_target, dict):
            batch = batch_or_target
            target_strip = batch["target"]
            boundary = _inner(batch["boundary_band_mask"], self.outer_width)
            decay = _inner(batch["decay_mask"], self.outer_width)
        else:
            target_strip = batch_or_target
            boundary = torch.ones_like(target_strip[:, :1, :, self.outer_width:])
            decay = boundary
        pred = outputs["corrected_inner"]
        target = _inner(target_strip, self.outer_width)
        full_mask = torch.ones_like(boundary)
        inner_weight = (0.15 + 0.85 * decay).clamp(0.0, 1.0)
        outside_boundary = (1.0 - boundary).clamp(0.0, 1.0)
        rec_map = charbonnier(pred - target)
        l_rec = rec_map.mean()
        seam_weight = boundary * (0.35 + 0.65 * decay)
        l_seam = _masked_mean(rec_map.mean(dim=1, keepdim=True), seam_weight)
        low_pred = gaussian_blur_tensor(pred, self.low_sigma)
        low_target = gaussian_blur_tensor(target, self.low_sigma)
        l_low = (low_pred - low_target).abs().mean()
        gp = sobel_gradients(pred)
        gt = sobel_gradients(target)
        grad_err = (gp[0] - gt[0]).abs() + (gp[1] - gt[1]).abs()
        l_grad = _masked_mean(grad_err.mean(dim=1, keepdim=True), full_mask)
        chroma_pred = _color_opponent(pred)
        chroma_target = _color_opponent(target)
        chroma_err = (chroma_pred - chroma_target).abs().mean(dim=1, keepdim=True)
        l_chroma = 0.7 * _masked_mean(chroma_err, seam_weight) + 0.3 * _masked_mean(chroma_err, inner_weight)
        stats_dims = (-2, -1)
        def _stats_term(mask: torch.Tensor) -> torch.Tensor:
            pred_mean = (pred * mask).sum(dim=stats_dims) / mask.sum(dim=stats_dims).clamp_min(1e-6)
            target_mean = (target * mask).sum(dim=stats_dims) / mask.sum(dim=stats_dims).clamp_min(1e-6)
            pred_centered = (pred - pred_mean.unsqueeze(-1).unsqueeze(-1)) * mask
            target_centered = (target - target_mean.unsqueeze(-1).unsqueeze(-1)) * mask
            pred_std = torch.sqrt(pred_centered.square().sum(dim=stats_dims) / mask.sum(dim=stats_dims).clamp_min(1e-6) + 1e-6)
            target_std = torch.sqrt(target_centered.square().sum(dim=stats_dims) / mask.sum(dim=stats_dims).clamp_min(1e-6) + 1e-6)
            return (pred_mean - target_mean).abs().mean() + (pred_std - target_std).abs().mean()

        l_stats = 0.65 * _stats_term(seam_weight) + 0.35 * _stats_term(inner_weight)
        lab_pred = _rgb_to_lab(pred)
        lab_target = _rgb_to_lab(target)
        lab_low_pred = gaussian_blur_tensor(lab_pred, self.low_sigma)
        lab_low_target = gaussian_blur_tensor(lab_target, self.low_sigma)
        lab_err = (lab_pred - lab_target).abs().mean(dim=1, keepdim=True)
        l_lab = 0.65 * _masked_mean(lab_err, seam_weight) + 0.35 * (lab_low_pred - lab_low_target).abs().mean()
        l_gate = (
            0.5 * _masked_mean(outputs["confidence"], boundary)
            + 1.5 * _masked_mean_or_zero(outputs["confidence"], outside_boundary)
            + 0.25 * tv_loss(outputs["confidence"])
        )
        l_field = (
            tv_loss(outputs["gain_lowres"])
            + tv_loss(outputs["gamma_lowres"])
            + tv_loss(outputs["bias_lowres"])
            + tv_loss(outputs["detail_lowres"])
            + tv_loss(outputs["gate_lowres"])
            + outputs["mix_lowres"].diff(dim=-1).abs().mean()
            + outputs["mix_lowres"].diff(dim=-2).abs().mean()
        )
        identity = torch.eye(3, device=pred.device, dtype=pred.dtype).view(1, 3, 3, 1, 1)
        l_matrix = (outputs["color_matrix"] - identity).abs().mean() + outputs["bias"].abs().mean()
        detail_abs = outputs["detail"].abs().mean(dim=1, keepdim=True)
        l_detail = 0.5 * _masked_mean(detail_abs, boundary) + 1.25 * _masked_mean_or_zero(detail_abs, outside_boundary)
        w = self.weights
        total = (
            w["rec"] * l_rec
            + w["seam"] * l_seam
            + w["low"] * l_low
            + w["grad"] * l_grad
            + w["chroma"] * l_chroma
            + w["stats"] * l_stats
            + w["lab"] * l_lab
            + w["gate"] * l_gate
            + w["field"] * l_field
            + w["detail"] * l_detail
            + w["matrix"] * l_matrix
        )
        return {
            "total": total,
            "l_rec": l_rec,
            "l_seam": l_seam,
            "l_low": l_low,
            "l_grad": l_grad,
            "l_chroma": l_chroma,
            "l_stats": l_stats,
            "l_lab": l_lab,
            "l_gate": l_gate,
            "l_field": l_field,
            "l_detail": l_detail,
            "l_matrix": l_matrix,
        }
