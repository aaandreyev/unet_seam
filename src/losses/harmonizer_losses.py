from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.blocks import gaussian_blur_tensor
from src.models.harmonizer_blocks import tv_loss


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff.square() + eps * eps)


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernels_x = kernel_x.repeat(x.shape[1], 1, 1, 1)
    kernels_y = kernel_y.repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_x, groups=x.shape[1])
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_y, groups=x.shape[1])
    return gx, gy


def _inner(x: torch.Tensor, outer_width: int) -> torch.Tensor:
    return x[..., outer_width:]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def _color_opponent(x: torch.Tensor) -> torch.Tensor:
    rg = x[:, 0:1] - x[:, 1:2]
    yb = 0.5 * (x[:, 0:1] + x[:, 1:2]) - x[:, 2:3]
    return torch.cat([rg, yb], dim=1)


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
            "rec": 1.0,
            "seam": 1.5,
            "low": 1.0,
            "grad": 0.35,
            "chroma": 0.25,
            "stats": 0.15,
            "gate": 0.02,
            "field": 0.05,
            "detail": 0.05,
            "matrix": 0.05,
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
        l_chroma = _masked_mean((chroma_pred - chroma_target).abs().mean(dim=1, keepdim=True), seam_weight)
        stats_dims = (-2, -1)
        pred_mean = (pred * seam_weight).sum(dim=stats_dims) / seam_weight.sum(dim=stats_dims).clamp_min(1e-6)
        target_mean = (target * seam_weight).sum(dim=stats_dims) / seam_weight.sum(dim=stats_dims).clamp_min(1e-6)
        pred_centered = (pred - pred_mean.unsqueeze(-1).unsqueeze(-1)) * seam_weight
        target_centered = (target - target_mean.unsqueeze(-1).unsqueeze(-1)) * seam_weight
        pred_std = torch.sqrt(pred_centered.square().sum(dim=stats_dims) / seam_weight.sum(dim=stats_dims).clamp_min(1e-6) + 1e-6)
        target_std = torch.sqrt(target_centered.square().sum(dim=stats_dims) / seam_weight.sum(dim=stats_dims).clamp_min(1e-6) + 1e-6)
        l_stats = (pred_mean - target_mean).abs().mean() + (pred_std - target_std).abs().mean()
        l_gate = outputs["confidence"].mean() + 0.25 * tv_loss(outputs["confidence"])
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
        l_detail = outputs["detail"].abs().mean()
        w = self.weights
        total = (
            w["rec"] * l_rec
            + w["seam"] * l_seam
            + w["low"] * l_low
            + w["grad"] * l_grad
            + w["chroma"] * l_chroma
            + w["stats"] * l_stats
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
            "l_gate": l_gate,
            "l_field": l_field,
            "l_detail": l_detail,
            "l_matrix": l_matrix,
        }
