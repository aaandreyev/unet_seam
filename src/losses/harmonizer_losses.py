from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.blocks import gaussian_blur_tensor
from src.models.harmonizer_blocks import tv_loss


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernels_x = kernel_x.repeat(x.shape[1], 1, 1, 1)
    kernels_y = kernel_y.repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_x, groups=x.shape[1])
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_y, groups=x.shape[1])
    return gx, gy


def _inner(target: torch.Tensor, outer_width: int) -> torch.Tensor:
    return target[..., outer_width:]


def seam_weight(height: int, inner_width: int, tau: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u = torch.arange(inner_width, device=device, dtype=dtype).view(1, 1, 1, inner_width)
    w = torch.exp(-u / tau)
    w = torch.where(u < 32, w, torch.zeros_like(w))
    return w.expand(1, 1, height, inner_width)


def curve_smoothness_loss(curves: torch.Tensor) -> torch.Tensor:
    d1 = curves[..., 1:] - curves[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]
    return d2.abs().mean()


def curve_identity_loss(curves: torch.Tensor) -> torch.Tensor:
    k = curves.shape[-1]
    identity = torch.linspace(0.0, 1.0, k, device=curves.device, dtype=curves.dtype).view(1, 1, k)
    return (curves - identity).abs().mean()


class HarmonizerLossComputer:
    def __init__(
        self,
        outer_width: int = 128,
        seam_tau: float = 12.0,
        low_sigma: float = 5.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.outer_width = outer_width
        self.seam_tau = seam_tau
        self.low_sigma = low_sigma
        default_weights = {
            "rec": 1.0,
            "seam": 2.0,
            "low": 0.75,
            "grad": 0.25,
            "curve_smooth": 0.02,
            "curve_id": 0.01,
            "tv": 0.02,
            "mean": 0.01,
        }
        self.weights = default_weights | (weights or {})

    def __call__(self, outputs: dict[str, torch.Tensor], target_strip: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = outputs["corrected_inner"]
        target = _inner(target_strip, self.outer_width)
        height, inner_width = pred.shape[-2:]
        w_seam = seam_weight(height, inner_width, self.seam_tau, pred.device, pred.dtype)
        l_rec = charbonnier(pred - target).mean()
        l_seam = (charbonnier(pred - target) * w_seam).sum() / (w_seam.sum() * pred.shape[1]).clamp_min(1.0)
        l_low = (gaussian_blur_tensor(pred, self.low_sigma) - gaussian_blur_tensor(target, self.low_sigma)).abs().mean()
        gp = sobel_gradients(pred)
        gt = sobel_gradients(target)
        l_grad = (gp[0] - gt[0]).abs().mean() + (gp[1] - gt[1]).abs().mean()
        l_curve_smooth = curve_smoothness_loss(outputs["curves"])
        l_curve_id = curve_identity_loss(outputs["curves"])
        l_tv = tv_loss(outputs["shading"])
        l_mean = outputs["shading"].mean().abs()
        w = self.weights
        total = (
            w["rec"] * l_rec
            + w["seam"] * l_seam
            + w["low"] * l_low
            + w["grad"] * l_grad
            + w["curve_smooth"] * l_curve_smooth
            + w["curve_id"] * l_curve_id
            + w["tv"] * l_tv
            + w["mean"] * l_mean
        )
        return {
            "total": total,
            "l_rec": l_rec,
            "l_seam": l_seam,
            "l_low": l_low,
            "l_grad": l_grad,
            "l_curve_smooth": l_curve_smooth,
            "l_curve_id": l_curve_id,
            "l_tv": l_tv,
            "l_mean": l_mean,
        }
