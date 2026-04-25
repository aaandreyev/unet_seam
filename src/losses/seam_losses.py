from __future__ import annotations

import torch

from src.losses.lowfreq import multiscale_lowfreq_loss
from src.losses.perceptual import BoundaryLPIPSLoss
from src.losses.residual_guard import residual_l1_loss, residual_magnitude_loss, residual_smoothness_loss


def charbonnier(diff: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(diff * diff + 1e-6)


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernels_x = kernel_x.repeat(x.shape[1], 1, 1, 1)
    kernels_y = kernel_y.repeat(x.shape[1], 1, 1, 1)
    gx = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_x, groups=x.shape[1])
    gy = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"), kernels_y, groups=x.shape[1])
    return gx, gy


class SeamLossComputer:
    def __init__(
        self,
        lpips_enabled: bool = True,
        lpips_max_batch: int = 4,
        lpips_resize: int | None = 256,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.lpips_hf = BoundaryLPIPSLoss(enabled=lpips_enabled, max_batch=lpips_max_batch, resize=lpips_resize)
        self.lowfreq_sigmas = (2.0, 4.0, 8.0, 16.0, 32.0)
        default_weights = {
            "inner": 1.0,
            "boundary": 4.0,
            "lowfreq": 0.5,
            "grad": 0.5,
            "identity": 2.0,
            "lpips_hf": 0.05,
            "smooth": 0.10,
            "residual_l1": 0.50,
            "residual_over_cap": 0.05,
        }
        self.weights = default_weights | (weights or {})

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, input_rgb: torch.Tensor, inner_mask: torch.Tensor, boundary_band: torch.Tensor, residual: torch.Tensor) -> dict[str, torch.Tensor]:
        outer_mask = 1.0 - inner_mask
        diff_inner = (pred - target) * inner_mask
        diff_boundary = (pred - target) * boundary_band
        l_inner = charbonnier(diff_inner).mean()
        l_boundary = charbonnier(diff_boundary).mean()
        l_lowfreq = multiscale_lowfreq_loss(pred, target, inner_mask, self.lowfreq_sigmas)
        grad_pred = sobel_gradients(pred)
        grad_target = sobel_gradients(target)
        l_grad = (grad_pred[0] - grad_target[0]).abs().mean() + (grad_pred[1] - grad_target[1]).abs().mean()
        l_identity = ((pred - input_rgb).abs() * outer_mask).mean()
        l_lpips_hf = self.lpips_hf(pred, target, boundary_band)
        l_smooth = residual_smoothness_loss(residual)
        l_residual_l1 = residual_l1_loss(residual, inner_mask)
        l_mag = residual_magnitude_loss(residual)
        w = self.weights
        total = (
            w["inner"] * l_inner
            + w["boundary"] * l_boundary
            + w["lowfreq"] * l_lowfreq
            + w["grad"] * l_grad
            + w["identity"] * l_identity
            + w["lpips_hf"] * l_lpips_hf
            + w["smooth"] * l_smooth
            + w["residual_l1"] * l_residual_l1
            + w["residual_over_cap"] * l_mag
        )
        return {
            "total": total,
            "l_inner": l_inner,
            "l_boundary": l_boundary,
            "l_lowfreq_ms": l_lowfreq,
            "l_grad": l_grad,
            "l_identity": l_identity,
            "l_lpips_hf": l_lpips_hf,
            "l_residual_smooth": l_smooth,
            "l_residual_l1": l_residual_l1,
            "l_residual_magnitude": l_mag,
        }
