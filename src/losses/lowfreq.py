from __future__ import annotations

import torch

from src.models.blocks import gaussian_blur_tensor


def multiscale_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, inner_mask: torch.Tensor, sigmas: tuple[float, ...]) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    denom = inner_mask.sum().clamp_min(1.0)
    for sigma in sigmas:
        p = gaussian_blur_tensor(pred, sigma)
        t = gaussian_blur_tensor(target, sigma)
        total = total + ((p - t).abs() * inner_mask).sum() / denom
    return total / max(len(sigmas), 1)
