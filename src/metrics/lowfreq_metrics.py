from __future__ import annotations

import torch

from src.models.blocks import gaussian_blur_tensor


def lowfreq_mae(pred: torch.Tensor, target: torch.Tensor, sigma: float = 16.0) -> torch.Tensor:
    return (gaussian_blur_tensor(pred, sigma) - gaussian_blur_tensor(target, sigma)).abs().mean()
