from __future__ import annotations

import torch


def residual_smoothness_loss(residual: torch.Tensor) -> torch.Tensor:
    dx = residual[..., :, 1:] - residual[..., :, :-1]
    dy = residual[..., 1:, :] - residual[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def residual_magnitude_loss(residual: torch.Tensor, cap: float = 0.3) -> torch.Tensor:
    over = (residual.abs() - cap).clamp(min=0.0)
    return over.mean()


def residual_l1_loss(residual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    values = residual.abs()
    if mask is not None:
        values = values * mask
    return values.mean()
