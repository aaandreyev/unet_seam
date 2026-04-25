from __future__ import annotations

import torch
import torch.nn.functional as F


def sobel_gradients(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("expected BCHW RGB tensor")
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return gx, gy


def sobel_magnitude(rgb: torch.Tensor) -> torch.Tensor:
    gx, gy = sobel_gradients(rgb)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def gradient_cosine_similarity(a_rgb: torch.Tensor, b_rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    agx, agy = sobel_gradients(a_rgb)
    bgx, bgy = sobel_gradients(b_rgb)
    a = torch.cat([agx.flatten(1), agy.flatten(1)], dim=1)
    b = torch.cat([bgx.flatten(1), bgy.flatten(1)], dim=1)
    return F.cosine_similarity(a, b, dim=1, eps=eps)


def keep_structurally_matched_strip(
    generated_strip: torch.Tensor,
    target_strip: torch.Tensor,
    outer_width: int = 128,
    band_px: int = 32,
    threshold: float = 0.6,
) -> bool:
    if generated_strip.ndim == 3:
        generated_strip = generated_strip.unsqueeze(0)
    if target_strip.ndim == 3:
        target_strip = target_strip.unsqueeze(0)
    band = slice(outer_width, outer_width + band_px)
    score = gradient_cosine_similarity(generated_strip[..., band], target_strip[..., band])
    return bool(torch.all(score >= threshold).item())
