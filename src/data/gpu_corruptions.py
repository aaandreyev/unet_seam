from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _r(B: int, lo: float, hi: float, device: torch.device,
       gen: torch.Generator | None = None) -> Tensor:
    return torch.rand(B, 1, 1, 1, device=device, generator=gen) * (hi - lo) + lo


def _prob(B: int, p: float, device: torch.device,
          gen: torch.Generator | None = None) -> Tensor:
    return (torch.rand(B, 1, 1, 1, device=device, generator=gen) < p).float()


def _gaussian_kernel(size: int, sigma: float, C: int, device: torch.device) -> Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1)).view(1, 1, size, size)
    return k.expand(C, 1, size, size).contiguous()


class GPUCorruption(nn.Module):
    """Randomized photometric + spatial corruptions on GPU.

    Mirrors CPU corruption families A/B/C/D but fully vectorized across the batch.
    All ops are in-place-free PyTorch — no PIL, no numpy, no Python loops.
    """

    def __init__(self, p_spatial: float = 0.35, p_pipeline: float = 0.25) -> None:
        super().__init__()
        self.p_spatial = p_spatial
        self.p_pipeline = p_pipeline

    @torch.no_grad()
    def forward(self, inner: Tensor, gen: torch.Generator | None = None) -> Tensor:
        """
        Args:
            inner: B×3×H×W float in [0,1] — inner half of each strip.
            gen:   optional RNG (pass per-step generator for reproducibility).
        Returns:
            Corrupted B×3×H×W clamped to [0,1].
        """
        B, C, H, W = inner.shape
        dev = inner.device
        x = inner.clone()

        def r(lo: float, hi: float) -> Tensor:
            return _r(B, lo, hi, dev, gen)

        def p(prob: float) -> Tensor:
            return _prob(B, prob, dev, gen)

        # ── Family A: global photometric ─────────────────────────────────────────

        # Exposure (log-uniform ±~1 stop)
        x = (x * torch.exp(r(-0.9, 0.9))).clamp(0, 1)

        # Brightness
        x = (x + r(-0.18, 0.18)).clamp(0, 1)

        # Contrast (scale around per-image mean)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mu) * r(0.5, 1.6) + mu).clamp(0, 1)

        # Gamma
        x = x.clamp(1e-7, 1.0).pow(r(0.5, 2.0))

        # Per-channel gain (color cast / tint)
        gains = torch.rand(B, 3, 1, 1, device=dev, generator=gen) * (1.25 - 0.75) + 0.75
        x = (x * gains).clamp(0, 1)

        # Saturation
        lum = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        x = (lum + (x - lum) * r(0.2, 1.8)).clamp(0, 1)

        # Temperature (warm R↑B↓ / cool R↓B↑)
        t = r(-0.12, 0.12)
        x = torch.cat([(x[:, 0:1] + t).clamp(0, 1), x[:, 1:2],
                       (x[:, 2:3] - t).clamp(0, 1)], dim=1)

        # Black-point lift
        bp = r(0.0, 0.12)
        x = (x * (1 - bp) + bp).clamp(0, 1)

        # White-point compression
        x = (x * r(0.85, 1.0)).clamp(0, 1)

        # ── Family B: tonal curves ────────────────────────────────────────────────

        # Shadow adjustment (weight strongest in dark areas)
        sw = (1 - x).pow(2)
        x = (x + r(-0.08, 0.15) * sw).clamp(0, 1)

        # S-curve / inverse-S via polynomial  y = x + α·x(1−x)(2x−1)
        alpha = r(-0.6, 0.6)
        x = (x + alpha * x * (1 - x) * (2 * x - 1)).clamp(0, 1)

        # Highlight scale
        hw = x.pow(2)
        x = (x + r(-0.15, 0.15) * hw).clamp(0, 1)

        # ── Family C: smooth spatial fields (~35 % of samples) ───────────────────

        m_c = p(self.p_spatial)                                  # B×1×1×1 mask

        # Smooth illumination / shading field  (4×4 → H×W bilinear)
        lum_lr = torch.rand(B, 1, 4, 4, device=dev, generator=gen) * 0.5 - 0.25
        lum_field = F.interpolate(lum_lr, size=(H, W), mode="bilinear", align_corners=False)
        x = (x * (1 + lum_field * m_c)).clamp(0, 1)

        # Smooth color-temperature field
        t_lr = torch.rand(B, 1, 4, 4, device=dev, generator=gen) * 0.2 - 0.1
        t_field = F.interpolate(t_lr, size=(H, W), mode="bilinear", align_corners=False) * m_c
        x = torch.cat([(x[:, 0:1] + t_field).clamp(0, 1),
                       x[:, 1:2],
                       (x[:, 2:3] - t_field).clamp(0, 1)], dim=1)

        # Horizontal luminance ramp (seam-direction drift)
        grad = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W)
        x = (x + grad * r(-0.2, 0.2) * m_c).clamp(0, 1)

        # Vertical luminance ramp
        vgrad = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1)
        x = (x + vgrad * r(-0.15, 0.15) * m_c).clamp(0, 1)

        # ── Family D: pipeline artifacts (~25 % of samples) ──────────────────────

        m_d = p(self.p_pipeline)

        # Gaussian blur (fixed 5×5, sigma≈1.2)
        k = _gaussian_kernel(5, 1.2, C, dev)
        blurred = F.conv2d(x, k, padding=2, groups=C)
        x = torch.where(m_d > 0, blurred, x)

        # Additive Gaussian noise
        noise_std = r(0.0, 0.025) * m_d
        noise = torch.randn(B, C, H, W, device=dev, generator=gen)
        x = (x + noise * noise_std).clamp(0, 1)

        return x
