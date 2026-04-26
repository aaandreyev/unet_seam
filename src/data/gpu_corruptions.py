from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gaussian_kernel(size: int, sigma: float, C: int, device: torch.device) -> Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1)).view(1, 1, size, size)
    return k.expand(C, 1, size, size).contiguous()


def _planner_field_batched(
    B: int, H: int, W: int, dev: torch.device, gen: torch.Generator | None, magnitude: float
) -> Tensor:
    """Batched version of corruptions._field: ax*xx + ay*yy, per sample."""
    ax = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * magnitude
    ay = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * magnitude
    yy = torch.linspace(-1.0, 1.0, H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    return ax * xx + ay * yy


class GPUCorruption(nn.Module):
    """Randomized photometric + spatial corruptions on GPU.

    Aligns with CPU `apply_random_corruptions`:
    - Photometric: each op is included independently with its own p (mild A/B-style stack).
    - Family C: at most *one* spatial op per sample, with probability ~0.50 and
      weighted toward color-drift / illumination fields.
    - Family D: at most *one* of blur / microcontrast / noise / jpeg per sample, p ~0.20.

    The previous version gated C and D on a single batch-wide coin flip and stacked multiple
    C (or blur+noise) on one sample, which made GPU corruption much heavier than the dataset.
    """

    def __init__(self, p_c: float = 0.5, p_d: float = 0.2) -> None:
        super().__init__()
        self.p_c = p_c
        self.p_d = p_d

    @torch.no_grad()
    def forward(self, inner: Tensor, gen: torch.Generator | None = None) -> Tensor:
        """
        Args:
            inner: B×3×H×W float in [0,1] — inner half to corrupt.
            gen:   optional RNG for reproducibility.
        Returns:
            Corrupted B×3×H×W clamped to [0,1].
        """
        B, C, H, W = inner.shape
        dev = inner.device
        x = inner.clone()

        def rand(lo: float, hi: float) -> Tensor:
            """B×1×1×1 uniform sample."""
            return torch.rand(B, 1, 1, 1, device=dev, generator=gen) * (hi - lo) + lo

        def apply(original: Tensor, modified: Tensor, p: float) -> Tensor:
            """Apply modified to original with per-sample probability p."""
            mask = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) < p).float()
            return original + mask * (modified - original)

        # ── Family A: global photometric (each op ~p=0.3–0.4) ─────────────────────

        # Exposure (log-uniform ±1 stop)
        x = apply(x, (x * torch.exp(rand(-1.0, 1.0))).clamp(0, 1), p=0.40)

        # Brightness
        x = apply(x, (x + rand(-0.15, 0.15)).clamp(0, 1), p=0.35)

        # Contrast (scale around image mean)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = apply(x, ((x - mu) * rand(0.6, 1.5) + mu).clamp(0, 1), p=0.35)

        # Gamma
        x = apply(x, x.clamp(1e-7, 1).pow(rand(0.5, 2.0)), p=0.30)

        # Per-channel gain (color cast)
        gains = torch.rand(B, 3, 1, 1, device=dev, generator=gen) * (1.2 - 0.8) + 0.8
        x = apply(x, (x * gains).clamp(0, 1), p=0.35)

        # Saturation
        lum = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        x = apply(x, (lum + (x - lum) * rand(0.3, 1.7)).clamp(0, 1), p=0.30)

        # Temperature (warm/cool shift)
        t = rand(-0.10, 0.10)
        warm = torch.cat([(x[:, 0:1] + t).clamp(0, 1), x[:, 1:2],
                          (x[:, 2:3] - t).clamp(0, 1)], dim=1)
        x = apply(x, warm, p=0.30)

        # Black-point lift
        x = apply(x, (x + rand(0.0, 0.10) * (1 - x)).clamp(0, 1), p=0.25)

        # White-point compression
        x = apply(x, (x * rand(0.88, 1.0)).clamp(0, 1), p=0.25)

        # ── Family B: tonal curves (~p=0.20 each) ────────────────────────────────

        # Shadow lift / crush
        shadow_w = (1 - x).pow(2)
        x = apply(x, (x + rand(-0.08, 0.12) * shadow_w).clamp(0, 1), p=0.20)

        # S-curve / inverse-S  (polynomial: y = x + α·x(1−x)(2x−1))
        alpha = rand(-0.5, 0.5)
        x = apply(x, (x + alpha * x * (1 - x) * (2 * x - 1)).clamp(0, 1), p=0.20)

        # Highlight scale
        hi_w = x.pow(2)
        x = apply(x, (x + rand(-0.10, 0.10) * hi_w).clamp(0, 1), p=0.20)

        # ── Family C: at most one spatial op per sample (p ≈ 0.35) ──────────────
        m_c = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) < self.p_c).float()
        choice_c = torch.multinomial(
            torch.tensor([0.10, 0.08, 0.24, 0.32, 0.26], device=dev, dtype=torch.float32),
            B,
            replacement=True,
            generator=gen,
        )
        # unsqueeze(1): (B,1,1,1,1) so (B,1,1,1,1) * (B,5,1,1,1) broadcasts on class dim, not batch
        m_k = m_c.unsqueeze(1) * (F.one_hot(choice_c, 5).view(B, 5, 1, 1, 1).float())

        # k=0 horizontal luma
        h_grad = torch.linspace(-1.0, 1.0, W, device=dev, dtype=x.dtype).view(1, 1, 1, W)
        u0 = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.24 - 0.12)
        eff0 = (x + h_grad * u0).clamp(0, 1)
        x = x + m_k[:, 0] * (eff0 - x)

        # k=1 vertical luma
        v_grad = torch.linspace(-1.0, 1.0, H, device=dev, dtype=x.dtype).view(1, 1, H, 1)
        u1 = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.24 - 0.12)
        eff1 = (x + v_grad * u1).clamp(0, 1)
        x = x + m_k[:, 1] * (eff1 - x)

        # k=2 illumination field (CPU: x * (1 + _field(..., 0.10)))
        field2 = _planner_field_batched(B, H, W, dev, gen, 0.10)
        eff2 = (x * (1.0 + field2)).clamp(0, 1)
        x = x + m_k[:, 2] * (eff2 - x)

        # k=3 temperature field
        field3 = _planner_field_batched(B, H, W, dev, gen, 0.10)
        eff3 = torch.cat(
            [
                (x[:, 0:1] + field3).clamp(0, 1),
                x[:, 1:2],
                (x[:, 2:3] - field3 * 0.8).clamp(0, 1),
            ],
            dim=1,
        )
        x = x + m_k[:, 3] * (eff3 - x)

        # k=4 saturation field
        field4 = _planner_field_batched(B, H, W, dev, gen, 0.20)
        luma4 = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        sat_mul = (1.0 + field4).clamp(0.7, 1.3)
        eff4 = (luma4 + (x - luma4) * sat_mul).clamp(0, 1)
        x = x + m_k[:, 4] * (eff4 - x)

        # ── Family D: at most one of blur / microcontrast / noise / jpeg (p ≈ 0.25) ─
        m_d = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) < self.p_d).float()
        choice_d = torch.multinomial(
            torch.tensor([0.28, 0.18, 0.18, 0.36], device=dev, dtype=torch.float32),
            B,
            replacement=True,
            generator=gen,
        )
        m_dk = m_d.unsqueeze(1) * (F.one_hot(choice_d, 4).view(B, 4, 1, 1, 1).float())

        # d=0 Gaussian blur
        k_blur = _gaussian_kernel(5, 1.2, C, dev)
        blurred0 = F.conv2d(x, k_blur, padding=2, groups=C)
        x = x + m_dk[:, 0] * (blurred0 - x)

        # d=1 microcontrast
        k_sharp = _gaussian_kernel(5, 1.0, C, dev)
        blurred1 = F.conv2d(x, k_sharp, padding=2, groups=C)
        amount = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.1
        eff_m = (x + (x - blurred1) * amount).clamp(0, 1)
        x = x + m_dk[:, 1] * (eff_m - x)

        # d=2 noise
        sig = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.01
        noise = torch.randn(B, C, H, W, device=dev, generator=gen) * sig
        eff_n = (x + noise).clamp(0, 1)
        x = x + m_dk[:, 2] * (eff_n - x)

        # d=3 JPEG-like quantize
        levels = 64.0 + torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 95.0
        eff_j = (torch.round(x.clamp(0, 1) * levels) / levels).clamp(0, 1)
        x = x + m_dk[:, 3] * (eff_j - x)

        return x
