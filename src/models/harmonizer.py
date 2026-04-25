from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.harmonizer_blocks import NAFEncoderLite, resize_inner


def monotonic_knots(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    # raw: B x 3 x (K-1). Return B x 3 x K with first=0, last=1.
    delta = F.softplus(raw) + eps
    cumulative = torch.cumsum(delta, dim=-1)
    normalized = cumulative / cumulative[..., -1:].clamp_min(eps)
    zeros = torch.zeros(*raw.shape[:2], 1, device=raw.device, dtype=raw.dtype)
    return torch.cat([zeros, normalized], dim=-1)


def apply_monotonic_curves(inner_rgb: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    b, c, h, w = inner_rgb.shape
    if c != 3:
        raise ValueError("inner_rgb must have 3 channels")
    k = knots.shape[-1]
    x = inner_rgb.clamp(0.0, 1.0) * (k - 1)
    idx0 = torch.floor(x).to(torch.long).clamp(0, k - 2)
    idx1 = idx0 + 1
    t = x - idx0.to(x.dtype)
    table = knots.view(b, c, k, 1, 1).expand(b, c, k, h, w)
    y0 = torch.gather(table, 2, idx0.unsqueeze(2)).squeeze(2)
    y1 = torch.gather(table, 2, idx1.unsqueeze(2)).squeeze(2)
    return y0 * (1.0 - t) + y1 * t


def reconstruct_corrected_strip(
    strip_rgb: torch.Tensor,
    knots: torch.Tensor,
    shading_lowres: torch.Tensor,
    outer_width: int = 128,
    alpha: float = 0.20,
) -> dict[str, torch.Tensor]:
    height = strip_rgb.shape[-2]
    inner_width = strip_rgb.shape[-1] - outer_width
    outer = strip_rgb[..., :outer_width]
    inner = strip_rgb[..., outer_width:]
    curved = apply_monotonic_curves(inner, knots)
    shading = resize_inner(shading_lowres, height, inner_width)
    gain = torch.exp(alpha * torch.tanh(shading))
    corrected_inner = (curved * gain).clamp(0.0, 1.0)
    corrected_strip = torch.cat([outer, corrected_inner], dim=-1)
    return {
        "corrected_inner": corrected_inner,
        "corrected_strip": corrected_strip,
        "curve_applied_inner": curved,
        "shading": shading,
        "gain": gain,
    }


class SeamHarmonizerV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        blocks: tuple[int, ...] = (2, 2, 4, 6),
        num_knots: int = 16,
        outer_width: int = 128,
        alpha: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = blocks
        self.num_knots = num_knots
        self.outer_width = outer_width
        self.alpha = alpha
        self.encoder = NAFEncoderLite(in_channels=in_channels, channels=channels, blocks=blocks)
        self.curve_head = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.SiLU(),
            nn.Linear(256, 3 * (num_knots - 1)),
        )
        self.shading_head = nn.Sequential(
            nn.Conv2d(channels[2], 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        last = self.curve_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        last_conv = self.shading_head[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        pooled = feats[-1].mean(dim=(-2, -1))
        raw_curves = self.curve_head(pooled).view(x.shape[0], 3, self.num_knots - 1)
        knots = monotonic_knots(raw_curves)
        shading_lowres = self.shading_head(feats[2])
        # Stage 3 is 256x64 for 1024x256 input; target low-res shading is 256x32.
        shading_lowres = F.interpolate(shading_lowres, size=(256, 32), mode="bilinear", align_corners=False)
        recon = reconstruct_corrected_strip(
            x[:, :3],
            knots,
            shading_lowres,
            outer_width=self.outer_width,
            alpha=self.alpha,
        )
        return {
            "raw_curves": raw_curves,
            "curves": knots,
            "shading_lowres": shading_lowres,
            **recon,
        }
