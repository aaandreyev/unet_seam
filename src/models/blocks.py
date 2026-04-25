from __future__ import annotations

import functools

import torch
from torch import nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=8)
def _gaussian_kernel_1d(sigma: float, radius: int) -> torch.Tensor:
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(xs**2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(round(sigma * 3)))
    radius = min(radius, max(1, x.shape[-1] // 2 - 1), max(1, x.shape[-2] // 2 - 1))
    kernel = _gaussian_kernel_1d(sigma, radius).to(dtype=x.dtype, device=x.device)
    c = x.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel_x, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel_y, groups=c)
    return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res = ResBlock(out_channels, groups=groups)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.proj = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.res = ResBlock(out_channels, groups=groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.res(x)


class FiLMGenerator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = x.mean(dim=(-2, -1))
        gamma, beta = self.net(stats).chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
