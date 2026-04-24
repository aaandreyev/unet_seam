from __future__ import annotations

import torch
from torch import nn

from src.data.strip_geometry import build_decay_mask
from src.models.blocks import DownBlock, FiLMGenerator, ResBlock, UpBlock, gaussian_blur_tensor


class SeamResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        base_channels: int = 32,
        groups: int = 8,
        residual_cap: float = 0.3,
        residual_mode: str = "full",
        low_freq_sigma: float = 6.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.residual_cap = residual_cap
        self.residual_mode = residual_mode
        self.low_freq_sigma = low_freq_sigma

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = DownBlock(base_channels, 32, groups=groups)
        self.enc2 = DownBlock(32, 64, groups=groups)
        self.enc3 = DownBlock(64, 128, groups=groups)
        self.enc4 = DownBlock(128, 256, groups=groups)

        self.lf_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 256, kernel_size=3, stride=16, padding=1),
        )
        self.bottleneck = nn.Sequential(ResBlock(256, groups=groups), ResBlock(256, groups=groups))
        self.film = FiLMGenerator(256)
        self.dec4 = UpBlock(256, 256, 256, groups=groups)
        self.dec3 = UpBlock(256, 128, 128, groups=groups)
        self.dec2 = UpBlock(128, 64, 64, groups=groups)
        self.dec1 = UpBlock(64, 32, 32, groups=groups)
        self.head = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]
        seam_x = int((x[:, 3:4, 0, :].argmax(dim=-1).min().item()))
        stem = self.stem(x)
        x1, skip1 = self.enc1(stem)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        lf = gaussian_blur_tensor(rgb, sigma=self.low_freq_sigma)
        lf = self.lf_branch(lf)
        bn = self.bottleneck(x4 + lf)
        gamma, beta = self.film(bn)
        bn = bn * (1.0 + gamma) + beta
        dec = self.dec4(bn, skip4)
        dec = self.dec3(dec, skip3)
        dec = self.dec2(dec, skip2)
        dec = self.dec1(dec, skip1)
        raw = self.head(dec)
        scale = torch.clamp(self.residual_scale, 0.0, 1.0)
        residual = torch.tanh(raw) * self.residual_cap * scale
        if self.residual_mode == "low_freq_only":
            residual = gaussian_blur_tensor(residual, sigma=self.low_freq_sigma)
        inner_width = x.shape[-1] - seam_x
        decay = build_decay_mask(x.shape[-2], x.shape[-1], seam_x, inner_width).to(x.device, x.dtype)
        return residual * decay
