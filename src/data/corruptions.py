from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


GROUPS = {
    "A": ["exposure", "brightness", "contrast", "gamma", "saturation", "hue", "temperature", "tint", "channel_gains", "black_point", "white_point"],
    "B": ["shadow_lift", "shadow_crush", "highlight_compress", "highlight_boost", "midtone", "s_curve", "reverse_s_curve"],
    "C": ["horizontal_luma_gradient", "vertical_luma_gradient", "illumination_field", "temperature_field", "saturation_field"],
    "D": ["blur", "microcontrast", "noise", "jpeg_like"],
}


@dataclass
class CorruptionResult:
    image: torch.Tensor
    ops: list[str]


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _apply_gamma(x: torch.Tensor, gamma: float) -> torch.Tensor:
    return x.clamp(1e-6, 1.0).pow(gamma)


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(sigma * 3)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(xs**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel1d(sigma, x.device, x.dtype)
    pad = min(kernel.numel() // 2, max(1, x.shape[-1] // 2 - 1), max(1, x.shape[-2] // 2 - 1))
    if pad * 2 + 1 != kernel.numel():
        kernel = kernel[kernel.numel() // 2 - pad : kernel.numel() // 2 + pad + 1]
        kernel = kernel / kernel.sum()
    kernel_x = kernel.view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kernel_x, groups=x.shape[1])
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), kernel_y, groups=x.shape[1])
    return x


def _field(shape: torch.Size, magnitude: float, generator: torch.Generator) -> torch.Tensor:
    _, _, h, w = shape
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)
    ax = torch.rand(1, generator=generator).item() * magnitude
    ay = torch.rand(1, generator=generator).item() * magnitude
    return ax * xx + ay * yy


def _maybe_add(chosen: list[str], group: list[str], probability: float, generator: torch.Generator) -> None:
    if torch.rand(1, generator=generator).item() < probability:
        name = group[int(torch.randint(0, len(group), (1,), generator=generator).item())]
        if name not in chosen:
            chosen.append(name)


def _pick_unique(pool: list[str], chosen: list[str], generator: torch.Generator) -> str:
    while True:
        name = pool[int(torch.randint(0, len(pool), (1,), generator=generator).item())]
        if name not in chosen:
            return name


def apply_random_corruptions(inner: torch.Tensor, generator: torch.Generator) -> CorruptionResult:
    x = inner.clone()
    ops: list[str] = []
    n_ops = int(torch.multinomial(torch.tensor([0.2, 0.4, 0.3, 0.1]), 1, generator=generator).item()) + 2
    chosen = [_pick_unique(GROUPS["A"] + GROUPS["B"], [], generator)]
    _maybe_add(chosen, GROUPS["C"], 0.35, generator)
    _maybe_add(chosen, GROUPS["D"], 0.25, generator)
    candidates = GROUPS["A"] + GROUPS["B"]
    while len(chosen) < n_ops:
        chosen.append(_pick_unique(candidates, chosen, generator))
    chosen = chosen[:n_ops]
    for name in chosen:
        if name == "brightness":
            delta = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
            x = x + delta
        elif name == "exposure":
            ev = torch.empty(1).uniform_(-0.3, 0.5, generator=generator).item()
            x = x * (2.0**ev)
        elif name == "contrast":
            contrast = torch.empty(1).uniform_(0.8, 1.25, generator=generator).item()
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * contrast + mean
        elif name == "gamma":
            gamma = torch.empty(1).uniform_(0.85, 1.2, generator=generator).item()
            x = _apply_gamma(x, gamma)
        elif name == "saturation":
            sat = torch.empty(1).uniform_(0.75, 1.35, generator=generator).item()
            luma = _rgb_to_luma(x)
            x = luma + (x - luma) * sat
        elif name == "hue":
            angle = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
            luma = _rgb_to_luma(x)
            centered = x - luma
            x = luma + centered.roll(shifts=1, dims=1) * angle + centered * (1.0 - abs(angle))
        elif name == "temperature":
            t = torch.empty(1).uniform_(-0.06, 0.06, generator=generator).item()
            x[:, 0:1] += t
            x[:, 2:3] -= t
        elif name == "tint":
            t = torch.empty(1).uniform_(-0.05, 0.05, generator=generator).item()
            x[:, 1:2] += t
        elif name == "channel_gains":
            gains = torch.empty((1, 3, 1, 1)).uniform_(0.9, 1.1, generator=generator)
            x = x * gains
        elif name == "black_point":
            black = torch.empty(1).uniform_(-0.04, 0.06, generator=generator).item()
            x = (x - black) / max(1.0 - black, 1e-3)
        elif name == "white_point":
            white = torch.empty(1).uniform_(0.92, 1.08, generator=generator).item()
            x = x / max(white, 1e-3)
        elif name == "shadow_lift":
            amount = torch.empty(1).uniform_(0.0, 0.12, generator=generator).item()
            x = x + (1.0 - x) * amount * (1.0 - x).pow(2)
        elif name == "shadow_crush":
            amount = torch.empty(1).uniform_(0.0, 0.16, generator=generator).item()
            x = x - amount * (1.0 - x).pow(2)
        elif name == "highlight_compress":
            amount = torch.empty(1).uniform_(0.0, 0.12, generator=generator).item()
            x = x - amount * x.pow(2)
        elif name == "highlight_boost":
            amount = torch.empty(1).uniform_(0.0, 0.12, generator=generator).item()
            x = x + amount * x.pow(2)
        elif name == "midtone":
            amount = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
            x = x + amount * torch.sin(x * math.pi)
        elif name == "s_curve":
            amount = torch.empty(1).uniform_(0.10, 0.30, generator=generator).item()
            x = x + amount * (x - 0.5) * (1.0 - (2.0 * x - 1.0).abs())
        elif name == "reverse_s_curve":
            amount = torch.empty(1).uniform_(0.10, 0.30, generator=generator).item()
            x = x - amount * (x - 0.5) * (1.0 - (2.0 * x - 1.0).abs())
        elif name == "horizontal_luma_gradient":
            xx = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, x.shape[-1])
            x = x + xx * torch.empty(1).uniform_(-0.12, 0.12, generator=generator).item()
        elif name == "vertical_luma_gradient":
            yy = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, x.shape[-2], 1)
            x = x + yy * torch.empty(1).uniform_(-0.12, 0.12, generator=generator).item()
        elif name == "illumination_field":
            x = x * (1.0 + _field(x.shape, 0.10, generator).to(device=x.device, dtype=x.dtype))
        elif name == "temperature_field":
            field = _field(x.shape, 0.10, generator).to(device=x.device, dtype=x.dtype)
            x[:, 0:1] += field
            x[:, 2:3] -= field * 0.8
        elif name == "saturation_field":
            field = _field(x.shape, 0.20, generator).to(device=x.device, dtype=x.dtype)
            luma = _rgb_to_luma(x)
            x = luma + (x - luma) * (1.0 + field).clamp(0.7, 1.3)
        elif name == "blur":
            sigma = torch.empty(1).uniform_(0.0, 1.5, generator=generator).item()
            x = _gaussian_blur(x, sigma)
        elif name == "noise":
            sigma = torch.empty(1).uniform_(0.0, 0.01, generator=generator).item()
            x = x + torch.randn_like(x, generator=generator) * sigma
        elif name == "microcontrast":
            amount = torch.empty(1).uniform_(0.0, 0.1, generator=generator).item()
            blur = _gaussian_blur(x, 1.0)
            x = x + (x - blur) * amount
        elif name == "jpeg_like":
            levels = int(torch.randint(64, 160, (1,), generator=generator).item())
            x = torch.round(x.clamp(0.0, 1.0) * float(levels)) / float(levels)
        ops.append(name)
    return CorruptionResult(x.clamp(0.0, 1.0), ops)
