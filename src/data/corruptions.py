from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


GROUPS = {
    "A": ["brightness", "exposure", "gain"],
    "B": ["contrast", "gamma"],
    "C": ["temperature", "tint", "channel_gains", "saturation"],
    "D": ["shadow", "highlight", "midtone"],
    "E": ["brightness_gradient", "color_gradient", "illumination_poly", "vignette"],
    "F": ["blur", "jpeg_like", "noise", "microcontrast"],
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


def apply_random_corruptions(inner: torch.Tensor, generator: torch.Generator) -> CorruptionResult:
    x = inner.clone()
    ops: list[str] = []
    candidates = [
        "brightness",
        "exposure",
        "gain",
        "contrast",
        "gamma",
        "temperature",
        "tint",
        "channel_gains",
        "saturation",
        "shadow",
        "highlight",
        "midtone",
        "brightness_gradient",
        "color_gradient",
        "illumination_poly",
        "vignette",
        "blur",
        "noise",
        "microcontrast",
    ]
    n_ops = int(torch.multinomial(torch.tensor([0.2, 0.4, 0.3, 0.1]), 1, generator=generator).item()) + 2
    chosen: list[str] = []
    while len(chosen) < n_ops:
        idx = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        name = candidates[idx]
        if name not in chosen:
            chosen.append(name)
    if not any(name in GROUPS["A"] + GROUPS["B"] + GROUPS["C"] for name in chosen):
        chosen[0] = "exposure"
    if not any(name in GROUPS["D"] for name in chosen):
        chosen[min(1, len(chosen) - 1)] = "midtone"
    for name in chosen:
        if name == "brightness":
            delta = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
            x = x + delta
        elif name == "exposure":
            ev = torch.empty(1).uniform_(-0.3, 0.5, generator=generator).item()
            x = x * (2.0**ev)
        elif name == "gain":
            gain = torch.empty(1).uniform_(0.85, 1.2, generator=generator).item()
            x = x * gain
        elif name == "contrast":
            contrast = torch.empty(1).uniform_(0.8, 1.25, generator=generator).item()
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * contrast + mean
        elif name == "gamma":
            gamma = torch.empty(1).uniform_(0.85, 1.2, generator=generator).item()
            x = _apply_gamma(x, gamma)
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
        elif name == "saturation":
            sat = torch.empty(1).uniform_(0.75, 1.35, generator=generator).item()
            luma = _rgb_to_luma(x)
            x = luma + (x - luma) * sat
        elif name == "shadow":
            amount = torch.empty(1).uniform_(0.0, 0.12, generator=generator).item()
            x = x + (1.0 - x) * amount * (1.0 - x).pow(2)
        elif name == "highlight":
            amount = torch.empty(1).uniform_(0.0, 0.12, generator=generator).item()
            x = x - amount * x.pow(2)
        elif name == "midtone":
            amount = torch.empty(1).uniform_(-0.08, 0.08, generator=generator).item()
            x = x + amount * torch.sin(x * math.pi)
        elif name == "brightness_gradient":
            x = x + _field(x.shape, 0.12, generator)
        elif name == "color_gradient":
            field = _field(x.shape, 0.12, generator)
            x[:, 0:1] += field
            x[:, 2:3] -= field * 0.8
        elif name == "illumination_poly":
            field = _field(x.shape, 0.10, generator)
            x = x * (1.0 + field)
        elif name == "vignette":
            _, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1)
            xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)
            rr = (xx * xx + yy * yy).sqrt()
            scale = torch.empty(1).uniform_(0.0, 0.15, generator=generator).item()
            x = x * (1.0 - rr * scale)
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
        ops.append(name)
    return CorruptionResult(x.clamp(0.0, 1.0), ops)
