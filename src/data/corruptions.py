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


# Central corruption tuning config. Adjust ranges/weights here instead of editing code below.
CORRUPTION_CFG = {
    "op_count_weights": [0.2, 0.4, 0.3, 0.1],  # maps to 2/3/4/5 ops
    "group_c_probability": 0.5,
    "group_d_probability": 0.20,
    "group_c_weights": [0.10, 0.08, 0.24, 0.32, 0.26],
    "group_d_weights": [0.28, 0.18, 0.18, 0.36],
    "spatial_probability": {
        "ab": 0.55,
        "d": 0.65,
    },
    "field_bank": {
        "macro_mix": {
            "tonal_macro": 0.70,
            "tonal_fine": 0.30,
            "chroma_macro": 0.55,
            "chroma_fine": 0.45,
            "detail_macro": 0.25,
            "detail_fine": 0.75,
            "degrade_macro": 0.35,
            "degrade_fine": 0.65,
            "seam_macro": 0.55,
            "seam_pref": 0.45,
            "region_field": 0.65,
            "region_seam": 0.35,
        },
        "region_sigmoid_scale": (1.8, 4.5),
    },
    "procedural_field": {
        "coarse_h": (10, 32, 48),
        "coarse_w": (8, 24, 12),
        "noise_std": 0.25,
        "gradient_weight": (-1.0, 1.0),
        "sin_count": (1, 4),
        "sin_fx": (0.5, 3.0),
        "sin_fy": (0.3, 2.0),
        "sin_amp": (0.10, 0.40),
        "blob_count": (1, 6),
        "blob_center": (-1.0, 1.0),
        "blob_sigma": (0.20, 0.75),
        "blob_amp": (0.15, 0.55),
        "blur_sigma": (0.6, 1.8),
    },
    "spatial_helpers": {
        "additive_span_scale": (0.35, 1.25),
        "factor_span_scale": (0.35, 1.25),
        "amount_span_scale": (0.35, 1.25),
    },
    "ops": {
        "brightness": {
            "base": (-0.10, 0.10),
            "spatial_field_mix": (0.8, 0.2),
            "spatial_floor": 0.015,
        },
        "exposure": {
            "base": (-0.50, 0.50),
            "spatial_floor": 0.07,
        },
        "contrast": {
            "base": (0.75, 1.40),
            "spatial_field_mix": (0.85, 0.15),
            "spatial_floor": 0.10,
            "clamp": (0.60, 1.80),
        },
        "gamma": {
            "base": (0.75, 1.40),
            "spatial_floor": 0.08,
            "clamp": (0.60, 1.80),
        },
        "saturation": {
            "base": (0.65, 1.60),
            "spatial_floor": 0.15,
            "clamp": (0.45, 2.00),
        },
        "hue": {
            "base": (-0.12, 0.12),
            "spatial_floor": 0.025,
            "clamp": (-0.18, 0.18),
        },
        "temperature": {
            "base": (-0.10, 0.10),
            "spatial_floor": 0.02,
            "clamp": (-0.16, 0.16),
        },
        "tint": {
            "base": (-0.08, 0.08),
            "spatial_floor": 0.018,
            "clamp": (-0.14, 0.14),
        },
        "channel_gains": {
            "base": (0.80, 1.35),
            "spatial_weight": (0.6, 1.4),
            "spatial_amp": (0.04, 0.10),
            "clamp": (0.65, 1.50),
        },
        "black_point": {
            "base": (-0.10, 0.15),
            "spatial_floor": 0.02,
            "clamp": (-0.12, 0.18),
        },
        "white_point": {
            "base": (0.88, 1.18),
            "spatial_floor": 0.04,
            "clamp": (0.80, 1.25),
        },
        "shadow_lift": {
            "base": (0.00, 0.16),
            "field_mix": (0.75, 0.25),
            "max_value": 0.28,
        },
        "shadow_crush": {
            "base": (0.00, 0.18),
            "field_mix": (0.75, 0.25),
            "max_value": 0.30,
        },
        "highlight_compress": {
            "base": (0.00, 0.16),
            "max_value": 0.28,
        },
        "highlight_boost": {
            "base": (0.00, 0.18),
            "max_value": 0.28,
        },
        "midtone": {
            "base": (-0.15, 0.15),
            "spatial_floor": 0.025,
            "clamp": (-0.22, 0.22),
        },
        "s_curve": {
            "base": (0.12, 0.35),
            "max_value": 0.50,
        },
        "reverse_s_curve": {
            "base": (0.12, 0.35),
            "max_value": 0.50,
        },
        "horizontal_luma_gradient": {
            "base": (-0.18, 0.18),
            "field_mix": (0.80, 0.20),
        },
        "vertical_luma_gradient": {
            "base": (-0.18, 0.18),
            "field_mix": (0.80, 0.20),
        },
        "illumination_field": {
            "amp": (0.05, 0.16),
            "field_mix": (0.70, 0.20),
        },
        "temperature_field": {
            "amp": (0.05, 0.16),
            "field_mix": (0.75, 0.25),
            "blue_scale": 0.8,
        },
        "saturation_field": {
            "amp": (0.08, 0.24),
            "field_mix": (0.75, 0.25),
            "clamp": (0.75, 1.40),
        },
        "blur": {
            "base": (0.00, 0.90),
            "mix_floor": 0.15,
            "mix_scale": 0.85,
            "mix_strength": (0.35, 0.80),
        },
        "noise": {
            "base": (0.00, 0.015),
            "sigma_floor": 0.002,
            "mix_floor": 0.15,
            "mix_scale": 0.85,
        },
        "microcontrast": {
            "base": (0.00, 0.16),
            "blur_sigma": 1.0,
            "max_value": 0.22,
        },
        "jpeg_like": {
            "levels": (96, 192),
            "mix_floor": 0.10,
            "mix_scale": 0.90,
            "mix_strength": (0.25, 0.70),
        },
    },
}


@dataclass
class CorruptionResult:
    image: torch.Tensor
    ops: list[str]


@dataclass
class ArtifactFieldBank:
    tonal: torch.Tensor
    chroma: torch.Tensor
    detail: torch.Tensor
    degrade: torch.Tensor
    region: torch.Tensor
    seam_bias: torch.Tensor


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _apply_gamma(x: torch.Tensor, gamma: float | torch.Tensor) -> torch.Tensor:
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


def _pick_cfg(name: str) -> dict:
    return CORRUPTION_CFG["ops"][name]


def _field(shape: torch.Size, magnitude: float, generator: torch.Generator) -> torch.Tensor:
    _, _, h, w = shape
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)
    ax = torch.rand(1, generator=generator).item() * magnitude
    ay = torch.rand(1, generator=generator).item() * magnitude
    return ax * xx + ay * yy


def _pick_unique(pool: list[str], chosen: list[str], generator: torch.Generator) -> str:
    while True:
        name = pool[int(torch.randint(0, len(pool), (1,), generator=generator).item())]
        if name not in chosen:
            return name


def _pick_weighted_unique(pool: list[str], weights: list[float], chosen: list[str], generator: torch.Generator) -> str:
    weight_t = torch.tensor(weights, dtype=torch.float32)
    while True:
        idx = int(torch.multinomial(weight_t, 1, generator=generator).item())
        name = pool[idx]
        if name not in chosen:
            return name


def _u(generator: torch.Generator, lo: float, hi: float) -> float:
    return float(torch.empty(1).uniform_(lo, hi, generator=generator).item())


def _normalize_signed(field: torch.Tensor) -> torch.Tensor:
    centered = field - field.mean(dim=(-2, -1), keepdim=True)
    scale = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return centered / scale


def _resized_grid(h: int, w: int, coarse_h: int, coarse_w: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy = torch.linspace(-1.0, 1.0, coarse_h, device=device, dtype=dtype).view(1, 1, coarse_h, 1)
    xx = torch.linspace(-1.0, 1.0, coarse_w, device=device, dtype=dtype).view(1, 1, 1, coarse_w)
    return yy, xx


def _procedural_field(shape: torch.Size, generator: torch.Generator, *, coarse_h: int | None = None, coarse_w: int | None = None) -> torch.Tensor:
    cfg = CORRUPTION_CFG["procedural_field"]
    b, _, h, w = shape
    coarse_h = coarse_h or max(cfg["coarse_h"][0], min(cfg["coarse_h"][1], h // cfg["coarse_h"][2]))
    coarse_w = coarse_w or max(cfg["coarse_w"][0], min(cfg["coarse_w"][1], w // cfg["coarse_w"][2]))
    device = torch.device("cpu")
    dtype = torch.float32
    yy, xx = _resized_grid(h, w, coarse_h, coarse_w, device, dtype)
    field = torch.randn((b, 1, coarse_h, coarse_w), generator=generator, dtype=dtype) * cfg["noise_std"]
    field += _u(generator, *cfg["gradient_weight"]) * xx
    field += _u(generator, *cfg["gradient_weight"]) * yy
    for _ in range(int(torch.randint(cfg["sin_count"][0], cfg["sin_count"][1], (1,), generator=generator).item())):
        fx = _u(generator, *cfg["sin_fx"])
        fy = _u(generator, *cfg["sin_fy"])
        phase = _u(generator, -math.pi, math.pi)
        amp = _u(generator, *cfg["sin_amp"])
        field += amp * torch.sin(fx * math.pi * xx + fy * math.pi * yy + phase)
    for _ in range(int(torch.randint(cfg["blob_count"][0], cfg["blob_count"][1], (1,), generator=generator).item())):
        cx = _u(generator, *cfg["blob_center"])
        cy = _u(generator, *cfg["blob_center"])
        sx = _u(generator, *cfg["blob_sigma"])
        sy = _u(generator, *cfg["blob_sigma"])
        amp = _u(generator, *cfg["blob_amp"]) * (1.0 if torch.rand(1, generator=generator).item() < 0.5 else -1.0)
        blob = torch.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        field += amp * blob
    field = _gaussian_blur(field, _u(generator, *cfg["blur_sigma"]))
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)
    return _normalize_signed(field)


def _build_artifact_field_bank(shape: torch.Size, generator: torch.Generator) -> ArtifactFieldBank:
    cfg = CORRUPTION_CFG["field_bank"]
    mix = cfg["macro_mix"]
    _, _, h, _ = shape
    macro = _procedural_field(shape, generator)
    tonal = _normalize_signed(mix["tonal_macro"] * macro + mix["tonal_fine"] * _procedural_field(shape, generator))
    chroma = _normalize_signed(mix["chroma_macro"] * macro + mix["chroma_fine"] * _procedural_field(shape, generator))
    detail = _normalize_signed(
        mix["detail_macro"] * macro
        + mix["detail_fine"] * _procedural_field(shape, generator, coarse_h=max(16, min(48, h // 24)), coarse_w=max(10, min(32, shape[-1] // 6)))
    )
    degrade = _normalize_signed(mix["degrade_macro"] * macro + mix["degrade_fine"] * _procedural_field(shape, generator))
    xs = torch.linspace(1.0, -1.0, shape[-1], dtype=torch.float32).view(1, 1, 1, shape[-1])
    seam_pref = xs.expand(shape[0], 1, h, shape[-1])
    seam_bias = _normalize_signed(mix["seam_macro"] * macro + mix["seam_pref"] * seam_pref)
    region_logits = mix["region_field"] * _procedural_field(shape, generator) + mix["region_seam"] * seam_bias
    region = torch.sigmoid(region_logits * _u(generator, *cfg["region_sigmoid_scale"]))
    return ArtifactFieldBank(tonal=tonal, chroma=chroma, detail=detail, degrade=degrade, region=region, seam_bias=seam_bias)


def _spatial_additive(base: float, field: torch.Tensor, generator: torch.Generator, floor: float) -> torch.Tensor:
    scale = CORRUPTION_CFG["spatial_helpers"]["additive_span_scale"]
    span = max(abs(base) * _u(generator, *scale), floor)
    return base + field * span


def _spatial_factor(base: float, field: torch.Tensor, generator: torch.Generator, *, floor: float, min_value: float, max_value: float) -> torch.Tensor:
    scale = CORRUPTION_CFG["spatial_helpers"]["factor_span_scale"]
    span = max(abs(base - 1.0) * _u(generator, *scale), floor)
    return (base + field * span).clamp(min_value, max_value)


def _spatial_amount(base: float, field: torch.Tensor, generator: torch.Generator, max_value: float) -> torch.Tensor:
    scale = CORRUPTION_CFG["spatial_helpers"]["amount_span_scale"]
    span = max(base * _u(generator, *scale), max_value * 0.10)
    return (base + field * span).clamp(0.0, max_value)


def _apply_corruption_op(name: str, x: torch.Tensor, generator: torch.Generator, fields: ArtifactFieldBank, *, use_spatial: bool | None = None) -> torch.Tensor:
    cfg = _pick_cfg(name)
    if use_spatial is None:
        use_spatial = name in GROUPS["C"] or (torch.rand(1, generator=generator).item() < (CORRUPTION_CFG["spatial_probability"]["d"] if name in GROUPS["D"] else CORRUPTION_CFG["spatial_probability"]["ab"]))
    bank = fields
    x = x.to(dtype=torch.float32)
    if name == "brightness":
        delta = _u(generator, *cfg["base"])
        x = x + (_spatial_additive(delta, cfg["spatial_field_mix"][0] * bank.tonal + cfg["spatial_field_mix"][1] * bank.seam_bias, generator, cfg["spatial_floor"]) if use_spatial else delta)
    elif name == "exposure":
        ev = _u(generator, *cfg["base"])
        if use_spatial:
            ev_map = _spatial_additive(ev, bank.tonal, generator, cfg["spatial_floor"])
            x = x * torch.pow(torch.full_like(ev_map, 2.0), ev_map)
        else:
            x = x * (2.0**ev)
    elif name == "contrast":
        contrast = _u(generator, *cfg["base"])
        mean = x.mean(dim=(-2, -1), keepdim=True)
        if use_spatial:
            contrast_map = _spatial_factor(contrast, cfg["spatial_field_mix"][0] * bank.tonal + cfg["spatial_field_mix"][1] * bank.region, generator, floor=cfg["spatial_floor"], min_value=cfg["clamp"][0], max_value=cfg["clamp"][1])
            x = (x - mean) * contrast_map + mean
        else:
            x = (x - mean) * contrast + mean
    elif name == "gamma":
        gamma = _u(generator, *cfg["base"])
        x = _apply_gamma(x, _spatial_factor(gamma, bank.tonal, generator, floor=cfg["spatial_floor"], min_value=cfg["clamp"][0], max_value=cfg["clamp"][1]) if use_spatial else gamma)
    elif name == "saturation":
        sat = _u(generator, *cfg["base"])
        luma = _rgb_to_luma(x)
        x = luma + (x - luma) * (_spatial_factor(sat, bank.chroma, generator, floor=cfg["spatial_floor"], min_value=cfg["clamp"][0], max_value=cfg["clamp"][1]) if use_spatial else sat)
    elif name == "hue":
        angle = _u(generator, *cfg["base"])
        luma = _rgb_to_luma(x)
        centered = x - luma
        if use_spatial:
            angle_map = _spatial_additive(angle, bank.chroma, generator, cfg["spatial_floor"]).clamp(cfg["clamp"][0], cfg["clamp"][1])
            x = luma + centered.roll(shifts=1, dims=1) * angle_map + centered * (1.0 - angle_map.abs())
        else:
            x = luma + centered.roll(shifts=1, dims=1) * angle + centered * (1.0 - abs(angle))
    elif name == "temperature":
        t = _u(generator, *cfg["base"])
        if use_spatial:
            t_map = _spatial_additive(t, bank.chroma, generator, cfg["spatial_floor"]).clamp(cfg["clamp"][0], cfg["clamp"][1])
            x[:, 0:1] += t_map
            x[:, 2:3] -= t_map
        else:
            x[:, 0:1] += t
            x[:, 2:3] -= t
    elif name == "tint":
        t = _u(generator, *cfg["base"])
        x[:, 1:2] += _spatial_additive(t, bank.chroma, generator, cfg["spatial_floor"]).clamp(cfg["clamp"][0], cfg["clamp"][1]) if use_spatial else t
    elif name == "channel_gains":
        gains = torch.empty((1, 3, 1, 1)).uniform_(*cfg["base"], generator=generator).to(dtype=x.dtype)
        if use_spatial:
            weights = torch.empty((1, 3, 1, 1)).uniform_(*cfg["spatial_weight"], generator=generator).to(dtype=x.dtype)
            gain_map = (gains + bank.chroma * weights * _u(generator, *cfg["spatial_amp"])).clamp(cfg["clamp"][0], cfg["clamp"][1])
            x = x * gain_map
        else:
            x = x * gains
    elif name == "black_point":
        black = _u(generator, *cfg["base"])
        black_val = _spatial_additive(black, bank.tonal, generator, cfg["spatial_floor"]).clamp(cfg["clamp"][0], cfg["clamp"][1]) if use_spatial else black
        x = (x - black_val) / ((1.0 - black_val).clamp_min(1e-3) if isinstance(black_val, torch.Tensor) else max(1.0 - black_val, 1e-3))
    elif name == "white_point":
        white = _u(generator, *cfg["base"])
        white_val = _spatial_factor(white, bank.tonal, generator, floor=cfg["spatial_floor"], min_value=cfg["clamp"][0], max_value=cfg["clamp"][1]) if use_spatial else white
        x = x / (white_val.clamp_min(1e-3) if isinstance(white_val, torch.Tensor) else max(white_val, 1e-3))
    elif name == "shadow_lift":
        amount = _u(generator, *cfg["base"])
        amount_val = _spatial_amount(amount, cfg["field_mix"][0] * bank.tonal + cfg["field_mix"][1] * bank.region, generator, cfg["max_value"]) if use_spatial else amount
        x = x + (1.0 - x) * amount_val * (1.0 - x).pow(2)
    elif name == "shadow_crush":
        amount = _u(generator, *cfg["base"])
        amount_val = _spatial_amount(amount, cfg["field_mix"][0] * bank.tonal + cfg["field_mix"][1] * bank.region, generator, cfg["max_value"]) if use_spatial else amount
        x = x - amount_val * (1.0 - x).pow(2)
    elif name == "highlight_compress":
        amount = _u(generator, *cfg["base"])
        amount_val = _spatial_amount(amount, bank.tonal, generator, cfg["max_value"]) if use_spatial else amount
        x = x - amount_val * x.pow(2)
    elif name == "highlight_boost":
        amount = _u(generator, *cfg["base"])
        amount_val = _spatial_amount(amount, bank.tonal, generator, cfg["max_value"]) if use_spatial else amount
        x = x + amount_val * x.pow(2)
    elif name == "midtone":
        amount = _u(generator, *cfg["base"])
        amount_val = _spatial_additive(amount, bank.tonal, generator, cfg["spatial_floor"]).clamp(cfg["clamp"][0], cfg["clamp"][1]) if use_spatial else amount
        x = x + amount_val * torch.sin(x * math.pi)
    elif name == "s_curve":
        amount = _u(generator, *cfg["base"])
        x = x + (_spatial_amount(amount, bank.tonal, generator, cfg["max_value"]) if use_spatial else amount) * (x - 0.5) * (1.0 - (2.0 * x - 1.0).abs())
    elif name == "reverse_s_curve":
        amount = _u(generator, *cfg["base"])
        x = x - (_spatial_amount(amount, bank.tonal, generator, cfg["max_value"]) if use_spatial else amount) * (x - 0.5) * (1.0 - (2.0 * x - 1.0).abs())
    elif name == "horizontal_luma_gradient":
        xx = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, x.shape[-1])
        amp = _u(generator, *cfg["base"])
        pattern = _normalize_signed(cfg["field_mix"][0] * xx + cfg["field_mix"][1] * bank.tonal.to(device=x.device, dtype=x.dtype))
        x = x + pattern * amp
    elif name == "vertical_luma_gradient":
        yy = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, x.shape[-2], 1)
        amp = _u(generator, *cfg["base"])
        pattern = _normalize_signed(cfg["field_mix"][0] * yy + cfg["field_mix"][1] * bank.tonal.to(device=x.device, dtype=x.dtype))
        x = x + pattern * amp
    elif name == "illumination_field":
        amp = _u(generator, *cfg["amp"])
        field = _normalize_signed(cfg["field_mix"][0] * bank.seam_bias + cfg["field_mix"][1] * bank.tonal).to(device=x.device, dtype=x.dtype)
        x = x * (1.0 + field * amp)
    elif name == "temperature_field":
        amp = _u(generator, *cfg["amp"])
        field = _normalize_signed(cfg["field_mix"][0] * bank.chroma + cfg["field_mix"][1] * bank.seam_bias).to(device=x.device, dtype=x.dtype) * amp
        x[:, 0:1] += field
        x[:, 2:3] -= field * cfg["blue_scale"]
    elif name == "saturation_field":
        field = _normalize_signed(cfg["field_mix"][0] * bank.chroma + cfg["field_mix"][1] * bank.tonal).to(device=x.device, dtype=x.dtype)
        luma = _rgb_to_luma(x)
        x = luma + (x - luma) * (1.0 + field * _u(generator, *cfg["amp"])).clamp(cfg["clamp"][0], cfg["clamp"][1])
    elif name == "blur":
        sigma = _u(generator, *cfg["base"])
        blurred = _gaussian_blur(x, sigma)
        if use_spatial:
            mix = (cfg["mix_floor"] + cfg["mix_scale"] * bank.region).to(device=x.device, dtype=x.dtype) * _u(generator, *cfg["mix_strength"])
            x = x + mix.clamp(0.0, 1.0) * (blurred - x)
        else:
            x = blurred
    elif name == "noise":
        sigma = _u(generator, *cfg["base"])
        if use_spatial:
            sigma_map = (cfg["mix_floor"] + cfg["mix_scale"] * bank.degrade.abs()).to(device=x.device, dtype=x.dtype) * max(sigma, cfg["sigma_floor"])
            x = x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma_map
        else:
            x = x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
    elif name == "microcontrast":
        amount = _u(generator, *cfg["base"])
        blur = _gaussian_blur(x, cfg["blur_sigma"])
        x = x + (x - blur) * (_spatial_amount(amount, bank.detail, generator, cfg["max_value"]) if use_spatial else amount)
    elif name == "jpeg_like":
        lo, hi = cfg["levels"]
        levels = int(torch.randint(lo, hi, (1,), generator=generator).item())
        quantized = torch.round(x.clamp(0.0, 1.0) * float(levels)) / float(levels)
        if use_spatial:
            mix = (cfg["mix_floor"] + cfg["mix_scale"] * bank.region).to(device=x.device, dtype=x.dtype) * _u(generator, *cfg["mix_strength"])
            x = x + mix.clamp(0.0, 1.0) * (quantized - x)
        else:
            x = quantized
    else:
        raise ValueError(f"unsupported corruption op: {name}")
    return x.clamp(0.0, 1.0)


def apply_random_corruptions(inner: torch.Tensor, generator: torch.Generator) -> CorruptionResult:
    x = inner.clone().to(dtype=torch.float32)
    ops: list[str] = []
    fields = _build_artifact_field_bank(x.shape, generator)
    n_ops = int(torch.multinomial(torch.tensor(CORRUPTION_CFG["op_count_weights"]), 1, generator=generator).item()) + 2
    chosen = [_pick_unique(GROUPS["A"] + GROUPS["B"], [], generator)]
    if torch.rand(1, generator=generator).item() < CORRUPTION_CFG["group_c_probability"]:
        chosen.append(_pick_weighted_unique(GROUPS["C"], CORRUPTION_CFG["group_c_weights"], chosen, generator))
    if torch.rand(1, generator=generator).item() < CORRUPTION_CFG["group_d_probability"]:
        chosen.append(_pick_weighted_unique(GROUPS["D"], CORRUPTION_CFG["group_d_weights"], chosen, generator))
    candidates = GROUPS["A"] + GROUPS["B"]
    while len(chosen) < n_ops:
        chosen.append(_pick_unique(candidates, chosen, generator))
    for name in chosen[:n_ops]:
        x = _apply_corruption_op(name, x, generator, fields)
        ops.append(name)
    return CorruptionResult(x.clamp(0.0, 1.0).to(dtype=inner.dtype, device=inner.device), ops)
