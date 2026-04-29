from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom


def _resize_array(src: np.ndarray, width: int, height: int, order: int) -> np.ndarray:
    if src.ndim == 2:
        zoom_factors = (height / src.shape[0], width / src.shape[1])
    else:
        zoom_factors = (height / src.shape[0], width / src.shape[1], 1.0)
    return zoom(src, zoom_factors, order=order, mode="nearest").astype(src.dtype, copy=False)


def _fast_distance_transform(mask_bin: np.ndarray) -> np.ndarray:
    return distance_transform_edt(mask_bin.astype(bool)).astype(np.float32)


def _distance_to_seed_map(seed: np.ndarray) -> np.ndarray:
    if not np.any(seed):
        return np.full(seed.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~seed.astype(bool)).astype(np.float32)


def _to_lab(img_u8: np.ndarray) -> np.ndarray:
    rgb = img_u8.astype(np.float32) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.empty_like(rgb, dtype=np.float32)
    xyz[..., 0] = rgb[..., 0] * 0.4124564 + rgb[..., 1] * 0.3575761 + rgb[..., 2] * 0.1804375
    xyz[..., 1] = rgb[..., 0] * 0.2126729 + rgb[..., 1] * 0.7151522 + rgb[..., 2] * 0.0721750
    xyz[..., 2] = rgb[..., 0] * 0.0193339 + rgb[..., 1] * 0.1191920 + rgb[..., 2] * 0.9503041
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / white
    delta = 6.0 / 29.0
    xyz_f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(xyz_f, dtype=np.float32)
    lab[..., 0] = 116.0 * xyz_f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (xyz_f[..., 0] - xyz_f[..., 1])
    lab[..., 2] = 200.0 * (xyz_f[..., 1] - xyz_f[..., 2])
    return lab


def _lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0
    xyz = np.empty_like(lab, dtype=np.float32)
    xyz[..., 0] = np.where(fx > delta, fx**3, 3 * delta**2 * (fx - 4.0 / 29.0))
    xyz[..., 1] = np.where(fy > delta, fy**3, 3 * delta**2 * (fy - 4.0 / 29.0))
    xyz[..., 2] = np.where(fz > delta, fz**3, 3 * delta**2 * (fz - 4.0 / 29.0))
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz * white
    rgb = np.empty_like(xyz, dtype=np.float32)
    rgb[..., 0] = xyz[..., 0] * 3.2404542 + xyz[..., 1] * -1.5371385 + xyz[..., 2] * -0.4985314
    rgb[..., 1] = xyz[..., 0] * -0.9692660 + xyz[..., 1] * 1.8760108 + xyz[..., 2] * 0.0415560
    rgb[..., 2] = xyz[..., 0] * 0.0556434 + xyz[..., 1] * -0.2040259 + xyz[..., 2] * 1.0572252
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _gaussian_blur_aniso(src: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    sigma_x = max(float(sigma_x), 1e-3)
    sigma_y = max(float(sigma_y), 1e-3)
    if src.ndim == 2:
        return gaussian_filter(src.astype(np.float32), sigma=(sigma_y, sigma_x), mode="nearest").astype(np.float32)
    out = np.empty_like(src, dtype=np.float32)
    for c in range(src.shape[2]):
        out[:, :, c] = gaussian_filter(src[:, :, c].astype(np.float32), sigma=(sigma_y, sigma_x), mode="nearest")
    return out


def _gaussian_distance_weight(dist: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    weight = np.exp(-0.5 * np.square(dist.astype(np.float32) / sigma))
    weight[~np.isfinite(dist)] = 0.0
    return weight.astype(np.float32)


def _compute_harmonize_delta(
    img_u8: np.ndarray,
    mask_np: np.ndarray,
    *,
    strip_width: int,
    blur_sigma: float,
    mask_threshold: float,
    alpha_np: np.ndarray | None = None,
    protect_mask_np: np.ndarray | None = None,
    corner_spread: int = 0,
) -> np.ndarray | None:
    h, w = img_u8.shape[:2]
    lab = _to_lab(img_u8)

    mask_f = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
    mask_bin = (mask_f >= mask_threshold).astype(np.uint8)
    if mask_bin.sum() == 0 or mask_bin.sum() == h * w:
        return None

    dist_in = _fast_distance_transform(mask_bin)
    dist_out = _fast_distance_transform(1 - mask_bin)
    inner_strip = (mask_bin == 1) & (dist_in > 0) & (dist_in <= strip_width)
    outer_strip = (mask_bin == 0) & (dist_out > 0) & (dist_out <= strip_width)
    if inner_strip.sum() == 0 or outer_strip.sum() == 0:
        return None

    effective_sigma = max(float(blur_sigma), 0.5)
    strip_cross_sigma = max(float(strip_width) * 0.35, 0.75)
    corner_mix_sigma = float(corner_spread) if corner_spread > 0 else max(float(strip_width) * 0.35, 2.0)
    alpha_valid = np.clip(alpha_np.astype(np.float32), 0.0, 1.0) if alpha_np is not None else np.ones((h, w), dtype=np.float32)
    protect_valid = 1.0 - np.clip(protect_mask_np.astype(np.float32), 0.0, 1.0) if protect_mask_np is not None else np.ones((h, w), dtype=np.float32)
    pixel_valid = np.clip(alpha_valid * protect_valid, 0.0, 1.0)

    eps = 1e-4
    inner_edge = inner_strip & (dist_in <= 1.5)
    outer_edge = outer_strip & (dist_out <= 1.5)

    outside_up = np.zeros((h, w), dtype=bool)
    outside_up[1:, :] = mask_bin[:-1, :] == 0
    outside_down = np.zeros((h, w), dtype=bool)
    outside_down[:-1, :] = mask_bin[1:, :] == 0
    outside_left = np.zeros((h, w), dtype=bool)
    outside_left[:, 1:] = mask_bin[:, :-1] == 0
    outside_right = np.zeros((h, w), dtype=bool)
    outside_right[:, :-1] = mask_bin[:, 1:] == 0

    inside_up = np.zeros((h, w), dtype=bool)
    inside_up[1:, :] = mask_bin[:-1, :] == 1
    inside_down = np.zeros((h, w), dtype=bool)
    inside_down[:-1, :] = mask_bin[1:, :] == 1
    inside_left = np.zeros((h, w), dtype=bool)
    inside_left[:, 1:] = mask_bin[:, :-1] == 1
    inside_right = np.zeros((h, w), dtype=bool)
    inside_right[:, :-1] = mask_bin[:, 1:] == 1

    side_specs = {
        "top": {
            "inner_strip": inner_strip & outside_up,
            "outer_strip": outer_strip & inside_down,
            "geom_seed": (inner_edge & outside_up) | (outer_edge & inside_down),
            "sigma_x": effective_sigma,
            "sigma_y": strip_cross_sigma,
        },
        "bottom": {
            "inner_strip": inner_strip & outside_down,
            "outer_strip": outer_strip & inside_up,
            "geom_seed": (inner_edge & outside_down) | (outer_edge & inside_up),
            "sigma_x": effective_sigma,
            "sigma_y": strip_cross_sigma,
        },
        "left": {
            "inner_strip": inner_strip & outside_left,
            "outer_strip": outer_strip & inside_right,
            "geom_seed": (inner_edge & outside_left) | (outer_edge & inside_right),
            "sigma_x": strip_cross_sigma,
            "sigma_y": effective_sigma,
        },
        "right": {
            "inner_strip": inner_strip & outside_right,
            "outer_strip": outer_strip & inside_left,
            "geom_seed": (inner_edge & outside_right) | (outer_edge & inside_left),
            "sigma_x": strip_cross_sigma,
            "sigma_y": effective_sigma,
        },
    }

    side_fields: list[dict[str, np.ndarray | bool]] = []
    for spec in side_specs.values():
        if not np.any(spec["geom_seed"]):
            continue

        inner_side = spec["inner_strip"]
        outer_side = spec["outer_strip"]

        inner_pack = np.zeros((h, w, 4), dtype=np.float32)
        inner_pack[inner_side, :3] = lab[inner_side] * pixel_valid[inner_side, None]
        inner_pack[inner_side, 3] = pixel_valid[inner_side]

        outer_pack = np.zeros((h, w, 4), dtype=np.float32)
        outer_pack[outer_side, :3] = lab[outer_side] * pixel_valid[outer_side, None]
        outer_pack[outer_side, 3] = pixel_valid[outer_side]

        inner_pack_s = _gaussian_blur_aniso(inner_pack, spec["sigma_x"], spec["sigma_y"])
        outer_pack_s = _gaussian_blur_aniso(outer_pack, spec["sigma_x"], spec["sigma_y"])

        inner_wt_s = inner_pack_s[:, :, 3]
        outer_wt_s = outer_pack_s[:, :, 3]
        inner_mean = np.where(inner_wt_s[:, :, None] > eps, inner_pack_s[:, :, :3] / np.maximum(inner_wt_s[:, :, None], eps), 0.0)
        outer_mean = np.where(outer_wt_s[:, :, None] > eps, outer_pack_s[:, :, :3] / np.maximum(outer_wt_s[:, :, None], eps), 0.0)
        side_valid = (inner_wt_s > eps) & (outer_wt_s > eps)
        side_delta = np.where(side_valid[:, :, None], outer_mean - inner_mean, 0.0).astype(np.float32)
        valid_seed = spec["geom_seed"] & side_valid
        geom_dist = _distance_to_seed_map(spec["geom_seed"])

        if np.any(valid_seed):
            valid_dist, indices = distance_transform_edt(~valid_seed.astype(bool), return_indices=True)
            idx_y = indices[0].astype(np.int32)
            idx_x = indices[1].astype(np.int32)
            nearest_delta = side_delta[idx_y, idx_x]
            side_fields.append({
                "geom_dist": geom_dist,
                "valid_dist": valid_dist.astype(np.float32),
                "nearest_delta": nearest_delta,
                "has_valid": True,
            })
        else:
            side_fields.append({
                "geom_dist": geom_dist,
                "valid_dist": np.full((h, w), np.inf, dtype=np.float32),
                "nearest_delta": np.zeros((h, w, 3), dtype=np.float32),
                "has_valid": False,
            })

    if not side_fields:
        return None

    min_geom_dist = np.full((h, w), np.inf, dtype=np.float32)
    for field in side_fields:
        min_geom_dist = np.minimum(min_geom_dist, field["geom_dist"])

    delta_acc = np.zeros((h, w, 3), dtype=np.float32)
    geom_weight_sum = np.zeros((h, w), dtype=np.float32)
    for field in side_fields:
        angle_delta = np.maximum(field["geom_dist"] - min_geom_dist, 0.0)
        geom_weight = _gaussian_distance_weight(angle_delta, corner_mix_sigma)
        geom_weight_sum += geom_weight
        if not field["has_valid"]:
            continue
        validity_gap = np.maximum(field["valid_dist"] - field["geom_dist"], 0.0)
        valid_weight = geom_weight * _gaussian_distance_weight(validity_gap, corner_mix_sigma)
        delta_acc += field["nearest_delta"] * valid_weight[:, :, None]

    return np.where(
        geom_weight_sum[:, :, None] > 1e-8,
        delta_acc / np.maximum(geom_weight_sum[:, :, None], 1e-8),
        0.0,
    ).astype(np.float32)


def harmonize_by_mask_np(
    img_u8: np.ndarray,
    mask_np: np.ndarray,
    *,
    mode: str = "inside",
    strip_width: int = 8,
    blur_sigma: float = 20.0,
    falloff: int = 64,
    correction_strength: float = 1.0,
    luminance_strength: float = 1.0,
    chroma_strength: float = 1.0,
    mask_threshold: float = 0.5,
    alpha_np: np.ndarray | None = None,
    protect_mask_np: np.ndarray | None = None,
    corner_spread: int = 0,
    max_workdim: int = 640,
) -> np.ndarray:
    h, w = img_u8.shape[:2]
    mask_f = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
    mask_bin = (mask_f >= mask_threshold).astype(np.uint8)
    if mask_bin.sum() == 0 or mask_bin.sum() == h * w:
        return img_u8.copy()

    work_scale = min(1.0, max_workdim / float(max(h, w)))
    if work_scale < 1.0:
        work_h = max(1, int(round(h * work_scale)))
        work_w = max(1, int(round(w * work_scale)))
        work_img_u8 = _resize_array(img_u8, work_w, work_h, order=1).astype(np.uint8)
        work_mask = _resize_array(mask_f, work_w, work_h, order=1).astype(np.float32)
        work_alpha = _resize_array(alpha_np.astype(np.float32), work_w, work_h, order=1).astype(np.float32) if alpha_np is not None else None
        work_protect = _resize_array(protect_mask_np.astype(np.float32), work_w, work_h, order=1).astype(np.float32) if protect_mask_np is not None else None
        delta_smooth = _compute_harmonize_delta(
            work_img_u8,
            work_mask,
            strip_width=max(1, int(round(strip_width * work_scale))),
            blur_sigma=max(float(blur_sigma) * work_scale, 0.5),
            mask_threshold=mask_threshold,
            alpha_np=work_alpha,
            protect_mask_np=work_protect,
            corner_spread=int(round(corner_spread * work_scale)),
        )
        if delta_smooth is None:
            return img_u8.copy()
        delta_smooth = _resize_array(delta_smooth, w, h, order=1).astype(np.float32)
    else:
        delta_smooth = _compute_harmonize_delta(
            img_u8,
            mask_f,
            strip_width=strip_width,
            blur_sigma=blur_sigma,
            mask_threshold=mask_threshold,
            alpha_np=alpha_np,
            protect_mask_np=protect_mask_np,
            corner_spread=corner_spread,
        )
        if delta_smooth is None:
            return img_u8.copy()

    lab = _to_lab(img_u8)
    alpha_valid = np.clip(alpha_np.astype(np.float32), 0.0, 1.0) if alpha_np is not None else np.ones((h, w), dtype=np.float32)
    protect_valid = 1.0 - np.clip(protect_mask_np.astype(np.float32), 0.0, 1.0) if protect_mask_np is not None else np.ones((h, w), dtype=np.float32)
    alpha_f32 = np.clip(alpha_valid * protect_valid, 0.0, 1.0).astype(np.float32)
    channel_scale = np.array(
        [
            correction_strength * luminance_strength,
            correction_strength * chroma_strength,
            correction_strength * chroma_strength,
        ],
        dtype=np.float32,
    )

    result_lab = lab.copy()
    scale = 0.5 if mode == "both" else 1.0
    if mode in ("inside", "both"):
        dist_in = _fast_distance_transform(mask_bin)
        t_in = np.clip(dist_in / max(float(falloff), 1.0), 0.0, 1.0)
        falloff_in = (0.5 * (1.0 + np.cos(np.pi * t_in))).astype(np.float32)
        weight_in = (falloff_in * mask_f * alpha_f32)[:, :, None] * channel_scale[None, None, :]
        result_lab += delta_smooth * weight_in * scale

    if mode in ("outside", "both"):
        dist_out = _fast_distance_transform(1 - mask_bin)
        t_out = np.clip(dist_out / max(float(falloff), 1.0), 0.0, 1.0)
        falloff_out = (0.5 * (1.0 + np.cos(np.pi * t_out))).astype(np.float32)
        weight_out = (falloff_out * (1.0 - mask_f) * alpha_f32)[:, :, None] * channel_scale[None, None, :]
        result_lab -= delta_smooth * weight_out * scale

    rgb_u8 = _lab_to_rgb_u8(result_lab)
    if protect_mask_np is not None:
        protect = np.clip(protect_mask_np.astype(np.float32), 0.0, 1.0)[..., None]
        rgb_u8 = np.where(protect > 0.5, img_u8, rgb_u8)
    return rgb_u8


def harmonize_by_mask_torch(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "inside",
    strip_width: int = 8,
    blur_sigma: float = 20.0,
    falloff: int = 64,
    correction_strength: float = 1.0,
    luminance_strength: float = 1.0,
    chroma_strength: float = 1.0,
    mask_threshold: float = 0.5,
    protect_mask: torch.Tensor | None = None,
    corner_spread: int = 0,
    max_workdim: int = 640,
) -> torch.Tensor:
    device = image.device
    batch = image.shape[0]
    results = []
    for b in range(batch):
        frame = image[b].detach().cpu().numpy()
        h, w = frame.shape[:2]
        if frame.shape[2] == 4:
            alpha_np = np.clip(frame[:, :, 3], 0.0, 1.0).astype(np.float32)
            img_u8 = (frame[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
        else:
            alpha_np = None
            img_u8 = (frame[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)

        mask_tensor = mask[b] if mask.ndim == 3 else mask[b, 0]
        mask_np = mask_tensor.detach().cpu().numpy().astype(np.float32)
        if mask_np.shape != (h, w):
            mask_np = _resize_array(mask_np, w, h, order=1).astype(np.float32)

        protect_np = None
        if protect_mask is not None:
            protect_tensor = protect_mask[b] if protect_mask.ndim == 3 else protect_mask[b, 0]
            protect_np = protect_tensor.detach().cpu().numpy().astype(np.float32)
            if protect_np.shape != (h, w):
                protect_np = _resize_array(protect_np, w, h, order=1).astype(np.float32)

        result_rgb = harmonize_by_mask_np(
            img_u8,
            mask_np,
            mode=mode,
            strip_width=strip_width,
            blur_sigma=blur_sigma,
            falloff=falloff,
            correction_strength=correction_strength,
            luminance_strength=luminance_strength,
            chroma_strength=chroma_strength,
            mask_threshold=mask_threshold,
            alpha_np=alpha_np,
            protect_mask_np=protect_np,
            corner_spread=corner_spread,
            max_workdim=max_workdim,
        ).astype(np.float32) / 255.0
        result_t = torch.from_numpy(result_rgb)
        if alpha_np is not None:
            result_t = torch.cat((result_t, torch.from_numpy(alpha_np).unsqueeze(-1)), dim=-1)
        results.append(result_t)
    return torch.stack(results).to(device=device, dtype=image.dtype)
