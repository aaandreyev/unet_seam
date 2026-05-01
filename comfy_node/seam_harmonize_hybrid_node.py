from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    from .model_loader import load_model
    from .strip_ops import mask_bbox, rectangularity
except ImportError:
    from model_loader import load_model
    from strip_ops import mask_bbox, rectangularity

from src.infer.correct_full_frame import apply_corrector_to_full_frame
from src.infer.cv_mask_harmonize import harmonize_by_mask_torch


class SeamHarmonizerHybridNode:
    @classmethod
    def INPUT_TYPES(cls):
        default_model = str((Path(__file__).resolve().parents[1] / "outputs/exports/seam_harmonizer_v3.safetensors"))
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "MASK": ("MASK",),
                "model_path": ("STRING", {"default": default_model}),
                "route_mode": (["auto", "ml", "cv", "blend"], {"default": "auto"}),
                "region": (["inside", "outside", "both"], {"default": "inside"}),
                "inner_width": ("INT", {"default": 128}),
                "inner_falloff_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0}),
                "strip_width": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "blur_sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "falloff": ("INT", {"default": 64, "min": 1, "max": 2048, "step": 1}),
                "correction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "luminance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "corner_spread": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 4}),
                "max_workdim": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 32}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "debug_previews": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "PROTECT_MASK": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "seam"

    @staticmethod
    def _debug_root() -> Path:
        return Path.cwd() / "outputs" / "debug_previews"

    @staticmethod
    def _ml_sides(image: torch.Tensor, bbox: tuple[int, int, int, int], process_left: bool, process_right: bool, process_top: bool, process_bottom: bool) -> list[str]:
        x0, y0, x1, y1 = bbox
        sides = []
        if process_left and x0 > 0:
            sides.append("left")
        if process_right and image.shape[-1] - x1 > 0:
            sides.append("right")
        if process_top and y0 > 0:
            sides.append("top")
        if process_bottom and image.shape[-2] - y1 > 0:
            sides.append("bottom")
        return sides

    @staticmethod
    def _blend_ml_cv(input_rgb: torch.Tensor, ml_rgb: torch.Tensor, cv_rgb: torch.Tensor, mask: torch.Tensor, region: str) -> torch.Tensor:
        mask_bin = (mask > 0.5).float()
        if region == "outside":
            return cv_rgb

        residual = (cv_rgb - input_rgb).detach().cpu().numpy().astype("float32")
        lowfreq = gaussian_filter(residual, sigma=(0.0, 0.0, 12.0, 12.0), mode="nearest")
        lowfreq_t = torch.from_numpy(lowfreq).to(device=ml_rgb.device, dtype=ml_rgb.dtype)

        dist_in = torch.from_numpy(distance_transform_edt(mask_bin[0, 0].detach().cpu().numpy() > 0.5)).to(device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
        seam_weight = torch.exp(-0.5 * torch.square(dist_in / 8.0)).clamp(0.0, 1.0).pow(0.75)
        if region == "both":
            dist_out = torch.from_numpy(distance_transform_edt((1.0 - mask_bin[0, 0]).detach().cpu().numpy() > 0.5)).to(device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
            outer_weight = torch.exp(-0.5 * torch.square(dist_out / 8.0)).clamp(0.0, 1.0).pow(0.75)
            signed_mask = mask_bin - (1.0 - mask_bin)
            return ml_rgb + lowfreq_t * 0.25 * signed_mask * torch.maximum(seam_weight, outer_weight)
        return ml_rgb + lowfreq_t * 0.25 * seam_weight * mask_bin

    @staticmethod
    def _apply_protect_mask(result: torch.Tensor, original: torch.Tensor, protect_mask: torch.Tensor | None) -> torch.Tensor:
        if protect_mask is None:
            return result
        protect = protect_mask.unsqueeze(1).float() if protect_mask.ndim == 3 else protect_mask.float()
        if protect.shape[-2:] != result.shape[-2:]:
            protect = F.interpolate(protect, size=result.shape[-2:], mode="nearest")
        protect = (protect > 0.5).to(dtype=result.dtype, device=result.device)
        if result.shape[1] != original.shape[1]:
            return result
        return result * (1.0 - protect) + original * protect

    def run(
        self,
        IMAGE,
        MASK,
        model_path,
        route_mode,
        region,
        inner_width,
        inner_falloff_px,
        strength,
        strip_width,
        blur_sigma,
        falloff,
        correction_strength,
        luminance_strength,
        chroma_strength,
        mask_threshold,
        corner_spread,
        max_workdim,
        process_left,
        process_right,
        process_top,
        process_bottom,
        debug_previews,
        PROTECT_MASK=None,
    ):
        image_bchw = IMAGE.permute(0, 3, 1, 2).contiguous()
        rgb = image_bchw[:, :3]
        alpha = image_bchw[:, 3:] if image_bchw.shape[1] > 3 else None
        mask = MASK.unsqueeze(1).float()
        if mask.shape[-2:] != rgb.shape[-2:]:
            mask = F.interpolate(mask, size=rgb.shape[-2:], mode="nearest")
        mask = (mask > mask_threshold).float()
        bbox = mask_bbox(mask)
        rect = rectangularity(mask)
        sides = self._ml_sides(rgb, bbox, process_left, process_right, process_top, process_bottom)
        ml_eligible = region == "inside" and rect >= 0.96 and bool(sides)

        selected_route = route_mode
        if route_mode == "auto":
            selected_route = "blend" if ml_eligible else "cv"
        elif route_mode == "blend" and not ml_eligible:
            selected_route = "cv"
        elif route_mode == "ml" and not ml_eligible:
            raise RuntimeError("ML route requires a near-rectangular mask, inside region, and at least one side with outer context.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        debug = {
            "route_mode": route_mode,
            "selected_route": selected_route,
            "bbox": list(bbox),
            "rectangularity": float(rect),
            "ml_eligible": bool(ml_eligible),
            "sides": sides,
            "region": region,
            "inner_falloff_px": int(inner_falloff_px),
        }

        ml_rgb = None
        ml_debug = None
        if selected_route in {"ml", "blend"}:
            model, _sidecar = load_model(model_path, device=device)
            ml_rgb, ml_debug = apply_corrector_to_full_frame(
                model,
                rgb,
                mask,
                bbox,
                sides,
                inner_width,
                strength,
                inner_falloff_px or None,
            )
            ml_rgb = self._apply_protect_mask(ml_rgb, rgb, PROTECT_MASK)

        cv_input = IMAGE if alpha is None else IMAGE
        cv_result = None
        if selected_route in {"cv", "blend"}:
            cv_result = harmonize_by_mask_torch(
                cv_input,
                MASK,
                mode=region,
                strip_width=strip_width,
                blur_sigma=blur_sigma,
                falloff=falloff,
                correction_strength=correction_strength * strength,
                luminance_strength=luminance_strength,
                chroma_strength=chroma_strength,
                mask_threshold=mask_threshold,
                protect_mask=PROTECT_MASK,
                corner_spread=corner_spread,
                max_workdim=max_workdim,
            )

        if selected_route == "ml":
            result = ml_rgb
            if alpha is not None:
                result = torch.cat((result, alpha), dim=1)
        elif selected_route == "cv":
            result = cv_result.permute(0, 3, 1, 2).contiguous()
        else:
            cv_rgb = cv_result[..., :3].permute(0, 3, 1, 2).contiguous()
            blended_rgb = self._blend_ml_cv(rgb, ml_rgb, cv_rgb, mask, region)
            blended_rgb = self._apply_protect_mask(blended_rgb, rgb, PROTECT_MASK)
            if alpha is not None:
                result = torch.cat((blended_rgb, alpha), dim=1)
            else:
                result = blended_rgb

        debug["ml_debug"] = ml_debug["per_side"] if ml_debug else None
        if debug_previews:
            self._write_debug(debug, image_bchw, result)
        return (result.permute(0, 2, 3, 1).contiguous(),)

    def _write_debug(self, debug: dict, image: torch.Tensor, corrected: torch.Tensor) -> None:
        root = self._debug_root() / datetime.now().strftime("%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        self._save_tensor(image[0], root / "input.png")
        self._save_tensor(corrected[0], root / "corrected.png")
        diff = (corrected[:, :3] - image[:, :3]).abs()
        self._save_tensor((diff[0] / diff[0].amax().clamp_min(1e-6)), root / "diff.png")
        (root / "summary.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")

    @staticmethod
    def _save_tensor(x: torch.Tensor, path: Path) -> None:
        rgb = x[:3].detach().cpu().clamp(0, 1)
        arr = (rgb.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(path)
