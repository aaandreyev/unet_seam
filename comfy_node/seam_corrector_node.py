from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .model_loader import load_model
    from .strip_ops import mask_bbox
except ImportError:
    from model_loader import load_model
    from strip_ops import mask_bbox
from src.infer.correct_full_frame import apply_corrector_to_full_frame


class SeamHarmonizerV3Node:
    @classmethod
    def INPUT_TYPES(cls):
        default_model = str((Path(__file__).resolve().parents[1] / "outputs/exports/seam_harmonizer_v3.safetensors"))
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "MASK": ("MASK",),
                "model_path": ("STRING", {"default": default_model}),
                "inner_width": ("INT", {"default": 128}),
                "inner_falloff_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "debug_previews": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "seam"

    @staticmethod
    def _debug_root() -> Path:
        return Path.cwd() / "outputs" / "debug_previews"

    def run(
        self,
        IMAGE,
        MASK,
        model_path,
        inner_width,
        inner_falloff_px,
        strength,
        process_left,
        process_right,
        process_top,
        process_bottom,
        debug_previews,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _sidecar = load_model(model_path, device=device)
        image = IMAGE.permute(0, 3, 1, 2).contiguous()
        mask = MASK.unsqueeze(1).float()
        original_mask_shape = [int(mask.shape[-2]), int(mask.shape[-1])]
        image_hw = [int(image.shape[-2]), int(image.shape[-1])]
        mask_resized = False
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
            mask_resized = True
        mask = (mask > 0.5).float()
        bbox = mask_bbox(mask)
        x0, y0, x1, y1 = bbox
        base_meta = {
            "bbox": [x0, y0, x1, y1],
            "mask_mean": float(mask.mean().item()),
            "inner_width": int(inner_width),
            "inner_falloff_px": int(inner_falloff_px),
            "strength": float(strength),
            "model_path": str(model_path),
            "original_mask_shape": original_mask_shape,
            "image_shape_hw": image_hw,
            "mask_resized_to_image": mask_resized,
        }
        sides = []
        if process_left and x0 > 0:
            sides.append("left")
        if process_right and image.shape[-1] - x1 > 0:
            sides.append("right")
        if process_top and y0 > 0:
            sides.append("top")
        if process_bottom and image.shape[-2] - y1 > 0:
            sides.append("bottom")
        if not sides:
            if debug_previews:
                self._write_debug({"per_side": {}, "reason": "no_processable_sides"}, image, image, extra={**base_meta, "sides": sides})
            return (IMAGE,)
        corrected, debug = apply_corrector_to_full_frame(
            model,
            image,
            mask,
            bbox,
            sides,
            inner_width,
            strength,
            inner_falloff_px or None,
        )
        if debug_previews:
            self._write_debug(debug, image, corrected, extra={**base_meta, "sides": sides})
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

    def _write_debug(self, debug: dict, image: torch.Tensor, corrected: torch.Tensor, extra: dict | None = None) -> None:
        root = self._debug_root() / datetime.now().strftime("%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        self._save_tensor(image[0], root / "input.png")
        self._save_tensor(corrected[0], root / "corrected.png")
        diff = (corrected - image).abs()
        self._save_tensor((diff[0] / diff[0].amax().clamp_min(1e-6)), root / "diff.png")
        merged = debug.get("merged_delta")
        if merged is not None:
            self._save_tensor((merged[0] + 0.5).clamp(0.0, 1.0), root / "merged_delta.png")
        for side, delta in debug.get("side_deltas", {}).items():
            self._save_tensor((delta[0] + 0.5).clamp(0.0, 1.0), root / f"side_{side}_delta.png")
        for side, confidence in debug.get("side_confidences", {}).items():
            self._save_tensor(confidence[0].repeat(3, 1, 1), root / f"confidence_{side}.png")
        for side, weight in debug.get("weights", {}).items():
            self._save_tensor(weight[0].repeat(3, 1, 1), root / f"weight_map_{side}.png")
        for key in ("gain_lowres", "gamma_lowres", "gate_lowres"):
            value = debug.get(key)
            if isinstance(value, torch.Tensor):
                for i in range(min(value.shape[0], 4)):
                    preview = value[i].repeat(3, 1, 1)
                    preview = (preview - preview.min()) / (preview.max() - preview.min()).clamp_min(1e-6)
                    self._save_tensor(preview, root / f"{key}_{i}.png")
        summary = {
            "per_side": debug.get("per_side", {}),
            "reason": debug.get("reason", "applied"),
            "debug_root": str(root),
            "max_abs_change": float(diff.max().item()),
            "mean_abs_change": float(diff.mean().item()),
        }
        if extra:
            summary.update(extra)
            bbox = extra.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                mask_preview = torch.zeros(3, image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype)
                x0, y0, x1, y1 = [int(v) for v in bbox]
                mask_preview[:, y0:y1, x0:x1] = 1.0
                self._save_tensor(mask_preview, root / "mask.png")
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _save_tensor(x: torch.Tensor, path: Path) -> None:
        arr = (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(path)
