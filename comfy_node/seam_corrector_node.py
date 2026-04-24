from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from comfy_node.model_loader import load_model
from comfy_node.strip_ops import mask_bbox, rectangularity
from src.infer.correct_full_frame import apply_corrector_to_full_frame


class SeamResidualCorrectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "MASK": ("MASK",),
                "model_path": ("STRING", {"default": "outputs/exports/seam_residual_corrector_v1.safetensors"}),
                "inner_width": ("INT", {"default": 128}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
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

    def run(self, IMAGE, MASK, model_path, inner_width, strength, process_left, process_right, process_top, process_bottom, debug_previews):
        model, sidecar = load_model(model_path, device="cpu")
        if inner_width not in sidecar["strip"]["supported_inner_widths"]:
            raise RuntimeError(f"Unsupported inner_width={inner_width}")
        image = IMAGE.permute(0, 3, 1, 2).contiguous()
        mask = MASK.unsqueeze(1).float()
        if rectangularity(mask) < 0.9:
            raise RuntimeError("v1 supports only rectangular masks")
        bbox = mask_bbox(mask)
        x0, y0, x1, y1 = bbox
        if min(x1 - x0, y1 - y0) < 64:
            return (IMAGE,)
        if mask.mean().item() > 0.98:
            return (IMAGE,)
        sides = []
        if process_left and x0 >= 32:
            sides.append("left")
        if process_right and image.shape[-1] - x1 >= 32:
            sides.append("right")
        if process_top and y0 >= 32:
            sides.append("top")
        if process_bottom and image.shape[-2] - y1 >= 32:
            sides.append("bottom")
        corrected, debug = apply_corrector_to_full_frame(model, image, mask, bbox, sides, inner_width, strength)
        if debug_previews:
            self._write_debug(debug, image, corrected)
        corrected[:, :, :, :x0] = image[:, :, :, :x0]
        corrected[:, :, :, x1:] = image[:, :, :, x1:]
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

    def _write_debug(self, debug: dict, image: torch.Tensor, corrected: torch.Tensor) -> None:
        root = Path("outputs/debug_previews") / datetime.now().strftime("%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        self._save_tensor(image[0], root / "input.png")
        self._save_tensor(corrected[0], root / "corrected.png")
        merged = debug.get("merged_residual")
        if merged is not None:
            self._save_tensor((merged[0] + 0.5).clamp(0.0, 1.0), root / "merged_residual.png")
        for side, residual in debug.get("side_residuals", {}).items():
            self._save_tensor((residual[0] + 0.5).clamp(0.0, 1.0), root / f"side_{side}_residual.png")
        for side, weight in debug.get("weights", {}).items():
            self._save_tensor(weight[0].repeat(3, 1, 1), root / f"weight_map_{side}.png")
        (root / "summary.json").write_text(json.dumps({"per_side": debug.get("per_side", {})}, indent=2), encoding="utf-8")

    @staticmethod
    def _save_tensor(x: torch.Tensor, path: Path) -> None:
        arr = (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(path)
