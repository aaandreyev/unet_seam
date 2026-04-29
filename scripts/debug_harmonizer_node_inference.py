#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from comfy_node.model_loader import load_model
from comfy_node.strip_ops import mask_bbox, rectangularity
from src.infer.correct_full_frame import apply_corrector_to_full_frame


def _load_rgb(path: Path) -> torch.Tensor:
    arr = torch.from_numpy(__import__("numpy").array(Image.open(path).convert("RGB"))).float() / 255.0
    return arr.permute(2, 0, 1).unsqueeze(0).contiguous()


def _load_mask(path: Path) -> torch.Tensor:
    arr = torch.from_numpy(__import__("numpy").array(Image.open(path).convert("L"))).float() / 255.0
    return (arr > 0.5).float().unsqueeze(0).unsqueeze(0).contiguous()


def _save_tensor(x: torch.Tensor, path: Path) -> None:
    arr = (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def _save_gray(x: torch.Tensor, path: Path) -> None:
    x = x.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
    arr = (x.squeeze().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--inner-width", type=int, default=128)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if rectangularity(mask) < 0.9:
        raise RuntimeError("Only rectangular masks are supported by this debug script")
    bbox = mask_bbox(mask)
    model, sidecar = load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    sides = []
    x0, y0, x1, y1 = bbox
    if x0 >= 32:
        sides.append("left")
    if image.shape[-1] - x1 >= 32:
        sides.append("right")
    if y0 >= 32:
        sides.append("top")
    if image.shape[-2] - y1 >= 32:
        sides.append("bottom")

    corrected, debug = apply_corrector_to_full_frame(
        model,
        image,
        mask,
        bbox,
        sides,
        args.inner_width,
        strength=args.strength,
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_tensor(image[0], out_dir / "input.png")
    _save_tensor(corrected[0], out_dir / "corrected.png")
    _save_gray(mask[0], out_dir / "mask.png")
    _save_tensor((debug["merged_delta"][0] + 0.5).clamp(0.0, 1.0), out_dir / "merged_delta.png")
    for side, delta in debug.get("side_deltas", {}).items():
        _save_tensor((delta[0] + 0.5).clamp(0.0, 1.0), out_dir / f"side_{side}_delta.png")
    for side, weight in debug.get("weights", {}).items():
        _save_gray(weight[0], out_dir / f"weight_{side}.png")
    for side, confidence in debug.get("side_confidences", {}).items():
        _save_gray(confidence[0], out_dir / f"confidence_{side}.png")
    summary = {
        "image": str(args.image),
        "mask": str(args.mask),
        "model": args.model,
        "bbox": bbox,
        "sides": sides,
        "sidecar_model": sidecar.get("model_name"),
        "per_side": debug.get("per_side", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
