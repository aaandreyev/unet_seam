"""Render a fixed visual bench for a single checkpoint.

Picks N reproducible "hard" cases from the val split, runs the harmonizer, and
saves a per-case PNG grid plus a summary.json with quick numerical triggers.

Triggers are diagnostic, NOT auto-rejection: dark_line, halo_band, outside_drift.
The bench exists because boundary metrics alone miss visible artifacts (narrow
dark line, halo, glare). Look at the PNGs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.models.harmonizer import SeamHarmonizerV3
from src.train.checkpoint import load_checkpoint
from src.utils.device import pick_device


def _build_model(ckpt: dict, device: torch.device) -> SeamHarmonizerV3:
    train_cfg = ckpt.get("config") or {}
    model_cfg = train_cfg.get("model") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    model = SeamHarmonizerV3(
        in_channels=int(model_cfg.get("in_channels", 9)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int(dataset_cfg.get("outer_width", 128)),
        boundary_band_px=int(dataset_cfg.get("boundary_band_px", 24)),
        correction_limits=model_cfg.get("correction_limits"),
    ).to(device)
    state = ckpt.get("ema") or ckpt.get("model")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(json.dumps({"event": "load_partial", "missing": list(result.missing_keys)[:6], "unexpected": list(result.unexpected_keys)[:6]}, ensure_ascii=False), flush=True)
    model.eval()
    return model


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0.0, 1.0).cpu().float().numpy() * 255.0 + 0.5).astype(np.uint8)


def _grid_image(panels: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    panels_padded = []
    for p in panels:
        if p.ndim == 2:
            p = np.stack([p, p, p], axis=-1)
        if p.shape[0] < h:
            pad = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        panels_padded.append(p)
    sep = np.full((h, gap, 3), 64, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, p in enumerate(panels_padded):
        parts.append(p)
        if i < len(panels_padded) - 1:
            parts.append(sep)
    return np.concatenate(parts, axis=1)


def _triggers(corrected: torch.Tensor, input_inner: torch.Tensor, outer_width: int, boundary_band_px: int) -> dict[str, float]:
    """Cheap numerical signals for halo / dark line / outside drift.
    All on the inner half of the strip; coordinate 0 is the seam (inner_width starts at outer_width).
    """
    inner_corrected = corrected[..., outer_width:]
    inner_input = input_inner
    h, w = inner_corrected.shape[-2:]
    luma = 0.2126 * inner_corrected[..., 0, :, :] + 0.7152 * inner_corrected[..., 1, :, :] + 0.0722 * inner_corrected[..., 2, :, :]
    band_w = min(boundary_band_px, w - 1)
    band_luma = luma[..., :band_w].mean(dim=-1)
    rest_luma = luma[..., band_w:].mean(dim=-1)
    dark_line = float((band_luma - rest_luma).mean().item())
    halo_lo = min(8, w - 1)
    halo_hi = min(32, w)
    halo_zone = (inner_corrected[..., halo_lo:halo_hi] - inner_input[..., halo_lo:halo_hi]).abs().mean()
    halo_band = float(halo_zone.item())
    outside_band_lo = min(64, w - 1)
    outside_drift = float((inner_corrected[..., outside_band_lo:] - inner_input[..., outside_band_lo:]).abs().mean().item())
    return {"dark_line": dark_line, "halo_band": halo_band, "outside_drift": outside_drift}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl", type=Path)
    parser.add_argument("--out", default="outputs/visual_bench", type=Path)
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--diff-gain", type=float, default=8.0, help="multiplier for visualizing |corrected-input|")
    args = parser.parse_args()

    device = pick_device()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model = _build_model(ckpt, device)

    train_cfg = ckpt.get("config") or {}
    dcfg = train_cfg.get("dataset") or {}
    outer_width = int(dcfg.get("outer_width", 128))
    inner_width = int(dcfg.get("inner_width", 128))
    boundary_band_px = int(dcfg.get("boundary_band_px", 24))
    spec = StripSpec(
        strip_height=int(dcfg.get("strip_height", 1024)),
        outer_width=outer_width,
        inner_width=inner_width,
        seam_jitter_px=0,
    )
    dataset = SyntheticStripDataset(
        args.manifest,
        split="val",
        strips_per_image=1,
        seed=args.seed,
        spec=spec,
        boundary_band_px=boundary_band_px,
        inner_widths=[inner_width],
        apply_corruption=True,
    )
    n = min(args.num_cases, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()

    out_dir = args.out / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        batch = collate_strip_batch([sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.inference_mode():
            out = model(batch["input"])
        corrected = out["corrected_strip"][0]
        input_strip = batch["input_rgb"][0]
        target = batch["target"][0]
        confidence = out["confidence"][0, 0]
        attention = out.get("attention_lowres", torch.zeros_like(confidence[None, None]))
        attn_full = torch.nn.functional.interpolate(attention, size=corrected.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        diff = ((corrected - input_strip).abs() * float(args.diff_gain)).clamp(0.0, 1.0)
        target_diff = ((target - input_strip).abs() * float(args.diff_gain)).clamp(0.0, 1.0)
        panels = [
            _to_uint8(input_strip.permute(1, 2, 0)),
            _to_uint8(corrected.permute(1, 2, 0)),
            _to_uint8(target.permute(1, 2, 0)),
            _to_uint8(diff.permute(1, 2, 0)),
            _to_uint8(target_diff.permute(1, 2, 0)),
            _to_uint8(confidence),
            _to_uint8(attn_full),
        ]
        grid = _grid_image(panels)
        Image.fromarray(grid).save(out_dir / f"case_{i:02d}.png")
        triggers = _triggers(corrected.cpu(), input_strip.cpu()[..., outer_width:], outer_width, boundary_band_px)
        summary.append({"case": i, "index": int(idx), **triggers})

    avg = {key: float(np.mean([s[key] for s in summary])) for key in ("dark_line", "halo_band", "outside_drift")}
    (out_dir / "summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "num_cases": n, "average": avg, "per_case": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"event": "visual_bench_done", "out": str(out_dir), "num_cases": n, "average": avg}, ensure_ascii=False))


if __name__ == "__main__":
    main()
