#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.color import rgb2hsv, rgb2lab
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/Users/andreyev-a/ComfyUI")
USER_ARGV = sys.argv[1:]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _bootstrap_comfy_argv(user_argv: list[str]) -> list[str]:
    device = "auto"
    cpu_vae = False
    idx = 0
    while idx < len(user_argv):
        arg = user_argv[idx]
        if arg == "--device" and idx + 1 < len(user_argv):
            device = user_argv[idx + 1].strip().lower()
            idx += 2
            continue
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1].strip().lower()
            idx += 1
            continue
        if arg == "--cpu-vae":
            cpu_vae = True
        idx += 1

    comfy_argv = [sys.argv[0], "--disable-all-custom-nodes"]
    if device == "cpu":
        comfy_argv.append("--cpu")
    elif device not in {"auto", "mps"}:
        raise ValueError(f"Unsupported --device={device!r}; expected auto, cpu, or mps")
    if cpu_vae:
        comfy_argv.append("--cpu-vae")
    return comfy_argv


sys.argv = _bootstrap_comfy_argv(USER_ARGV)

import comfy.options  # type: ignore
comfy.options.enable_args_parsing(True)
import comfy.sd  # type: ignore
import comfy.utils  # type: ignore


def _load_vae(path: Path):
    sd, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _crop_for_vae(vae, pixels: torch.Tensor) -> torch.Tensor:
    return vae.vae_encode_crop_pixels(pixels)


def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().clamp(0.0, 1.0).numpy()


def _psnr(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _deltae76_mean(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    lab_a = rgb2lab(a)
    lab_b = rgb2lab(b)
    de = np.linalg.norm(lab_a - lab_b, axis=-1)
    if mask is not None:
        return float(de[mask].mean()) if mask.any() else 0.0
    return float(de.mean())


def _band_masks(h: int, w: int, band: int = 32) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    border = (yy < band) | (yy >= h - band) | (xx < band) | (xx >= w - band)
    center = ~border
    return border, center


def _signed_channel_mean(diff: np.ndarray) -> list[float]:
    return [float(diff[..., i].mean()) for i in range(diff.shape[-1])]


def _roundtrip_one(vae, path: Path) -> dict[str, Any]:
    pixels = _load_image(path)
    cropped = _crop_for_vae(vae, pixels)
    with torch.inference_mode():
        latent = vae.encode(cropped)
        recon = vae.decode(latent)
    orig = _tensor_to_np(cropped[0])
    rec = _tensor_to_np(recon[0])
    diff = rec - orig
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    hsv_o = rgb2hsv(orig)
    hsv_r = rgb2hsv(rec)
    lab_o = rgb2lab(orig)
    lab_r = rgb2lab(rec)
    border_mask, center_mask = _band_masks(orig.shape[0], orig.shape[1], band=min(32, orig.shape[0] // 8, orig.shape[1] // 8))
    return {
        "path": str(path),
        "shape_h": int(orig.shape[0]),
        "shape_w": int(orig.shape[1]),
        "mae": float(abs_diff.mean()),
        "mse": mse,
        "psnr": _psnr(mse),
        "max_abs": float(abs_diff.max()),
        "deltae76_mean": _deltae76_mean(orig, rec),
        "deltae76_border_mean": _deltae76_mean(orig, rec, border_mask),
        "deltae76_center_mean": _deltae76_mean(orig, rec, center_mask),
        "rgb_shift_r": float(diff[..., 0].mean()),
        "rgb_shift_g": float(diff[..., 1].mean()),
        "rgb_shift_b": float(diff[..., 2].mean()),
        "lab_shift_L": float((lab_r[..., 0] - lab_o[..., 0]).mean()),
        "lab_shift_a": float((lab_r[..., 1] - lab_o[..., 1]).mean()),
        "lab_shift_b": float((lab_r[..., 2] - lab_o[..., 2]).mean()),
        "hue_shift_mean": float((hsv_r[..., 0] - hsv_o[..., 0]).mean()),
        "sat_shift_mean": float((hsv_r[..., 1] - hsv_o[..., 1]).mean()),
        "val_shift_mean": float((hsv_r[..., 2] - hsv_o[..., 2]).mean()),
        "sat_abs_mae": float(np.abs(hsv_r[..., 1] - hsv_o[..., 1]).mean()),
        "val_abs_mae": float(np.abs(hsv_r[..., 2] - hsv_o[..., 2]).mean()),
        "orig_mean_r": float(orig[..., 0].mean()),
        "orig_mean_g": float(orig[..., 1].mean()),
        "orig_mean_b": float(orig[..., 2].mean()),
        "recon_mean_r": float(rec[..., 0].mean()),
        "recon_mean_g": float(rec[..., 1].mean()),
        "recon_mean_b": float(rec[..., 2].mean()),
        "_orig": orig,
        "_recon": rec,
        "_diff": abs_diff,
    }


def _save_preview(row: dict[str, Any], out_dir: Path, rank: int) -> None:
    orig = Image.fromarray((row["_orig"] * 255.0).clip(0, 255).astype("uint8"))
    recon = Image.fromarray((row["_recon"] * 255.0).clip(0, 255).astype("uint8"))
    diff = Image.fromarray((row["_diff"] / max(row["_diff"].max(), 1e-6) * 255.0).clip(0, 255).astype("uint8"))
    w, h = orig.size
    canvas = Image.new("RGB", (w * 3, h + 28), color=(18, 24, 32))
    canvas.paste(orig, (0, 28))
    canvas.paste(recon, (w, 28))
    canvas.paste(diff, (w * 2, 28))
    canvas.save(out_dir / f"worst_{rank:02d}_{Path(row['path']).stem}.png")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mae",
        "mse",
        "psnr",
        "max_abs",
        "deltae76_mean",
        "deltae76_border_mean",
        "deltae76_center_mean",
        "rgb_shift_r",
        "rgb_shift_g",
        "rgb_shift_b",
        "lab_shift_L",
        "lab_shift_a",
        "lab_shift_b",
        "hue_shift_mean",
        "sat_shift_mean",
        "val_shift_mean",
        "sat_abs_mae",
        "val_abs_mae",
    ]
    out: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    out["worst_by_deltae76"] = [
        {
            "path": r["path"],
            "deltae76_mean": r["deltae76_mean"],
            "deltae76_border_mean": r["deltae76_border_mean"],
            "sat_shift_mean": r["sat_shift_mean"],
            "val_shift_mean": r["val_shift_mean"],
            "rgb_shift": [r["rgb_shift_r"], r["rgb_shift_g"], r["rgb_shift_b"]],
        }
        for r in sorted(rows, key=lambda x: x["deltae76_mean"], reverse=True)[:10]
    ]
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    plain_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(plain_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plain_rows)


def _write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], out_path: Path, vae_path: Path, data_dir: Path) -> None:
    s = summary
    border_gap = s["deltae76_border_mean"]["mean"] - s["deltae76_center_mean"]["mean"]
    drift_rgb = [s["rgb_shift_r"]["mean"], s["rgb_shift_g"]["mean"], s["rgb_shift_b"]["mean"]]
    lines = [
        "# VAE Roundtrip Analysis",
        "",
        f"- VAE: `{vae_path}`",
        f"- Dataset: `{data_dir}`",
        f"- Images analyzed: `{s['n']}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- Mean MAE: `{s['mae']['mean']:.6f}`",
        f"- Mean PSNR: `{s['psnr']['mean']:.2f}` dB",
        f"- Mean ΔE76: `{s['deltae76_mean']['mean']:.4f}`",
        f"- Mean border ΔE76: `{s['deltae76_border_mean']['mean']:.4f}`",
        f"- Mean center ΔE76: `{s['deltae76_center_mean']['mean']:.4f}`",
        f"- Border-center ΔE gap: `{border_gap:.4f}`",
        f"- Mean RGB shift: `R {drift_rgb[0]:+.6f}, G {drift_rgb[1]:+.6f}, B {drift_rgb[2]:+.6f}`",
        f"- Mean Lab shift: `L {s['lab_shift_L']['mean']:+.4f}, a {s['lab_shift_a']['mean']:+.4f}, b {s['lab_shift_b']['mean']:+.4f}`",
        f"- Mean saturation shift: `{s['sat_shift_mean']['mean']:+.6f}`",
        f"- Mean value shift: `{s['val_shift_mean']['mean']:+.6f}`",
        "",
        "## Interpretation",
        "",
    ]
    if abs(s["sat_shift_mean"]["mean"]) > 0.01 or abs(s["val_shift_mean"]["mean"]) > 0.01:
        lines.append("- There is a systematic global color/value drift in pure VAE roundtrip.")
    else:
        lines.append("- Global mean saturation/value drift is small; any visible seams are more likely from inpaint conditioning / latent mixing than from a pure global VAE tint.")
    if border_gap > 0.15:
        lines.append("- Border error is materially higher than center error, which suggests edge/crop/tile behavior contributes to visible seams.")
    else:
        lines.append("- Border error is close to center error, so the VAE does not show a strong edge-only failure mode on pure roundtrip.")
    if max(abs(x) for x in drift_rgb) > 0.005:
        lines.append("- RGB channel means show non-trivial signed bias, so there is measurable channel drift in the roundtrip.")
    else:
        lines.append("- Signed RGB drift is small; the dominant error is likely reconstruction loss rather than a strong fixed tint.")
    lines += [
        "",
        "## Worst Cases by ΔE76",
        "",
    ]
    for row in summary["worst_by_deltae76"]:
        lines.append(
            f"- `{Path(row['path']).name}`: ΔE76 `{row['deltae76_mean']:.4f}`, border ΔE76 `{row['deltae76_border_mean']:.4f}`, sat shift `{row['sat_shift_mean']:+.4f}`, value shift `{row['val_shift_mean']:+.4f}`"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_partial(rows: list[dict[str, Any]], out_dir: Path, vae_path: Path, data_dir: Path) -> None:
    if not rows:
        return
    summary = _aggregate(rows)
    _write_csv(rows, out_dir / "partial_per_image_metrics.csv")
    _write_markdown(summary, rows, out_dir / "partial_report.md", vae_path, data_dir)
    (out_dir / "partial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, default=REPO_ROOT / "vae" / "flux2-vae.safetensors")
    ap.add_argument("--data-dir", type=Path, default=Path("/Users/andreyev-a/pet_projects/seams_lora/hard_dataset/crops"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "vae_roundtrip_analysis")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--cpu-vae", action="store_true", help="Force Comfy VAE execution on CPU while keeping the rest of the bootstrap unchanged.")
    args = ap.parse_args(USER_ARGV)

    image_paths = sorted([p for p in args.data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = _load_vae(args.vae)
    rows: list[dict[str, Any]] = []
    print(json.dumps({"event": "vae_loaded", "vae": str(args.vae), "images": len(image_paths)}, ensure_ascii=False), flush=True)
    progress = tqdm(image_paths, desc="VAE roundtrip", unit="img", dynamic_ncols=True)
    rolling_mae: list[float] = []
    rolling_de: list[float] = []
    for idx, path in enumerate(progress, start=1):
        row = _roundtrip_one(vae, path)
        rows.append(row)
        rolling_mae.append(float(row["mae"]))
        rolling_de.append(float(row["deltae76_mean"]))
        rolling_mae = rolling_mae[-10:]
        rolling_de = rolling_de[-10:]
        progress.set_postfix(
            mae=f"{np.mean(rolling_mae):.4f}",
            de76=f"{np.mean(rolling_de):.2f}",
            last=path.name[:24],
        )
        if idx % max(args.save_every, 1) == 0 or idx == len(image_paths):
            _write_partial(rows, args.output_dir, args.vae, args.data_dir)
        if idx == 1 or idx % 10 == 0 or idx == len(image_paths):
            print(
                json.dumps(
                    {
                        "event": "vae_roundtrip_progress",
                        "idx": idx,
                        "total": len(image_paths),
                        "file": path.name,
                        "mae": round(row["mae"], 6),
                        "deltae76": round(row["deltae76_mean"], 4),
                        "rolling_mae10": round(float(np.mean(rolling_mae)), 6),
                        "rolling_de10": round(float(np.mean(rolling_de)), 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = _aggregate(rows)
    _write_csv(rows, args.output_dir / "per_image_metrics.csv")
    for rank, row in enumerate(sorted(rows, key=lambda x: x["deltae76_mean"], reverse=True)[:10], start=1):
        _save_preview(row, args.output_dir, rank)
    _write_markdown(summary, rows, args.output_dir / "report.md", args.vae, args.data_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "vae_roundtrip_done", "output_dir": str(args.output_dir), "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
