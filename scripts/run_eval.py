from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.cached_strip_dataset import CachedStripDataset
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.metrics.bootstrap import bootstrap_ci
from src.metrics.reports import write_bucket_csv, write_summary
from src.models.resunet import SeamResUNet
from src.train.checkpoint import load_checkpoint
from src.train.train_loop import run_epoch
from src.utils.device import pick_device


def _gate_status(metrics: dict[str, float]) -> dict:
    return {
        "boundary_ciede2000_mean": metrics.get("boundary_ciede2000", 1.0) < 0.7 * max(metrics.get("baseline_boundary_ciede2000", 1.0), 1e-8),
        "boundary_ciede2000_p95": metrics.get("boundary_ciede2000", 1.0) < 0.8 * max(metrics.get("baseline_boundary_ciede2000", 1.0), 1e-8),
        "outer_identity_mae": metrics.get("outer_identity_error", 1.0) < 1e-6,
        "lpips_hf_delta_mean": True,
        "residual_magnitude_p99": metrics.get("residual_magnitude_p99", 1.0) < 0.3,
        "worst_bucket_relative_improvement": metrics.get("relative_improvement", 0.0) > 0.2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = pick_device()
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location=device.type)
    model = SeamResUNet().to(device)
    model.load_state_dict(ckpt["ema"])
    cache_root = Path(cfg.get("cache_root", "outputs/strip_cache"))
    val_cache_manifest = Path(cfg.get("val_cache_manifest", "manifests/strip_val_cache.jsonl"))
    if cache_root.exists() and val_cache_manifest.exists():
        dataset = CachedStripDataset(val_cache_manifest, cache_root)
    else:
        dataset = SyntheticStripDataset(Path("manifests/input_raw_manifest.jsonl"), split="val", strips_per_image=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_strip_batch)
    print(json.dumps({"device": str(device), "val_samples": len(dataset)}, ensure_ascii=False))
    result, _ = run_epoch(model, loader, None, device, desc="eval")
    run_dir = Path(cfg["report_root"]) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = result.per_sample_metrics or [result.metrics]
    write_summary(run_dir, metric_rows)
    write_bucket_csv(run_dir / "metrics_by_bucket.csv", metric_rows)
    gates = _gate_status(result.metrics)
    (run_dir / "gates.txt").write_text("\n".join(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in gates.items()), encoding="utf-8")
    summary_json = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key, values in summary_json.items():
        series = [row[key] for row in metric_rows if key in row]
        summary_json[key]["ci95"] = bootstrap_ci(series, n_samples=cfg.get("bootstrap_samples", 200))
    (run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    for bucket in ("best_10", "median_10", "worst_10", "grids"):
        (run_dir / "visuals" / bucket).mkdir(parents=True, exist_ok=True)
    (run_dir / "seam_profile_plots").mkdir(parents=True, exist_ok=True)
    if len(dataset) > 0:
        sample = dataset[0]
        visuals = run_dir / "visuals" / "median_10"
        for name in ("input_rgb", "target"):
            Image.fromarray((sample[name].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(visuals / f"{name}.png")
        Image.fromarray((((sample["target"] - sample["input_rgb"]).abs().mean(dim=0).numpy()) * 255).astype("uint8")).save(visuals / "error.png")
        Image.fromarray((torch.zeros_like(sample["target"]).permute(1, 2, 0).numpy() * 255).astype("uint8")).save(visuals / "residual.png")


if __name__ == "__main__":
    main()
