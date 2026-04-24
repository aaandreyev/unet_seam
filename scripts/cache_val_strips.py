from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from src.data.manifest import write_jsonl
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def save_gray(tensor, path: Path) -> None:
    arr = (tensor.squeeze(0).numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


def _cache_sample(args: tuple[int, dict, dict, Path]) -> dict:
    idx, sample, row, out_root = args
    sample_id = f"{idx:06d}"
    sample_dir = out_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((sample["input_rgb"].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(sample_dir / "input.png")
    Image.fromarray((sample["target"].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(sample_dir / "target.png")
    Image.fromarray(np.zeros((sample["target"].shape[1], sample["target"].shape[2], 3), dtype="uint8")).save(sample_dir / "residual.png")
    err = (sample["target"] - sample["input_rgb"]).abs().mean(dim=0).numpy()
    Image.fromarray((err * 255).clip(0, 255).astype("uint8")).save(sample_dir / "error.png")
    save_gray(sample["mask"], sample_dir / "mask.png")
    save_gray(sample["distance"], sample_dir / "distance.png")
    meta = {
        "id": sample_id,
        "source_image": row["source_path"],
        "cluster_id": row.get("cluster_id"),
        "split": "val",
        "scene_tags": row.get("scene_tags", []),
        "strip": {
            "axis": sample["meta"]["axis"],
            "original_side": sample["meta"]["side"],
            "seam_x_frac_in_source": sample["meta"]["seam_x_frac_in_source"],
            "flip_h": sample["meta"]["flip_h"],
            "rotation_k": sample["meta"]["rotation_k"],
            "seam_jitter_px": sample["meta"]["seam_jitter_px"],
            "inner_width": sample["meta"]["inner_width"],
            "edge_padded_pixels": sample["meta"]["edge_padded_pixels"],
        },
        "corruption": {"ops": sample["meta"]["ops"]},
        "metrics_precomputed": {"baseline_boundary_mae": 0.0, "baseline_boundary_ciede2000": 0.0, "relative_improvement": 0.0},
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--out", default="outputs/strip_cache/val")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()
    dataset = SyntheticStripDataset(Path(args.manifest), split="val", strips_per_image=1)
    rows = []
    out_root = Path(args.out)
    total = min(args.count, len(dataset))
    tasks = [(idx, dataset[idx], dataset.rows[idx], out_root) for idx in range(total)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        progress = tqdm(executor.map(_cache_sample, tasks), total=total, desc="cache_val_strips", dynamic_ncols=True)
        for meta in progress:
            rows.append(meta)
            progress.set_postfix(done=len(rows), workers=args.workers)
    write_jsonl(Path("manifests/strip_val_cache.jsonl"), rows)


if __name__ == "__main__":
    main()
