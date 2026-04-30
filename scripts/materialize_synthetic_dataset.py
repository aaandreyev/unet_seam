from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import read_jsonl, write_jsonl
from src.data.corruptions import apply_random_corruptions
from src.data.harmonizer_input import build_harmonizer_input
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype("uint8")


def _tensor_to_mask_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.numpy().clip(0.0, 1.0) * 255.0).astype("uint8")


def _should_use_tqdm() -> bool:
    return sys.stdout.isatty() and os.environ.get("UNET_SEAM_PLAIN_PROGRESS") != "1"


def _maybe_log_plain_progress(
    *,
    label: str,
    done: int,
    total: int,
    started: float,
    last_log_at: float,
    force: bool = False,
    **metrics: Any,
) -> float:
    now = time.perf_counter()
    if not force and done < total and now - last_log_at < 2.0:
        return last_log_at
    elapsed = max(now - started, 1e-6)
    pct = (100.0 * done / total) if total else 100.0
    eta = ((total - done) * (elapsed / max(done, 1))) if total else 0.0
    extras = " ".join(f"{key}={value}" for key, value in metrics.items())
    line = f"{label}: {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.0f}s"
    if extras:
        line = f"{line} {extras}"
    print(line, flush=True)
    return now


def _build_dataset(cfg: dict[str, Any], manifest_path: Path) -> SyntheticStripDataset:
    dcfg = cfg["dataset"]
    return SyntheticStripDataset(
        manifest_path=manifest_path,
        strips_per_image=int(dcfg.get("strips_per_image", 25)),
        split=None,
        seed=int(cfg.get("seed", 42)),
        spec=StripSpec(
            strip_height=int(dcfg.get("strip_height", 1024)),
            outer_width=int(dcfg.get("outer_width", 128)),
            inner_width=int(dcfg.get("inner_width", 128)),
            seam_jitter_px=int(dcfg.get("seam_jitter_px", 0)),
        ),
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        inner_widths=[int(dcfg.get("inner_width", 128))],
        apply_corruption=True,
        corruption_cfg=dcfg.get("corruptions"),
    )


def _worker_init(config_path: str, manifest_path: str, export_root: str) -> None:
    global _DATASET, _EXPORT_ROOT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _DATASET = _build_dataset(cfg, Path(manifest_path))
    _EXPORT_ROOT = export_root


def _materialize_sample(dataset: SyntheticStripDataset, row_idx: int, sample_idx: int, base_image: torch.Tensor) -> dict[str, object]:
    row = dataset.rows[row_idx]
    cfg = dataset._config_for_index(sample_idx)
    image = dataset._augment_source_image(base_image, cfg)
    clean_strip = dataset._extract_clean_strip(image, cfg)
    if clean_strip.shape[-2:] != (dataset.spec.strip_height, dataset.spec.outer_width + cfg.inner_width):
        raise RuntimeError("unexpected strip shape")
    seam_x = dataset.spec.outer_width + cfg.seam_jitter_px
    pad_left = max(0, -cfg.seam_jitter_px)
    pad_right = max(0, cfg.seam_jitter_px)
    clean_strip = torch.nn.functional.pad(clean_strip, (pad_left, pad_right, 0, 0), mode="replicate")
    clean_strip = clean_strip[..., : dataset.spec.strip_height, : dataset.spec.outer_width + cfg.inner_width]
    target = clean_strip.clone()
    input_rgb = clean_strip.unsqueeze(0)
    corrupted_ops: list[dict[str, object]] = []
    if dataset.apply_corruption:
        inner = input_rgb[..., dataset.spec.outer_width :]
        corrupted = apply_random_corruptions(
            inner,
            torch.Generator().manual_seed(dataset.seed + sample_idx),
            corruption_cfg=dataset.corruption_cfg,
        )
        input_rgb[..., dataset.spec.outer_width :] = corrupted.image
        corrupted_ops = corrupted.ops
    built = build_harmonizer_input(
        input_rgb.squeeze(0),
        outer_width=dataset.spec.outer_width,
        boundary_band_px=dataset.boundary_band_px,
        seam_x=seam_x,
    )
    return {
        "input": _tensor_to_rgb_uint8(input_rgb.squeeze(0)),
        "target": _tensor_to_rgb_uint8(target),
        "mask": _tensor_to_mask_uint8(built["mask"][0]),
        "meta": {
            "image_id": row["id"],
            "axis": cfg.axis,
            "side": cfg.side,
            "rotation_k": cfg.rotation_k,
            "flip_h": cfg.flip_h,
            "seam_jitter_px": cfg.seam_jitter_px,
            "inner_width": cfg.inner_width,
            "edge_padded_pixels": 0,
            "ops": corrupted_ops,
            "scene_tags": row.get("scene_tags", []),
            "split": row.get("split"),
            "cluster_id": row.get("cluster_id"),
            "seam_x_frac_in_source": cfg.seam_x_frac,
            "seam_x": seam_x,
        },
    }


def _export_row_shard(row_idx: int) -> list[dict]:
    global _DATASET
    dataset: SyntheticStripDataset = _DATASET
    base_out = Path(_EXPORT_ROOT)
    row = dataset.rows[row_idx]
    source_image = dataset._load_image(row)
    shard_dir = base_out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_rel = Path("shards") / f"{row_idx:06d}.npz"
    shard_path = base_out / shard_rel
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict] = []
    base_idx = row_idx * dataset.strips_per_image
    for local_idx in range(dataset.strips_per_image):
        sample_idx = base_idx + local_idx
        materialized = _materialize_sample(dataset, row_idx, sample_idx, source_image)
        inputs.append(materialized["input"])
        targets.append(materialized["target"])
        masks.append(materialized["mask"])
        rows.append(
            {
                **materialized["meta"],
                "sample_index": sample_idx,
                "shard_path": str(shard_rel),
                "shard_index": local_idx,
                "outer_width": int(dataset.spec.outer_width),
                "strip_height": int(dataset.spec.strip_height),
                "boundary_band_px": int(dataset.boundary_band_px),
            }
        )
    np.savez(
        shard_path,
        inputs=np.stack(inputs, axis=0),
        targets=np.stack(targets, axis=0),
        masks=np.stack(masks, axis=0),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize full synthetic strip dataset from prepared clean source images.")
    parser.add_argument("--config", default="configs/finetune_harmonizer_v1.yaml")
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite to recreate")
        shutil.rmtree(out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    strips_per_image = int(yaml.safe_load(config_path.read_text(encoding="utf-8"))["dataset"].get("strips_per_image", 25))
    total_samples = len(rows) * strips_per_image
    worker_count = max(1, min(args.workers, len(rows)))

    global _EXPORT_ROOT
    _EXPORT_ROOT = str(out_dir)
    all_rows: list[dict] = []
    started = time.perf_counter()
    use_tqdm = _should_use_tqdm()
    last_log_at = started
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_init,
        initargs=(str(config_path), str(manifest_path), str(out_dir)),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        pending = {executor.submit(_export_row_shard, row_idx) for row_idx in range(len(rows))}
        progress = None
        if use_tqdm:
            progress = tqdm(
                total=total_samples,
                desc="materialize_dataset",
                dynamic_ncols=True,
                mininterval=0.5,
            )
        samples_done = 0
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            completed_samples = 0
            elapsed = max(time.perf_counter() - started, 1e-6)
            for future in done:
                chunk = future.result()
                all_rows.extend(chunk)
                samples_done += len(chunk)
                completed_samples += len(chunk)
            samples_per_sec = len(all_rows) / elapsed
            samples_left = max(total_samples - samples_done, 0)
            eta_sec = samples_left * (elapsed / max(samples_done, 1))
            if use_tqdm and progress is not None:
                if completed_samples:
                    progress.update(completed_samples)
                progress.set_postfix(
                    samples=len(all_rows),
                    workers=worker_count,
                    sps=round(samples_per_sec, 1),
                    eta_s=int(eta_sec),
                )
            else:
                last_log_at = _maybe_log_plain_progress(
                    label="materialize_dataset",
                    done=samples_done,
                    total=total_samples,
                    started=started,
                    last_log_at=last_log_at,
                    workers=worker_count,
                    sps=round(samples_per_sec, 1),
                    pending=len(pending),
                )
        if progress is not None:
            progress.close()
        elif samples_done != total_samples:
            _maybe_log_plain_progress(
                label="materialize_dataset",
                done=total_samples,
                total=total_samples,
                started=started,
                last_log_at=last_log_at,
                force=True,
                workers=worker_count,
                sps=round(len(all_rows) / max(time.perf_counter() - started, 1e-6), 1),
                pending=0,
            )

    all_rows.sort(key=lambda row: int(row["sample_index"]))
    write_jsonl(out_dir / "manifest.jsonl", all_rows)
    split_counts: dict[str, int] = {}
    for row in all_rows:
        split = str(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
    elapsed = max(time.perf_counter() - started, 1e-6)
    summary = {
        "out_dir": str(out_dir),
        "source_images": len(rows),
        "samples": len(all_rows),
        "shards": len(rows),
        "splits": split_counts,
        "workers": worker_count,
        "seconds": round(elapsed, 2),
        "samples_per_sec": round(len(all_rows) / elapsed, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
