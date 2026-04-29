from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    arr = (tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def _save_mask(path: Path, tensor: torch.Tensor) -> None:
    arr = (tensor.numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


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
    )


def _worker_init(config_path: str, manifest_path: str, export_root: str) -> None:
    global _DATASET, _EXPORT_ROOT
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _DATASET = _build_dataset(cfg, Path(manifest_path))
    _EXPORT_ROOT = export_root


def _export_row(row_idx: int) -> list[dict]:
    global _DATASET
    dataset: SyntheticStripDataset = _DATASET
    strips_per_image = dataset.strips_per_image
    base_out = Path(_EXPORT_ROOT)
    rows: list[dict] = []
    for local_idx in range(strips_per_image):
        idx = row_idx * strips_per_image + local_idx
        sample = dataset[idx]
        stem = f"{idx:08d}"
        input_rel = Path("inputs") / f"{stem}.png"
        target_rel = Path("targets") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        meta_rel = Path("meta") / f"{stem}.json"
        _save_rgb(base_out / input_rel, sample["input_rgb"])
        _save_rgb(base_out / target_rel, sample["target"])
        _save_mask(base_out / mask_rel, sample["mask"][0])
        row = {
            **sample["meta"],
            "sample_index": idx,
            "input_path": str(input_rel),
            "target_path": str(target_rel),
            "mask_path": str(mask_rel),
            "outer_width": int(dataset.spec.outer_width),
            "strip_height": int(dataset.spec.strip_height),
            "boundary_band_px": int(dataset.boundary_band_px),
        }
        (base_out / meta_rel).write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(row)
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
    for sub in ("inputs", "targets", "masks", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
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
    ) as executor:
        iterator = executor.map(_export_row, range(len(rows)), chunksize=1)
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(rows),
                desc="materialize_dataset",
                dynamic_ncols=True,
                mininterval=0.5,
            )
        rows_done = 0
        for chunk in iterator:
            all_rows.extend(chunk)
            rows_done += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            samples_per_sec = len(all_rows) / elapsed
            rows_left = max(len(rows) - rows_done, 0)
            eta_sec = rows_left * (elapsed / max(rows_done, 1))
            if use_tqdm:
                iterator.set_postfix(
                    samples=len(all_rows),
                    workers=worker_count,
                    sps=round(samples_per_sec, 1),
                    eta_s=int(eta_sec),
                )
            else:
                last_log_at = _maybe_log_plain_progress(
                    label="materialize_dataset",
                    done=rows_done,
                    total=len(rows),
                    started=started,
                    last_log_at=last_log_at,
                    samples=len(all_rows),
                    workers=worker_count,
                    sps=round(samples_per_sec, 1),
                )
        if not use_tqdm:
            _maybe_log_plain_progress(
                label="materialize_dataset",
                done=len(rows),
                total=len(rows),
                started=started,
                last_log_at=last_log_at,
                force=True,
                samples=len(all_rows),
                workers=worker_count,
                sps=round(len(all_rows) / max(time.perf_counter() - started, 1e-6), 1),
            )

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
        "splits": split_counts,
        "workers": worker_count,
        "seconds": round(elapsed, 2),
        "samples_per_sec": round(len(all_rows) / elapsed, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
