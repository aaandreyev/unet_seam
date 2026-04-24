from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# Before importing torch (via src.data) — main process and pool workers re-run this file.
warnings.filterwarnings("ignore", message=".*[Pp]ynvml.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*nvidia-ml-py.*", category=FutureWarning)

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.manifest import write_jsonl
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def _save_rgb(tensor, path: Path) -> None:
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


def _save_gray(tensor, path: Path) -> None:
    arr = (tensor.squeeze(0).numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


def _build_meta(sample_id: str, sample: dict, row: dict, split: str) -> dict:
    err = float((sample["target"] - sample["input_rgb"]).abs().mean().item())
    return {
        "id": sample_id,
        "source_image": row["source_path"],
        "cluster_id": row.get("cluster_id"),
        "split": split,
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
        "metrics_precomputed": {
            "baseline_boundary_mae": err,
            "baseline_boundary_ciede2000": 0.0,
            "relative_improvement": 0.0,
        },
    }


def _materialize_chunk(args: tuple[str, str, str, int, int, int, int, bool]) -> list[dict]:
    manifest_path, split, out_root, start, end, strips_per_image, seed, include_debug = args
    dataset = SyntheticStripDataset(Path(manifest_path), split=split, strips_per_image=strips_per_image, seed=seed)
    root = Path(out_root)
    rows = []
    split_rows = dataset.rows
    for idx in range(start, end):
        sample = dataset[idx]
        source_row = split_rows[idx // strips_per_image]
        sample_id = f"{split}_{idx:07d}"
        sample_dir = root / split / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_rgb(sample["input_rgb"], sample_dir / "input.png")
        _save_rgb(sample["target"], sample_dir / "target.png")
        _save_gray(sample["mask"], sample_dir / "mask.png")
        _save_gray(sample["distance"], sample_dir / "distance.png")
        if include_debug:
            error_map = (sample["target"] - sample["input_rgb"]).abs().mean(dim=0).numpy()
            Image.fromarray((error_map * 255.0).clip(0, 255).astype("uint8")).save(sample_dir / "error.png")
            Image.fromarray(np.zeros((sample["target"].shape[1], sample["target"].shape[2], 3), dtype="uint8")).save(sample_dir / "residual.png")
        meta = _build_meta(sample_id, sample, source_row, split)
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)
    return rows


def _chunk_ranges(total: int, workers: int, chunk_size: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if chunk_size <= 0:
        chunk_size = max(1, total // max(workers, 1))
    ranges = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        ranges.append((start, end))
        start = end
    return ranges


def _run_chunks_in_pool(
    executor_cls: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
    tasks: list[tuple],
    max_workers: int,
    desc: str,
    total_samples: int,
) -> list[dict]:
    """
    as_completed: progress updates as *any* chunk finishes (not only the first in order).
    Chunks are reassembled in task order so the manifest row order is deterministic.
    """
    if not tasks:
        return []
    n_tasks = len(tasks)
    with executor_cls(max_workers=max_workers) as executor:
        future_to_index: dict[object, int] = {executor.submit(_materialize_chunk, t): j for j, t in enumerate(tasks)}
        chunks: list[list[dict] | None] = [None] * n_tasks
        n_done_samples = 0
        with tqdm(
            total=n_tasks,
            desc=desc,
            unit="chunk",
            dynamic_ncols=True,
        ) as pbar:
            for fut in as_completed(future_to_index):
                j = future_to_index.pop(fut)
                rows = fut.result()
                chunks[j] = rows
                n_done_samples += len(rows)
                pbar.set_postfix_str(f"samples~{n_done_samples}/{total_samples}", refresh=True)
                pbar.update(1)
    return [m for ch in chunks if ch is not None for m in ch]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--out", default="outputs/strip_cache")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strips-per-image", type=int, default=25)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "bench"])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--executor", choices=["auto", "process", "thread"], default="auto")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        dataset = SyntheticStripDataset(Path(args.manifest), split=split, strips_per_image=args.strips_per_image, seed=args.seed)
        total = len(dataset)
        if args.max_samples_per_split > 0:
            total = min(total, args.max_samples_per_split)
        ranges = _chunk_ranges(total, args.workers, args.chunk_size)
        tasks = [
            (args.manifest, split, str(out_root), start, end, args.strips_per_image, args.seed, args.include_debug)
            for start, end in ranges
        ]
        n_chunks = len(tasks)
        print(
            f"materialize {split}: {total} samples, {n_chunks} chunks (chunk_size={args.chunk_size}, workers={args.workers})",
            flush=True,
        )
        if args.executor == "thread":
            all_rows = _run_chunks_in_pool(
                ThreadPoolExecutor, tasks, args.workers, f"materialize_{split}", total
            )
        elif args.executor == "process":
            all_rows = _run_chunks_in_pool(
                ProcessPoolExecutor, tasks, args.workers, f"materialize_{split}", total
            )
        else:
            try:
                all_rows = _run_chunks_in_pool(
                    ProcessPoolExecutor, tasks, args.workers, f"materialize_{split}", total
                )
            except PermissionError:
                all_rows = _run_chunks_in_pool(
                    ThreadPoolExecutor, tasks, args.workers, f"materialize_{split}_thread", total
                )
        write_jsonl(Path(f"manifests/strip_{split}_cache.jsonl"), all_rows)


if __name__ == "__main__":
    main()
