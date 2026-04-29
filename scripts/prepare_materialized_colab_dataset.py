from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm


RAW_EXTS = {".jpg", ".jpeg", ".png", ".txt"}


def _copy_one(args: tuple[Path, Path]) -> int:
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def _copy_tree(src_root: Path, dst_root: Path, workers: int) -> dict[str, int]:
    files = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTS]
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        progress = tqdm(
            executor.map(_copy_one, ((p, dst_root / p.relative_to(src_root)) for p in files), chunksize=8),
            total=len(files),
            desc="copy_raw_sources",
            dynamic_ncols=True,
        )
        for copied in progress:
            total_bytes += copied
            progress.set_postfix(gb=round(total_bytes / (1024**3), 2))
    return {"files": len(files), "bytes": total_bytes}


def main() -> None:
    ap = argparse.ArgumentParser(description="Colab helper: copy raw sources locally, prepare manifest, split, materialize synthetic dataset.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--drive-raw-root", type=Path, required=True)
    ap.add_argument("--local-data-root", type=Path, required=True)
    ap.add_argument("--train-config-name", type=str, default="finetune_harmonizer_v1.yaml")
    ap.add_argument("--copy-workers", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--prepare-workers", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--materialize-workers", type=int, default=max(4, os.cpu_count() or 4))
    args = ap.parse_args()

    pr = args.project_root.resolve()
    raw_src = args.drive_raw_root.resolve()
    local_root = args.local_data_root.resolve()
    local_raw = local_root / "input_raw"
    prepared_dir = local_root / "data/source_images"
    manifest = local_root / "manifests/input_raw_manifest.jsonl"
    excluded = local_root / "outputs/eval_reports/excluded_sources.jsonl"
    materialized_root = local_root / "materialized_dataset"

    for p in (local_raw, prepared_dir, manifest.parent, excluded.parent, materialized_root.parent):
        p.mkdir(parents=True, exist_ok=True)

    if local_raw.exists():
        shutil.rmtree(local_raw)
    local_raw.mkdir(parents=True, exist_ok=True)
    copy_stats = _copy_tree(raw_src, local_raw, workers=args.copy_workers)

    py = sys.executable
    subprocess.run(
        [
            py,
            "-m",
            "scripts.prepare_source",
            "--input",
            str(local_raw),
            "--output",
            str(prepared_dir),
            "--manifest",
            str(manifest),
            "--excluded-log",
            str(excluded),
            "--workers",
            str(args.prepare_workers),
        ],
        cwd=str(pr),
        check=True,
    )
    subprocess.run(
        [py, "-m", "scripts.build_split", "--manifest", str(manifest)],
        cwd=str(pr),
        check=True,
    )
    subprocess.run(
        [
            py,
            "-m",
            "scripts.materialize_synthetic_dataset",
            "--config",
            str(pr / "configs" / args.train_config_name),
            "--manifest",
            str(manifest),
            "--out",
            str(materialized_root),
            "--workers",
            str(args.materialize_workers),
            "--overwrite",
        ],
        cwd=str(pr),
        check=True,
    )
    summary = {
        "raw_copy": copy_stats,
        "prepared_manifest": str(manifest),
        "materialized_manifest": str(materialized_root / "manifest.jsonl"),
        "materialized_root": str(materialized_root),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
