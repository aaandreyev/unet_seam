from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm


RAW_EXTS = {".jpg", ".jpeg", ".png", ".txt"}


def _copy_one(args: tuple[Path, Path]) -> int:
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


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
    **metrics: object,
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


def _copy_tree(src_root: Path, dst_root: Path, workers: int) -> dict[str, int]:
    files = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTS]
    if not files:
        raise FileNotFoundError(
            f"no raw source files with supported extensions {sorted(RAW_EXTS)} found under {src_root}"
        )
    workers = max(1, min(workers, len(files)))
    total_bytes = 0
    started = time.perf_counter()
    use_tqdm = _should_use_tqdm()
    last_log_at = started
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_copy_one, ((p, dst_root / p.relative_to(src_root)) for p in files), chunksize=8)
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(files),
                desc="copy_raw_sources",
                dynamic_ncols=True,
                mininterval=0.5,
            )
        copied_files = 0
        for copied in iterator:
            total_bytes += copied
            copied_files += 1
            gb = round(total_bytes / (1024**3), 2)
            if use_tqdm:
                iterator.set_postfix(gb=gb)
            else:
                last_log_at = _maybe_log_plain_progress(
                    label="copy_raw_sources",
                    done=copied_files,
                    total=len(files),
                    started=started,
                    last_log_at=last_log_at,
                    gb=gb,
                    workers=workers,
                )
        if not use_tqdm:
            _maybe_log_plain_progress(
                label="copy_raw_sources",
                done=len(files),
                total=len(files),
                started=started,
                last_log_at=last_log_at,
                force=True,
                gb=round(total_bytes / (1024**3), 2),
                workers=workers,
            )
    elapsed = max(time.perf_counter() - started, 1e-6)
    return {
        "files": len(files),
        "bytes": total_bytes,
        "workers": workers,
        "seconds": round(elapsed, 2),
        "files_per_sec": round(len(files) / elapsed, 2),
        "gb_per_sec": round((total_bytes / (1024**3)) / elapsed, 3),
    }


def _run_stage(cmd: list[str], *, cwd: Path, env: dict[str, str], stage_name: str) -> float:
    print(f"[stage:start] {stage_name}", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    started = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"[stage:done] {stage_name} in {elapsed:.2f}s", flush=True)
    return elapsed


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
    if not pr.exists():
        raise FileNotFoundError(f"project root does not exist: {pr}")
    if not raw_src.exists():
        raise FileNotFoundError(f"drive raw root does not exist: {raw_src}")
    if not raw_src.is_dir():
        raise NotADirectoryError(f"drive raw root is not a directory: {raw_src}")
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
    print(f"[stage:start] copy_raw_sources from {raw_src} -> {local_raw}", flush=True)
    copy_stats = _copy_tree(raw_src, local_raw, workers=args.copy_workers)
    print(f"[stage:done] copy_raw_sources {json.dumps(copy_stats, ensure_ascii=False)}", flush=True)

    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pr) if not env.get("PYTHONPATH") else f"{str(pr)}{os.pathsep}{env['PYTHONPATH']}"
    stage_timings = {}
    stage_timings["prepare_source_sec"] = round(
        _run_stage(
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
            cwd=pr,
            env=env,
            stage_name="prepare_source",
        ),
        2,
    )
    stage_timings["build_split_sec"] = round(
        _run_stage(
            [py, "-m", "scripts.build_split", "--manifest", str(manifest)],
            cwd=pr,
            env=env,
            stage_name="build_split",
        ),
        2,
    )
    stage_timings["materialize_dataset_sec"] = round(
        _run_stage(
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
            cwd=pr,
            env=env,
            stage_name="materialize_synthetic_dataset",
        ),
        2,
    )
    summary = {
        "raw_copy": copy_stats,
        "prepared_manifest": str(manifest),
        "materialized_manifest": str(materialized_root / "manifest.jsonl"),
        "materialized_root": str(materialized_root),
        "stage_timings_sec": stage_timings,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
