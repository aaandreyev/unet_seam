from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.data.manifest import append_jsonl, write_jsonl
from src.data.preprocess import prepare_single_source
from src.utils.image_io import iter_image_files
from tqdm.auto import tqdm


def _prepare_one(args: tuple[Path, Path, str]) -> dict:
    path, output_dir, sample_id = args
    prepared = prepare_single_source(path, output_dir, sample_id)
    return {"row": prepared.row, "excluded": prepared.excluded_reason}


def _should_use_tqdm() -> bool:
    return sys.stdout.isatty() and os.environ.get("UNET_SEAM_PLAIN_PROGRESS") != "1"


def _maybe_log_plain_progress(
    *,
    done: int,
    total: int,
    started: float,
    last_log_at: float,
    force: bool = False,
    valid: int,
    excluded: int,
    workers: int,
) -> float:
    now = time.perf_counter()
    if not force and done < total and now - last_log_at < 2.0:
        return last_log_at
    elapsed = max(now - started, 1e-6)
    pct = (100.0 * done / total) if total else 100.0
    eta = ((total - done) * (elapsed / max(done, 1))) if total else 0.0
    print(
        f"prepare_source: {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.0f}s "
        f"valid={valid} excluded={excluded} workers={workers}",
        flush=True,
    )
    return now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_raw")
    parser.add_argument("--output", default="data/source_images")
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--excluded-log", default="outputs/eval_reports/excluded_sources.jsonl")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rows = []
    excluded_rows = []
    files = list(iter_image_files(input_dir))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        files = files[: args.limit]
    tasks = [(path, output_dir, f"{idx:06d}") for idx, path in enumerate(files)]
    use_tqdm = _should_use_tqdm()
    started = time.perf_counter()
    last_log_at = started
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_prepare_one, task) for task in tasks]
        progress = None
        if use_tqdm:
            progress = tqdm(
                total=len(tasks),
                desc="prepare_source",
                dynamic_ncols=True,
                mininterval=0.5,
            )
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["row"]:
                rows.append(result["row"])
            if result["excluded"]:
                excluded_rows.append(result["excluded"])
            if use_tqdm and progress is not None:
                progress.update(1)
                progress.set_postfix(valid=len(rows), excluded=len(excluded_rows), workers=args.workers)
            else:
                last_log_at = _maybe_log_plain_progress(
                    done=done,
                    total=len(tasks),
                    started=started,
                    last_log_at=last_log_at,
                    valid=len(rows),
                    excluded=len(excluded_rows),
                    workers=args.workers,
                )
        if progress is not None:
            progress.close()
        elif done != len(tasks):
            _maybe_log_plain_progress(
                done=len(tasks),
                total=len(tasks),
                started=started,
                last_log_at=last_log_at,
                force=True,
                valid=len(rows),
                excluded=len(excluded_rows),
                workers=args.workers,
            )
    write_jsonl(Path(args.manifest), rows)
    if excluded_rows:
        append_jsonl(Path(args.excluded_log), excluded_rows)


if __name__ == "__main__":
    main()
