from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.data.manifest import append_jsonl, write_jsonl
from src.data.preprocess import prepare_single_source
from src.utils.image_io import iter_image_files
from tqdm.auto import tqdm


def _prepare_one(args: tuple[Path, Path, str]) -> dict:
    path, output_dir, sample_id = args
    prepared = prepare_single_source(path, output_dir, sample_id)
    return {"row": prepared.row, "excluded": prepared.excluded_reason}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_raw")
    parser.add_argument("--output", default="data/source_images")
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--excluded-log", default="outputs/eval_reports/excluded_sources.jsonl")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rows = []
    excluded_rows = []
    files = list(iter_image_files(input_dir))
    tasks = [(path, output_dir, f"{idx:06d}") for idx, path in enumerate(files)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        progress = tqdm(
            executor.map(_prepare_one, tasks, chunksize=args.chunksize),
            total=len(tasks),
            desc="prepare_source",
            dynamic_ncols=True,
        )
        for result in progress:
            if result["row"]:
                rows.append(result["row"])
            if result["excluded"]:
                excluded_rows.append(result["excluded"])
            progress.set_postfix(valid=len(rows), excluded=len(excluded_rows), workers=args.workers)
    write_jsonl(Path(args.manifest), rows)
    if excluded_rows:
        append_jsonl(Path(args.excluded_log), excluded_rows)


if __name__ == "__main__":
    main()
