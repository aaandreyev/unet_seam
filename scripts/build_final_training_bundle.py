from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bundle-output", default="outputs/training_bundle/seam_harmonizer_training_bundle.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workers = args.workers
    py = sys.executable
    prep_cmd = [py, "-m", "scripts.prepare_source", "--input", str(root / "input_raw"), "--output", str(root / "data/source_images"), "--manifest", str(root / "manifests/input_raw_manifest.jsonl"), "--excluded-log", str(root / "outputs/eval_reports/excluded_sources.jsonl")]
    if workers > 0:
        prep_cmd += ["--workers", str(workers)]
    subprocess.run(prep_cmd, check=True)
    subprocess.run([py, "-m", "scripts.build_split", "--manifest", str(root / "manifests/input_raw_manifest.jsonl")], check=True)
    subprocess.run([py, "-m", "scripts.package_training_bundle", "--root", str(root), "--output", str(root / args.bundle_output)], check=True)


if __name__ == "__main__":
    main()
