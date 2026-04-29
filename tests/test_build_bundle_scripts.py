from __future__ import annotations

import tarfile
from pathlib import Path

from src.data.manifest import read_jsonl, write_jsonl


def _write_manifest(path: Path, count: int) -> None:
    rows = [
        {
            "id": f"{idx:06d}",
            "source_path": f"data/source_images/{idx:06d}.png",
            "phash64": f"{idx:016x}",
        }
        for idx in range(count)
    ]
    write_jsonl(path, rows)


def test_build_split_writes_outputs_next_to_manifest(tmp_path: Path) -> None:
    from scripts.build_split import main
    import sys

    manifest_path = tmp_path / "custom" / "input_raw_manifest.jsonl"
    _write_manifest(manifest_path, 5)

    old_argv = sys.argv
    sys.argv = ["build_split.py", "--manifest", str(manifest_path)]
    try:
        main()
    finally:
        sys.argv = old_argv

    rows = read_jsonl(manifest_path)
    assert len(rows) == 5
    assert all("split" in row for row in rows)
    assert (manifest_path.parent / "source_train.jsonl").exists()
    assert (manifest_path.parent / "source_val.jsonl").exists()
    assert (manifest_path.parent / "source_bench.jsonl").exists()


def test_package_training_bundle_uses_manifest_override(tmp_path: Path) -> None:
    from scripts.package_training_bundle import main
    import sys

    root = tmp_path / "repo"
    (root / "manifests").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "data/source_images").mkdir(parents=True)

    for name in (
        "model_harmonizer_v1.yaml",
        "train_harmonizer_v1.yaml",
        "eval_harmonizer_v1.yaml",
        "export_harmonizer_v1.yaml",
    ):
        (root / "configs" / name).write_text("x: 1\n", encoding="utf-8")

    manifest_path = root / "limited.jsonl"
    rows = []
    for idx in range(3):
        image_path = root / "data/source_images" / f"{idx:06d}.png"
        image_path.write_bytes(b"fake")
        rows.append(
            {
                "id": f"{idx:06d}",
                "source_path": str(image_path),
                "split": "train",
                "phash64": f"{idx:016x}",
            }
        )
    write_jsonl(manifest_path, rows)

    out_path = root / "bundle.tar.gz"
    old_argv = sys.argv
    sys.argv = [
        "package_training_bundle.py",
        "--root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--output",
        str(out_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    assert out_path.exists()
    with tarfile.open(out_path, "r:gz") as tf:
        manifest_member = tf.extractfile("manifests/input_raw_manifest.jsonl")
        assert manifest_member is not None
        manifest_rows = [
            line
            for line in manifest_member.read().decode("utf-8").splitlines()
            if line.strip()
        ]
    assert len(manifest_rows) == 3


def test_package_training_bundle_can_write_folder(tmp_path: Path) -> None:
    from scripts.package_training_bundle import main
    import sys

    root = tmp_path / "repo"
    (root / "manifests").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "data/source_images").mkdir(parents=True)

    for name in (
        "model_harmonizer_v1.yaml",
        "train_harmonizer_v1.yaml",
        "eval_harmonizer_v1.yaml",
        "export_harmonizer_v1.yaml",
    ):
        (root / "configs" / name).write_text("x: 1\n", encoding="utf-8")

    manifest_path = root / "limited.jsonl"
    rows = []
    for idx in range(2):
        image_path = root / "data/source_images" / f"{idx:06d}.png"
        image_path.write_bytes(b"fake")
        rows.append(
            {
                "id": f"{idx:06d}",
                "source_path": str(image_path),
                "split": "train",
                "phash64": f"{idx:016x}",
            }
        )
    write_jsonl(manifest_path, rows)

    out_dir = root / "bundle_dir"
    old_argv = sys.argv
    sys.argv = [
        "package_training_bundle.py",
        "--root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(out_dir),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "manifests/input_raw_manifest.jsonl").exists()
    assert len(read_jsonl(out_dir / "manifests/input_raw_manifest.jsonl")) == 2
    assert (out_dir / "data/source_images/000000.png").exists()
    assert (out_dir / "configs/train_harmonizer_v1.yaml").exists()


def test_prepare_source_limit_applies_before_processing(tmp_path: Path) -> None:
    from scripts.prepare_source import main
    import sys
    from PIL import Image
    import numpy as np

    input_dir = tmp_path / "input_raw"
    output_dir = tmp_path / "data/source_images"
    manifest_path = tmp_path / "manifests/input_raw_manifest.jsonl"
    excluded_log = tmp_path / "outputs/eval_reports/excluded_sources.jsonl"
    input_dir.mkdir(parents=True)

    for idx in range(5):
        image = np.full((32, 32, 3), idx, dtype="uint8")
        Image.fromarray(image).save(input_dir / f"{idx:06d}.png")

    old_argv = sys.argv
    sys.argv = [
        "prepare_source.py",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--excluded-log",
        str(excluded_log),
        "--workers",
        "1",
        "--limit",
        "2",
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    rows = read_jsonl(manifest_path)
    excluded_rows = read_jsonl(excluded_log)
    assert len(rows) + len(excluded_rows) == 2
    output_files = sorted(output_dir.glob("*.png"))
    assert len(output_files) <= 2
