from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.data.manifest import read_jsonl, write_jsonl
from src.data.materialized_strip_dataset import MaterializedStripDataset


def test_materialized_dataset_loads_triplets(tmp_path: Path) -> None:
    root = tmp_path / "mat"
    for sub in ("inputs", "targets", "masks"):
        (root / sub).mkdir(parents=True)
    input_arr = np.full((1024, 256, 3), 64, dtype="uint8")
    target_arr = np.full((1024, 256, 3), 128, dtype="uint8")
    mask_arr = np.zeros((1024, 256), dtype="uint8")
    mask_arr[:, 128:] = 255
    Image.fromarray(input_arr).save(root / "inputs/00000000.png")
    Image.fromarray(target_arr).save(root / "targets/00000000.png")
    Image.fromarray(mask_arr, mode="L").save(root / "masks/00000000.png")
    write_jsonl(
        root / "manifest.jsonl",
        [
            {
                "input_path": "inputs/00000000.png",
                "target_path": "targets/00000000.png",
                "mask_path": "masks/00000000.png",
                "split": "train",
                "seam_x": 128,
                "outer_width": 128,
            }
        ],
    )
    ds = MaterializedStripDataset(root / "manifest.jsonl", split="train", preload=True)
    sample = ds[0]
    assert sample["input"].shape == (9, 1024, 256)
    assert sample["target"].shape == (3, 1024, 256)
    assert sample["mask"].shape == (1, 1024, 256)


def test_materialize_script_exports_manifest_and_triplets(tmp_path: Path) -> None:
    from scripts.materialize_synthetic_dataset import main

    source = np.zeros((1024, 1024, 3), dtype="uint8")
    source[..., 1] = 127
    img_path = tmp_path / "source.png"
    Image.fromarray(source).save(img_path)
    manifest_path = tmp_path / "input_raw_manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "id": "000000",
                "source_path": str(img_path),
                "split": "train",
                "scene_tags": [],
                "cluster_id": 0,
            }
        ],
    )
    cfg = {
        "seed": 42,
        "dataset": {
            "source_manifest": str(manifest_path),
            "strips_per_image": 2,
            "strip_height": 1024,
            "outer_width": 128,
            "inner_width": 128,
            "seam_jitter_px": 0,
            "boundary_band_px": 24,
        },
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "out"
    old_argv = sys.argv
    sys.argv = [
        "materialize_synthetic_dataset.py",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_dir),
        "--workers",
        "1",
    ]
    try:
        main()
    finally:
        sys.argv = old_argv
    rows = read_jsonl(out_dir / "manifest.jsonl")
    assert len(rows) == 2
    assert (out_dir / "inputs/00000000.png").exists()
    assert (out_dir / "targets/00000000.png").exists()
    assert (out_dir / "masks/00000000.png").exists()

