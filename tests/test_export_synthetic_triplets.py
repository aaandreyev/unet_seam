from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.data.manifest import read_jsonl, write_jsonl
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def test_export_synthetic_triplets_matches_dataset(tmp_path: Path) -> None:
    from scripts.export_synthetic_triplets import main

    image = np.zeros((1024, 1024, 3), dtype="uint8")
    image[..., 0] = np.tile(np.arange(1024, dtype="uint8"), (1024, 1))
    image[..., 1] = 64
    image[..., 2] = 192
    image_path = tmp_path / "000000.png"
    Image.fromarray(image).save(image_path)

    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "id": "000000",
                "source_path": str(image_path),
                "split": "train",
                "scene_tags": [],
                "cluster_id": 0,
            }
        ],
    )

    config_path = tmp_path / "config.yaml"
    config = {
        "seed": 123,
        "dataset": {
            "source_manifest": str(manifest_path),
            "strips_per_image": 3,
            "val_strips_per_image": 1,
            "strip_height": 1024,
            "outer_width": 128,
            "inner_width": 128,
            "seam_jitter_px": 6,
            "boundary_band_px": 24,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "exported"
    old_argv = sys.argv
    sys.argv = [
        "export_synthetic_triplets.py",
        "--config",
        str(config_path),
        "--split",
        "train",
        "--limit",
        "2",
        "--out",
        str(out_dir),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    dataset = SyntheticStripDataset(
        manifest_path=manifest_path,
        strips_per_image=3,
        split="train",
        seed=123,
        spec=StripSpec(strip_height=1024, outer_width=128, inner_width=128, seam_jitter_px=6),
        boundary_band_px=24,
        inner_widths=[128],
        apply_corruption=True,
    )

    rows = read_jsonl(out_dir / "manifest.jsonl")
    assert len(rows) == 2
    assert rows[0]["input_path"] == "inputs/000000.png"
    assert rows[0]["target_path"] == "targets/000000.png"
    assert rows[0]["mask_path"] == "masks/000000.png"

    sample0 = dataset[0]
    input0 = np.asarray(Image.open(out_dir / "inputs/000000.png"))
    target0 = np.asarray(Image.open(out_dir / "targets/000000.png"))
    mask0 = np.asarray(Image.open(out_dir / "masks/000000.png"))

    expected_input0 = (sample0["input_rgb"].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    expected_target0 = (sample0["target"].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    expected_mask0 = (sample0["mask"][0].numpy().clip(0.0, 1.0) * 255.0).astype("uint8")

    assert np.array_equal(input0, expected_input0)
    assert np.array_equal(target0, expected_target0)
    assert np.array_equal(mask0, expected_mask0)

    meta0 = json.loads((out_dir / "meta/000000.json").read_text(encoding="utf-8"))
    assert meta0["index"] == 0
    assert meta0["image_id"] == "000000"
