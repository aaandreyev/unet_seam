from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.manifest import read_jsonl, write_jsonl
from src.data.materialized_strip_dataset import MaterializedStripDataset
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


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


def test_materialized_dataset_loads_repo_relative_triplets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    root = repo_root / "outputs" / "synthetic_triplets_100"
    for sub in ("inputs", "targets", "masks"):
        (root / sub).mkdir(parents=True)
    input_arr = np.full((1024, 256, 3), 64, dtype="uint8")
    target_arr = np.full((1024, 256, 3), 128, dtype="uint8")
    mask_arr = np.zeros((1024, 256), dtype="uint8")
    mask_arr[:, 128:] = 255
    Image.fromarray(input_arr).save(root / "inputs/00000000.png")
    Image.fromarray(target_arr).save(root / "targets/00000000.png")
    Image.fromarray(mask_arr).save(root / "masks/00000000.png")
    write_jsonl(
        root / "manifest.jsonl",
        [
            {
                "input_path": "outputs/synthetic_triplets_100/inputs/00000000.png",
                "target_path": "outputs/synthetic_triplets_100/targets/00000000.png",
                "mask_path": "outputs/synthetic_triplets_100/masks/00000000.png",
                "split": "train",
                "seam_x": 128,
                "outer_width": 128,
            }
        ],
    )
    ds = MaterializedStripDataset(root / "manifest.jsonl", split="train", preload=False)
    sample = ds[0]
    assert sample["input"].shape == (9, 1024, 256)
    assert sample["target"].shape == (3, 1024, 256)
    assert sample["mask"].shape == (1, 1024, 256)


def test_materialized_dataset_loads_sharded_triplets(tmp_path: Path) -> None:
    root = tmp_path / "mat"
    (root / "shards").mkdir(parents=True)
    input_arr = np.full((2, 1024, 256, 3), 64, dtype="uint8")
    target_arr = np.full((2, 1024, 256, 3), 128, dtype="uint8")
    mask_arr = np.zeros((2, 1024, 256), dtype="uint8")
    mask_arr[:, :, 128:] = 255
    np.savez(
        root / "shards/000000.npz",
        inputs=input_arr,
        targets=target_arr,
        masks=mask_arr,
    )
    write_jsonl(
        root / "manifest.jsonl",
        [
            {
                "shard_path": "shards/000000.npz",
                "shard_index": 0,
                "split": "train",
                "seam_x": 128,
                "outer_width": 128,
            },
            {
                "shard_path": "shards/000000.npz",
                "shard_index": 1,
                "split": "train",
                "seam_x": 128,
                "outer_width": 128,
            },
        ],
    )
    ds = MaterializedStripDataset(root / "manifest.jsonl", split="train", preload=True)
    sample = ds[1]
    assert sample["input"].shape == (9, 1024, 256)
    assert sample["target"].shape == (3, 1024, 256)
    assert sample["mask"].shape == (1, 1024, 256)
    assert torch.allclose(sample["target"][:, 0, 0], torch.tensor([128, 128, 128], dtype=torch.float32) / 255.0)


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
    assert (out_dir / "shards/000000.npz").exists()
    assert rows[0]["shard_path"] == "shards/000000.npz"
    assert rows[0]["shard_index"] == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers"] == 1
    assert rows[0]["sample_index"] == 0
    assert rows[1]["sample_index"] == 1


def test_sharded_materialization_matches_synthetic_dataset(tmp_path: Path) -> None:
    from scripts.materialize_synthetic_dataset import main

    source = np.zeros((1024, 1024, 3), dtype="uint8")
    source[..., 0] = 32
    source[..., 1] = 96
    source[..., 2] = 160
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

    loaded = MaterializedStripDataset(out_dir / "manifest.jsonl", split="train", preload=False)
    synthetic = SyntheticStripDataset(
        manifest_path,
        strips_per_image=2,
        split="train",
        seed=42,
        spec=StripSpec(strip_height=1024, outer_width=128, inner_width=128, seam_jitter_px=0),
        boundary_band_px=24,
        inner_widths=[128],
        apply_corruption=True,
    )
    for idx in range(2):
        materialized_sample = loaded[idx]
        synthetic_sample = synthetic[idx]
        expected_input_rgb = (synthetic_sample["input_rgb"] * 255.0).to(torch.uint8).to(torch.float32) / 255.0
        expected_target = (synthetic_sample["target"] * 255.0).to(torch.uint8).to(torch.float32) / 255.0
        expected_mask = (synthetic_sample["mask"] * 255.0).to(torch.uint8).to(torch.float32) / 255.0
        assert torch.allclose(materialized_sample["target"], expected_target)
        assert torch.allclose(materialized_sample["mask"], expected_mask)
        assert torch.allclose(materialized_sample["input"][:3], expected_input_rgb)
