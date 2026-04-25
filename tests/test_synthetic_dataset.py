from pathlib import Path

import numpy as np
from PIL import Image

from src.data.manifest import write_jsonl
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def test_dataset_shapes(tmp_path: Path):
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    img_path = tmp_path / "000000.png"
    Image.fromarray(img).save(img_path)
    write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "id": "000000",
                "source_path": str(img_path),
                "split": "train",
                "scene_tags": ["sky"],
                "cluster_id": 0,
            }
        ],
    )
    dataset = SyntheticStripDataset(tmp_path / "manifest.jsonl", strips_per_image=1, split="train")
    sample = dataset[0]
    assert sample["input"].shape[0] == 9
    assert sample["target"].shape[0] == 3
    assert sample["mask"].shape[0] == 1


def test_dataset_resolves_relative_manifest_paths(tmp_path: Path):
    root = tmp_path / "bundle"
    image_dir = root / "data/source_images"
    manifest_dir = root / "manifests"
    image_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    Image.fromarray(img).save(image_dir / "000000.png")
    write_jsonl(
        manifest_dir / "input_raw_manifest.jsonl",
        [{"id": "000000", "source_path": "data/source_images/000000.png", "split": "train"}],
    )
    dataset = SyntheticStripDataset(manifest_dir / "input_raw_manifest.jsonl", strips_per_image=1, split="train", inner_widths=[128])
    assert dataset[0]["input"].shape == (9, 1024, 256)


def test_dataset_mask_tracks_jittered_seam(tmp_path: Path):
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    img_path = tmp_path / "000001.png"
    Image.fromarray(img).save(img_path)
    write_jsonl(
        tmp_path / "manifest.jsonl",
        [{"id": "000001", "source_path": str(img_path), "split": "train"}],
    )
    dataset = SyntheticStripDataset(
        tmp_path / "manifest.jsonl",
        strips_per_image=8,
        split="train",
        spec=StripSpec(
            strip_height=1024,
            outer_width=128,
            inner_width=128,
            seam_jitter_px=6,
        ),
        inner_widths=[128],
        apply_corruption=False,
    )
    sample = dataset[3]
    seam_x = int(sample["meta"]["seam_x"])
    mask = sample["mask"][0]
    assert float(mask[:, :seam_x].max()) == 0.0
    assert float(mask[:, seam_x:].min()) == 1.0
