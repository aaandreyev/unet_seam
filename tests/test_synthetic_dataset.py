from pathlib import Path

import numpy as np
from PIL import Image

from src.data.manifest import write_jsonl
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
    assert sample["input"].shape[0] == 5
    assert sample["target"].shape[0] == 3
    assert sample["mask"].shape[0] == 1
