from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.data.synthetic_strip_dataset import SyntheticStripDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/input_raw_manifest.jsonl")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out", default="outputs/strip_cache/preview")
    args = parser.parse_args()
    dataset = SyntheticStripDataset(Path(args.manifest), strips_per_image=max(args.count, 1))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(args.count, len(dataset))):
        sample = dataset[idx]
        sample_dir = out_dir / f"{idx:06d}"
        sample_dir.mkdir(exist_ok=True)
        for name in ("input_rgb", "target"):
            arr = (sample[name].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            Image.fromarray(arr).save(sample_dir / f"{name}.png")


if __name__ == "__main__":
    main()
