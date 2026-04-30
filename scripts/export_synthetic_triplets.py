from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import write_jsonl
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset


def _build_dataset(
    config_path: Path,
    manifest_override: Path | None,
    split: str,
    seed_override: int | None,
    strips_per_image_override: int | None,
) -> SyntheticStripDataset:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dcfg = cfg["dataset"]
    source_manifest = manifest_override or Path(dcfg["source_manifest"])
    spec = StripSpec(
        strip_height=int(dcfg.get("strip_height", 1024)),
        outer_width=int(dcfg.get("outer_width", 128)),
        inner_width=int(dcfg.get("inner_width", 128)),
        seam_jitter_px=int(dcfg.get("seam_jitter_px", 0)),
    )
    strips_per_image = strips_per_image_override
    if strips_per_image is None:
        if split == "train":
            strips_per_image = int(dcfg.get("strips_per_image", 25))
        else:
            strips_per_image = int(dcfg.get("val_strips_per_image", 1))
    return SyntheticStripDataset(
        manifest_path=source_manifest,
        strips_per_image=strips_per_image,
        split=split,
        seed=int(seed_override if seed_override is not None else cfg.get("seed", 42)),
        spec=spec,
        boundary_band_px=int(dcfg.get("boundary_band_px", 24)),
        inner_widths=[int(dcfg.get("inner_width", 128))],
        apply_corruption=True,
        corruption_cfg=dcfg.get("corruptions"),
    )


def _save_rgb(path: Path, tensor) -> None:
    arr = (tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def _save_mask(path: Path, tensor) -> None:
    arr = (tensor.numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize SyntheticStripDataset samples into cached inputs/targets/masks triplets."
    )
    parser.add_argument("--config", default="configs/train_harmonizer_v1.yaml")
    parser.add_argument("--manifest", default=None, help="Optional source manifest override.")
    parser.add_argument("--split", default="train", choices=["train", "val", "bench"])
    parser.add_argument("--limit", type=int, required=True, help="Final number of triplets to export.")
    parser.add_argument("--out", default="outputs/synthetic_triplets")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strips-per-image", type=int, default=None)
    parser.add_argument(
        "--save-model-input",
        action="store_true",
        help="Also save the 9-channel model input tensor as .npy files.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    config_path = Path(args.config)
    manifest_override = Path(args.manifest) if args.manifest else None
    dataset = _build_dataset(
        config_path=config_path,
        manifest_override=manifest_override,
        split=args.split,
        seed_override=args.seed,
        strips_per_image_override=args.strips_per_image,
    )
    if len(dataset.rows) == 0:
        raise ValueError(f"no rows found for split={args.split!r} in manifest {dataset.manifest_root}")
    if len(dataset) < args.limit:
        needed = math.ceil(args.limit / max(len(dataset.rows), 1))
        raise ValueError(
            f"dataset only exposes {len(dataset)} samples with strips_per_image={dataset.strips_per_image}; "
            f"use --strips-per-image >= {needed} to export {args.limit} samples"
        )

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    inputs_dir = out_dir / "inputs"
    targets_dir = out_dir / "targets"
    masks_dir = out_dir / "masks"
    meta_dir = out_dir / "meta"
    model_input_dir = out_dir / "model_input"
    for path in (inputs_dir, targets_dir, masks_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_model_input:
        model_input_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx in tqdm(range(args.limit), desc="export_triplets", dynamic_ncols=True):
        sample = dataset[idx]
        stem = f"{idx:06d}"
        input_path = inputs_dir / f"{stem}.png"
        target_path = targets_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"
        meta_path = meta_dir / f"{stem}.json"

        _save_rgb(input_path, sample["input_rgb"])
        _save_rgb(target_path, sample["target"])
        _save_mask(mask_path, sample["mask"][0])

        meta = {
            **sample["meta"],
            "index": idx,
            "input_path": str(input_path.relative_to(out_dir)),
            "target_path": str(target_path.relative_to(out_dir)),
            "mask_path": str(mask_path.relative_to(out_dir)),
        }
        if args.save_model_input:
            model_input_path = model_input_dir / f"{stem}.npy"
            np.save(model_input_path, sample["input"].numpy())
            meta["model_input_path"] = str(model_input_path.relative_to(out_dir))
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(meta)

    write_jsonl(out_dir / "manifest.jsonl", rows)
    summary = {
        "out_dir": str(out_dir.resolve()),
        "triplets": args.limit,
        "split": args.split,
        "strips_per_image": dataset.strips_per_image,
        "save_model_input": bool(args.save_model_input),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
