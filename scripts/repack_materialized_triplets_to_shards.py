from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.manifest import read_jsonl, write_jsonl


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _resolve(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack materialized triplets from PNG files into sharded NPZ format."
    )
    parser.add_argument("--src", type=Path, required=True, help="Directory with manifest.jsonl and input/target/mask PNGs.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for sharded dataset.")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    src_manifest = src / "manifest.jsonl"
    if not src_manifest.exists():
        raise FileNotFoundError(src_manifest)
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite to recreate")
        shutil.rmtree(out)
    (out / "shards").mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(src_manifest)
    if not rows:
        raise ValueError(f"empty manifest: {src_manifest}")

    shard_rows: list[dict] = []
    shard_size = max(1, int(args.shard_size))

    for shard_idx, start in enumerate(range(0, len(rows), shard_size)):
        chunk = rows[start : start + shard_size]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        shard_rel = Path("shards") / f"{shard_idx:06d}.npz"
        shard_path = out / shard_rel

        for local_idx, row in enumerate(chunk):
            inputs.append(_load_rgb(_resolve(src, row["input_path"])))
            targets.append(_load_rgb(_resolve(src, row["target_path"])))
            masks.append(_load_mask(_resolve(src, row["mask_path"])))
            meta = dict(row)
            meta.pop("input_path", None)
            meta.pop("target_path", None)
            meta.pop("mask_path", None)
            meta["shard_path"] = str(shard_rel)
            meta["shard_index"] = local_idx
            shard_rows.append(meta)

        np.savez(
            shard_path,
            inputs=np.stack(inputs, axis=0),
            targets=np.stack(targets, axis=0),
            masks=np.stack(masks, axis=0),
        )

    write_jsonl(out / "manifest.jsonl", shard_rows)
    print(
        json.dumps(
            {
                "src": str(src),
                "out": str(out),
                "rows": len(rows),
                "shards": len(list((out / 'shards').glob('*.npz'))),
                "shard_size": shard_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
