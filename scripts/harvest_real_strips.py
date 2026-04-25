from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.manifest import read_jsonl, write_jsonl
from src.data.strip_geometry import StripSpec, extract_side_strip
from src.data.structural_filter import keep_structurally_matched_strip


def _load_rgb(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _save_rgb(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (x.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL rows with input_image_path, target_image_path, bbox, split")
    parser.add_argument("--output-root", default="outputs/real_strips")
    parser.add_argument("--output-manifest", default="manifests/real_paired_strips_manifest.jsonl")
    parser.add_argument("--structural-threshold", type=float, default=0.6)
    args = parser.parse_args()
    spec = StripSpec(seam_jitter_px=0)
    output_root = Path(args.output_root)
    rows_out = []
    for row in read_jsonl(Path(args.manifest)):
        generated = _load_rgb(row["input_image_path"])
        target = _load_rgb(row["target_image_path"])
        bbox = tuple(int(v) for v in row["bbox"])
        for side in ("left", "right", "top", "bottom"):
            gen_strip, gen_meta = extract_side_strip(generated, bbox, side, spec)
            target_strip, _ = extract_side_strip(target, bbox, side, spec)
            if not keep_structurally_matched_strip(gen_strip, target_strip, threshold=args.structural_threshold):
                continue
            sample_id = f"{row.get('id', len(rows_out))}_{side}"
            input_path = output_root / "input" / f"{sample_id}.png"
            target_path = output_root / "target" / f"{sample_id}.png"
            _save_rgb(gen_strip, input_path)
            _save_rgb(target_strip, target_path)
            rows_out.append(
                {
                    "id": sample_id,
                    "input_strip_path": str(input_path),
                    "target_strip_path": str(target_path),
                    "side": side,
                    "split": row.get("split", "train"),
                    "edge_padded_pixels": gen_meta["edge_padded_pixels"],
                    "scene_tags": row.get("scene_tags", []),
                }
            )
    write_jsonl(Path(args.output_manifest), rows_out)
    print({"written": len(rows_out), "manifest": args.output_manifest})


if __name__ == "__main__":
    main()
