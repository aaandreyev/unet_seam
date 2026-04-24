from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from safetensors.torch import save_file

from src.train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location="cpu")
    export_root = Path(cfg["export_root"])
    export_root.mkdir(parents=True, exist_ok=True)
    model_path = export_root / f"{cfg['model_name']}.safetensors"
    save_file(ckpt["ema"], str(model_path))
    sidecar = {
        "model_name": cfg["model_name"],
        "schema_version": 1,
        "architecture": {
            "in_channels": 5,
            "out_channels": 3,
            "base_channels": 32,
            "depth": 4,
            "bottleneck": {"gap_film": True, "lf_branch": True},
            "residual_cap_tanh_scale": 0.3,
            "decay_window": "cosine",
        },
        "strip": {
            "canonical_shape_chw": [5, 1024, 256],
            "outer_width": 128,
            "inner_width_default": 128,
            "supported_inner_widths": [96, 128, 160, 192],
            "seam_jitter_train_px": 16,
        },
        "preprocess": {
            "rgb_range": [0.0, 1.0],
            "expects_srgb_gamma": True,
            "channels_order": ["R", "G", "B", "inner_mask", "distance_to_seam"],
        },
        "orientation": {"canonical": "vertical_outer_left", "train_rotation_aug": True},
        "inference": {
            "strength_range": [0.0, 1.0],
            "strength_default": cfg["strength_default"],
            "hard_copy_outer": True,
            "clamp_output": [0.0, 1.0],
        },
        "training": {
            "dataset": "synthetic_only_v1",
            "epochs": 20,
            "ema_decay": 0.999,
            "git_hash": ckpt.get("git_hash", "nogit"),
            "commit_date": datetime.now(timezone.utc).isoformat(),
        },
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
