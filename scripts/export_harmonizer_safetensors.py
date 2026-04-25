from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from safetensors.torch import save_file

from src.train.checkpoint import load_checkpoint


def _model_config(train_cfg: dict) -> dict:
    model_cfg = train_cfg.get("model") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    return {
        "name": "seam_harmonizer_v1",
        "in_channels": 5,
        "channels": model_cfg.get("channels", [32, 64, 128, 192]),
        "blocks": model_cfg.get("blocks", [2, 2, 4, 6]),
        "num_knots": int(model_cfg.get("num_knots", 16)),
        "alpha": float(model_cfg.get("alpha", 0.20)),
        "heads": ["monotonic_rgb_curves", "low_frequency_shading"],
        "outer_width": int(dataset_cfg.get("outer_width", 128)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export_harmonizer_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location="cpu")
    train_cfg = ckpt.get("config") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    train_section = train_cfg.get("train") or {}
    export_root = Path(cfg["export_root"])
    export_root.mkdir(parents=True, exist_ok=True)
    model_path = export_root / f"{cfg['model_name']}.safetensors"
    save_file(ckpt["ema"], str(model_path))
    sidecar = {
        "model_name": cfg["model_name"],
        "schema_version": 1,
        "architecture": _model_config(train_cfg),
        "strip": {
            "canonical_shape_chw": [5, 1024, 256],
            "outer_width": int(dataset_cfg.get("outer_width", 128)),
            "inner_width_default": int(dataset_cfg.get("inner_width", 128)),
            "supported_inner_widths": [128],
            "seam_jitter_train_px": int(dataset_cfg.get("seam_jitter_px", 0)),
        },
        "preprocess": {
            "rgb_range": [0.0, 1.0],
            "expects_srgb_gamma": True,
            "channels_order": ["R", "G", "B", "inner_mask", "distance_from_seam_inner_ramp"],
            "normalization": None,
        },
        "orientation": {"canonical": "vertical_outer_left", "train_rotation_aug": True},
        "inference": {
            "strength_range": [0.0, 1.0],
            "strength_default": float(cfg.get("strength_default", 1.0)),
            "hard_copy_outer": True,
            "inner_taper": "cosine_from_seam_to_inner_edge",
            "corner_fusion": "weighted_average",
            "clamp_output": [0.0, 1.0],
        },
        "training": {
            "dataset": "synthetic_pretrain_v1",
            "epochs": train_section.get("num_epochs"),
            "batch_size": train_section.get("batch_size"),
            "val_batch_size": train_section.get("val_batch_size"),
            "ema_decay": (train_cfg.get("ema") or {}).get("decay", 0.999),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": ((ckpt.get("metrics") or {}).get("val") or {}),
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "sidecar": str(model_path.with_suffix(".json"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
