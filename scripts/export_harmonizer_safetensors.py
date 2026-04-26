from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml
from safetensors.torch import save_file

from src.models.factory import build_model_from_config
from src.train.checkpoint import load_checkpoint


def _contiguous_state_dict(sd: dict) -> dict:
    return {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


def _model_config(train_cfg: dict) -> dict:
    model_cfg = train_cfg.get("model") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    return {
        "name": "seam_harmonizer_v3",
        "in_channels": int(model_cfg.get("in_channels", 9)),
        "channels": model_cfg.get("channels", [32, 64, 128, 192]),
        "blocks": model_cfg.get("blocks", [2, 2, 4, 6]),
        "heads": ["local_gain", "local_gamma", "local_bias", "local_color_mix", "local_detail", "local_confidence"],
        "outer_width": int(dataset_cfg.get("outer_width", 128)),
        "boundary_band_px": int(dataset_cfg.get("boundary_band_px", 24)),
    }


def _validate_checkpoint_for_export(ckpt: dict) -> None:
    train_cfg = ckpt.get("config") or {}
    arch = ((train_cfg.get("model") or {}).get("architecture")) or ((train_cfg.get("model") or {}).get("name"))
    if arch != "seam_harmonizer_v3":
        raise RuntimeError(f"Unsupported checkpoint architecture for export: {arch!r}")
    model = build_model_from_config(
        {
            "model": train_cfg.get("model") or {},
            "strip": {
                "outer_width": int((train_cfg.get("dataset") or {}).get("outer_width", 128)),
                "boundary_band_px": int((train_cfg.get("dataset") or {}).get("boundary_band_px", 24)),
            },
        }
    )
    try:
        model.load_state_dict(ckpt["ema"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Checkpoint EMA weights are incompatible with the current export architecture. "
            "Train a fresh v3 checkpoint or use --load-weights followed by new training before export."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export_harmonizer_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = load_checkpoint(Path(cfg["checkpoint"]), map_location="cpu")
    _validate_checkpoint_for_export(ckpt)
    train_cfg = ckpt.get("config") or {}
    dataset_cfg = train_cfg.get("dataset") or {}
    train_section = train_cfg.get("train") or {}
    export_root = Path(cfg["export_root"])
    export_root.mkdir(parents=True, exist_ok=True)
    model_path = export_root / f"{cfg['model_name']}.safetensors"
    save_file(_contiguous_state_dict(ckpt["ema"]), str(model_path))
    sidecar = {
        "model_name": cfg["model_name"],
        "schema_version": 1,
        "architecture": _model_config(train_cfg),
        "strip": {
            "canonical_shape_chw": [9, 1024, 256],
            "outer_width": int(dataset_cfg.get("outer_width", 128)),
            "inner_width_default": int(dataset_cfg.get("inner_width", 128)),
            "supported_inner_widths": [128],
            "seam_jitter_train_px": int(dataset_cfg.get("seam_jitter_px", 0)),
            "boundary_band_px": int(dataset_cfg.get("boundary_band_px", 24)),
        },
        "preprocess": {
            "rgb_range": [0.0, 1.0],
            "expects_srgb_gamma": True,
            "channels_order": [
                "R",
                "G",
                "B",
                "inner_mask",
                "distance_from_seam_inner_ramp",
                "boundary_band_mask",
                "decay_mask",
                "luma",
                "gradient_magnitude",
            ],
            "normalization": None,
        },
        "orientation": {"canonical": "vertical_outer_left", "train_rotation_aug": True},
        "inference": {
            "strength_range": [0.0, 10.0],
            "strength_default": float(cfg.get("strength_default", 1.0)),
            "hard_copy_outer": True,
            "inner_taper": "cosine_from_seam_to_inner_edge",
            "corner_fusion": "weighted_average",
            "clamp_output": [0.0, 1.0],
        },
        "training": {
            "dataset": "synthetic_pretrain_v3",
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
