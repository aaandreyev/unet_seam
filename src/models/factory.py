from __future__ import annotations

from typing import Any

import torch

from src.models.harmonizer import SeamHarmonizerV1


def build_model_from_config(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", cfg.get("architecture", {}))
    arch = model_cfg.get("architecture") or model_cfg.get("name")
    if arch != "seam_harmonizer_v1":
        raise RuntimeError(f"Unsupported architecture: {arch}")
    return SeamHarmonizerV1(
        in_channels=int(model_cfg.get("in_channels", 5)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        num_knots=int(model_cfg.get("num_knots", 16)),
        alpha=float(model_cfg.get("alpha", 0.20)),
        outer_width=int((cfg.get("strip") or {}).get("outer_width", 128)),
    )
