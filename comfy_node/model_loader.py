from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.models.harmonizer import SeamHarmonizerV1


_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, dict]] = {}


def load_model(path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    key = (path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model_path = Path(path)
    sidecar_path = model_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _validate_sidecar(sidecar)
    state = load_file(str(model_path), device=device)
    arch = sidecar["architecture"]["name"]
    if arch != "seam_harmonizer_v1":
        raise RuntimeError(f"Unsupported architecture: {arch}")
    model = SeamHarmonizerV1(
        in_channels=sidecar["architecture"]["in_channels"],
        channels=tuple(sidecar["architecture"]["channels"]),
        blocks=tuple(sidecar["architecture"]["blocks"]),
        num_knots=sidecar["architecture"]["num_knots"],
        outer_width=sidecar["strip"]["outer_width"],
        alpha=sidecar["architecture"]["alpha"],
    )
    model.load_state_dict(state)
    model.eval().to(device)
    _MODEL_CACHE[key] = (model, sidecar)
    return model, sidecar


def _validate_sidecar(sidecar: dict) -> None:
    if sidecar["schema_version"] != 1:
        raise RuntimeError("Unsupported schema_version")
    if sidecar["architecture"]["in_channels"] != 5:
        raise RuntimeError("Model must have 5 input channels")
    if sidecar["strip"]["canonical_shape_chw"] != [5, 1024, 256]:
        raise RuntimeError("Canonical strip mismatch")
    if sidecar["strip"]["outer_width"] != 128:
        raise RuntimeError("outer_width must be 128")
    if sidecar["architecture"]["name"] != "seam_harmonizer_v1":
        raise RuntimeError("Only seam_harmonizer_v1 exports are supported")
    if sidecar["architecture"]["num_knots"] < 4:
        raise RuntimeError("harmonizer num_knots must be >= 4")
    if sidecar["architecture"]["alpha"] > 0.2:
        raise RuntimeError("harmonizer alpha must be <= 0.20")
    if not sidecar["inference"]["hard_copy_outer"]:
        raise RuntimeError("hard_copy_outer must be true")
