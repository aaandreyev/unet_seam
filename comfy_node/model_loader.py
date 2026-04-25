from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.models.factory import build_model_from_config


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
    model = build_model_from_config(sidecar)
    model.load_state_dict(state)
    model.eval().to(device)
    _MODEL_CACHE[key] = (model, sidecar)
    return model, sidecar


def _validate_sidecar(sidecar: dict) -> None:
    if sidecar["schema_version"] != 1:
        raise RuntimeError("Unsupported schema_version")
    if sidecar["architecture"]["in_channels"] != 9:
        raise RuntimeError("Model must have 9 input channels")
    if sidecar["strip"]["canonical_shape_chw"] != [9, 1024, 256]:
        raise RuntimeError("Canonical strip mismatch")
    if sidecar["strip"]["outer_width"] != 128:
        raise RuntimeError("outer_width must be 128")
    if sidecar["architecture"]["name"] != "seam_harmonizer_v3":
        raise RuntimeError("Only seam_harmonizer_v3 exports are supported")
    if not sidecar["inference"]["hard_copy_outer"]:
        raise RuntimeError("hard_copy_outer must be true")
