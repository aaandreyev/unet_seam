from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.models.factory import build_model_from_config


# Cache key: (path, device, mtime). mtime invalidates stale entries after export.
_MODEL_CACHE: dict[tuple[str, str, float], tuple[torch.nn.Module, dict]] = {}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pick_inference_device() -> str:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_path(path: str) -> Path:
    model_path = Path(path).expanduser()
    candidates = []
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.append((Path.cwd() / model_path).resolve())
        candidates.append((_PROJECT_ROOT / model_path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _filter_matching_state_dict(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    current = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    dropped_unexpected: list[str] = []
    dropped_mismatch: list[str] = []
    for key, value in state.items():
        if key not in current:
            dropped_unexpected.append(key)
            continue
        if not isinstance(value, torch.Tensor) or current[key].shape != value.shape:
            dropped_mismatch.append(key)
            continue
        matched[key] = value
    return matched, dropped_unexpected, dropped_mismatch


def load_model(path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    model_path = _resolve_model_path(path)
    mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    key = (path, device, mtime)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not model_path.exists():  # path resolved above, check again after mtime
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Expected export like '{_PROJECT_ROOT / 'outputs/exports/seam_harmonizer_v3.safetensors'}'."
        )
    sidecar_path = model_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar JSON not found: {sidecar_path}. "
            "The .safetensors export must be рядом with a same-name .json file."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _validate_sidecar(sidecar)
    state = load_file(str(model_path), device=device)
    model = build_model_from_config(sidecar)
    compatible_state, dropped_unexpected, dropped_mismatch = _filter_matching_state_dict(model, state)
    result = model.load_state_dict(compatible_state, strict=False)
    if result.missing_keys or dropped_unexpected or dropped_mismatch:
        print(
            f"[seam_harmonizer] partial load: missing={list(result.missing_keys)[:6]} "
            f"dropped_unexpected={dropped_unexpected[:6]} "
            f"dropped_mismatch={dropped_mismatch[:6]}",
            flush=True,
        )
    model.eval().to(device)
    _MODEL_CACHE[key] = (model, sidecar)
    return model, sidecar


def _validate_sidecar(sidecar: dict) -> None:
    if sidecar["schema_version"] != 1:
        raise RuntimeError("Unsupported schema_version")
    if sidecar["architecture"]["in_channels"] != 9:
        raise RuntimeError("Model must have 9 input channels")
    if sidecar["architecture"]["name"] != "seam_harmonizer_v3":
        raise RuntimeError("Only seam_harmonizer_v3 exports are supported")
