from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(dtype=torch.uint8, device="cpu")
    if isinstance(value, (bytes, bytearray)):
        return torch.tensor(list(value), dtype=torch.uint8)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.uint8)
    raise TypeError(f"unsupported RNG tensor type: {type(value)!r}")


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    try:
        if "torch" in state:
            torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "python" in state:
            random.setstate(state["python"])
        if torch.cuda.is_available() and "cuda" in state:
            cuda_states = [_as_cpu_byte_tensor(item) for item in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Skipping incompatible RNG state restore: {exc}", RuntimeWarning, stacklevel=2)


def config_hash(config: dict) -> str:
    return str(abs(hash(json.dumps(config, sort_keys=True))))


def git_hash() -> str:
    return os.environ.get("GIT_HASH", "nogit")


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_state: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: object | None,
    epoch: int,
    config: dict,
    metrics: dict,
) -> None:
    # torch.compile wraps the model; always save the raw module's state dict.
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "model": raw_model.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "config": config,
        "rng_state": capture_rng_state(),
        "config_hash": config_hash(config),
        "git_hash": git_hash(),
        "metrics": metrics,
    }
    save_checkpoint(path, payload)
