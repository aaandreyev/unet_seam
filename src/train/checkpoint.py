from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
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


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


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
    payload = {
        "model": model.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "rng_state": capture_rng_state(),
        "config_hash": config_hash(config),
        "git_hash": git_hash(),
        "metrics": metrics,
    }
    save_checkpoint(path, payload)
