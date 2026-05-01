"""Memory-safe checkpoint loading for M1/MPS (32 GB unified memory).

Rules:
- Always call release() after use.
- Never hold >2 full model state_dicts in memory simultaneously.
- Use load_ema() — training state (optimizer, scaler) is never loaded.
"""
from __future__ import annotations

import gc
import copy
from pathlib import Path
from typing import Any

import torch

# Head channel slices in coarse_head output (18 channels total)
HEAD_SLICES: dict[str, slice] = {
    "gain":   slice(0, 1),
    "gamma":  slice(1, 2),
    "bias":   slice(2, 5),
    "mix":    slice(5, 14),
    "detail": slice(14, 17),
    "gate":   slice(17, 18),
}


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_ema(path: Path | str, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    """Load only EMA weights. Frees the full checkpoint immediately."""
    path = Path(path)
    raw = torch.load(path, map_location=map_location, weights_only=False)
    ema: dict[str, torch.Tensor] = raw.get("ema") or raw.get("model") or {}
    meta: dict[str, Any] = {
        "epoch": raw.get("epoch"),
        "config": raw.get("config") or {},
        "metrics": (raw.get("metrics") or {}).get("val") or {},
        "path": str(path),
    }
    del raw
    free_memory()
    return ema, meta


def save_surgery_checkpoint(state_dict: dict, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": state_dict, "model": state_dict, "config": meta.get("config", {}),
                "epoch": meta.get("epoch"), "metrics": {"val": meta.get("metrics", {})}}, path)


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def transplant_head(
    base_state: dict[str, torch.Tensor],
    donor_state: dict[str, torch.Tensor],
    head: str,
) -> dict[str, torch.Tensor]:
    """Return a new state_dict with `head` weights taken from donor."""
    sl = HEAD_SLICES[head]
    out = clone_state(base_state)
    w_key = "coarse_head.2.weight"
    b_key = "coarse_head.2.bias"
    if w_key in donor_state and w_key in out:
        out[w_key][sl] = donor_state[w_key][sl].clone()
    if b_key in donor_state and b_key in out:
        out[b_key][sl] = donor_state[b_key][sl].clone()
    return out


def linear_merge(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Linear interpolation: (1-alpha)*a + alpha*b for every tensor."""
    return {k: (1.0 - alpha) * a[k] + alpha * b[k] for k in a if k in b}


def slerp_merge(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    alpha: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Spherical interpolation per-tensor (falls back to lerp for near-parallel)."""
    out = {}
    for k in a:
        if k not in b:
            out[k] = a[k].clone()
            continue
        v0, v1 = a[k].float().flatten(), b[k].float().flatten()
        dot = (v0 * v1).sum() / (v0.norm() * v1.norm() + eps)
        dot = dot.clamp(-1.0, 1.0)
        theta = dot.acos()
        if theta.abs() < eps:
            merged = (1 - alpha) * v0 + alpha * v1
        else:
            merged = (torch.sin((1 - alpha) * theta) / theta.sin()) * v0 + \
                     (torch.sin(alpha * theta) / theta.sin()) * v1
        out[k] = merged.reshape(a[k].shape).to(a[k].dtype)
    return out


def selective_head_merge(
    base: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    head: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Linearly blend only one head, keep everything else from base."""
    sl = HEAD_SLICES[head]
    out = clone_state(base)
    for key_suffix, sl_ in [("weight", sl), ("bias", sl)]:
        key = f"coarse_head.2.{key_suffix}"
        if key in donor and key in out:
            out[key][sl_] = (1 - alpha) * base[key][sl_] + alpha * donor[key][sl_]
    return out


def apply_gate_bias_to_state(
    state: dict[str, torch.Tensor],
    gate_bias_delta: float,
) -> dict[str, torch.Tensor]:
    """Shift gate channel bias in coarse_head. Does NOT copy full state — mutates a clone."""
    out = clone_state(state)
    key = "coarse_head.2.bias"
    if key in out:
        out[key][HEAD_SLICES["gate"]] += gate_bias_delta
    return out
