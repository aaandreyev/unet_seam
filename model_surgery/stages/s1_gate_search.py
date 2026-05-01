"""Stage 1: Inference-time parameter search (gate_bias, correction limits).

For each top-K checkpoint: search over gate_bias × gain_limit × detail_limit grids.
No weight changes needed — all modifications are to correction_limits at inference.
This is the cheapest and fastest improvement lever.

Outputs:
  outputs/s1_gate_search.json  — all eval results
  outputs/s1_best.pt           — best state_dict with best inference params
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from model_surgery.lib.checkpoint_io import free_memory, load_ema
from model_surgery.lib.eval_mini import build_loader, metrics_summary, quality_score, run_eval
from model_surgery.lib.reporting import RunLog, save_json


def _gate_bias_grid(cfg: dict) -> list[float]:
    lo, hi = cfg["gate_bias_range"]
    n = cfg["n_points"]
    return list(np.linspace(lo, hi, n))


def _apply_gate_bias(meta: dict, delta: float) -> dict:
    """Return a meta copy with gate_bias adjusted."""
    m = copy.deepcopy(meta)
    cl = (m.get("config") or {}).get("model", {}).get("correction_limits") or {}
    base = cl.get("gate_bias", -0.10)
    m.setdefault("config", {}).setdefault("model", {}).setdefault("correction_limits", {})
    m["config"]["model"]["correction_limits"]["gate_bias"] = base + delta
    return m


def _apply_limits(meta: dict, gain_limit: float | None, detail_limit: float | None) -> dict:
    m = copy.deepcopy(meta)
    cl = m.setdefault("config", {}).setdefault("model", {}).setdefault("correction_limits", {})
    if gain_limit is not None:
        cl["gain_limit"] = gain_limit
    if detail_limit is not None:
        cl["detail_limit"] = detail_limit
    return m


def gate_search(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    s_cfg = cfg["s1_gate_search"]
    eval_cfg = cfg["eval"]

    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=eval_cfg["outer_width"],
        inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"],
        boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"],
        seed=eval_cfg["seed"],
    )

    gate_biases = _gate_bias_grid(s_cfg)
    gain_limits = list(np.linspace(s_cfg["gain_limit_range"][0], s_cfg["gain_limit_range"][1],
                                   s_cfg["gain_n_points"]))
    detail_limits = list(np.linspace(s_cfg["detail_limit_range"][0], s_cfg["detail_limit_range"][1],
                                     s_cfg["detail_n_points"]))

    candidates = survey_rows[:top_k]
    total_evals = len(candidates) * len(gate_biases) * len(gain_limits) * len(detail_limits)

    log.log("gate_search_start", top_k=len(candidates), total_evals=total_evals)
    print(f"\n[S1] Gate/limits search: {len(candidates)} checkpoints × "
          f"{len(gate_biases)} gate × {len(gain_limits)} gain × {len(detail_limits)} detail "
          f"= {total_evals} evals")

    all_results: list[dict[str, Any]] = []
    best_q = float("inf")
    best_result: dict[str, Any] | None = None

    pbar = tqdm(total=total_evals, desc="S1 param search",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for row in candidates:
        ema_state, base_meta = load_ema(Path(row["path"]))
        base_name = row["name"]

        for gate_delta, gain_l, detail_l in itertools.product(gate_biases, gain_limits, detail_limits):
            meta_try = _apply_gate_bias(base_meta, gate_delta)
            meta_try = _apply_limits(meta_try, gain_l, detail_l)
            try:
                metrics = run_eval(ema_state, meta_try, loader,
                                   outer_width=eval_cfg["outer_width"])
            except Exception as e:
                pbar.update(1)
                continue

            q = quality_score(metrics)
            result = {
                "base": base_name,
                "base_path": row["path"],
                "gate_bias_delta": round(gate_delta, 4),
                "gain_limit": round(gain_l, 4),
                "detail_limit": round(detail_l, 4),
                "quality": q,
                **metrics,
            }
            all_results.append(result)

            if q < best_q:
                best_q = q
                best_result = result
                pbar.set_postfix_str(f"BEST {metrics_summary(metrics)}")

            pbar.update(1)

        del ema_state
        free_memory()

    pbar.close()
    all_results.sort(key=lambda r: r.get("quality", float("inf")))
    save_json(all_results[:200], out_dir / "s1_gate_search.json")

    if best_result:
        # Save best corrected state
        ema_state, best_meta = load_ema(Path(best_result["base_path"]))
        best_meta_patched = _apply_gate_bias(best_meta, best_result["gate_bias_delta"])
        best_meta_patched = _apply_limits(best_meta_patched,
                                          best_result["gain_limit"], best_result["detail_limit"])
        from model_surgery.lib.checkpoint_io import save_surgery_checkpoint
        save_surgery_checkpoint(ema_state, best_meta_patched, out_dir / "s1_best.pt")
        del ema_state
        free_memory()
        log.log("gate_search_done", best_quality=best_q, best=best_result)
        print(f"\n[S1] Best: {best_result['base']} "
              f"gate_delta={best_result['gate_bias_delta']:.3f} "
              f"gain={best_result['gain_limit']:.2f} "
              f"detail={best_result['detail_limit']:.2f}  Q={best_q:.3f}")

    return all_results
