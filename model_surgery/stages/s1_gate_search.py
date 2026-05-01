"""Stage 1: Inference-time parameter search (gate_bias, correction limits).

Key optimisation: model is built ONCE per checkpoint, then correction_limits are mutated
in-place for each parameter combination. This avoids 400 redundant model builds + weight
loads per checkpoint — the dominant cost in the naive approach.

Speedup: ~20-50× over naive (1 build vs 400 builds per checkpoint).

Outputs:
  outputs/s1_gate_search.json  — all eval results sorted by quality
  outputs/s1_best.pt           — best state_dict with best inference params embedded in meta
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from model_surgery.lib.checkpoint_io import (
    free_memory, load_ema, save_surgery_checkpoint,
)
from model_surgery.lib.eval_mini import (
    _pick_device, build_loader, build_model,
    cache_coarse_outputs, eval_from_cache, eval_with_preloaded,
    preload_batches, metrics_summary, quality_score,
)
from model_surgery.lib.reporting import RunLog, save_json


def _gate_bias_grid(cfg: dict) -> list[float]:
    lo, hi = cfg["gate_bias_range"]
    return list(np.linspace(lo, hi, cfg["n_points"]))


def gate_search(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    s_cfg = cfg["s1_gate_search"]
    eval_cfg = cfg["eval"]
    outer_width = eval_cfg["outer_width"]

    mat_dir = Path(eval_cfg["materialized_dir"]) if eval_cfg.get("materialized_dir") else None
    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=outer_width,
        inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"],
        boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"],
        seed=eval_cfg["seed"],
        materialized_dir=mat_dir,
    )

    gate_biases = _gate_bias_grid(s_cfg)
    gain_limits = list(np.linspace(
        s_cfg["gain_limit_range"][0], s_cfg["gain_limit_range"][1], s_cfg["gain_n_points"]))
    detail_limits = list(np.linspace(
        s_cfg["detail_limit_range"][0], s_cfg["detail_limit_range"][1], s_cfg["detail_n_points"]))

    candidates = survey_rows[:top_k]
    combos = list(itertools.product(gate_biases, gain_limits, detail_limits))
    total_evals = len(candidates) * len(combos)

    log.log("gate_search_start", top_k=len(candidates), total_evals=total_evals,
            note="model built once per checkpoint, limits mutated in-place")
    print(f"\n[S1] Gate/limits search: {len(candidates)} checkpoints × "
          f"{len(gate_biases)} gate × {len(gain_limits)} gain × {len(detail_limits)} detail "
          f"= {total_evals} evals")
    print(f"     Model built ONCE per checkpoint → ~{len(combos)}× faster than naive")

    device = _pick_device()

    # Preload all eval batches to device ONCE — eliminates 1600 × N_batches H→D transfers
    print("[S1] Preloading eval batches to device...", flush=True)
    preloaded = preload_batches(loader, device)
    print(f"[S1] {len(preloaded)} batches preloaded to {device}", flush=True)

    all_results: list[dict[str, Any]] = []
    best_q = float("inf")
    best_result: dict[str, Any] | None = None
    best_limits: dict[str, float] | None = None

    pbar = tqdm(total=total_evals, desc="S1 param search",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for row in candidates:
        ema_state, base_meta = load_ema(Path(row["path"]))
        base_name = row["name"]

        # Phase 1: Build model ONCE, run encoder/decoder ONCE per batch → cache coarse outputs
        model = build_model(ema_state, base_meta, device)
        base_limits = dict(model.correction_limits)
        print(f"[S1] Caching coarse outputs for {base_name}...", flush=True)
        cached = cache_coarse_outputs(model, preloaded)
        del model, ema_state  # free GPU memory — reconstruction is cheap
        free_memory()

        # Phase 2: For each combo, run reconstruction only (no encoder/decoder)
        for gate_delta, gain_l, detail_l in combos:
            limits = {
                **base_limits,
                "gate_bias": base_limits.get("gate_bias", -0.10) + gate_delta,
                "gain_limit": gain_l,
                "detail_limit": detail_l,
            }
            try:
                metrics = eval_from_cache(cached, limits, outer_width)
            except Exception:
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
                best_limits = {
                    "gate_bias": limits["gate_bias"],
                    "gain_limit": gain_l,
                    "detail_limit": detail_l,
                }
                pbar.set_postfix_str(f"BEST {metrics_summary(metrics)}")

            pbar.update(1)

        del cached
        free_memory()

    pbar.close()
    all_results.sort(key=lambda r: r.get("quality", float("inf")))
    save_json(all_results[:200], out_dir / "s1_gate_search.json")

    if best_result and best_limits:
        # Full re-eval of best candidate with CIEDE2000 (fast search used proxy metrics)
        print("[S1] Full re-eval of best candidate (with CIEDE2000)...", flush=True)
        ema_state, best_meta = load_ema(Path(best_result["base_path"]))
        best_model = build_model(ema_state, best_meta, device)
        best_model.correction_limits.update(best_limits)
        full_metrics = eval_with_preloaded(best_model, preloaded, outer_width)
        full_q = quality_score(full_metrics)
        del best_model
        free_memory()
        print(f"[S1] Full Q={full_q:.3f}  {metrics_summary(full_metrics)}")

        # Bake best limits into meta and save
        cl = best_meta.setdefault("config", {}).setdefault("model", {}).setdefault("correction_limits", {})
        cl.update(best_limits)
        save_surgery_checkpoint(ema_state, best_meta, out_dir / "s1_best.pt")
        del ema_state
        free_memory()
        log.log("gate_search_done", best_quality=full_q, proxy_quality=best_q,
                best=best_result, best_limits=best_limits, full_metrics=full_metrics)
        print(f"\n[S1] Best: {best_result['base']} "
              f"gate_delta={best_result['gate_bias_delta']:.3f} "
              f"gain={best_limits['gain_limit']:.2f} "
              f"detail={best_limits['detail_limit']:.2f}  Q(full)={full_q:.3f}")

    return all_results
