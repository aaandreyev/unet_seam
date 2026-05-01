"""Stage 5: Cyclic improvement loop.

Starts from the best checkpoint found so far (from S0-S4).
Each cycle applies all non-training surgery operations: gate search, merge, transplant.
Keeps state if score improves. Stops after `patience` stale cycles.

Optional: if score plateaus, triggers a short targeted fine-tune (1-2 epochs).

Outputs:
  outputs/s5_cycle_log.json    — full cycle history
  outputs/best_model.pt        — final best model
  outputs/best_model_meta.json — meta (quality, params, provenance)

Op tuple format: (label, state, meta, owns_state)
  owns_state=True  → this cycle created the state (merge/transplant) → safe to del after
  owns_state=False → state is a shared reference (gate/limit variants) → never del
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from model_surgery.lib.checkpoint_io import (
    HEAD_SLICES, apply_gate_bias_to_state, clone_state, free_memory, linear_merge,
    load_ema, save_surgery_checkpoint, selective_head_merge,
    slerp_merge, transplant_head,
)
from model_surgery.lib.eval_mini import (
    build_loader, metrics_summary, quality_score, run_eval,
)
from model_surgery.lib.reporting import RunLog, save_json


def _mat_dir(eval_cfg: dict) -> Path | None:
    md = eval_cfg.get("materialized_dir")
    return Path(md) if md else None


# Op tuple type: (label, state, meta, owns_state)
Op = tuple[str, dict, dict, bool]


def _build_operations(
    current_state: dict,
    current_meta: dict,
    pool_states: list[tuple[dict, dict, str]],
) -> list[Op]:
    """Build ops for one cycle. Returns (label, state, meta, owns_state).

    Gate/limit variants: owns_state=False (shared ref to current_state, no clone needed).
    Merge/transplant: owns_state=True (newly allocated tensor dict).

    Memory budget per cycle (full 85MB model):
      Gate/limit (240 ops): 0 extra MB — shared reference
      Merge (7α × N_pool): N_pool × 85MB each created then freed immediately
      Head blend (3α × 6heads × N_pool): same
      Transplant (6heads × N_pool): same
    Maximum concurrent: 1 newly allocated state at a time inside the loop.
    """
    ops: list[Op] = []
    gate_biases = list(np.linspace(-0.05, -1.2, 12))
    gain_limits = [1.4, 1.6, 1.8, 2.0, 2.2]
    detail_limits = [0.18, 0.25, 0.32, 0.40]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    heads = list(HEAD_SLICES.keys())

    # 1. Gate bias + correction limits: share state, only vary meta
    for gb in gate_biases:
        meta_gb = copy.deepcopy(current_meta)
        cl = meta_gb.setdefault("config", {}).setdefault("model", {}).setdefault("correction_limits", {})
        cl["gate_bias"] = cl.get("gate_bias", -0.10) + gb
        for gl in gain_limits:
            for dl in detail_limits:
                m2 = copy.deepcopy(meta_gb)
                m2["config"]["model"]["correction_limits"]["gain_limit"] = gl
                m2["config"]["model"]["correction_limits"]["detail_limit"] = dl
                # owns_state=False: current_state is shared, must NOT be deleted
                ops.append((f"gate_gb={gb:.2f}_gl={gl:.1f}_dl={dl:.2f}",
                             current_state, m2, False))

    # 2. Linear merge: creates new tensor dict → owns_state=True
    for pool_state, _, pool_name in pool_states:
        for alpha in alphas:
            merged = linear_merge(current_state, pool_state, alpha)
            ops.append((f"merge_linear_α={alpha:.2f}_{pool_name[:20]}",
                         merged, current_meta, True))

    # 3. Selective head merge: creates new tensor dict → owns_state=True
    for pool_state, _, pool_name in pool_states:
        for head in heads:
            for alpha in [0.3, 0.5, 0.7]:
                merged = selective_head_merge(current_state, pool_state, head, alpha)
                ops.append((f"head_blend_{head}_α={alpha:.1f}_{pool_name[:20]}",
                             merged, current_meta, True))

    # 4. Head transplant: creates new tensor dict → owns_state=True
    for pool_state, _, pool_name in pool_states:
        for head in heads:
            transplanted = transplant_head(current_state, pool_state, head)
            ops.append((f"transplant_{head}_from={pool_name[:20]}",
                         transplanted, current_meta, True))

    return ops


def _run_finetune(
    state: dict, meta: dict, cfg: dict, out_dir: Path, log: RunLog,
) -> tuple[dict, dict] | None:
    """Run a short targeted fine-tune. Returns (new_state, new_meta) or None on failure."""
    ft_cfg = cfg["s5_cycle"]["finetune_on_plateau"]
    if not ft_cfg.get("enabled", False):
        return None
    try:
        import subprocess
        import sys
        import yaml

        base_cfg_path = Path("configs/finetune_harmonizer_stage8_soft.yaml")
        if not base_cfg_path.exists():
            log.log("finetune_skip", reason="stage8 config not found")
            return None
        base_ft = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        base_ft["train"]["lr"] = ft_cfg["lr"]
        base_ft["train"]["num_epochs"] = ft_cfg["epochs"]
        base_ft["train"]["freeze_correction_epochs"] = ft_cfg.get("freeze_correction_epochs", 0)
        # Log dir scoped to this surgery run to avoid clobbering regular training logs
        base_ft.setdefault("logging", {})["log_dir"] = str(out_dir / "finetune_logs")
        tmp_cfg = out_dir / "_tmp_finetune.yaml"
        tmp_cfg.write_text(yaml.safe_dump(base_ft, sort_keys=False), encoding="utf-8")

        tmp_ckpt = out_dir / "_tmp_finetune_base.pt"
        save_surgery_checkpoint(state, meta, tmp_ckpt)

        # train_harmonizer.py saves best to outputs/checkpoints/best_harmonizer_quality.pt
        # We rename it immediately after training to avoid race conditions.
        ft_out_ckpt = out_dir / "_finetune_result.pt"

        cmd = [sys.executable, "-m", "scripts.train_harmonizer",
               "--config", str(tmp_cfg),
               "--load-weights", str(tmp_ckpt),
               "--additional-epochs", str(ft_cfg["epochs"])]
        log.log("finetune_start", epochs=ft_cfg["epochs"], lr=ft_cfg["lr"])
        print(f"\n[S5] Starting {ft_cfg['epochs']}-epoch fine-tune from current best...")
        result = subprocess.run(cmd, capture_output=False, timeout=7200)
        if result.returncode != 0:
            log.log("finetune_failed", returncode=result.returncode)
            return None

        canonical = Path("outputs/checkpoints/best_harmonizer_quality.pt")
        if canonical.exists():
            import shutil
            shutil.copy2(canonical, ft_out_ckpt)
            new_state, new_meta = load_ema(ft_out_ckpt)
            log.log("finetune_done", saved_to=str(ft_out_ckpt))
            return new_state, new_meta
        log.log("finetune_no_output", expected=str(canonical))
    except Exception as e:
        log.log("finetune_error", error=str(e))
    return None


def cycle_loop(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    s1_results: list | None = None,
    s3_results: list | None = None,
    s4_results: list | None = None,
) -> dict[str, Any]:
    s_cfg = cfg["s5_cycle"]
    eval_cfg = cfg["eval"]
    max_cycles = s_cfg["max_cycles"]
    patience = s_cfg["patience"]
    top_k_base = s_cfg["top_k_base"]
    mat_dir = _mat_dir(eval_cfg)

    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"],
        materialized_dir=mat_dir,
    )
    full_loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["full_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"] + 1,
        materialized_dir=mat_dir,
    )

    # Load initial pool (top_k_base checkpoints)
    pool_entries: list[tuple[dict, dict, str]] = []
    for row in survey_rows[:max(top_k_base, 6)]:
        st, me = load_ema(Path(row["path"]))
        pool_entries.append((st, me, row["name"]))
    free_memory()

    # Start from globally best checkpoint
    best_row = survey_rows[0]
    current_state, current_meta = load_ema(Path(best_row["path"]))
    current_q = float(best_row.get("quality_score") or float("inf"))

    # Also load surgery-improved models from prior stages; promote if better
    for pt_name in ["s4_best.pt", "s3_best.pt", "s1_best.pt"]:
        pt_path = out_dir / pt_name
        if pt_path.exists():
            st, me = load_ema(pt_path)
            m = run_eval(st, me, loader, outer_width=eval_cfg["outer_width"])
            q = quality_score(m)
            if q < current_q:
                del current_state
                free_memory()
                current_state, current_meta, current_q = st, me, q
                print(f"[S5] Starting from {pt_name}  Q={q:.3f}")
            else:
                pool_entries.append((st, me, pt_name))

    log.log("cycle_start", initial_q=current_q, max_cycles=max_cycles)
    print(f"\n[S5] Cycle loop — initial Q={current_q:.3f}  target={cfg['target_quality_score']}")
    print(f"     patience={patience}  max_cycles={max_cycles}  pool={len(pool_entries)} ckpts")

    history: list[dict] = []
    stale_count = 0
    target_q = float(cfg["target_quality_score"])
    t_start = time.monotonic()
    last_improvement_op: str | None = None

    cycle_bar = tqdm(range(max_cycles), desc="S5 cycles",
                     bar_format="{l_bar}{bar}| cycle {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for cycle_idx in cycle_bar:
        cycle_bar.set_postfix_str(f"best_Q={current_q:.3f} stale={stale_count}/{patience}")

        ops = _build_operations(current_state, current_meta, pool_entries)
        cycle_best_q = current_q
        cycle_best_state: dict | None = None
        cycle_best_meta: dict | None = None
        cycle_best_label: str | None = None

        op_bar = tqdm(ops, desc=f"  cycle {cycle_idx+1} ops", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for label, cand_state, cand_meta, owns_state in op_bar:
            try:
                m = run_eval(cand_state, cand_meta, loader, outer_width=eval_cfg["outer_width"])
                q = quality_score(m)
                if q < cycle_best_q:
                    cycle_best_q = q
                    # Free previous cycle_best_state only if we own it
                    if cycle_best_state is not None:
                        del cycle_best_state
                        free_memory()
                    # Take ownership: clone if we don't own it (shared ref), keep if we do
                    cycle_best_state = clone_state(cand_state) if not owns_state else cand_state
                    cycle_best_meta = cand_meta
                    cycle_best_label = label
                    op_bar.set_postfix_str(f"NEW_BEST Q={q:.3f} {label[:40]}")
                elif owns_state:
                    del cand_state
                # Never del shared-ref states (owns_state=False)
                free_memory()
            except Exception:
                if owns_state:
                    del cand_state
                free_memory()

        op_bar.close()

        if cycle_best_state is not None and cycle_best_q < current_q:
            improvement = current_q - cycle_best_q
            last_improvement_op = cycle_best_label
            log.log("cycle_improvement", cycle=cycle_idx + 1,
                    prev_q=current_q, new_q=cycle_best_q,
                    improvement=improvement, op=cycle_best_label)
            del current_state
            free_memory()
            current_state = cycle_best_state
            current_meta = cycle_best_meta
            current_q = cycle_best_q
            stale_count = 0
            pool_entries.append((clone_state(current_state), copy.deepcopy(current_meta),
                                  f"cycle{cycle_idx+1}"))
            print(f"\n[S5] Cycle {cycle_idx+1}: IMPROVED → Q={current_q:.3f} "
                  f"(Δ={improvement:.3f}) via {cycle_best_label}")
        else:
            if cycle_best_state is not None:
                del cycle_best_state
            free_memory()
            stale_count += 1
            log.log("cycle_stale", cycle=cycle_idx + 1, stale_count=stale_count, q=current_q)

        history.append({"cycle": cycle_idx + 1, "quality": current_q,
                        "stale": stale_count, "op": last_improvement_op,
                        "elapsed_s": round(time.monotonic() - t_start, 1)})
        save_json(history, out_dir / "s5_cycle_log.json")

        if current_q <= target_q:
            print(f"\n[S5] TARGET REACHED: Q={current_q:.3f} ≤ {target_q}")
            log.log("target_reached", cycle=cycle_idx + 1, q=current_q)
            break

        plateau_every = s_cfg["finetune_on_plateau"]["plateau_cycles"]
        if stale_count > 0 and stale_count % plateau_every == 0:
            print(f"\n[S5] Plateau for {stale_count} cycles — attempting fine-tune...")
            ft_result = _run_finetune(current_state, current_meta, cfg, out_dir, log)
            if ft_result is not None:
                ft_state, ft_meta = ft_result
                ft_m = run_eval(ft_state, ft_meta, loader, outer_width=eval_cfg["outer_width"])
                ft_q = quality_score(ft_m)
                if ft_q < current_q:
                    del current_state
                    free_memory()
                    current_state, current_meta, current_q = ft_state, ft_meta, ft_q
                    stale_count = 0
                    pool_entries.append((clone_state(current_state),
                                         copy.deepcopy(current_meta), "finetune"))
                    print(f"[S5] Fine-tune improved: Q={current_q:.3f}")
                else:
                    del ft_state
                    free_memory()

        if stale_count >= patience:
            print(f"\n[S5] Patience exhausted ({stale_count} stale cycles). Stopping.")
            log.log("patience_exhausted", cycle=cycle_idx + 1, final_q=current_q)
            break

    cycle_bar.close()

    print(f"\n[S5] Running full eval on best model (Q={current_q:.3f})...")
    full_metrics = run_eval(current_state, current_meta, full_loader,
                            outer_width=eval_cfg["outer_width"])
    full_q = quality_score(full_metrics)
    print(f"[S5] Full eval: {metrics_summary(full_metrics)}")

    save_surgery_checkpoint(current_state, current_meta, out_dir / "best_model.pt")
    final_meta = {
        "quality_score": full_q,
        "metrics": full_metrics,
        "provenance": last_improvement_op,
        "cycles_run": len(history),
        "config": current_meta.get("config"),
    }
    save_json(final_meta, out_dir / "best_model_meta.json")

    for st, me, _ in pool_entries:
        del st, me
    del current_state
    free_memory()

    log.log("cycle_loop_done", final_quality=full_q, cycles=len(history))
    print(f"\n[S5] DONE. Final Q={full_q:.3f}  Saved → {out_dir/'best_model.pt'}")
    return final_meta
