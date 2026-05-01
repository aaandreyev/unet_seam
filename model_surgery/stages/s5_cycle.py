"""Stage 5: Cyclic improvement loop.

Starts from the best checkpoint found so far (from S0-S4).
Each cycle applies all non-training surgery operations: gate search, merge, transplant.
Keeps state if score improves. Stops after `patience` stale cycles.

Optional: if score plateaus, triggers a short targeted fine-tune (1-2 epochs).

Outputs:
  outputs/s5_cycle_log.json   — full cycle history
  outputs/best_model.pt       — final best model
  outputs/best_model_meta.json — meta (quality, params, provenance)
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
    HEAD_SLICES, clone_state, free_memory, linear_merge,
    load_ema, save_surgery_checkpoint, selective_head_merge,
    slerp_merge, transplant_head,
)
from model_surgery.lib.eval_mini import (
    build_loader, metrics_summary, quality_score, run_eval,
)
from model_surgery.lib.reporting import RunLog, save_json


# ---- helpers ----------------------------------------------------------------

def _all_surgery_rows(survey_rows: list[dict], s1_results, s3_results, s4_results) -> list[dict]:
    """Pool all candidate records from previous stages."""
    extra = []
    for results, key in [(s3_results, "s3_merge"), (s4_results, "s4_surgery")]:
        for r in (results or [])[:20]:
            path = None
            out_dir = None
            # These are eval records — we need to find their saved .pt
            # They are intermediate; we'll just use their source paths
        pass
    return survey_rows


def _build_operations(
    current_state: dict, current_meta: dict,
    pool_states: list[tuple[dict, dict, str]],
    s_cfg: dict,
) -> list[tuple[str, dict, dict]]:
    """Build a list of (label, candidate_state, candidate_meta) to evaluate this cycle."""
    ops = []
    gate_biases = list(np.linspace(-0.05, -1.2, 12))
    gain_limits = [1.4, 1.6, 1.8, 2.0, 2.2]
    detail_limits = [0.18, 0.25, 0.32, 0.40]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    heads = list(HEAD_SLICES.keys())

    # 1. Gate bias variants of current best
    for gb in gate_biases:
        from model_surgery.lib.checkpoint_io import apply_gate_bias_to_state
        meta_try = copy.deepcopy(current_meta)
        cl = meta_try.setdefault("config", {}).setdefault("model", {}).setdefault("correction_limits", {})
        base_gb = cl.get("gate_bias", -0.10)
        cl["gate_bias"] = base_gb + gb
        for gl in gain_limits:
            for dl in detail_limits:
                m2 = copy.deepcopy(meta_try)
                m2["config"]["model"]["correction_limits"]["gain_limit"] = gl
                m2["config"]["model"]["correction_limits"]["detail_limit"] = dl
                ops.append((f"gate_gb={gb:.2f}_gl={gl:.1f}_dl={dl:.2f}",
                            clone_state(current_state), m2))

    # 2. Linear merge with pool checkpoints
    for (pool_state, pool_meta, pool_name) in pool_states:
        for alpha in alphas:
            merged = linear_merge(current_state, pool_state, alpha)
            ops.append((f"merge_linear_α={alpha:.2f}_with={pool_name[:20]}",
                        merged, current_meta))

    # 3. Selective head merge
    for (pool_state, pool_meta, pool_name) in pool_states:
        for head in heads:
            for alpha in [0.3, 0.5, 0.7]:
                merged = selective_head_merge(current_state, pool_state, head, alpha)
                ops.append((f"head_blend_{head}_α={alpha:.1f}_donor={pool_name[:20]}",
                            merged, current_meta))

    # 4. Head transplant from pool
    for (pool_state, pool_meta, pool_name) in pool_states:
        for head in heads:
            transplanted = transplant_head(current_state, pool_state, head)
            ops.append((f"transplant_{head}_from={pool_name[:20]}",
                        transplanted, current_meta))

    return ops


def _run_finetune(state: dict, meta: dict, cfg: dict, out_dir: Path, log: RunLog) -> tuple[dict, dict] | None:
    """Run a short targeted fine-tune. Returns (new_state, new_meta) or None on failure."""
    ft_cfg = cfg["s5_cycle"]["finetune_on_plateau"]
    if not ft_cfg.get("enabled", False):
        return None

    try:
        import subprocess, sys, json, tempfile, yaml
        # Build a temp finetune config
        base_cfg_path = Path("configs/finetune_harmonizer_stage8_soft.yaml")
        base_ft = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        base_ft["train"]["lr"] = ft_cfg["lr"]
        base_ft["train"]["num_epochs"] = ft_cfg["epochs"]
        base_ft["train"]["freeze_correction_epochs"] = ft_cfg.get("freeze_correction_epochs", 0)
        tmp_cfg = out_dir / "_tmp_finetune.yaml"
        tmp_cfg.write_text(yaml.safe_dump(base_ft, sort_keys=False), encoding="utf-8")

        # Save current state as the load-weights checkpoint
        tmp_ckpt = out_dir / "_tmp_finetune_base.pt"
        save_surgery_checkpoint(state, meta, tmp_ckpt)
        out_ckpt = out_dir / "_tmp_finetune_out"
        out_ckpt.mkdir(exist_ok=True)

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

        ft_ckpt = Path("outputs/checkpoints/best_harmonizer_quality.pt")
        if ft_ckpt.exists():
            new_state, new_meta = load_ema(ft_ckpt)
            log.log("finetune_done", ckpt=str(ft_ckpt))
            return new_state, new_meta
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

    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"],
    )
    full_loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["full_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"] + 1,
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

    # Also load any surgery-improved models from prev stages
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
    print(f"     patience={patience}  max_cycles={max_cycles}")

    history: list[dict] = []
    stale_count = 0
    target_q = float(cfg["target_quality_score"])
    t_start = time.monotonic()

    cycle_bar = tqdm(range(max_cycles), desc="S5 cycles",
                     bar_format="{l_bar}{bar}| cycle {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for cycle_idx in cycle_bar:
        cycle_bar.set_postfix_str(f"best_Q={current_q:.3f} stale={stale_count}/{patience}")

        ops = _build_operations(current_state, current_meta, pool_entries, s_cfg)
        cycle_best_q = current_q
        cycle_best_state = None
        cycle_best_meta = None
        cycle_best_label = None

        op_bar = tqdm(ops, desc=f"  cycle {cycle_idx+1} ops", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for label, cand_state, cand_meta in op_bar:
            try:
                m = run_eval(cand_state, cand_meta, loader, outer_width=eval_cfg["outer_width"])
                q = quality_score(m)
                if q < cycle_best_q:
                    cycle_best_q = q
                    if cycle_best_state is not None:
                        del cycle_best_state
                    cycle_best_state = cand_state
                    cycle_best_meta = cand_meta
                    cycle_best_label = label
                    op_bar.set_postfix_str(f"NEW_BEST Q={q:.3f} {label[:40]}")
                else:
                    del cand_state
                free_memory()
            except Exception:
                del cand_state
                free_memory()

        op_bar.close()

        if cycle_best_state is not None and cycle_best_q < current_q:
            improvement = current_q - cycle_best_q
            log.log("cycle_improvement", cycle=cycle_idx + 1,
                    prev_q=current_q, new_q=cycle_best_q,
                    improvement=improvement, op=cycle_best_label)
            del current_state
            free_memory()
            current_state = cycle_best_state
            current_meta = cycle_best_meta
            current_q = cycle_best_q
            stale_count = 0
            # Add improved model to pool
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
                        "stale": stale_count, "op": cycle_best_label,
                        "elapsed_s": round(time.monotonic() - t_start, 1)})
        save_json(history, out_dir / "s5_cycle_log.json")

        # Check target
        if current_q <= target_q:
            print(f"\n[S5] TARGET REACHED: Q={current_q:.3f} ≤ {target_q}")
            log.log("target_reached", cycle=cycle_idx + 1, q=current_q)
            break

        # Plateau → try fine-tune
        if stale_count > 0 and stale_count % s_cfg["finetune_on_plateau"]["plateau_cycles"] == 0:
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

        # Hard stop on patience
        if stale_count >= patience:
            print(f"\n[S5] Patience exhausted ({stale_count} stale cycles). Stopping.")
            log.log("patience_exhausted", cycle=cycle_idx + 1, final_q=current_q)
            break

    cycle_bar.close()

    # Final full eval
    print(f"\n[S5] Running full eval on best model (Q={current_q:.3f})...")
    full_metrics = run_eval(current_state, current_meta, full_loader, outer_width=eval_cfg["outer_width"])
    full_q = quality_score(full_metrics)
    print(f"[S5] Full eval: {metrics_summary(full_metrics)}")

    # Save
    save_surgery_checkpoint(current_state, current_meta, out_dir / "best_model.pt")
    final_meta = {
        "quality_score": full_q,
        "metrics": full_metrics,
        "provenance": cycle_best_label,
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
