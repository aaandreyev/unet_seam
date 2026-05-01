"""Stage 2: Layer ablation — zero out each output head, measure metric impact.

For each checkpoint in top-K: for each head group (gain, gamma, bias, mix, detail, gate):
  - Zero out those channels in coarse_head.2.weight and .bias
  - Run mini eval
  - Delta vs baseline = head's contribution to each metric

Outputs:
  outputs/s2_ablation.json  — per-checkpoint per-head deltas
  outputs/s2_ablation.txt   — human-readable table
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from model_surgery.lib.checkpoint_io import HEAD_SLICES, free_memory, load_ema
from model_surgery.lib.eval_mini import (
    _pick_device, build_loader, build_model,
    eval_with_preloaded, preload_batches, quality_score,
)
from model_surgery.lib.reporting import RunLog, save_json


DISPLAY_METRICS = [
    "boundary_mae_16", "boundary_ciede2000_16", "lowfreq_mae",
    "overcorrection_mae", "confidence_mean", "gain_abs_log_mean", "detail_abs_mean",
]


def ablation(
    survey_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    log: RunLog,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    s_cfg = cfg["s2_ablation"]
    eval_cfg = cfg["eval"]
    heads = [h["name"] for h in s_cfg["heads"]]
    candidates = survey_rows[:top_k]
    total = len(candidates) * (1 + len(heads))  # baseline + one per head

    mat_dir = Path(eval_cfg["materialized_dir"]) if eval_cfg.get("materialized_dir") else None
    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=s_cfg["n_strips"],
        outer_width=eval_cfg["outer_width"], inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"], boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"], seed=eval_cfg["seed"],
        materialized_dir=mat_dir,
    )

    log.log("ablation_start", top_k=len(candidates), heads=heads, total_evals=total)
    print(f"\n[S2] Ablation: {len(candidates)} checkpoints × {len(heads)+1} variants = {total} evals")

    device = _pick_device()
    preloaded = preload_batches(loader, device)

    all_results: list[dict[str, Any]] = []
    pbar = tqdm(total=total, desc="S2 ablation",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for row in candidates:
        ema_state, meta = load_ema(Path(row["path"]))
        base_name = row["name"]

        # Build model ONCE — zero heads in-place and restore, no state_dict cloning needed
        model = build_model(ema_state, meta, device)

        # Baseline (model unmodified)
        baseline_m = eval_with_preloaded(model, preloaded, outer_width=eval_cfg["outer_width"])
        baseline_q = quality_score(baseline_m)
        pbar.update(1)
        pbar.set_postfix_str(f"{base_name[:30]} baseline Q={baseline_q:.3f}")

        ckpt_result: dict[str, Any] = {"checkpoint": base_name, "baseline_quality": baseline_q,
                                        "baseline_metrics": baseline_m, "heads": {}}

        for head in heads:
            sl = HEAD_SLICES[head]
            w_key = "coarse_head.2.weight"
            b_key = "coarse_head.2.bias"
            # Save and zero in-place
            saved_w = model.coarse_head[2].weight.data[sl].clone()
            saved_b = model.coarse_head[2].bias.data[sl].clone()
            model.coarse_head[2].weight.data[sl] = 0.0
            model.coarse_head[2].bias.data[sl] = 0.0

            m = eval_with_preloaded(model, preloaded, outer_width=eval_cfg["outer_width"])

            # Restore original weights
            model.coarse_head[2].weight.data[sl] = saved_w
            model.coarse_head[2].bias.data[sl] = saved_b

            q = quality_score(m)
            delta_q = q - baseline_q
            deltas = {k: m.get(k, 0.0) - baseline_m.get(k, 0.0) for k in DISPLAY_METRICS}
            ckpt_result["heads"][head] = {
                "quality_without": q,
                "delta_quality": delta_q,
                "contribution_label": "critical" if delta_q > 5 else "important" if delta_q > 1 else "minor",
                "metric_deltas": deltas,
            }
            pbar.update(1)

        all_results.append(ckpt_result)
        del model, ema_state
        free_memory()

    pbar.close()
    save_json(all_results, out_dir / "s2_ablation.json")

    # Pretty text report
    lines = ["\n=== S2 ABLATION REPORT ===\n"]
    for r in all_results:
        lines.append(f"Checkpoint: {r['checkpoint']}")
        lines.append(f"  Baseline Q={r['baseline_quality']:.3f}")
        lines.append(f"  {'head':<10} {'ΔQUALITY':>10} {'importance':<12} "
                     + "  ".join(f"{k[:8]:>9}" for k in DISPLAY_METRICS))
        for head, hd in r["heads"].items():
            dq = hd["delta_quality"]
            dm = hd["metric_deltas"]
            dm_str = "  ".join(f"{dm.get(k, 0):+9.4f}" for k in DISPLAY_METRICS)
            lines.append(f"  {head:<10} {dq:>+10.3f} {hd['contribution_label']:<12} {dm_str}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (out_dir / "s2_ablation.txt").write_text(report, encoding="utf-8")
    log.log("ablation_done", n_checkpoints=len(all_results))
    return all_results
