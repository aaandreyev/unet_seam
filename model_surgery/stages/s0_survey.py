"""Stage 0: Survey all checkpoints across all runs.

Outputs:
  outputs/s0_survey.json   — list of all candidate records
  outputs/s0_survey.csv    — same as table
  outputs/s0_survey.txt    — pretty comparison table (printed + saved)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from model_surgery.lib.checkpoint_io import free_memory, load_ema
from model_surgery.lib.eval_mini import ReusableModelEvaluator, _pick_device, build_loader, preload_batches, quality_score
from model_surgery.lib.reporting import METRIC_COLS, RunLog, comparison_table, save_csv, save_json


def _parse_eval_summary(run_dir: Path) -> dict[str, float]:
    p = run_dir / "eval_reports" / "summary_harmonizer.json"
    if not p.exists():
        return {}
    import json
    return json.loads(p.read_text(encoding="utf-8")).get("metrics", {})


def survey(cfg: dict[str, Any], out_dir: Path, log: RunLog) -> list[dict[str, Any]]:
    runs_root = Path(cfg["runs_dir"])
    local_ckpt_dir = Path(cfg["local_checkpoints_dir"])
    eval_cfg = cfg["eval"]

    # Collect all .pt paths
    candidates: list[Path] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates.extend(sorted(ckpt_dir.glob("*.pt")))
    # Also local checkpoints dir
    if local_ckpt_dir.exists():
        candidates.extend(sorted(local_ckpt_dir.glob("*.pt")))

    candidates = sorted(set(candidates))
    log.log("survey_start", n_checkpoints=len(candidates))
    print(f"\n[S0] Surveying {len(candidates)} checkpoints across all runs...")
    print("[S0] Ranking with live surgery mini-eval on the materialized dataset, not stale training summaries.")

    mat_dir = Path(eval_cfg["materialized_dir"]) if eval_cfg.get("materialized_dir") else None
    loader = build_loader(
        manifest=Path(cfg["manifest"]),
        n_strips=eval_cfg["mini_strips"],
        outer_width=eval_cfg["outer_width"],
        inner_width=eval_cfg["inner_width"],
        strip_height=eval_cfg["strip_height"],
        boundary_band_px=eval_cfg["boundary_band_px"],
        batch_size=eval_cfg["batch_size"],
        seed=eval_cfg["seed"],
        materialized_dir=mat_dir,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        materialized_preload=bool(eval_cfg.get("materialized_preload", False)),
    )
    device = _pick_device()
    preloaded = preload_batches(loader, device)
    evaluator = ReusableModelEvaluator(preloaded, device, outer_width=eval_cfg["outer_width"])

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for pt_path in tqdm(candidates, desc="S0 survey", unit="ckpt",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        if str(pt_path) in seen_paths:
            continue
        seen_paths.add(str(pt_path))

        try:
            ema_state, meta = load_ema(pt_path)
        except Exception as e:
            log.log("survey_skip", path=str(pt_path), reason=str(e))
            continue

        m = meta.get("metrics") or {}
        # Supplement with eval summary if available
        run_dir = pt_path.parent.parent
        eval_m = _parse_eval_summary(run_dir)
        merged_m = {**eval_m, **m}  # legacy reference metrics only

        live_metrics = evaluator.evaluate(ema_state, meta, fast=False)
        q = quality_score(live_metrics)
        summary_q = quality_score(merged_m) if merged_m else None
        row: dict[str, Any] = {
            "name": f"{run_dir.name}/{pt_path.name}" if run_dir.is_relative_to(runs_root) else pt_path.name,
            "path": str(pt_path),
            "run": run_dir.name,
            "epoch": meta.get("epoch"),
            "quality_score": q,
            "summary_quality_score": summary_q,
        }
        row.update({k: live_metrics.get(k) for k in METRIC_COLS if k != "quality_score"})
        rows.append(row)
        del ema_state
        free_memory()

    # Sort by quality (lower = better)
    rows.sort(key=lambda r: r.get("quality_score") or float("inf"))

    table = comparison_table(rows)
    print("\n" + table)

    save_json(rows, out_dir / "s0_survey.json")
    save_csv(rows, out_dir / "s0_survey.csv")
    (out_dir / "s0_survey.txt").write_text(table, encoding="utf-8")

    evaluator.close()
    free_memory()

    log.log("survey_done", n_valid=len(rows),
            best_path=rows[0]["path"] if rows else None,
            best_quality=rows[0].get("quality_score") if rows else None)
    print(f"\n[S0] Done. Best: {rows[0]['name']} Q={rows[0].get('quality_score', '?'):.3f}")
    return rows
