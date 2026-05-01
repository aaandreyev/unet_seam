"""Main orchestrator for model surgery pipeline.

Usage:
    python -m model_surgery.run                          # all stages, default config
    python -m model_surgery.run --config model_surgery/config.yaml
    python -m model_surgery.run --stages s0,s1,s3       # specific stages
    python -m model_surgery.run --stages s5              # cycle only (needs s0 cache)
    python -m model_surgery.run --top-k 4               # override top-K checkpoints

Outputs land in model_surgery/outputs/ (or config.output_dir).

Runtime estimates (M1 Pro, mini_strips=60, batch_size=4):
  S0 survey:     ~1 min per 10 checkpoints
  S1 gate search: ~2 hours (4 ckpts × 20 gate × 5 gain × 4 detail = 1600 evals × ~4s)
  S2 ablation:   ~20 min
  S3 merge:      ~3 hours
  S4 surgery:    ~2 hours
  S5 cycle:      ~4-8 hours (depends on patience)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Model surgery pipeline for SeamHarmonizer")
    parser.add_argument("--config", default="model_surgery/config.yaml")
    parser.add_argument("--stages", default="s0,s1,s2,s3,s4,s5",
                        help="Comma-separated list of stages to run")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top-K checkpoints for all stages")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="Override s5 max_cycles")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.top_k:
        cfg["s5_cycle"]["top_k_base"] = args.top_k
    if args.max_cycles:
        cfg["s5_cycle"]["max_cycles"] = args.max_cycles

    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stages_to_run = {s.strip() for s in args.stages.split(",")}

    from model_surgery.lib.reporting import RunLog
    log_path = out_dir / "run.jsonl"
    t0 = time.monotonic()

    print("=" * 70)
    print("  SeamHarmonizer Model Surgery Pipeline")
    print(f"  Stages: {', '.join(sorted(stages_to_run))}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    with RunLog(log_path) as log:
        log.log("pipeline_start", stages=list(stages_to_run), config=cfg)

        survey_rows = []
        s1_results = None
        s3_results = None
        s4_results = None

        # ── S0: Survey ────────────────────────────────────────────────────────
        survey_cache = out_dir / "s0_survey.json"
        if "s0" in stages_to_run:
            from model_surgery.stages.s0_survey import survey
            survey_rows = survey(cfg, out_dir, log)
        elif survey_cache.exists():
            from model_surgery.lib.reporting import load_json
            survey_rows = load_json(survey_cache)
            print(f"[S0] Loaded {len(survey_rows)} rows from cache")
        else:
            print("[S0] ERROR: no survey data — run with s0 first", file=sys.stderr)
            sys.exit(1)

        if not survey_rows:
            print("No valid checkpoints found. Exiting.", file=sys.stderr)
            sys.exit(1)

        # ── S1: Gate search ───────────────────────────────────────────────────
        if "s1" in stages_to_run and cfg["stages"].get("s1_gate_search", True):
            from model_surgery.stages.s1_gate_search import gate_search
            s1_results = gate_search(survey_rows, cfg, out_dir, log, top_k=args.top_k or 4)

        # ── S2: Ablation ──────────────────────────────────────────────────────
        if "s2" in stages_to_run and cfg["stages"].get("s2_ablation", True):
            from model_surgery.stages.s2_ablation import ablation
            ablation(survey_rows, cfg, out_dir, log, top_k=args.top_k or 3)

        # ── S3: Merge ─────────────────────────────────────────────────────────
        if "s3" in stages_to_run and cfg["stages"].get("s3_merge", True):
            from model_surgery.stages.s3_merge import merge
            s3_results = merge(survey_rows, cfg, out_dir, log, top_k=args.top_k or 5)

        # ── S4: Surgery ───────────────────────────────────────────────────────
        if "s4" in stages_to_run and cfg["stages"].get("s4_surgery", True):
            from model_surgery.stages.s4_surgery import surgery
            s4_results = surgery(survey_rows, cfg, out_dir, log,
                                 top_k_base=args.top_k or 4,
                                 top_k_donors=args.top_k or 8)

        # ── S5: Cycle ─────────────────────────────────────────────────────────
        if "s5" in stages_to_run and cfg["stages"].get("s5_cycle", True):
            from model_surgery.stages.s5_cycle import cycle_loop
            final = cycle_loop(survey_rows, cfg, out_dir, log,
                               s1_results=s1_results,
                               s3_results=s3_results,
                               s4_results=s4_results)

        elapsed = round(time.monotonic() - t0, 1)
        log.log("pipeline_done", elapsed_s=elapsed)
        print(f"\n{'='*70}")
        print(f"  Pipeline complete in {elapsed/60:.1f} min")
        best_pt = out_dir / "best_model.pt"
        if best_pt.exists():
            print(f"  Best model → {best_pt}")
        print("=" * 70)


if __name__ == "__main__":
    main()
