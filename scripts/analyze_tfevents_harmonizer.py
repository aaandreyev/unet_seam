#!/usr/bin/env python3
"""Summarize TensorBoard scalars from harmonizer training (event file or logdir).

Usage:
  python scripts/analyze_tfevents_harmonizer.py path/to/events.out.tfevents.*
  python scripts/analyze_tfevents_harmonizer.py outputs/logs/tensorboard_harmonizer --json out.json

Requires: tensorboard (see requirements.txt)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# Default loss weights (HarmonizerLossComputer) for contribution estimates
_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "rec": 1.0,
    "seam": 1.5,
    "low": 1.0,
    "grad": 0.35,
    "chroma": 0.25,
    "stats": 0.15,
    "gate": 0.02,
    "field": 0.05,
    "detail": 0.05,
    "matrix": 0.05,
}


def _find_event_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    files = sorted(p.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No events.out.tfevents.* under {p}")
    return files


def _load_scalars(path: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(str(path), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def _merge_runs(files: list[Path]) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, list[tuple[int, float]]] = {}
    for f in files:
        part = _load_scalars(f)
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)
    for k in merged:
        merged[k].sort(key=lambda t: (t[0], t[1]))
    return merged


def _infer_val_steps(data: dict[str, list[tuple[int, float]]]) -> list[int]:
    keys = [k for k in data if k.startswith("val/")]
    if not keys:
        return []
    steps = sorted({s for k in keys for s, _ in data[k]})
    return steps


def _infer_batches_per_epoch(val_steps: list[int], train_max_step: int) -> int | None:
    if len(val_steps) < 2:
        return None
    d = val_steps[1] - val_steps[0]
    if d > 0 and (train_max_step == 0 or val_steps[-1] % d < 2 or val_steps[-1] == (len(val_steps) * d)):
        return d
    return d


def _table_val_epochs(
    data: dict[str, list[tuple[int, float]]], val_steps: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "val/metric/"
    loss_p = "val/loss/"
    for i, st in enumerate(val_steps, start=1):
        row: dict[str, Any] = {"epoch_end_step": st, "epoch_idx": i}

        def pick(tag: str) -> float | None:
            pts = data.get(tag, [])
            for s, v in pts:
                if s == st:
                    return v
            return None

        for key in (
            "total",
            "l_rec",
            "l_seam",
            "l_low",
            "l_grad",
            "l_chroma",
            "l_stats",
            "l_gate",
            "l_field",
            "l_detail",
            "l_matrix",
        ):
            v = pick(f"{loss_p}{key}")
            if v is not None:
                row[f"loss_{key}"] = v
        for suffix in (
            "boundary_mae_16",
            "baseline_boundary_mae_16",
            "boundary_ciede2000_16",
            "baseline_boundary_ciede2000_16",
            "lowfreq_mae",
            "gradient_mae",
            "quality_score",
            "confidence_mean",
            "detail_abs_mean",
            "gain_abs_log_mean",
        ):
            v = pick(prefix + suffix)
            if v is not None:
                row[suffix] = v
        bm = row.get("boundary_mae_16")
        bb = row.get("baseline_boundary_mae_16")
        if bm is not None and bb is not None and bb > 1e-8:
            row["mae16_vs_baseline_ratio"] = bm / bb
            row["mae16_improvement_pct"] = (1.0 - bm / bb) * 100.0
        de = row.get("boundary_ciede2000_16")
        bde = row.get("baseline_boundary_ciede2000_16")
        if de is not None and bde is not None and bde > 1e-8:
            row["de16_vs_baseline_ratio"] = de / bde
            row["de16_improvement_pct"] = (1.0 - de / bde) * 100.0
        rows.append(row)
    return rows


def _train_summary(data: dict[str, list[tuple[int, float]]], tag: str) -> dict[str, float] | None:
    pts = data.get(tag)
    if not pts:
        return None
    vals = [v for _, v in pts]
    n = len(vals)
    if n == 0:
        return None
    head = max(1, n // 10)
    tail = max(1, n // 10)
    return {
        "n": float(n),
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean_head10pct": statistics.mean(vals[:head]),
        "mean_tail10pct": statistics.mean(vals[-tail:]),
        "delta_last_minus_first": vals[-1] - vals[0],
    }


def _weighted_loss_breakdown_at_step(
    data: dict[str, list[tuple[int, float]]], step: int
) -> dict[str, Any] | None:
    """Estimate contribution of each term to total at a given global step (train)."""
    mapping = {
        "l_rec": "rec",
        "l_seam": "seam",
        "l_low": "low",
        "l_grad": "grad",
        "l_chroma": "chroma",
        "l_stats": "stats",
        "l_gate": "gate",
        "l_field": "field",
        "l_detail": "detail",
        "l_matrix": "matrix",
    }
    raw: dict[str, float] = {}
    for lk, wk in mapping.items():
        tag = f"train/loss/{lk}"
        for s, v in data.get(tag, []):
            if s == step:
                raw[wk] = v
                break
    if not raw:
        return None
    total_w = 0.0
    parts: dict[str, float] = {}
    for wk, v in raw.items():
        w = _DEFAULT_LOSS_WEIGHTS.get(wk, 0.0)
        parts[wk] = w * v
        total_w += parts[wk]
    out: dict[str, Any] = {"step": step, "weighted_sum": total_w, "components": {}}
    for wk, contrib in parts.items():
        out["components"][wk] = {
            "raw": raw[wk],
            "weight": _DEFAULT_LOSS_WEIGHTS.get(wk, 0.0),
            "weighted": contrib,
            "pct_of_weighted_sum": (100.0 * contrib / total_w) if total_w > 1e-12 else 0.0,
        }
    return out


def _best_epoch(rows: list[dict[str, Any]], key: str, minimize: bool = True) -> dict[str, Any] | None:
    candidates = [r for r in rows if key in r and r[key] is not None and not math.isnan(r[key])]
    if not candidates:
        return None
    if minimize:
        best = min(candidates, key=lambda r: r[key])
    else:
        best = max(candidates, key=lambda r: r[key])
    return {"key": key, "minimize": minimize, "epoch_idx": best.get("epoch_idx"), "value": best[key]}


def _volatility(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r]
    if len(vals) < 2:
        return None
    return statistics.pstdev(vals)


def analyze(
    data: dict[str, list[tuple[int, float]]],
) -> dict[str, Any]:
    val_steps = _infer_val_steps(data)
    train_steps = [s for s, _ in data.get("train/loss/total", [])]
    train_max = max(train_steps) if train_steps else 0
    bpe = _infer_batches_per_epoch(val_steps, train_max) if val_steps else None

    val_rows = _table_val_epochs(data, val_steps)
    report: dict[str, Any] = {
        "tags": sorted(data.keys()),
        "n_tags": len(data),
        "val_checkpoints": len(val_steps),
        "inferred_batches_per_epoch": bpe,
        "train_global_step_max": train_max,
        "val_epochs": val_rows,
        "train_summaries": {},
    }
    for t in (
        "train/loss/total",
        "train/metric/boundary_mae_16",
        "train/metric/lowfreq_mae",
        "train/metric/gradient_mae",
    ):
        s = _train_summary(data, t)
        if s:
            report["train_summaries"][t] = s

    if val_rows:
        report["best_epochs"] = [
            _best_epoch(val_rows, "quality_score", True),
            _best_epoch(val_rows, "boundary_ciede2000_16", True),
            _best_epoch(val_rows, "boundary_mae_16", True),
        ]
        report["val_volatility_pstdev"] = {}
        for k in ("boundary_mae_16", "boundary_ciede2000_16", "quality_score"):
            v = _volatility(val_rows, k)
            if v is not None:
                report["val_volatility_pstdev"][k] = v
        lt = [r["loss_total"] for r in val_rows if r.get("loss_total") is not None]
        if len(lt) > 1:
            report["val_volatility_pstdev"]["loss_total"] = statistics.pstdev(lt)

    # loss breakdown at last train log step (last point of train/loss/total)
    tt = data.get("train/loss/total", [])
    if tt:
        last_step = tt[-1][0]
        br = _weighted_loss_breakdown_at_step(data, last_step)
        if br:
            report["train_loss_breakdown_last_step"] = br

    # Correlation: train boundary_mae vs total (simple Pearson on aligned steps)
    t_mae = {s: v for s, v in data.get("train/metric/boundary_mae_16", [])}
    t_tot = {s: v for s, v in data.get("train/loss/total", [])}
    common = sorted(set(t_mae) & set(t_tot))
    if len(common) > 5:
        xs = [t_mae[s] for s in common]
        ys = [t_tot[s] for s in common]
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs)) or 1e-12
        deny = math.sqrt(sum((y - my) ** 2 for y in ys)) or 1e-12
        report["correlation_train_mae16_vs_loss_total"] = num / (denx * deny)

    return report


def _print_human(r: dict[str, Any]) -> None:
    print("=== TensorBoard harmonizer report ===\n")
    print(f"Tags: {r['n_tags']}  |  train max step: {r.get('train_global_step_max')}")
    if r.get("inferred_batches_per_epoch"):
        print(f"Inferred batches/epoch (from val steps): {r['inferred_batches_per_epoch']}")
    print(f"Val checkpoints logged: {r['val_checkpoints']}\n")

    if r.get("train_summaries"):
        print("--- Train (cumulative mean per log; first vs last) ---")
        for tag, s in r["train_summaries"].items():
            print(f"  {tag}")
            print(
                f"    first={s['first']:.6f} last={s['last']:.6f} min={s['min']:.6f} max={s['max']:.6f} "
                f"Δ(last-first)={s['delta_last_minus_first']:.6f}"
            )
            print(
                f"    mean first 10% logs={s['mean_head10pct']:.6f}  mean last 10%={s['mean_tail10pct']:.6f}"
            )
        print()

    if r.get("val_epochs"):
        print("--- Val (end of each epoch) ---")
        for row in r["val_epochs"]:
            ep = row.get("epoch_idx")
            print(f"  Epoch {ep} (step {row['epoch_end_step']}):")
            if "loss_total" in row:
                print(f"    val loss total: {row['loss_total']:.6f}")
            for k in (
                "boundary_mae_16",
                "baseline_boundary_mae_16",
                "mae16_improvement_pct",
                "boundary_ciede2000_16",
                "baseline_boundary_ciede2000_16",
                "de16_improvement_pct",
                "quality_score",
                "lowfreq_mae",
                "confidence_mean",
                "detail_abs_mean",
                "gain_abs_log_mean",
            ):
                if k in row and row[k] is not None:
                    if "pct" in k or "improvement" in k:
                        print(f"    {k}: {row[k]:.2f}")
                    else:
                        print(f"    {k}: {row[k]:.6f}")
            print()

    if r.get("best_epochs"):
        print("--- Best epochs ---")
        for b in r["best_epochs"]:
            if b:
                print(f"  By {b['key']}: epoch {b['epoch_idx']} → {b['value']:.6f} (minimize={b['minimize']})")
        print()

    if r.get("val_volatility_pstdev"):
        print("--- Val metric volatility (pstdev across epochs) ---")
        for k, v in r["val_volatility_pstdev"].items():
            if v is not None:
                print(f"  {k}: {v:.6f}")
        print()

    if r.get("correlation_train_mae16_vs_loss_total") is not None:
        c = r["correlation_train_mae16_vs_loss_total"]
        print(f"--- Pearson r (train mae16 vs loss total, aligned steps): {c:.4f} ---")
        print(
            "    (Often near 1.0: both are logged cumulative means over the same steps and trend together.)\n"
        )

    if r.get("train_loss_breakdown_last_step"):
        br = r["train_loss_breakdown_last_step"]
        print(f"--- Weighted loss breakdown @ train step {br['step']} (default yaml weights) ---")
        for name, d in sorted(br["components"].items(), key=lambda x: -x[1]["weighted"]):
            print(
                f"  {name:14s} raw={d['raw']:.6f}  w={d['weight']}  "
                f"contrib={d['weighted']:.6f}  ({d['pct_of_weighted_sum']:.1f}%)"
            )
        print()

    print("=== End ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze harmonizer TensorBoard event file(s)")
    ap.add_argument("path", type=Path, help="Event file or directory with events.out.tfevents.*")
    ap.add_argument("--json", type=Path, default=None, help="Write full report JSON here")
    args = ap.parse_args()
    files = _find_event_files(args.path)
    data = _merge_runs(files)
    if not data:
        print("No scalar data found.", file=sys.stderr)
        sys.exit(1)
    report = analyze(data)
    report["source_files"] = [str(f) for f in files]
    _print_human(report)
    if args.json:
        # JSON-serializable: report may have mixed types
        def _json_default(o: Any) -> Any:
            if isinstance(o, (int, float, str, bool)) or o is None:
                return o
            if isinstance(o, dict):
                return {str(k): _json_default(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_json_default(x) for x in o]
            return str(o)

        args.json.write_text(json.dumps(_json_default(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
