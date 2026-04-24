from __future__ import annotations

import csv
import json
from pathlib import Path

from src.metrics.bootstrap import bootstrap_ci


def write_summary(run_dir: Path, metric_rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key in metric_rows[0].keys():
        values = [float(row[key]) for row in metric_rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "ci95": bootstrap_ci(values),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_bucket_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
