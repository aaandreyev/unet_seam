"""Reporting: comparison tables, logging, CSV/JSON output."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_COLS = [
    "boundary_mae_16", "boundary_ciede2000_16",
    "baseline_boundary_mae_16", "baseline_boundary_ciede2000_16",
    "lowfreq_mae", "overcorrection_mae",
    "confidence_mean", "confidence_alignment_mae",
    "detail_abs_mean", "gain_abs_log_mean",
    "delta_luma_profile_mae", "delta_chroma_profile_mae",
    "quality_score",
]
SHORT = {
    "boundary_mae_16": "mae16",
    "boundary_ciede2000_16": "ΔE16",
    "lowfreq_mae": "lowf",
    "overcorrection_mae": "ovrcor",
    "confidence_mean": "conf",
    "confidence_alignment_mae": "conf_al",
    "detail_abs_mean": "detail",
    "gain_abs_log_mean": "gain_lg",
    "quality_score": "Q",
}


def _fmt(v: Any, width: int = 7) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "  n/a  "[:width]
    return f"{v:{width}.4f}" if isinstance(v, float) else f"{str(v):<{width}}"


def comparison_table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    cols = cols or METRIC_COLS
    short_cols = [SHORT.get(c, c[:10]) for c in cols]
    header = f"{'name':<45} {'ep':>3}  " + "  ".join(f"{s:>8}" for s in short_cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        name = row.get("name", "?")[:44]
        ep = str(row.get("epoch", "?"))[:3]
        vals = "  ".join(_fmt(row.get(c), 8) for c in cols)
        lines.append(f"{name:<45} {ep:>3}  {vals}")
    return "\n".join(lines)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = list({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class RunLog:
    """Append-only JSONL log for surgery session."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._t0 = time.monotonic()

    def log(self, event: str, **kwargs: Any) -> None:
        row = {"ts": datetime.utcnow().isoformat(), "elapsed_s": round(time.monotonic() - self._t0, 1),
               "event": event, **kwargs}
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
