from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/harmonizer_metrics_dashboard.py")
    spec = importlib.util.spec_from_file_location("harmonizer_metrics_dashboard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_collect_best_rows_keeps_distinct_metric_optima():
    mod = _load_module()
    rows = [
        {
            "epoch_idx": 1,
            "boundary_mae_16": 0.030,
            "baseline_boundary_mae_16": 0.060,
            "boundary_ciede2000_16": 3.0,
            "baseline_boundary_ciede2000_16": 5.0,
            "quality_score": 8.0,
        },
        {
            "epoch_idx": 2,
            "boundary_mae_16": 0.020,
            "baseline_boundary_mae_16": 0.060,
            "boundary_ciede2000_16": 3.6,
            "baseline_boundary_ciede2000_16": 5.0,
            "quality_score": 7.2,
        },
        {
            "epoch_idx": 3,
            "boundary_mae_16": 0.026,
            "baseline_boundary_mae_16": 0.060,
            "boundary_ciede2000_16": 2.4,
            "baseline_boundary_ciede2000_16": 5.0,
            "quality_score": 6.5,
        },
    ]
    best = mod._collect_best_rows(rows)
    assert best["mae"]["epoch_idx"] == 2
    assert best["de"]["epoch_idx"] == 3
    assert best["quality"]["epoch_idx"] == 3
    assert best["score"] is not None
    assert "integrated_score_0_95" in best["score"]


def test_goal_status_clamps_distance_for_degradation():
    mod = _load_module()
    goals = mod._goal_status_from_best(
        {
            "boundary_mae_16": 0.090,
            "baseline_boundary_mae_16": 0.050,
            "boundary_ciede2000_16": 8.0,
            "baseline_boundary_ciede2000_16": 4.0,
        }
    )
    assert goals
    assert all(0.0 <= goal["distance_pct"] <= 100.0 for goal in goals)
    assert not any(goal["reached"] for goal in goals)


def test_build_run_segments_tracks_multiple_event_files(monkeypatch, tmp_path: Path):
    mod = _load_module()
    f1 = tmp_path / "events.out.tfevents.1"
    f2 = tmp_path / "events.out.tfevents.2"
    f1.write_text("")
    f2.write_text("")

    scalars = {
        f1: {
            "train/loss/total": [(10, 1.0), (20, 0.8)],
            "train/metric/boundary_mae_16": [(10, 0.05), (20, 0.04)],
            "train/metric/lowfreq_mae": [(10, 0.03)],
            "train/metric/gradient_mae": [(20, 0.02)],
            "val/metric/boundary_mae_16": [(20, 0.04)],
        },
        f2: {
            "train/loss/total": [(30, 0.7), (40, 0.6)],
            "train/metric/boundary_mae_16": [(30, 0.03), (40, 0.02)],
            "train/metric/lowfreq_mae": [(40, 0.015)],
            "train/metric/gradient_mae": [(40, 0.010)],
            "val/metric/boundary_mae_16": [(40, 0.02)],
        },
    }

    monkeypatch.setattr(mod, "_load_scalars", lambda path: scalars[path])
    segments = mod._build_run_segments([f1, f2])

    assert len(segments) == 2
    assert segments[0]["start_step"] == 10
    assert segments[0]["end_step"] == 20
    assert segments[1]["start_step"] == 30
    assert segments[1]["train_boundary"] == 30
    assert segments[1]["val_boundary_epoch"] == 2
    assert segments[1]["markers"]["train/loss/total"] == {"x": 40.0, "y": 0.6}
