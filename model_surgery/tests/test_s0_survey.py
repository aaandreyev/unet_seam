from __future__ import annotations

import json
from pathlib import Path

from model_surgery.lib.reporting import RunLog
from model_surgery.stages.s0_survey import survey


def test_survey_ranks_by_live_eval_not_stale_summary(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = runs_dir / "run_a" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    best_pt = ckpt_dir / "best.pt"
    other_pt = ckpt_dir / "other.pt"
    best_pt.write_bytes(b"x")
    other_pt.write_bytes(b"y")

    (runs_dir / "run_a" / "eval_reports").mkdir(parents=True)
    (runs_dir / "run_a" / "eval_reports" / "summary_harmonizer.json").write_text(
        json.dumps({"metrics": {"boundary_mae_16": 999.0}}), encoding="utf-8"
    )

    cfg = {
        "runs_dir": str(runs_dir),
        "local_checkpoints_dir": str(tmp_path / "local_checkpoints"),
        "manifest": "manifests/input_raw_manifest.jsonl",
        "eval": {
            "materialized_dir": None,
            "mini_strips": 2,
            "outer_width": 128,
            "inner_width": 128,
            "strip_height": 1024,
            "boundary_band_px": 24,
            "batch_size": 2,
            "seed": 1,
            "num_workers": 0,
            "materialized_preload": False,
        },
    }

    def fake_load_ema(path: Path):
        return {"weights": path.name}, {"metrics": {"boundary_mae_16": 111.0}}

    class DummyEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def evaluate(self, ema_state, meta, *, fast: bool):
            if ema_state["weights"] == "best.pt":
                return {"q": 5.0, "boundary_mae_16": 0.05}
            return {"q": 9.0, "boundary_mae_16": 0.09}

        def close(self) -> None:
            return None

    monkeypatch.setattr("model_surgery.stages.s0_survey.build_loader", lambda **kwargs: object())
    monkeypatch.setattr("model_surgery.stages.s0_survey._pick_device", lambda: "cpu")
    monkeypatch.setattr("model_surgery.stages.s0_survey.preload_batches", lambda loader, device: [{}])
    monkeypatch.setattr("model_surgery.stages.s0_survey.ReusableModelEvaluator", DummyEvaluator)
    monkeypatch.setattr("model_surgery.stages.s0_survey.load_ema", fake_load_ema)
    monkeypatch.setattr(
        "model_surgery.stages.s0_survey.quality_score",
        lambda metrics: metrics.get("q", metrics.get("boundary_mae_16")),
    )
    monkeypatch.setattr("model_surgery.stages.s0_survey.free_memory", lambda: None)

    with RunLog(tmp_path / "run.jsonl") as log:
        rows = survey(cfg, tmp_path / "out", log)

    assert rows[0]["path"] == str(best_pt)
    assert rows[0]["quality_score"] == 5.0
    assert rows[0]["summary_quality_score"] == 111.0
    assert rows[0]["boundary_mae_16"] == 0.05


def test_survey_skips_live_eval_failures(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = runs_dir / "run_a" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    good_pt = ckpt_dir / "good.pt"
    bad_pt = ckpt_dir / "bad.pt"
    good_pt.write_bytes(b"x")
    bad_pt.write_bytes(b"y")

    cfg = {
        "runs_dir": str(runs_dir),
        "local_checkpoints_dir": str(tmp_path / "local_checkpoints"),
        "manifest": "manifests/input_raw_manifest.jsonl",
        "eval": {
            "materialized_dir": None,
            "mini_strips": 2,
            "outer_width": 128,
            "inner_width": 128,
            "strip_height": 1024,
            "boundary_band_px": 24,
            "batch_size": 2,
            "seed": 1,
            "num_workers": 0,
            "materialized_preload": False,
        },
    }

    def fake_load_ema(path: Path):
        return {"weights": path.name}, {"metrics": {"boundary_mae_16": 111.0}}

    class DummyEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def evaluate(self, ema_state, meta, *, fast: bool):
            if ema_state["weights"] == "bad.pt":
                raise RuntimeError("legacy incompatible checkpoint")
            return {"q": 5.0, "boundary_mae_16": 0.05}

        def close(self) -> None:
            return None

    monkeypatch.setattr("model_surgery.stages.s0_survey.build_loader", lambda **kwargs: object())
    monkeypatch.setattr("model_surgery.stages.s0_survey._pick_device", lambda: "cpu")
    monkeypatch.setattr("model_surgery.stages.s0_survey.preload_batches", lambda loader, device: [{}])
    monkeypatch.setattr("model_surgery.stages.s0_survey.ReusableModelEvaluator", DummyEvaluator)
    monkeypatch.setattr("model_surgery.stages.s0_survey.load_ema", fake_load_ema)
    monkeypatch.setattr(
        "model_surgery.stages.s0_survey.quality_score",
        lambda metrics: metrics.get("q", metrics.get("boundary_mae_16")),
    )
    monkeypatch.setattr("model_surgery.stages.s0_survey.free_memory", lambda: None)

    with RunLog(tmp_path / "run.jsonl") as log:
        rows = survey(cfg, tmp_path / "out", log)

    assert len(rows) == 1
    assert rows[0]["path"] == str(good_pt)
