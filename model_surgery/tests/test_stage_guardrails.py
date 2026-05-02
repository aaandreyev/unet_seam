from __future__ import annotations

from pathlib import Path

from model_surgery.lib.reporting import RunLog
from model_surgery.stages.s1_gate_search import gate_search
from model_surgery.stages.s3_merge import merge
from model_surgery.stages.s4_surgery import surgery


def test_s1_does_not_save_degrading_candidate(monkeypatch, tmp_path):
    cfg = {
        "manifest": "manifests/input_raw_manifest.jsonl",
        "eval": {
            "mini_strips": 2,
            "outer_width": 128,
            "inner_width": 128,
            "strip_height": 1024,
            "boundary_band_px": 24,
            "batch_size": 2,
            "seed": 1,
            "materialized_dir": None,
        },
        "s1_gate_search": {
            "gate_bias_range": [-0.1, 0.1],
            "n_points": 1,
            "gain_limit_range": [1.5, 1.5],
            "gain_n_points": 1,
            "detail_limit_range": [0.2, 0.2],
            "detail_n_points": 1,
        },
    }
    survey_rows = [{"path": "a.pt", "name": "a", "quality_score": 5.0}]
    saved: list[Path] = []

    class DummyModel:
        def __init__(self) -> None:
            self.correction_limits = {"gate_bias": -0.1, "gain_limit": 1.6, "detail_limit": 0.2}

    monkeypatch.setattr("model_surgery.stages.s1_gate_search.build_loader", lambda **kwargs: object())
    monkeypatch.setattr("model_surgery.stages.s1_gate_search._pick_device", lambda: "cpu")
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.preload_batches", lambda loader, device: [{}])
    monkeypatch.setattr(
        "model_surgery.stages.s1_gate_search.load_ema",
        lambda path: ({"weights": 1}, {"config": {"model": {"correction_limits": {"gate_bias": -0.1}}}}),
    )
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.build_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.cache_coarse_outputs", lambda *args, **kwargs: [{}])
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.eval_from_cache", lambda *args, **kwargs: {"q": 7.0})
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.eval_with_preloaded", lambda *args, **kwargs: {"q": 8.0})
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.quality_score", lambda metrics: metrics["q"])
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.metrics_summary", lambda metrics: f"Q={metrics['q']}")
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.save_surgery_checkpoint", lambda *args: saved.append(args[-1]))
    monkeypatch.setattr("model_surgery.stages.s1_gate_search.free_memory", lambda: None)

    with RunLog(tmp_path / "run.jsonl") as log:
        gate_search(survey_rows, cfg, tmp_path, log, top_k=1)

    assert saved == []


def test_s3_does_not_save_degrading_candidate(monkeypatch, tmp_path):
    cfg = {
        "manifest": "manifests/input_raw_manifest.jsonl",
        "eval": {
            "mini_strips": 2,
            "outer_width": 128,
            "inner_width": 128,
            "strip_height": 1024,
            "boundary_band_px": 24,
            "batch_size": 2,
            "seed": 1,
            "materialized_dir": None,
            "num_workers": 0,
            "materialized_preload": False,
        },
        "s3_merge": {
            "alphas": [0.1],
            "selective_heads": ["gain"],
            "use_fast_proxy": True,
            "rerank_top_k": 1,
        },
    }
    survey_rows = [
        {"path": "a.pt", "name": "a", "quality_score": 5.0},
        {"path": "b.pt", "name": "b", "quality_score": 6.0},
    ]
    saved: list[Path] = []

    class DummyEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def evaluate(self, state_dict, meta, *, fast: bool):
            return {"q": 9.0 if fast else 10.0}

        def close(self) -> None:
            return None

    monkeypatch.setattr("model_surgery.stages.s3_merge.build_loader", lambda **kwargs: object())
    monkeypatch.setattr("model_surgery.stages.s3_merge._pick_device", lambda: "cpu")
    monkeypatch.setattr("model_surgery.stages.s3_merge.preload_batches", lambda loader, device: [{}])
    monkeypatch.setattr("model_surgery.stages.s3_merge.ReusableModelEvaluator", DummyEvaluator)
    monkeypatch.setattr("model_surgery.stages.s3_merge.load_ema", lambda path: ({"weights": path.name}, {"config": {}}))
    monkeypatch.setattr("model_surgery.stages.s3_merge.linear_merge", lambda *args, **kwargs: {"merged": "linear"})
    monkeypatch.setattr("model_surgery.stages.s3_merge.slerp_merge", lambda *args, **kwargs: {"merged": "slerp"})
    monkeypatch.setattr("model_surgery.stages.s3_merge.selective_head_merge", lambda *args, **kwargs: {"merged": "head"})
    monkeypatch.setattr("model_surgery.stages.s3_merge.quality_score", lambda metrics: metrics["q"])
    monkeypatch.setattr("model_surgery.stages.s3_merge.metrics_summary", lambda metrics: f"Q={metrics['q']}")
    monkeypatch.setattr("model_surgery.stages.s3_merge.save_surgery_checkpoint", lambda *args: saved.append(args[-1]))
    monkeypatch.setattr("model_surgery.stages.s3_merge.free_memory", lambda: None)

    with RunLog(tmp_path / "run.jsonl") as log:
        merge(survey_rows, cfg, tmp_path, log, top_k=2)

    assert saved == []


def test_s4_does_not_save_degrading_candidate(monkeypatch, tmp_path):
    cfg = {
        "manifest": "manifests/input_raw_manifest.jsonl",
        "eval": {
            "mini_strips": 2,
            "outer_width": 128,
            "inner_width": 128,
            "strip_height": 1024,
            "boundary_band_px": 24,
            "batch_size": 2,
            "seed": 1,
            "materialized_dir": None,
            "num_workers": 0,
            "materialized_preload": False,
        },
        "s4_surgery": {
            "heads_to_transplant": ["gain"],
            "use_fast_proxy": True,
            "rerank_top_k": 1,
        },
    }
    survey_rows = [
        {"path": "a.pt", "name": "a", "quality_score": 5.0},
        {"path": "b.pt", "name": "b", "quality_score": 6.0},
    ]
    saved: list[Path] = []

    class DummyEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def evaluate(self, state_dict, meta, *, fast: bool):
            return {"q": 9.0 if fast else 10.0}

        def close(self) -> None:
            return None

    monkeypatch.setattr("model_surgery.stages.s4_surgery.build_loader", lambda **kwargs: object())
    monkeypatch.setattr("model_surgery.stages.s4_surgery._pick_device", lambda: "cpu")
    monkeypatch.setattr("model_surgery.stages.s4_surgery.preload_batches", lambda loader, device: [{}])
    monkeypatch.setattr("model_surgery.stages.s4_surgery.ReusableModelEvaluator", DummyEvaluator)
    monkeypatch.setattr("model_surgery.stages.s4_surgery.load_ema", lambda path: ({"weights": path.name}, {"config": {}}))
    monkeypatch.setattr("model_surgery.stages.s4_surgery.transplant_head", lambda *args, **kwargs: {"merged": "head"})
    monkeypatch.setattr("model_surgery.stages.s4_surgery.quality_score", lambda metrics: metrics["q"])
    monkeypatch.setattr("model_surgery.stages.s4_surgery.metrics_summary", lambda metrics: f"Q={metrics['q']}")
    monkeypatch.setattr("model_surgery.stages.s4_surgery.save_surgery_checkpoint", lambda *args: saved.append(args[-1]))
    monkeypatch.setattr("model_surgery.stages.s4_surgery.free_memory", lambda: None)

    with RunLog(tmp_path / "run.jsonl") as log:
        surgery(survey_rows, cfg, tmp_path, log, top_k_base=1, top_k_donors=2)

    assert saved == []
