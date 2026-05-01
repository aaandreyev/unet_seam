"""Tests for eval_mini: model building, quality_score, metrics_summary."""
from __future__ import annotations

import pytest
import torch

from model_surgery.lib.eval_mini import (
    ReusableModelEvaluator, _infer_channels_blocks, build_model,
    cache_coarse_outputs, eval_from_cache, eval_with_preloaded_fast,
    metrics_summary, quality_score,
)
from src.models.harmonizer import DEFAULT_CORRECTION_LIMITS, SeamHarmonizerV3


@pytest.fixture
def dummy_meta():
    return {
        "config": {
            "model": {
                "architecture": "seam_harmonizer_v3",
                "in_channels": 9, "channels": [8, 12, 16, 24],
                "blocks": [1, 1, 1, 1],
                "correction_limits": {"gate_bias": -0.10},
            },
            "dataset": {"outer_width": 128, "boundary_band_px": 24},
        },
        "epoch": 1,
        "metrics": {},
    }


@pytest.fixture
def tiny_state():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    return m.state_dict()


def test_build_model_returns_seam_harmonizer(tiny_state, dummy_meta):
    device = torch.device("cpu")
    model = build_model(tiny_state, dummy_meta, device)
    assert isinstance(model, SeamHarmonizerV3)


def test_build_model_is_eval_mode(tiny_state, dummy_meta):
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    assert not model.training


def test_build_model_correct_architecture(tiny_state, dummy_meta):
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    assert model.channels == (8, 12, 16, 24)
    assert model.blocks == (1, 1, 1, 1)


def test_build_model_legacy_state_dict_loads_gracefully(dummy_meta):
    """State dict missing attention_head should load without exception."""
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    legacy_state = {k: v for k, v in m.state_dict().items()
                    if not k.startswith("attention_head")}
    device = torch.device("cpu")
    model = build_model(legacy_state, dummy_meta, device)
    assert isinstance(model, SeamHarmonizerV3)


def test_build_model_unexpected_keys_raises(dummy_meta):
    """State dict with completely wrong keys must raise."""
    bad_state = {"wrong.key": torch.zeros(1)}
    with pytest.raises((RuntimeError, Exception)):
        build_model(bad_state, dummy_meta, torch.device("cpu"))


def test_quality_score_lower_is_better():
    good = {"boundary_mae_16": 0.010, "boundary_ciede2000_16": 1.8,
            "baseline_boundary_mae_16": 0.04, "baseline_boundary_ciede2000_16": 5.0,
            "lowfreq_mae": 0.015, "overcorrection_mae": 0.004,
            "confidence_mean": 0.25, "confidence_alignment_mae": 0.10,
            "detail_abs_mean": 0.003, "gain_abs_log_mean": 0.04,
            "delta_luma_profile_mae": 0.012, "delta_chroma_profile_mae": 0.013}
    bad = {"boundary_mae_16": 0.030, "boundary_ciede2000_16": 4.5,
           "baseline_boundary_mae_16": 0.04, "baseline_boundary_ciede2000_16": 5.0,
           "lowfreq_mae": 0.045, "overcorrection_mae": 0.012,
           "confidence_mean": 0.60, "confidence_alignment_mae": 0.30,
           "detail_abs_mean": 0.010, "gain_abs_log_mean": 0.15,
           "delta_luma_profile_mae": 0.025, "delta_chroma_profile_mae": 0.025}
    assert quality_score(good) < quality_score(bad)


def test_quality_score_with_empty_metrics():
    q = quality_score({})
    assert isinstance(q, float)
    assert q >= 0 or q != q  # non-negative or nan (when baseline=0)


def test_quality_score_missing_penalty_metrics_dont_inflate():
    """Older checkpoints without overcorrection_mae etc. must not be penalised."""
    good_seam = {
        "boundary_mae_16": 0.0145, "boundary_ciede2000_16": 2.15,
        "baseline_boundary_mae_16": 0.040, "baseline_boundary_ciede2000_16": 4.67,
        "lowfreq_mae": 0.018, "confidence_mean": 0.38,
        # No overcorrection_mae, delta_*_profile_mae — like stage3 checkpoint
    }
    bad_seam = {
        "boundary_mae_16": 0.030, "boundary_ciede2000_16": 3.5,
        "baseline_boundary_mae_16": 0.040, "baseline_boundary_ciede2000_16": 4.67,
        "lowfreq_mae": 0.035, "confidence_mean": 0.55,
        "overcorrection_mae": 0.008, "delta_luma_profile_mae": 0.02,
        "delta_chroma_profile_mae": 0.02,
    }
    q_good = quality_score(good_seam)
    q_bad = quality_score(bad_seam)
    assert q_good < q_bad, (
        f"Better seam should rank higher (lower Q): good_seam Q={q_good:.1f}, bad_seam Q={q_bad:.1f}"
    )


def test_quality_score_ignores_unknown_keys():
    m = {"boundary_mae_16": 0.02, "unknown_metric": 999.0}
    q = quality_score(m)
    assert isinstance(q, float)


def test_metrics_summary_contains_key_values():
    m = {"boundary_ciede2000_16": 2.35, "boundary_mae_16": 0.0155,
         "confidence_mean": 0.28, "lowfreq_mae": 0.019,
         "overcorrection_mae": 0.006,
         "baseline_boundary_mae_16": 0.04, "baseline_boundary_ciede2000_16": 4.5}
    s = metrics_summary(m)
    assert "2.350" in s or "2.35" in s
    assert "0.0155" in s or "0.016" in s
    assert "0.28" in s
    assert "Q=" in s


def test_metrics_summary_handles_missing_keys():
    s = metrics_summary({})
    assert "Q=" in s
    assert "nan" in s.lower() or "nan" not in s  # just shouldn't crash


def test_quality_score_no_nan_when_ciede2000_missing():
    """Missing CIEDE2000 must not produce nan (inf/inf case)."""
    m = {"boundary_mae_16": 0.02, "baseline_boundary_mae_16": 0.04,
         "lowfreq_mae": 0.015, "confidence_mean": 0.25}
    q = quality_score(m)
    assert q == q, "quality_score must not be nan when CIEDE2000 is absent"
    assert q != float("inf"), "quality_score must not be inf"


def test_quality_score_no_nan_empty():
    q = quality_score({})
    assert q == q, "quality_score({}) must not be nan"


# --- architecture inference ---

def test_infer_channels_blocks_from_state_dict(tiny_state):
    ch, bl = _infer_channels_blocks(tiny_state)
    assert ch == (8, 12, 16, 24)
    assert len(bl) == 4
    assert all(b >= 1 for b in bl)


def test_build_model_with_wrong_meta_channels_falls_back(tiny_state):
    """build_model must recover from size mismatch by inferring architecture."""
    meta_wrong = {
        "config": {
            "model": {"channels": [32, 64, 128, 192], "blocks": [2, 2, 4, 6]},
            "dataset": {"outer_width": 128, "boundary_band_px": 24},
        }
    }
    # tiny_state has channels=(8,12,16,24) but meta says (32,64,128,192) → size mismatch
    model = build_model(tiny_state, meta_wrong, torch.device("cpu"))
    assert model.channels == (8, 12, 16, 24)


# --- cache_coarse_outputs + eval_from_cache ---

def _make_batch(outer_width=128, inner_width=64, height=128, batch=2):
    total_width = outer_width + inner_width
    strip = torch.rand(batch, 9, height, total_width)
    rgb = strip[:, :3]
    return {
        "input": strip,
        "input_rgb": rgb,
        "target": torch.rand(batch, 3, height, total_width),
    }


def test_cache_coarse_outputs_returns_required_keys(tiny_state, dummy_meta):
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    batch = _make_batch()
    cached = cache_coarse_outputs(model, [batch])
    assert len(cached) == 1
    for key in ("gain_lowres", "gamma_lowres", "bias_lowres", "mix_lowres",
                "detail_lowres", "gate_lowres", "attention_lowres", "x_rgb",
                "input_rgb", "target"):
        assert key in cached[0], f"Missing key: {key}"


def test_eval_from_cache_returns_metrics(tiny_state, dummy_meta):
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    batch = _make_batch()
    cached = cache_coarse_outputs(model, [batch])
    limits = dict(DEFAULT_CORRECTION_LIMITS)
    metrics = eval_from_cache(cached, limits, outer_width=128)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert "boundary_mae_16" in metrics


def test_eval_from_cache_varies_with_limits(tiny_state, dummy_meta):
    """Different correction limits must produce different metrics."""
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    batch = _make_batch()
    cached = cache_coarse_outputs(model, [batch, _make_batch()])

    limits_a = {**DEFAULT_CORRECTION_LIMITS, "gate_bias": -0.10}
    limits_b = {**DEFAULT_CORRECTION_LIMITS, "gate_bias": -2.00}
    ma = eval_from_cache(cached, limits_a, outer_width=128)
    mb = eval_from_cache(cached, limits_b, outer_width=128)
    # Heavy negative gate_bias suppresses confidence → different correction → different metrics
    assert ma != mb


def test_eval_with_preloaded_fast_returns_metrics(tiny_state, dummy_meta):
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    batches = [_make_batch(), _make_batch()]
    metrics = eval_with_preloaded_fast(model, batches, outer_width=128)
    assert "boundary_mae_16" in metrics
    assert "lowfreq_mae" in metrics
    assert "boundary_ciede2000_16" not in metrics


def test_reusable_model_evaluator_reuses_shell_and_supports_fast_and_full(tiny_state, dummy_meta):
    batches = [_make_batch(), _make_batch()]
    evaluator = ReusableModelEvaluator(batches, torch.device("cpu"), outer_width=128)
    fast_metrics = evaluator.evaluate(tiny_state, dummy_meta, fast=True)
    full_metrics = evaluator.evaluate(tiny_state, dummy_meta, fast=False)
    assert "boundary_mae_16" in fast_metrics
    assert "boundary_ciede2000_16" not in fast_metrics
    assert "boundary_ciede2000_16" in full_metrics
    assert len(evaluator._models) == 1
    evaluator.close()


def test_cache_coarse_does_not_require_model_after(tiny_state, dummy_meta):
    """eval_from_cache must work after the model is deleted."""
    model = build_model(tiny_state, dummy_meta, torch.device("cpu"))
    batch = _make_batch()
    cached = cache_coarse_outputs(model, [batch])
    del model  # free model — cache must be self-contained
    limits = dict(DEFAULT_CORRECTION_LIMITS)
    metrics = eval_from_cache(cached, limits, outer_width=128)
    assert "boundary_mae_16" in metrics
