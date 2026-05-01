"""Tests for eval_mini: model building, quality_score, metrics_summary."""
from __future__ import annotations

import pytest
import torch

from model_surgery.lib.eval_mini import build_model, metrics_summary, quality_score
from src.models.harmonizer import SeamHarmonizerV3


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
