"""Tests for stage9 improvements:
- Block 1: finetune_harmonizer_stage9.yaml valid and has expected values
- Block 2: train_harmonizer saves best_de / best_mae checkpoints
- Block 3: analyze_tfevents gain_reg default matches HarmonizerLossComputer
- Block 5: dashboard detects monotone-growth trends for gain / conf_align
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dashboard():
    path = Path("scripts/harmonizer_metrics_dashboard.py")
    spec = importlib.util.spec_from_file_location("harmonizer_metrics_dashboard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_analyze():
    path = Path("scripts/analyze_tfevents_harmonizer.py")
    spec = importlib.util.spec_from_file_location("analyze_tfevents_harmonizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Block 1: stage9 config
# ---------------------------------------------------------------------------

STAGE9_CFG = Path("configs/finetune_harmonizer_stage9.yaml")


def test_stage9_config_exists():
    assert STAGE9_CFG.exists(), "finetune_harmonizer_stage9.yaml must exist"


def test_stage9_config_is_valid_yaml():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "loss" in cfg
    assert "train" in cfg
    assert "scheduler" in cfg


def test_stage9_gain_limit_reduced():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    limits = cfg["model"]["correction_limits"]
    assert limits["gain_limit"] <= 1.20, (
        f"gain_limit must be ≤ 1.20 to prevent overcorrection; got {limits['gain_limit']}"
    )
    assert limits["gamma_limit"] <= 1.20


def test_stage9_loss_weights_rebalanced():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    w = cfg["loss"]["weights"]
    # seam must dominate over attn
    assert w["seam"] > w["attn"] * 5, (
        f"seam weight ({w['seam']}) should be >> attn ({w['attn']})"
    )
    # gain_reg must be substantially higher than old default (0.20 → now ≥ 0.50)
    assert w["gain_reg"] >= 0.50, f"gain_reg must be ≥ 0.50; got {w['gain_reg']}"
    # overcorrection must be substantially higher than old 0.22
    assert w["overcorr"] >= 0.50, f"overcorr weight must be ≥ 0.50; got {w['overcorr']}"
    # attn must be reduced from 0.20
    assert w["attn"] <= 0.10, f"attn weight must be ≤ 0.10; got {w['attn']}"


def test_stage9_lr_is_lower_than_v1():
    cfg_v1 = yaml.safe_load(Path("configs/finetune_harmonizer_v1.yaml").read_text(encoding="utf-8"))
    cfg_s9 = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    assert cfg_s9["train"]["lr"] < cfg_v1["train"]["lr"], (
        "stage9 LR must be lower than finetune_v1 to avoid oscillating at the plateau"
    )


def test_stage9_has_freeze_correction_epochs():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    assert cfg["train"].get("freeze_correction_epochs", 0) >= 1, (
        "stage9 must have freeze_correction_epochs ≥ 1 for phase-A stabilisation"
    )


def test_stage9_encoder_lr_scale_is_small():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    scale = cfg["train"].get("encoder_lr_scale", 1.0)
    assert scale <= 0.2, (
        f"encoder_lr_scale should be ≤ 0.2 (encoder is pre-trained); got {scale}"
    )


def test_stage9_min_lr_scale_lower_than_v1():
    cfg_v1 = yaml.safe_load(Path("configs/finetune_harmonizer_v1.yaml").read_text(encoding="utf-8"))
    cfg_s9 = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    s9_min = cfg_s9["scheduler"].get("min_lr_scale", 1.0)
    v1_min = cfg_v1["scheduler"].get("min_lr_scale", 1.0)
    assert s9_min < v1_min, (
        f"stage9 min_lr_scale ({s9_min}) must be lower than v1 ({v1_min}) for deeper cosine descent"
    )


# ---------------------------------------------------------------------------
# Block 2: checkpoint selection in train_harmonizer
# ---------------------------------------------------------------------------

def test_train_harmonizer_exposes_best_de_and_mae_tracking():
    """train_harmonizer must initialise best_de and best_mae and write their checkpoints."""
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert "best_de = float" in src, "best_de tracking must be initialised"
    assert "best_mae = float" in src, "best_mae tracking must be initialised"
    assert "best_harmonizer_de.pt" in src, "checkpoint best_harmonizer_de.pt must be saved"
    assert "best_harmonizer_mae.pt" in src, "checkpoint best_harmonizer_mae.pt must be saved"


def test_train_harmonizer_restores_de_and_mae_on_resume():
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert "boundary_ciede2000_16" in src, (
        "best_de must be initialised from checkpoint metrics on resume"
    )
    assert "boundary_mae_16" in src and "best_mae" in src, (
        "best_mae must be initialised from checkpoint metrics on resume"
    )


def test_notebook_primary_checkpoint_is_best_de():
    nb = Path("colab/seam_harmonizer_train_eval_colab.ipynb").read_text(encoding="utf-8")
    assert "best_harmonizer_de.pt" in nb, (
        "Notebook PRIMARY_CHECKPOINT should be best_harmonizer_de.pt (lowest ΔE@16)"
    )
    # Must not still default to quality (was wrong default)
    assert "PRIMARY_CHECKPOINT = 'best_harmonizer_quality.pt'" not in nb, (
        "Notebook must no longer default PRIMARY_CHECKPOINT to best_harmonizer_quality.pt"
    )


def test_notebook_train_config_is_stage9():
    nb = Path("colab/seam_harmonizer_train_eval_colab.ipynb").read_text(encoding="utf-8")
    assert "finetune_harmonizer_stage9.yaml" in nb, (
        "Notebook TRAIN_CONFIG_NAME should reference finetune_harmonizer_stage9.yaml"
    )


# ---------------------------------------------------------------------------
# Block 3: gain_reg default in analyze_tfevents
# ---------------------------------------------------------------------------

def test_analyze_gain_reg_default_is_nonzero():
    mod = _load_analyze()
    assert mod._DEFAULT_LOSS_WEIGHTS["gain_reg"] == pytest.approx(0.20), (
        "_DEFAULT_LOSS_WEIGHTS['gain_reg'] must be 0.20 to match HarmonizerLossComputer default"
    )


def test_analyze_gain_reg_default_matches_loss_computer():
    from src.losses.harmonizer_losses import HarmonizerLossComputer
    computer = HarmonizerLossComputer()
    mod = _load_analyze()
    assert computer.weights["gain_reg"] == pytest.approx(mod._DEFAULT_LOSS_WEIGHTS["gain_reg"]), (
        "analyze script gain_reg default must match HarmonizerLossComputer default"
    )


# ---------------------------------------------------------------------------
# Block 5: monotone growth detection in dashboard
# ---------------------------------------------------------------------------

def _make_epoch_rows(values: list[float], key: str = "gain_abs_log_mean") -> list[dict]:
    return [{"epoch_idx": i + 1, key: v} for i, v in enumerate(values)]


def test_detect_monotone_growth_returns_none_for_flat():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.10, 0.10, 0.10, 0.10])
    assert mod._detect_monotone_growth(rows, "gain_abs_log_mean") is None


def test_detect_monotone_growth_returns_none_for_decreasing():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.20, 0.18, 0.15, 0.12])
    assert mod._detect_monotone_growth(rows, "gain_abs_log_mean") is None


def test_detect_monotone_growth_returns_none_for_short_streak():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.10, 0.12, 0.15])  # streak=3 exactly
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=4)
    assert result is None


def test_detect_monotone_growth_detects_3_epoch_streak():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.10, 0.15, 0.20, 0.25])
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert result is not None
    assert result["streak"] >= 3
    assert result["start_val"] == pytest.approx(0.10)
    assert result["end_val"] == pytest.approx(0.25)


def test_detect_monotone_growth_detects_longest_streak():
    mod = _load_dashboard()
    # ep1:0.07, ep2:0.09 (grow 2), ep3:0.06 (dip, reset), ep4→ep7:0.10,0.15,0.22,0.30 (grow 5)
    rows = _make_epoch_rows([0.07, 0.09, 0.06, 0.10, 0.15, 0.22, 0.30])
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert result is not None
    assert result["streak"] == 5  # ep3→ep7: values 0.06→0.10→0.15→0.22→0.30
    assert result["start_val"] == pytest.approx(0.06)
    assert result["end_val"] == pytest.approx(0.30)


def test_detect_monotone_growth_returns_streak_metadata():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.076, 0.112, 0.145, 0.170, 0.190, 0.202, 0.212, 0.219, 0.224, 0.227])
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert result is not None
    assert "streak" in result
    assert "start_ep" in result
    assert "end_ep" in result
    assert "start_val" in result
    assert "end_val" in result
    assert result["streak"] == 10  # all 10 epochs grow


def test_detect_monotone_growth_handles_missing_key():
    mod = _load_dashboard()
    rows = [{"epoch_idx": 1}, {"epoch_idx": 2}, {"epoch_idx": 3}]
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert result is None


def test_detect_monotone_growth_handles_nan():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.10, float("nan"), 0.20, 0.30, 0.40])
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    # nan epoch is skipped; remaining 0.10, 0.20, 0.30, 0.40 form a 4-streak
    assert result is not None
    assert result["streak"] >= 3


def test_detect_monotone_growth_too_few_epochs():
    mod = _load_dashboard()
    rows = _make_epoch_rows([0.10, 0.20])
    result = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert result is None


def test_dashboard_insight_contains_gain_warning_for_real_log_data():
    """Integration: actual tfevents data must trigger the gain-growing alert."""
    mod = _load_dashboard()
    # Simulate the actual gain_abs_log_mean trend from the last training run (ep1→ep10 all grow)
    rows = [
        {"epoch_idx": i + 1, "gain_abs_log_mean": v, "confidence_alignment_mae": ca}
        for i, (v, ca) in enumerate([
            (0.076, 0.271), (0.112, 0.295), (0.145, 0.315),
            (0.170, 0.330), (0.189, 0.338), (0.202, 0.343),
            (0.212, 0.346), (0.219, 0.348), (0.224, 0.350), (0.227, 0.351),
        ])
    ]
    gain_trend = mod._detect_monotone_growth(rows, "gain_abs_log_mean", min_streak=3)
    assert gain_trend is not None, "Must detect gain growing across all 10 epochs"
    assert gain_trend["streak"] == 10

    conf_trend = mod._detect_monotone_growth(rows, "confidence_alignment_mae", min_streak=3)
    assert conf_trend is not None, "Must detect conf_align growing across all 10 epochs"
    assert conf_trend["streak"] == 10


def test_collect_best_rows_still_works_after_refactor():
    """Regression: _collect_best_rows must still find optima correctly."""
    mod = _load_dashboard()
    rows = [
        {
            "epoch_idx": 1,
            "boundary_mae_16": 0.020,
            "baseline_boundary_mae_16": 0.060,
            "boundary_ciede2000_16": 2.8,
            "baseline_boundary_ciede2000_16": 6.0,
        },
        {
            "epoch_idx": 2,
            "boundary_mae_16": 0.018,
            "baseline_boundary_mae_16": 0.060,
            "boundary_ciede2000_16": 2.5,
            "baseline_boundary_ciede2000_16": 6.0,
        },
    ]
    best = mod._collect_best_rows(rows)
    assert best["mae"]["epoch_idx"] == 2
    assert best["de"]["epoch_idx"] == 2
