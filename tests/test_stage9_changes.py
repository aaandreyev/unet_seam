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
    # gain_reg must be higher than old 0.20 but not catastrophically high.
    # At 0.80 it becomes 75% of total loss (raw gain_reg ≈ 0.357 from tfevents).
    # At 0.40 it becomes ~55%, still dominant but allows seam correction to compete.
    assert 0.30 <= w["gain_reg"] <= 0.60, (
        f"gain_reg must be in [0.30, 0.60] — too low doesn't fix growth, "
        f"too high (≥0.60) dominates loss and kills seam correction; got {w['gain_reg']}"
    )
    # overcorrection must be substantially higher than old 0.22
    assert w["overcorr"] >= 0.50, f"overcorr weight must be ≥ 0.50; got {w['overcorr']}"
    # attn must be reduced from 0.20 but kept meaningful for phase-A freeze
    assert 0.10 <= w["attn"] <= 0.18, (
        f"attn weight must be in [0.10, 0.18] — too low weakens phase-A training; got {w['attn']}"
    )


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
    assert "PRIMARY_CHECKPOINT = 'best_harmonizer_de.pt'" in nb, (
        "Notebook PRIMARY_CHECKPOINT should be best_harmonizer_de.pt (lowest ΔE@16)"
    )
    # Must not still default to quality for PRIMARY_CHECKPOINT
    assert "PRIMARY_CHECKPOINT = 'best_harmonizer_quality.pt'" not in nb, (
        "Notebook must no longer default PRIMARY_CHECKPOINT to best_harmonizer_quality.pt"
    )


def test_notebook_load_weights_checkpoint_is_quality_for_old_runs():
    """LOAD_WEIGHTS_CHECKPOINT must be best_harmonizer_quality.pt.

    Old runs (pre-stage9) only produced best_harmonizer_quality.pt.
    Using best_harmonizer_de.pt here would FileNotFoundError on the first stage9 run.
    """
    nb = Path("colab/seam_harmonizer_train_eval_colab.ipynb").read_text(encoding="utf-8")
    assert "LOAD_WEIGHTS_CHECKPOINT = 'best_harmonizer_quality.pt'" in nb, (
        "LOAD_WEIGHTS_CHECKPOINT must be best_harmonizer_quality.pt for loading from pre-stage9 runs"
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


# ---------------------------------------------------------------------------
# Additional fixes: _quality, best tracking, metrics, scheduler, freeze_phase
# ---------------------------------------------------------------------------

def test_quality_function_excludes_conf_align():
    """_quality must not use confidence_alignment_mae (metric measures wrong output)."""
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    # The function body after the docstring should not use conf_align
    # Find _quality function body
    start = src.index("def _quality(")
    end = src.index("\ndef ", start + 1)
    fn_body = src[start:end]
    assert "conf_align" not in fn_body, (
        "_quality() must not use confidence_alignment_mae — the metric tracks gate vs fixed-0.08 "
        "target, while attn loss trains attention_lowres vs percentile-normalized target. "
        "The metric grows monotonically regardless of seam quality improvement."
    )


def test_train_harmonizer_stores_best_values_in_checkpoint_metrics():
    """last_harmonizer.pt must include best_de/best_mae/best_quality in metrics dict."""
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert '"best_quality": best_quality' in src or "'best_quality': best_quality" in src, (
        "metrics dict saved to last_harmonizer.pt must include best_quality for correct resume"
    )
    assert '"best_de": best_de' in src or "'best_de': best_de" in src, (
        "metrics dict saved to last_harmonizer.pt must include best_de for correct resume"
    )
    assert '"best_mae": best_mae' in src or "'best_mae': best_mae" in src, (
        "metrics dict saved to last_harmonizer.pt must include best_mae for correct resume"
    )


def test_train_harmonizer_restores_best_from_stored_keys_on_resume():
    """On resume, best_de/best_mae should prefer stored keys over last-epoch metrics."""
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert '_stored.get("best_de")' in src or "_stored.get('best_de')" in src, (
        "best_de must be restored from stored checkpoint key, not just last-epoch val metrics"
    )
    assert '_stored.get("best_mae")' in src or "_stored.get('best_mae')" in src, (
        "best_mae must be restored from stored checkpoint key, not just last-epoch val metrics"
    )


def test_train_harmonizer_freeze_phase_floor_is_zero():
    """freeze_phase_floor must be 0 (epoch-0-relative) not start_epoch.

    Using start_epoch triggers re-freeze of correction heads on every resume,
    wasting 2 epochs of training budget that was intended for the initial run only.
    """
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert "freeze_phase_floor = 0" in src, (
        "freeze_phase_floor must be 0 (absolute), not start_epoch. "
        "With start_epoch, any resume would re-freeze correction heads for 2 epochs."
    )
    assert "freeze_phase_floor = start_epoch" not in src, (
        "freeze_phase_floor = start_epoch is wrong: causes re-freeze on every resume"
    )


def test_train_harmonizer_scheduler_not_loaded_with_additional_epochs():
    """Scheduler state must not be loaded when --additional-epochs is specified.

    With additional_epochs, total_steps doubles. Loading the old state would place
    LR at mid-schedule (~52% of peak) instead of near-zero end-of-schedule — a ×10 LR jump.
    """
    src = Path("scripts/train_harmonizer.py").read_text(encoding="utf-8")
    assert "additional_epochs is None" in src, (
        "Scheduler state must only be restored when additional_epochs is None. "
        "When additional_epochs is set, total_steps changes and the old schedule state "
        "would cause a large LR jump (×10) at the resume point."
    )


def test_overcorrection_metric_uses_abs_before_blur():
    """overcorrection_mae must compute abs-then-blur to match the loss computation.

    Old code used blur-then-abs (sigma=7.0) which allows sign cancellation across channels
    and uses a different sigma than the loss (5.0). This misalignment means tuning the
    overcorr loss weight does not reliably improve the reported metric.
    """
    src = Path("src/metrics/harmonizer_metrics.py").read_text(encoding="utf-8")
    # The new code should have abs() BEFORE gaussian_blur_tensor for overcorrection
    assert "gaussian_blur_tensor(delta_pred.abs()" in src, (
        "overcorrection_mae must use abs-then-blur (matching the loss), not blur-then-abs"
    )
    assert "overcorr_pred_mag" in src, (
        "overcorrection_mae computation must use a separate abs-then-blur variable"
    )


def test_attention_alignment_mae_metric_added():
    """attention_alignment_mae must be computed against attention_lowres (the actual supervised output)."""
    src = Path("src/metrics/harmonizer_metrics.py").read_text(encoding="utf-8")
    assert "attention_alignment_mae" in src, (
        "attention_alignment_mae must be added to track the attention head quality "
        "using the same normalization as the attn loss (85th percentile)"
    )
    assert "attention_lowres" in src, (
        "attention_alignment_mae must compare attention_lowres, not confidence (gate)"
    )


def test_write_colab_runtime_yamls_default_checkpoint_is_best_de():
    src = Path("scripts/write_colab_runtime_yamls.py").read_text(encoding="utf-8")
    assert "best_harmonizer_de.pt" in src, (
        "write_colab_runtime_yamls.py --primary-checkpoint default must be best_harmonizer_de.pt"
    )


def test_stage9_no_dead_train_seed_key():
    cfg = yaml.safe_load(STAGE9_CFG.read_text(encoding="utf-8"))
    train_section = cfg.get("train", {})
    assert "seed" not in train_section, (
        "train.seed is dead config — train_harmonizer.py reads cfg.get('seed') (top-level), "
        "not train.seed. Remove to avoid confusion."
    )


def test_metrics_imports_F():
    """harmonizer_metrics.py must import torch.nn.functional for attention_alignment_mae."""
    src = Path("src/metrics/harmonizer_metrics.py").read_text(encoding="utf-8")
    assert "import torch.nn.functional as F" in src, (
        "harmonizer_metrics.py must import F for F.interpolate used in attention_alignment_mae"
    )
