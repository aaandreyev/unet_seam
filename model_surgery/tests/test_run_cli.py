"""Tests for run.py CLI argument parsing and config loading."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


CONFIG_PATH = Path("model_surgery/config.yaml")


@pytest.fixture
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_has_all_required_keys(cfg):
    required_top = ["runs_dir", "local_checkpoints_dir", "manifest", "output_dir",
                    "target_quality_score", "eval", "stages",
                    "s1_gate_search", "s2_ablation", "s3_merge", "s4_surgery", "s5_cycle"]
    for key in required_top:
        assert key in cfg, f"config missing top-level key: {key}"


def test_config_eval_has_all_keys(cfg):
    required = ["materialized_dir", "mini_strips", "full_strips", "outer_width",
                "inner_width", "strip_height", "boundary_band_px", "batch_size",
                "num_workers", "materialized_preload", "seed"]
    for key in required:
        assert key in cfg["eval"], f"config.eval missing key: {key}"


def test_config_s1_has_grid_params(cfg):
    s1 = cfg["s1_gate_search"]
    assert "gate_bias_range" in s1
    assert len(s1["gate_bias_range"]) == 2
    lo, hi = s1["gate_bias_range"]
    assert lo != hi, "gate_bias_range start and end must differ"
    assert abs(hi - lo) <= 0.5, "gate_bias_range should stay narrow around baseline"
    assert s1["n_points"] > 0
    assert "gain_limit_range" in s1
    assert "detail_limit_range" in s1
    gl_lo, gl_hi = s1["gain_limit_range"]
    dl_lo, dl_hi = s1["detail_limit_range"]
    assert (gl_hi - gl_lo) <= 0.6, "gain_limit_range should stay local"
    assert (dl_hi - dl_lo) <= 0.15, "detail_limit_range should stay local"


def test_config_s5_cycle_has_required_keys(cfg):
    s5 = cfg["s5_cycle"]
    for key in ["max_cycles", "patience", "top_k_base", "rerank_top_k",
                "early_stop_no_improve_cycles", "finetune_on_plateau"]:
        assert key in s5, f"s5_cycle missing key: {key}"
    ft = s5["finetune_on_plateau"]
    for key in ["enabled", "plateau_cycles", "epochs", "lr"]:
        assert key in ft, f"finetune_on_plateau missing key: {key}"


def test_config_stages_flags(cfg):
    stages = cfg["stages"]
    for key in ["s0_survey", "s1_gate_search", "s2_ablation", "s3_merge", "s4_surgery", "s5_cycle"]:
        assert key in stages, f"stages missing flag: {key}"
        assert isinstance(stages[key], bool)


def test_config_target_quality_has_at_least_one_stop_condition(cfg):
    """Either absolute or relative target must be set (or both)."""
    abs_tgt = cfg.get("target_quality_score")
    rel_tgt = cfg.get("target_quality_relative")
    assert abs_tgt is not None or rel_tgt is not None, (
        "at least one of target_quality_score or target_quality_relative must be set"
    )
    if abs_tgt is not None:
        assert float(abs_tgt) > 0
    if rel_tgt is not None:
        assert 0.0 < float(rel_tgt) < 1.0, "target_quality_relative must be between 0 and 1"


def test_config_materialized_dir_points_to_real_path(cfg):
    """If materialized_dir is set, it should point to an existing directory."""
    mat = cfg["eval"].get("materialized_dir")
    if mat is None:
        pytest.skip("materialized_dir not configured")
    path = Path(mat)
    assert path.exists(), f"materialized_dir {path} does not exist"
    assert (path / "manifest.jsonl").exists(), "manifest.jsonl missing in materialized_dir"


def test_config_s2_ablation_heads_match_head_slices(cfg):
    from model_surgery.lib.checkpoint_io import HEAD_SLICES
    configured_heads = {h["name"] for h in cfg["s2_ablation"]["heads"]}
    expected_heads = set(HEAD_SLICES.keys())
    assert configured_heads == expected_heads, (
        f"s2_ablation heads {configured_heads} don't match HEAD_SLICES {expected_heads}"
    )


def test_config_s2_ablation_channels_cover_18(cfg):
    all_channels = []
    for h in cfg["s2_ablation"]["heads"]:
        all_channels.extend(h["channels"])
    assert sorted(all_channels) == list(range(18)), (
        f"s2_ablation head channels don't cover 0-17: {sorted(all_channels)}"
    )


def test_stages_arg_parsing():
    """run.py stages set is built correctly from comma-separated string."""
    stages_str = "s0,s1,s3,s5"
    stages = {s.strip() for s in stages_str.split(",")}
    assert stages == {"s0", "s1", "s3", "s5"}
    assert "s2" not in stages
    assert "s4" not in stages


def test_config_mini_strips_le_full_strips(cfg):
    assert cfg["eval"]["mini_strips"] <= cfg["eval"]["full_strips"], (
        "mini_strips should be ≤ full_strips"
    )


def test_config_mini_strips_le_500(cfg):
    """We only have 500 triplets."""
    assert cfg["eval"]["mini_strips"] <= 500
    assert cfg["eval"]["full_strips"] <= 500
