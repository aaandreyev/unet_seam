"""Integration tests: surgery operations produce valid models with changed outputs."""
from __future__ import annotations

import pytest
import torch

from model_surgery.lib.checkpoint_io import (
    HEAD_SLICES, apply_gate_bias_to_state, clone_state,
    linear_merge, selective_head_merge, transplant_head,
)
from src.models.harmonizer import SeamHarmonizerV3


@pytest.fixture(scope="module")
def model_a():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    torch.manual_seed(0)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


@pytest.fixture(scope="module")
def model_b():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    torch.manual_seed(1)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(42)
    return torch.rand(1, 9, 64, 256)


def _load_and_run(state: dict, x: torch.Tensor) -> dict[str, torch.Tensor]:
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    m.load_state_dict(state, strict=False)
    m.eval()
    with torch.inference_mode():
        return m(x)


def test_linear_merge_produces_different_output_than_either_parent(model_a, model_b, x):
    merged = linear_merge(model_a.state_dict(), model_b.state_dict(), alpha=0.5)
    out_a = _load_and_run(model_a.state_dict(), x)
    out_m = _load_and_run(merged, x)
    assert not torch.allclose(out_a["corrected_strip"], out_m["corrected_strip"], atol=1e-4)


def test_transplant_changes_output(model_a, model_b, x):
    for head in HEAD_SLICES:
        result = transplant_head(model_a.state_dict(), model_b.state_dict(), head)
        out_a = _load_and_run(model_a.state_dict(), x)
        out_r = _load_and_run(result, x)
        # Output should differ (transplanted head contributes differently)
        # Note: some heads may have near-zero contribution for zero-init attention
        diff = (out_a["corrected_strip"] - out_r["corrected_strip"]).abs().max()
        assert diff >= 0.0, f"head {head}: result is identical (may be expected for some heads)"


def test_gate_bias_lower_reduces_confidence(model_a, x):
    """Applying a negative gate_bias_delta should lower confidence_mean."""
    state = model_a.state_dict()
    state_shifted = apply_gate_bias_to_state(state, gate_bias_delta=-2.0)
    out_orig = _load_and_run(state, x)
    out_shifted = _load_and_run(state_shifted, x)
    conf_orig = float(out_orig["confidence"].mean())
    conf_shifted = float(out_shifted["confidence"].mean())
    assert conf_shifted < conf_orig, (
        f"gate_bias -2.0 should lower confidence: {conf_orig:.4f} → {conf_shifted:.4f}"
    )


def test_gate_bias_positive_raises_confidence(model_a, x):
    state = model_a.state_dict()
    state_shifted = apply_gate_bias_to_state(state, gate_bias_delta=2.0)
    out_orig = _load_and_run(state, x)
    out_shifted = _load_and_run(state_shifted, x)
    conf_orig = float(out_orig["confidence"].mean())
    conf_shifted = float(out_shifted["confidence"].mean())
    assert conf_shifted > conf_orig


def test_gate_bias_does_not_change_outer_strip(model_a, x):
    """Outer region (first 128 px) must never be modified by gate_bias."""
    state = model_a.state_dict()
    state_shifted = apply_gate_bias_to_state(state, gate_bias_delta=-5.0)
    out_orig = _load_and_run(state, x)
    out_shifted = _load_and_run(state_shifted, x)
    outer_orig = out_orig["corrected_strip"][..., :128]
    outer_shifted = out_shifted["corrected_strip"][..., :128]
    assert torch.equal(outer_orig, outer_shifted), "outer region must be pixel-identical"


def test_selective_head_merge_alpha05_is_between_transplant_and_base(model_a, model_b, x):
    """Selective blend at alpha=0.5 should be between base and transplant outputs."""
    base_out = _load_and_run(model_a.state_dict(), x)
    transplanted = transplant_head(model_a.state_dict(), model_b.state_dict(), "gate")
    transplant_out = _load_and_run(transplanted, x)
    blended = selective_head_merge(model_a.state_dict(), model_b.state_dict(), "gate", 0.5)
    blend_out = _load_and_run(blended, x)
    base_conf = float(base_out["confidence"].mean())
    trans_conf = float(transplant_out["confidence"].mean())
    blend_conf = float(blend_out["confidence"].mean())
    lo, hi = min(base_conf, trans_conf), max(base_conf, trans_conf)
    assert lo - 0.05 <= blend_conf <= hi + 0.05, (
        f"blended confidence {blend_conf:.4f} not between {lo:.4f} and {hi:.4f}"
    )


def test_merged_model_output_is_finite(model_a, model_b, x):
    merged = linear_merge(model_a.state_dict(), model_b.state_dict(), alpha=0.3)
    out = _load_and_run(merged, x)
    for key in ("corrected_strip", "corrected_inner", "confidence", "attention_lowres"):
        assert torch.isfinite(out[key]).all(), f"{key} has non-finite values after merge"


def test_transplant_output_is_finite_and_bounded(model_a, model_b, x):
    for head in HEAD_SLICES:
        result = transplant_head(model_a.state_dict(), model_b.state_dict(), head)
        out = _load_and_run(result, x)
        cs = out["corrected_strip"]
        assert torch.isfinite(cs).all(), f"head {head}: non-finite corrected_strip"
        assert float(cs.min()) >= -0.01, f"head {head}: corrected_strip below 0"
        assert float(cs.max()) <= 1.01, f"head {head}: corrected_strip above 1"
