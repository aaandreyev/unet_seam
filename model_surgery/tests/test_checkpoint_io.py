"""Tests for checkpoint_io: merge, transplant, gate_bias, memory safety."""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import pytest
import torch

from model_surgery.lib.checkpoint_io import (
    HEAD_SLICES,
    apply_gate_bias_to_state,
    clone_state,
    free_memory,
    linear_merge,
    load_ema,
    save_surgery_checkpoint,
    selective_head_merge,
    slerp_merge,
    transplant_head,
)
from src.models.harmonizer import SeamHarmonizerV3


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model_a():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    for p in m.parameters():
        torch.nn.init.normal_(p, mean=0.1, std=0.05)
    return m


@pytest.fixture(scope="module")
def tiny_model_b():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    for p in m.parameters():
        torch.nn.init.normal_(p, mean=-0.1, std=0.05)
    return m


@pytest.fixture(scope="module")
def state_a(tiny_model_a):
    return tiny_model_a.state_dict()


@pytest.fixture(scope="module")
def state_b(tiny_model_b):
    return tiny_model_b.state_dict()


@pytest.fixture(scope="module")
def saved_ckpt(tiny_model_a, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ckpts")
    meta = {"config": {"model": {"architecture": "seam_harmonizer_v3",
                                  "in_channels": 9, "channels": [8, 12, 16, 24],
                                  "blocks": [1, 1, 1, 1], "correction_limits": {"gate_bias": -0.10}},
                        "dataset": {"outer_width": 128, "boundary_band_px": 24}},
            "epoch": 5, "metrics": {"boundary_mae_16": 0.02}}
    path = tmp / "test.pt"
    save_surgery_checkpoint(tiny_model_a.state_dict(), meta, path)
    return path, meta


# ── HEAD_SLICES ───────────────────────────────────────────────────────────────

def test_head_slices_cover_18_channels():
    total = sum(s.stop - s.start for s in HEAD_SLICES.values())
    assert total == 18, f"expected 18 channels total, got {total}"


def test_head_slices_no_overlap():
    ranges = [(s.start, s.stop) for s in HEAD_SLICES.values()]
    for i, (a0, a1) in enumerate(ranges):
        for j, (b0, b1) in enumerate(ranges):
            if i == j:
                continue
            assert not (a0 < b1 and b0 < a1), f"heads overlap at indices {a0}:{a1} vs {b0}:{b1}"


def test_head_slices_keys():
    assert set(HEAD_SLICES.keys()) == {"gain", "gamma", "bias", "mix", "detail", "gate"}


# ── clone_state ───────────────────────────────────────────────────────────────

def test_clone_state_deep_copy(state_a):
    cloned = clone_state(state_a)
    for k in cloned:
        assert cloned[k] is not state_a[k]
        assert torch.equal(cloned[k], state_a[k])


def test_clone_state_modification_does_not_affect_original(state_a):
    cloned = clone_state(state_a)
    key = next(iter(cloned))
    cloned[key].fill_(999.0)
    assert not torch.equal(cloned[key], state_a[key])


# ── linear_merge ──────────────────────────────────────────────────────────────

def test_linear_merge_alpha0_equals_a(state_a, state_b):
    merged = linear_merge(state_a, state_b, alpha=0.0)
    for k in state_a:
        assert torch.allclose(merged[k], state_a[k], atol=1e-6)


def test_linear_merge_alpha1_equals_b(state_a, state_b):
    merged = linear_merge(state_a, state_b, alpha=1.0)
    for k in state_b:
        assert torch.allclose(merged[k], state_b[k], atol=1e-6)


def test_linear_merge_alpha_half_is_midpoint(state_a, state_b):
    merged = linear_merge(state_a, state_b, alpha=0.5)
    for k in state_a:
        expected = 0.5 * state_a[k] + 0.5 * state_b[k]
        assert torch.allclose(merged[k], expected, atol=1e-5)


def test_linear_merge_does_not_mutate_inputs(state_a, state_b):
    snap_a = {k: v.clone() for k, v in state_a.items()}
    linear_merge(state_a, state_b, alpha=0.3)
    for k in snap_a:
        assert torch.equal(snap_a[k], state_a[k])


# ── slerp_merge ───────────────────────────────────────────────────────────────

def test_slerp_merge_alpha0_near_a(state_a, state_b):
    merged = slerp_merge(state_a, state_b, alpha=0.0)
    for k in state_a:
        assert torch.allclose(merged[k].float(), state_a[k].float(), atol=1e-4)


def test_slerp_merge_alpha1_near_b(state_a, state_b):
    merged = slerp_merge(state_a, state_b, alpha=1.0)
    for k in state_b:
        assert torch.allclose(merged[k].float(), state_b[k].float(), atol=1e-4)


def test_slerp_merge_output_dtype_preserved(state_a, state_b):
    merged = slerp_merge(state_a, state_b, alpha=0.5)
    for k in state_a:
        assert merged[k].dtype == state_a[k].dtype


# ── transplant_head ───────────────────────────────────────────────────────────

def test_transplant_head_replaces_only_target_channels(state_a, state_b):
    for head in HEAD_SLICES:
        result = transplant_head(state_a, state_b, head)
        sl = HEAD_SLICES[head]
        # Transplanted channels match donor
        assert torch.equal(result["coarse_head.2.weight"][sl],
                           state_b["coarse_head.2.weight"][sl]), f"head {head} weight not transplanted"
        assert torch.equal(result["coarse_head.2.bias"][sl],
                           state_b["coarse_head.2.bias"][sl]), f"head {head} bias not transplanted"
        # Other channels still match base
        for other, other_sl in HEAD_SLICES.items():
            if other == head:
                continue
            assert torch.equal(result["coarse_head.2.weight"][other_sl],
                               state_a["coarse_head.2.weight"][other_sl])


def test_transplant_head_does_not_mutate_base(state_a, state_b):
    snap = clone_state(state_a)
    transplant_head(state_a, state_b, "gain")
    assert torch.equal(state_a["coarse_head.2.weight"], snap["coarse_head.2.weight"])


def test_transplant_head_non_coarse_layers_unchanged(state_a, state_b):
    result = transplant_head(state_a, state_b, "gain")
    for k in state_a:
        if "coarse_head.2" not in k:
            assert torch.equal(result[k], state_a[k]), f"{k} should be unchanged"


# ── selective_head_merge ──────────────────────────────────────────────────────

def test_selective_head_merge_alpha0_equals_base(state_a, state_b):
    result = selective_head_merge(state_a, state_b, "gain", alpha=0.0)
    sl = HEAD_SLICES["gain"]
    assert torch.allclose(result["coarse_head.2.weight"][sl],
                          state_a["coarse_head.2.weight"][sl], atol=1e-6)


def test_selective_head_merge_alpha1_equals_transplant(state_a, state_b):
    result_blend = selective_head_merge(state_a, state_b, "gate", alpha=1.0)
    result_transplant = transplant_head(state_a, state_b, "gate")
    sl = HEAD_SLICES["gate"]
    assert torch.allclose(result_blend["coarse_head.2.weight"][sl],
                          result_transplant["coarse_head.2.weight"][sl], atol=1e-5)


def test_selective_head_merge_non_head_layers_unchanged(state_a, state_b):
    result = selective_head_merge(state_a, state_b, "detail", alpha=0.5)
    for k in state_a:
        if "coarse_head.2" not in k:
            assert torch.equal(result[k], state_a[k])


# ── apply_gate_bias_to_state ─────────────────────────────────────────────────

def test_gate_bias_shifts_gate_channel_only(state_a):
    delta = -0.5
    result = apply_gate_bias_to_state(state_a, delta)
    sl = HEAD_SLICES["gate"]
    expected = state_a["coarse_head.2.bias"][sl] + delta
    assert torch.allclose(result["coarse_head.2.bias"][sl], expected, atol=1e-6)
    # Other channels unchanged
    for head, h_sl in HEAD_SLICES.items():
        if head == "gate":
            continue
        assert torch.equal(result["coarse_head.2.bias"][h_sl],
                           state_a["coarse_head.2.bias"][h_sl])


def test_gate_bias_does_not_affect_weights(state_a):
    result = apply_gate_bias_to_state(state_a, -0.3)
    assert torch.equal(result["coarse_head.2.weight"], state_a["coarse_head.2.weight"])


def test_gate_bias_zero_delta_identical(state_a):
    result = apply_gate_bias_to_state(state_a, 0.0)
    for k in state_a:
        assert torch.equal(result[k], state_a[k])


# ── save / load_ema ───────────────────────────────────────────────────────────

def test_save_load_roundtrip(saved_ckpt, state_a):
    path, meta = saved_ckpt
    loaded_state, loaded_meta = load_ema(path)
    for k in state_a:
        assert k in loaded_state
        assert torch.allclose(loaded_state[k], state_a[k], atol=1e-6)
    assert loaded_meta["epoch"] == 5


def test_load_ema_returns_only_weights_no_optimizer(saved_ckpt):
    path, _ = saved_ckpt
    loaded_state, _ = load_ema(path)
    for k in loaded_state:
        assert isinstance(loaded_state[k], torch.Tensor)
        assert not k.startswith("optimizer")


def test_save_surgery_checkpoint_creates_file(tmp_path, state_a):
    meta = {"config": {}, "epoch": 1, "metrics": {}}
    out = tmp_path / "out.pt"
    save_surgery_checkpoint(state_a, meta, out)
    assert out.exists()
    # tiny model (8-12-16-24 channels) is small; just verify it's a non-trivial file
    assert out.stat().st_size > 10_000


def test_load_ema_missing_file_raises():
    with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
        load_ema(Path("/nonexistent/path.pt"))


# ── free_memory ───────────────────────────────────────────────────────────────

def test_free_memory_runs_without_error():
    free_memory()  # Should not raise regardless of device state


def test_free_memory_gc_collects():
    # Create a tensor, delete it, call free_memory — no error
    t = torch.randn(100, 100)
    del t
    free_memory()
