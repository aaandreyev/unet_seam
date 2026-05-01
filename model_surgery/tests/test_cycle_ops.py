"""Tests for s5_cycle._build_operations — the core of the cyclic loop."""
from __future__ import annotations

import gc

import pytest
import torch

from model_surgery.lib.checkpoint_io import HEAD_SLICES, clone_state
from model_surgery.stages.s5_cycle import _build_operations
from src.models.harmonizer import SeamHarmonizerV3


@pytest.fixture(scope="module")
def base_state():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    torch.manual_seed(0)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.state_dict()


@pytest.fixture(scope="module")
def pool_state_1():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    torch.manual_seed(1)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.state_dict()


@pytest.fixture(scope="module")
def pool_state_2():
    m = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    torch.manual_seed(2)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.state_dict()


@pytest.fixture(scope="module")
def base_meta():
    return {"config": {"model": {"in_channels": 9, "channels": [8, 12, 16, 24],
                                  "blocks": [1, 1, 1, 1],
                                  "correction_limits": {"gate_bias": -0.10}},
                        "dataset": {"outer_width": 128, "boundary_band_px": 24}},
            "epoch": 0, "metrics": {}}


@pytest.fixture(scope="module")
def pool_entries(pool_state_1, pool_state_2, base_meta):
    return [
        (pool_state_1, base_meta, "pool_a"),
        (pool_state_2, base_meta, "pool_b"),
    ]


def test_build_operations_returns_list(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    assert isinstance(ops, list)
    assert len(ops) > 0


def test_build_operations_tuple_format(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    for op in ops:
        assert len(op) == 4, f"op must be 4-tuple (label, state, meta, owns_state), got {len(op)}"
        label, state, meta, owns_state = op
        assert isinstance(label, str)
        assert isinstance(state, dict)
        assert isinstance(meta, dict)
        assert isinstance(owns_state, bool)


def test_gate_ops_share_state_reference(base_state, base_meta, pool_entries):
    """Gate/limit variants must share base_state reference (owns_state=False, no clone)."""
    ops = _build_operations(base_state, base_meta, pool_entries)
    gate_ops = [op for op in ops if op[0].startswith("gate_")]
    assert len(gate_ops) > 0, "expected gate ops"
    for label, state, meta, owns_state in gate_ops:
        assert owns_state is False, f"gate op '{label}' must have owns_state=False"
        assert state is base_state, f"gate op '{label}' must share base_state reference"


def test_gate_ops_have_different_meta(base_state, base_meta, pool_entries):
    """Each gate op must have its own meta (different gate_bias value)."""
    ops = _build_operations(base_state, base_meta, pool_entries)
    gate_ops = [op for op in ops if op[0].startswith("gate_")]
    metas = [op[2] for op in gate_ops]
    gate_biases = [m["config"]["model"]["correction_limits"]["gate_bias"] for m in metas]
    assert len(set(gate_biases)) > 1, "gate ops should have different gate_bias values"


def test_gate_ops_do_not_mutate_base_meta(base_state, base_meta, pool_entries):
    """Building ops must not modify original base_meta."""
    original_gb = base_meta["config"]["model"]["correction_limits"]["gate_bias"]
    _build_operations(base_state, base_meta, pool_entries)
    assert base_meta["config"]["model"]["correction_limits"]["gate_bias"] == original_gb


def test_merge_ops_own_state(base_state, base_meta, pool_entries):
    """Linear merge ops must create new tensors (owns_state=True)."""
    ops = _build_operations(base_state, base_meta, pool_entries)
    merge_ops = [op for op in ops if "merge_linear" in op[0]]
    assert len(merge_ops) > 0
    for label, state, meta, owns_state in merge_ops:
        assert owns_state is True, f"merge op '{label}' must have owns_state=True"
        assert state is not base_state, "merge must create new state dict"


def test_transplant_ops_own_state(base_state, base_meta, pool_entries):
    """Transplant ops must create new tensor dicts."""
    ops = _build_operations(base_state, base_meta, pool_entries)
    transplant_ops = [op for op in ops if "transplant_" in op[0]]
    assert len(transplant_ops) > 0
    for label, state, meta, owns_state in transplant_ops:
        assert owns_state is True, f"transplant op '{label}' must have owns_state=True"


def test_head_blend_ops_own_state(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    blend_ops = [op for op in ops if "head_blend_" in op[0]]
    assert len(blend_ops) > 0
    for label, state, meta, owns_state in blend_ops:
        assert owns_state is True


def test_all_heads_covered_in_transplants(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    transplant_heads = {op[0].split("_")[1] for op in ops if op[0].startswith("transplant_")}
    assert transplant_heads == set(HEAD_SLICES.keys())


def test_all_heads_covered_in_blends(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    blend_heads = {op[0].split("_")[2] for op in ops if op[0].startswith("head_blend_")}
    assert blend_heads == set(HEAD_SLICES.keys())


def test_ops_states_are_finite(base_state, base_meta, pool_entries):
    ops = _build_operations(base_state, base_meta, pool_entries)
    # Check only owned states (shared refs are already tested elsewhere)
    for label, state, meta, owns_state in ops:
        if owns_state:
            for k, v in state.items():
                assert torch.isfinite(v).all(), f"op '{label}', key '{k}' has non-finite values"


def test_no_memory_leak_from_gate_ops(base_state, base_meta, pool_entries):
    """Gate ops share state by reference — creating 240 of them shouldn't allocate 240 state_dicts."""
    import tracemalloc
    tracemalloc.start()
    # Baseline
    snapshot1 = tracemalloc.take_snapshot()
    ops = _build_operations(base_state, base_meta, pool_entries)
    gate_ops = [op for op in ops if op[0].startswith("gate_")]
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    # Gate ops count should be substantial (at least 100)
    assert len(gate_ops) >= 100, f"expected many gate ops, got {len(gate_ops)}"
    # Memory delta should be small (metas only, no tensor clones)
    # Each meta is ~few KB; 240 metas ≈ 240KB. State_dict clone would be ~85MB each.
    stats = snapshot2.compare_to(snapshot1, "lineno")
    total_delta_mb = sum(s.size_diff for s in stats) / 1024 / 1024
    # Allow up to 20MB for metas + overhead (way less than 240×85MB=20GB)
    assert total_delta_mb < 20, (
        f"Gate ops allocated {total_delta_mb:.1f}MB — likely cloning state instead of sharing ref"
    )
