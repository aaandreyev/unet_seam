"""Tests for the phase-A freeze of correction heads and encoder LR scaling."""
from __future__ import annotations

import torch

from scripts.train_harmonizer import (
    apply_phase_a_freeze,
    build_param_groups,
)
from src.models.harmonizer import SeamHarmonizerV3


CORRECTION_PREFIXES = ("coarse_head", "coarse_adapter")


def _build_tiny_model() -> SeamHarmonizerV3:
    return SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))


def test_freeze_correction_when_active_disables_correction_grads():
    model = _build_tiny_model()
    apply_phase_a_freeze(model, freeze=True)
    for name, p in model.named_parameters():
        if name.startswith(CORRECTION_PREFIXES):
            assert p.requires_grad is False, f"{name} should be frozen"
        else:
            assert p.requires_grad is True, f"{name} should be trainable"


def test_freeze_correction_when_inactive_unfreezes_everything():
    model = _build_tiny_model()
    apply_phase_a_freeze(model, freeze=True)
    apply_phase_a_freeze(model, freeze=False)
    for name, p in model.named_parameters():
        assert p.requires_grad is True, f"{name} should be trainable after unfreeze"


def test_attention_head_remains_trainable_during_phase_a():
    model = _build_tiny_model()
    apply_phase_a_freeze(model, freeze=True)
    attn_params = [p for n, p in model.named_parameters() if n.startswith("attention_head")]
    assert len(attn_params) > 0
    assert all(p.requires_grad for p in attn_params), "attention_head must train during phase A"


def test_param_groups_apply_encoder_lr_scale():
    model = _build_tiny_model()
    base_lr = 1e-4
    groups = build_param_groups(model, base_lr=base_lr, encoder_lr_scale=0.2)
    assert len(groups) == 2
    encoder_group = next(g for g in groups if g.get("name") == "encoder")
    other_group = next(g for g in groups if g.get("name") == "other")
    assert encoder_group["lr"] == base_lr * 0.2
    assert other_group["lr"] == base_lr
    enc_param_ids = {id(p) for p in encoder_group["params"]}
    other_param_ids = {id(p) for p in other_group["params"]}
    assert enc_param_ids and other_param_ids
    assert enc_param_ids.isdisjoint(other_param_ids)
    for n, p in model.named_parameters():
        target = encoder_group["params"] if n.startswith("encoder") else other_group["params"]
        assert any(id(q) == id(p) for q in target), f"{n} not assigned to expected group"


def test_param_groups_default_scale_is_uniform():
    model = _build_tiny_model()
    groups = build_param_groups(model, base_lr=2e-4, encoder_lr_scale=1.0)
    for g in groups:
        assert g["lr"] == 2e-4


def test_optimizer_skips_frozen_params():
    """Frozen params must not move under AdamW step."""
    model = _build_tiny_model()
    apply_phase_a_freeze(model, freeze=True)
    groups = build_param_groups(model, base_lr=1e-3, encoder_lr_scale=1.0)
    opt = torch.optim.AdamW(
        [{"params": [p for p in g["params"] if p.requires_grad], "lr": g["lr"], "name": g["name"]} for g in groups],
        weight_decay=1e-4,
    )
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(CORRECTION_PREFIXES)}
    x = torch.rand(1, 9, 128, 256)
    out = model(x)
    out["corrected_strip"].sum().backward()
    opt.step()
    for n, p in model.named_parameters():
        if n.startswith(CORRECTION_PREFIXES):
            assert torch.equal(snapshot[n], p), f"{n} moved despite freeze"
