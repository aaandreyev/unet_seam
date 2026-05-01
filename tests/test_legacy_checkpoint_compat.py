"""Smoke tests that legacy state_dicts (without attention_head) load through every
inference / eval / export entrypoint via strict=False without raising."""
from __future__ import annotations

from pathlib import Path

import torch

from comfy_node.model_loader import _filter_matching_state_dict
from src.models.harmonizer import SeamHarmonizerV3


def _legacy_state_dict() -> dict[str, torch.Tensor]:
    """Build a state_dict that mimics a pre-attention checkpoint."""
    src = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    return {k: v.clone() for k, v in src.state_dict().items() if not k.startswith("attention_head")}


def test_load_legacy_into_fresh_model_strict_false_passes():
    model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    result = model.load_state_dict(_legacy_state_dict(), strict=False)
    assert result.missing_keys, "expected attention_head to appear in missing_keys"
    assert all(k.startswith("attention_head") for k in result.missing_keys)
    assert result.unexpected_keys == []


def test_load_new_into_fresh_model_strict_true_passes():
    """Once attention_head is part of the architecture, fresh-saved state_dicts
    must round-trip with strict=True (no missing/unexpected keys)."""
    src = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    state = src.state_dict()
    dst = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    dst.load_state_dict(state, strict=True)


def test_inference_scripts_use_strict_false():
    """Guard against accidental regression: scan the 5 known load points and ensure
    each call to load_state_dict on a SeamHarmonizerV3 uses strict=False."""
    project_root = Path(__file__).resolve().parents[1]
    targets = [
        project_root / "scripts" / "run_eval_harmonizer.py",
        project_root / "scripts" / "export_harmonizer_safetensors.py",
        project_root / "scripts" / "verify_harmonizer_export.py",
        project_root / "scripts" / "search_seam_fusion_recipes.py",
        project_root / "comfy_node" / "model_loader.py",
    ]
    failures = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if "load_state_dict" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "load_state_dict(" in line and "strict=False" not in line and "strict=True" not in line:
                # the following line might continue the call — allow multi-line by joining a window
                window = "\n".join(text.splitlines()[lineno - 1:lineno + 4])
                if "strict=False" not in window:
                    failures.append(f"{path.name}:{lineno}: {line.strip()}")
    assert failures == [], "load_state_dict without strict=False:\n" + "\n".join(failures)


def test_model_loader_filters_future_extra_heads_nonfatally():
    model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    state = model.state_dict()
    future_state = dict(state)
    future_state["future_head.weight"] = torch.zeros(1)
    future_state["attention_head.0.weight"] = state["attention_head.0.weight"].clone()
    matched, dropped_unexpected, dropped_mismatch = _filter_matching_state_dict(model, future_state)
    assert "future_head.weight" in dropped_unexpected
    assert "attention_head.0.weight" in matched
    result = model.load_state_dict(matched, strict=False)
    assert result.unexpected_keys == []
