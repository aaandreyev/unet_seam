from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    path = Path("scripts/search_seam_fusion_recipes.py")
    spec = importlib.util.spec_from_file_location("search_seam_fusion_recipes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_candidate_grid_has_large_search_space():
    module = _load_module()
    candidates = module._candidate_grid()
    assert len(candidates) >= 200
    assert {c["family"] for c in candidates} == {"residual_mlcv", "residual_inputcv", "local_rgb_mix"}


def test_apply_candidate_preserves_shape():
    module = _load_module()
    input_rgb = torch.zeros(2, 3, 32, 256)
    ml_rgb = torch.full_like(input_rgb, 0.4)
    cv_rgb = torch.full_like(input_rgb, 0.2)
    mask = torch.zeros(2, 1, 32, 256)
    mask[:, :, :, 128:] = 1.0
    candidate = {"family": "residual_mlcv", "corridor": 8.0, "power": 1.0, "sigma": 4.0, "scale": 0.5}
    fused = module._apply_candidate(candidate, input_rgb, ml_rgb, cv_rgb, mask)
    assert fused.shape == ml_rgb.shape
    assert torch.allclose(fused[:, :, :, :128], torch.full_like(fused[:, :, :, :128], 0.4), atol=1e-4)


def test_scores_prefer_lower_errors():
    module = _load_module()
    better = {
        "boundary_mae_16": 0.01,
        "lowfreq_mae": 0.01,
        "delta_luma_profile_mae": 0.01,
        "overcorrection_mae": 0.01,
        "boundary_ciede2000_16": 1.0,
    }
    worse = {
        "boundary_mae_16": 0.02,
        "lowfreq_mae": 0.02,
        "delta_luma_profile_mae": 0.02,
        "overcorrection_mae": 0.02,
        "boundary_ciede2000_16": 2.0,
    }
    assert module._coarse_score(better) < module._coarse_score(worse)
    assert module._full_score(better) < module._full_score(worse)
