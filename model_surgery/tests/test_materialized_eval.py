"""Tests for materialized triplet eval path (synthetic_triplets_500)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_surgery.lib.eval_mini import build_loader, run_eval
from src.models.harmonizer import SeamHarmonizerV3

TRIPLETS_DIR = Path("outputs/synthetic_triplets_500")
MANIFEST = Path("manifests/input_raw_manifest.jsonl")


@pytest.mark.skipif(not (TRIPLETS_DIR / "manifest.jsonl").exists(),
                    reason="synthetic_triplets_500 not found — run export first")
class TestMaterializedLoader:

    def test_loader_returns_batches(self):
        loader = build_loader(
            manifest=MANIFEST, n_strips=10,
            outer_width=128, inner_width=128, strip_height=1024,
            boundary_band_px=24, batch_size=2, seed=42,
            materialized_dir=TRIPLETS_DIR,
        )
        batch = next(iter(loader))
        assert "input" in batch
        assert "target" in batch
        assert batch["input"].shape[1] == 9  # 9-channel input

    def test_loader_respects_n_strips(self):
        loader = build_loader(
            manifest=MANIFEST, n_strips=16,
            outer_width=128, inner_width=128, strip_height=1024,
            boundary_band_px=24, batch_size=4, seed=42,
            materialized_dir=TRIPLETS_DIR,
        )
        total = sum(b["input"].shape[0] for b in loader)
        assert total == 16

    def test_loader_reproducible_with_same_seed(self):
        kwargs = dict(manifest=MANIFEST, n_strips=8, outer_width=128, inner_width=128,
                      strip_height=1024, boundary_band_px=24, batch_size=2, seed=99,
                      materialized_dir=TRIPLETS_DIR)
        loader_a = build_loader(**kwargs)
        loader_b = build_loader(**kwargs)
        for ba, bb in zip(loader_a, loader_b):
            assert torch.equal(ba["input"], bb["input"])

    def test_loader_different_seeds_give_different_batches(self):
        kwargs = dict(manifest=MANIFEST, n_strips=20, outer_width=128, inner_width=128,
                      strip_height=1024, boundary_band_px=24, batch_size=4,
                      materialized_dir=TRIPLETS_DIR)
        loader_a = build_loader(**kwargs, seed=1)
        loader_b = build_loader(**kwargs, seed=2)
        batch_a = next(iter(loader_a))["input"]
        batch_b = next(iter(loader_b))["input"]
        assert not torch.equal(batch_a, batch_b)

    def test_full_eval_run_on_materialized(self):
        model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
        state = model.state_dict()
        meta = {"config": {"model": {"in_channels": 9, "channels": [8, 12, 16, 24],
                                      "blocks": [1, 1, 1, 1]},
                            "dataset": {"outer_width": 128, "boundary_band_px": 24}},
                "epoch": 0, "metrics": {}}
        loader = build_loader(
            manifest=MANIFEST, n_strips=8, outer_width=128, inner_width=128,
            strip_height=1024, boundary_band_px=24, batch_size=2, seed=42,
            materialized_dir=TRIPLETS_DIR,
        )
        metrics = run_eval(state, meta, loader, outer_width=128)
        assert "boundary_mae_16" in metrics
        assert "boundary_ciede2000_16" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_materialized_loader_faster_than_synthetic(self):
        """Materialized loader should build without expensive on-the-fly corruption."""
        import time
        t0 = time.monotonic()
        loader = build_loader(
            manifest=MANIFEST, n_strips=20, outer_width=128, inner_width=128,
            strip_height=1024, boundary_band_px=24, batch_size=4, seed=42,
            materialized_dir=TRIPLETS_DIR,
        )
        # Just iterate to verify no crash and measure
        count = sum(1 for _ in loader)
        elapsed = time.monotonic() - t0
        assert count > 0
        # Should complete quickly (no corruption generation)
        assert elapsed < 60.0, f"Materialized loader too slow: {elapsed:.1f}s"
