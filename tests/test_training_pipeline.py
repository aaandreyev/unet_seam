from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.data.gpu_corruptions import GPUCorruption
from src.data.harmonizer_input import build_harmonizer_input
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch
from src.models.harmonizer import SeamHarmonizerV3

MANIFEST = Path("manifests/input_raw_manifest.jsonl")
OUTER_W = 128
INNER_W = 128
H = 1024
W = OUTER_W + INNER_W
SPEC = StripSpec(strip_height=H, outer_width=OUTER_W, inner_width=INNER_W, seam_jitter_px=0)


@pytest.fixture(scope="module")
def gpu_corruption():
    return GPUCorruption()


@pytest.fixture(scope="module")
def train_ds():
    return SyntheticStripDataset(
        MANIFEST,
        split="train",
        strips_per_image=2,
        seed=42,
        spec=SPEC,
        inner_widths=[INNER_W],
        apply_corruption=False,
    )


@pytest.fixture(scope="module")
def val_ds():
    return SyntheticStripDataset(
        MANIFEST,
        split="val",
        strips_per_image=1,
        seed=42,
        spec=SPEC,
        inner_widths=[INNER_W],
        apply_corruption=True,
    )


@pytest.fixture(scope="module")
def model():
    return SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))


def _corrupted_batch(batch: dict, gpu_corruption: GPUCorruption) -> dict:
    clean_strip = batch["input"][:, :3]
    inner = clean_strip[:, :, :, OUTER_W:]
    corrupted_inner = gpu_corruption(inner)
    corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
    return {**batch, **build_harmonizer_input(corrupted_strip, outer_width=OUTER_W, boundary_band_px=24)}


class TestDatasetIntegrity:
    def test_train_no_corruption_input_equals_target(self, train_ds):
        sample = train_ds[0]
        assert torch.allclose(sample["input"][:3], sample["target"], atol=1e-6)

    def test_val_corruption_changes_inner(self, val_ds):
        sample = val_ds[0]
        inner_input = sample["input"][:3, :, OUTER_W:]
        inner_target = sample["target"][:, :, OUTER_W:]
        assert not torch.allclose(inner_input, inner_target, atol=1e-5)

    def test_input_channels(self, train_ds):
        sample = train_ds[0]
        assert sample["input"].shape[0] == 9

    def test_input_shape(self, train_ds):
        sample = train_ds[0]
        assert sample["input"].shape == (9, H, W)


class TestGPUCorruptionPipeline:
    def test_gpu_corruption_rebuilds_v3_input(self, gpu_corruption, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        updated = _corrupted_batch(batch, gpu_corruption)
        assert updated["input"].shape == (2, 9, H, W)
        assert torch.allclose(updated["input"][:, :3, :, :OUTER_W], batch["target"][:, :, :, :OUTER_W], atol=1e-6)


class TestModelOutputShapes:
    def test_forward_output_keys(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        required_keys = {
            "gain_lowres",
            "gamma_lowres",
            "bias_lowres",
            "mix_lowres",
            "detail_lowres",
            "gate_lowres",
            "corrected_inner",
            "corrected_strip",
            "confidence",
            "gain",
            "detail",
            "color_matrix",
        }
        assert required_keys.issubset(out.keys())

    def test_corrected_ranges_and_outer_copy(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        assert out["corrected_inner"].min() >= 0.0
        assert out["corrected_inner"].max() <= 1.0
        assert torch.allclose(out["corrected_strip"][:, :, :, :OUTER_W], batch["input"][:, :3, :, :OUTER_W], atol=1e-5)


class TestLossAndMetrics:
    def test_all_loss_terms_present(self, model, train_ds, gpu_corruption):
        batch = _corrupted_batch(collate_strip_batch([train_ds[0], train_ds[1]]), gpu_corruption)
        with torch.no_grad():
            out = model(batch["input"])
        losses = HarmonizerLossComputer()(out, batch)
        expected = {"total", "l_rec", "l_seam", "l_low", "l_grad", "l_chroma", "l_stats", "l_lab", "l_gate", "l_field", "l_detail", "l_matrix"}
        assert expected.issubset(losses.keys())

    def test_metrics_finite(self, model, train_ds, gpu_corruption):
        batch = _corrupted_batch(collate_strip_batch([train_ds[0], train_ds[1]]), gpu_corruption)
        with torch.no_grad():
            out = model(batch["input"])
        metrics = evaluate_harmonizer_batch(out["corrected_strip"], batch["input_rgb"], batch["target"], out)
        for value in metrics.values():
            assert math.isfinite(value)
