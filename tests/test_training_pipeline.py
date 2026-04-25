"""
End-to-end correctness tests for the full training pipeline.

Verifies: GPU corruption, dataset integrity, target/mask/distance,
outer-half preservation, baseline metrics, and shape contracts.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.data.gpu_corruptions import GPUCorruption
from src.data.strip_geometry import StripSpec
from src.data.synthetic_strip_dataset import SyntheticStripDataset, collate_strip_batch
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.metrics.harmonizer_metrics import evaluate_harmonizer_batch
from src.models.harmonizer import SeamHarmonizerV1

MANIFEST = Path("manifests/input_raw_manifest.jsonl")
OUTER_W = 128
INNER_W = 128
H = 1024
W = OUTER_W + INNER_W

SPEC = StripSpec(strip_height=H, outer_width=OUTER_W, inner_width=INNER_W, seam_jitter_px=0)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gpu_corruption():
    return GPUCorruption()


@pytest.fixture(scope="module")
def train_ds():
    return SyntheticStripDataset(MANIFEST, split="train", strips_per_image=2,
                                 seed=42, spec=SPEC, inner_widths=[INNER_W],
                                 apply_corruption=False)


@pytest.fixture(scope="module")
def val_ds():
    return SyntheticStripDataset(MANIFEST, split="val", strips_per_image=1,
                                 seed=42, spec=SPEC, inner_widths=[INNER_W],
                                 apply_corruption=True)


@pytest.fixture(scope="module")
def model():
    return SeamHarmonizerV1()


# ── GPU Corruption tests ─────────────────────────────────────────────────────

class TestGPUCorruption:
    def test_output_shape(self, gpu_corruption):
        x = torch.rand(4, 3, H, INNER_W)
        out = gpu_corruption(x)
        assert out.shape == x.shape

    def test_output_range(self, gpu_corruption):
        x = torch.rand(8, 3, H, INNER_W)
        out = gpu_corruption(x)
        assert out.min() >= 0.0, "corruption output has values below 0"
        assert out.max() <= 1.0, "corruption output has values above 1"

    def test_corruption_changes_input(self, gpu_corruption):
        """Corruption must actually change the inner half."""
        x = torch.full((4, 3, H, INNER_W), 0.5)
        out = gpu_corruption(x)
        assert not torch.allclose(out, x, atol=1e-3), "corruption had no effect"

    def test_no_nan_inf(self, gpu_corruption):
        x = torch.rand(4, 3, H, INNER_W)
        out = gpu_corruption(x)
        assert not torch.isnan(out).any(), "NaN in corruption output"
        assert not torch.isinf(out).any(), "Inf in corruption output"

    def test_no_nan_on_black_white_inputs(self, gpu_corruption):
        """Extreme pixel values (0 and 1) must not cause NaN."""
        for val in [0.0, 1.0]:
            x = torch.full((2, 3, 64, 32), val)
            out = gpu_corruption(x)
            assert not torch.isnan(out).any(), f"NaN on constant input={val}"

    def test_different_samples_get_different_corruption(self, gpu_corruption):
        """Each sample in the batch should get independently randomized corruption."""
        x = torch.full((8, 3, 64, 32), 0.5)
        out = gpu_corruption(x)
        # Not all samples should be identical
        assert not torch.allclose(out[0], out[1], atol=1e-4), \
            "all samples received identical corruption — RNG not per-sample"

    def test_reproducible_with_generator(self, gpu_corruption):
        x = torch.rand(2, 3, 64, 32)
        gen1 = torch.Generator().manual_seed(0)
        gen2 = torch.Generator().manual_seed(0)
        out1 = gpu_corruption(x.clone(), gen=gen1)
        out2 = gpu_corruption(x.clone(), gen=gen2)
        assert torch.allclose(out1, out2), "same generator seed must give identical output"


# ── Dataset integrity tests ───────────────────────────────────────────────────

class TestDatasetIntegrity:
    def test_train_no_corruption_input_equals_target(self, train_ds):
        """With apply_corruption=False, input RGB == target (clean strip)."""
        sample = train_ds[0]
        input_rgb = sample["input"][:3]   # first 3 channels = RGB
        target = sample["target"]
        assert torch.allclose(input_rgb, target, atol=1e-6), \
            "train dataset: input_rgb != target when apply_corruption=False"

    def test_val_corruption_changes_inner(self, val_ds):
        """Val dataset (apply_corruption=True) must corrupt the inner half."""
        sample = val_ds[0]
        input_rgb = sample["input"][:3]
        target = sample["target"]
        inner_input = input_rgb[:, :, OUTER_W:]
        inner_target = target[:, :, OUTER_W:]
        assert not torch.allclose(inner_input, inner_target, atol=1e-5), \
            "val corruption had no effect on inner half"

    def test_val_outer_half_unchanged(self, val_ds):
        """Val corruption must never touch the outer (original) half."""
        sample = val_ds[0]
        outer_input = sample["input"][:3, :, :OUTER_W]
        outer_target = sample["target"][:, :, :OUTER_W]
        assert torch.allclose(outer_input, outer_target, atol=1e-6), \
            "val corruption modified the outer half"

    def test_val_deterministic(self, val_ds):
        """Same index must always produce the same corrupted sample."""
        s1 = val_ds[0]
        s2 = val_ds[0]
        assert torch.allclose(s1["input"], s2["input"]), \
            "val dataset not deterministic — corruption varies between accesses"

    def test_input_channels(self, train_ds):
        """Input must have exactly 5 channels: RGB + mask + distance."""
        sample = train_ds[0]
        assert sample["input"].shape[0] == 5, \
            f"expected 5 input channels, got {sample['input'].shape[0]}"

    def test_input_shape(self, train_ds):
        sample = train_ds[0]
        assert sample["input"].shape == (5, H, W), \
            f"unexpected input shape: {sample['input'].shape}"

    def test_target_shape(self, train_ds):
        sample = train_ds[0]
        assert sample["target"].shape == (3, H, W), \
            f"unexpected target shape: {sample['target'].shape}"

    def test_mask_binary(self, train_ds):
        """inner_mask must be exactly 0 or 1."""
        sample = train_ds[0]
        mask = sample["mask"]
        unique = mask.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0}), \
            f"mask has non-binary values: {unique}"

    def test_mask_outer_zero_inner_one(self, train_ds):
        """Outer half must be 0, inner half must be 1."""
        sample = train_ds[0]
        mask = sample["mask"].squeeze()   # H×W
        assert mask[:, :OUTER_W].max() == 0.0, "mask is non-zero on outer half"
        assert mask[:, OUTER_W:].min() == 1.0, "mask is not all-one on inner half"

    def test_distance_range(self, train_ds):
        """Distance-to-seam must be in [0, 1]."""
        sample = train_ds[0]
        dist = sample["distance"]
        assert dist.min() >= 0.0
        assert dist.max() <= 1.0 + 1e-5

    def test_distance_zero_at_seam(self, train_ds):
        """Distance must be 0 at the seam (column outer_width)."""
        sample = train_ds[0]
        dist = sample["distance"].squeeze()   # H×W
        seam_col = dist[:, OUTER_W]
        assert seam_col.max() < 1e-4, \
            "distance is not 0 at the seam line"

    def test_pixel_range(self, train_ds):
        """All pixel values must be in [0, 1]."""
        for idx in range(min(5, len(train_ds))):
            sample = train_ds[idx]
            rgb = sample["input"][:3]
            assert rgb.min() >= 0.0 and rgb.max() <= 1.0, \
                f"pixel values out of [0,1] at idx={idx}"

    def test_target_outer_equals_input_outer(self, val_ds):
        """Outer half of target must equal outer half of input (outer is original)."""
        sample = val_ds[0]
        outer_in = sample["input"][:3, :, :OUTER_W]
        outer_tgt = sample["target"][:, :, :OUTER_W]
        assert torch.allclose(outer_in, outer_tgt, atol=1e-6)


# ── GPU corruption pipeline integration tests ─────────────────────────────────

class TestGPUCorruptionPipeline:
    def test_gpu_corruption_only_changes_inner(self, gpu_corruption, train_ds):
        """GPU corruption must not modify the outer half of the strip."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        inner = clean_strip[:, :, :, OUTER_W:]
        corrupted_inner = gpu_corruption(inner)
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)

        outer_before = clean_strip[:, :, :, :OUTER_W]
        outer_after = corrupted_strip[:, :, :, :OUTER_W]
        assert torch.allclose(outer_before, outer_after, atol=1e-7), \
            "GPU corruption modified the outer half"

    def test_gpu_corruption_changes_inner(self, gpu_corruption, train_ds):
        """GPU corruption must change the inner half."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        inner = clean_strip[:, :, :, OUTER_W:]
        corrupted_inner = gpu_corruption(inner)
        assert not torch.allclose(inner, corrupted_inner, atol=1e-4), \
            "GPU corruption did not change the inner half"

    def test_input_rgb_updated_after_gpu_corruption(self, gpu_corruption, train_ds):
        """After GPU corruption, input_rgb in the batch must reflect the corrupted strip,
        not the clean strip — so baseline metrics are meaningful."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        inner = clean_strip[:, :, :, OUTER_W:]
        corrupted_inner = gpu_corruption(inner)
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        # Simulate what harmonizer_loop does
        updated_batch = {
            **batch,
            "input": torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1),
            "input_rgb": corrupted_strip,
        }
        # input_rgb inner must differ from clean target inner
        inp_inner = updated_batch["input_rgb"][:, :, :, OUTER_W:]
        tgt_inner = updated_batch["target"][:, :, :, OUTER_W:]
        assert not torch.allclose(inp_inner, tgt_inner, atol=1e-4), \
            "input_rgb was not updated: baseline metric would be trivially 0"

    def test_target_not_corrupted(self, gpu_corruption, train_ds):
        """Target must remain clean (original pixels) throughout the pipeline."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        target_before = batch["target"].clone()
        inner = batch["input"][:, :3, :, OUTER_W:]
        gpu_corruption(inner)   # must not affect target
        assert torch.allclose(batch["target"], target_before, atol=1e-7), \
            "target was modified by GPU corruption"

    def test_5ch_input_shape_after_corruption(self, gpu_corruption, train_ds):
        """After GPU corruption the input tensor must still be B×5×H×W."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        inner = clean_strip[:, :, :, OUTER_W:]
        corrupted_inner = gpu_corruption(inner)
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        new_input = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        B = new_input.shape[0]
        assert new_input.shape == (B, 5, H, W), \
            f"5-channel input shape broken: {new_input.shape}"


# ── Model output shape tests ─────────────────────────────────────────────────

class TestModelOutputShapes:
    def test_forward_output_keys(self, model, gpu_corruption, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        required_keys = {"curves", "shading_lowres", "shading", "corrected_inner",
                         "corrected_strip", "raw_curves"}
        assert required_keys.issubset(out.keys()), \
            f"missing output keys: {required_keys - out.keys()}"

    def test_curves_shape(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        B = batch["input"].shape[0]
        assert out["curves"].shape == (B, 3, 16), f"curves shape: {out['curves'].shape}"

    def test_shading_lowres_shape(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        B = batch["input"].shape[0]
        assert out["shading_lowres"].shape == (B, 1, 256, 32), \
            f"shading_lowres shape: {out['shading_lowres'].shape}"

    def test_corrected_inner_shape(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        B = batch["input"].shape[0]
        assert out["corrected_inner"].shape == (B, 3, H, INNER_W), \
            f"corrected_inner shape: {out['corrected_inner'].shape}"

    def test_corrected_strip_shape(self, model, train_ds):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        B = batch["input"].shape[0]
        assert out["corrected_strip"].shape == (B, 3, H, W), \
            f"corrected_strip shape: {out['corrected_strip'].shape}"

    def test_curves_monotonic(self, model, train_ds):
        """Curves must be strictly monotonically non-decreasing."""
        batch = collate_strip_batch([train_ds[0], train_ds[1], train_ds[2]])
        with torch.no_grad():
            out = model(batch["input"])
        diffs = out["curves"][..., 1:] - out["curves"][..., :-1]
        assert (diffs >= -1e-5).all(), "curves are not monotonic"

    def test_corrected_inner_range(self, model, train_ds):
        """Corrected inner values must be in [0, 1]."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out = model(batch["input"])
        ci = out["corrected_inner"]
        assert ci.min() >= 0.0, f"corrected_inner below 0: {ci.min()}"
        assert ci.max() <= 1.0, f"corrected_inner above 1: {ci.max()}"

    def test_outer_half_unchanged(self, model, train_ds):
        """Model must never modify the outer (original) pixels."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        outer_input = batch["input"][:, :3, :, :OUTER_W]
        with torch.no_grad():
            out = model(batch["input"])
        outer_output = out["corrected_strip"][:, :, :, :OUTER_W]
        assert torch.allclose(outer_input, outer_output, atol=1e-5), \
            "model modified the outer half"


# ── Loss correctness tests ─────────────────────────────────────────────────

class TestLossCorrectness:
    def test_all_loss_terms_present(self, model, train_ds, gpu_corruption):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        lc = HarmonizerLossComputer()
        losses = lc(out, batch["target"])
        expected = {"total", "l_rec", "l_seam", "l_low", "l_grad",
                    "l_curve_smooth", "l_curve_id", "l_tv", "l_mean"}
        assert expected.issubset(losses.keys()), f"missing loss keys: {expected - losses.keys()}"

    def test_no_nan_in_losses(self, model, train_ds, gpu_corruption):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        lc = HarmonizerLossComputer()
        losses = lc(out, batch["target"])
        for k, v in losses.items():
            assert not torch.isnan(v), f"NaN in loss component: {k}"
            assert not torch.isinf(v), f"Inf in loss component: {k}"

    def test_loss_positive(self, model, train_ds, gpu_corruption):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        lc = HarmonizerLossComputer()
        losses = lc(out, batch["target"])
        assert losses["total"].item() > 0, "total loss is zero or negative"


# ── Metric correctness tests ──────────────────────────────────────────────────

class TestMetricCorrectness:
    def test_baseline_mae_nonzero_with_corrupted_input(self, model, train_ds, gpu_corruption):
        """Baseline MAE must reflect actual corruption distance, not ~0."""
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        metrics = evaluate_harmonizer_batch(
            out["corrected_strip"], corrupted_strip, batch["target"],
            out["curves"], out["shading"],
        )
        assert metrics["baseline_boundary_mae_16"] > 0.001, \
            "baseline MAE is near 0 — input_rgb was probably clean instead of corrupted"

    def test_metrics_finite(self, model, train_ds, gpu_corruption):
        batch = collate_strip_batch([train_ds[0], train_ds[1]])
        clean_strip = batch["input"][:, :3]
        corrupted_inner = gpu_corruption(clean_strip[:, :, :, OUTER_W:])
        corrupted_strip = torch.cat([clean_strip[:, :, :, :OUTER_W], corrupted_inner], dim=-1)
        inp = torch.cat([corrupted_strip, batch["input"][:, 3:]], dim=1)
        with torch.no_grad():
            out = model(inp)
        metrics = evaluate_harmonizer_batch(
            out["corrected_strip"], corrupted_strip, batch["target"],
            out["curves"], out["shading"],
        )
        for k, v in metrics.items():
            assert math.isfinite(v), f"metric {k} is not finite: {v}"
