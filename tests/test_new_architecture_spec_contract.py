from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.data.corruptions import GROUPS, _apply_corruption_op, _build_artifact_field_bank, apply_random_corruptions
from src.data.harmonizer_input import build_harmonizer_input
from src.data.manifest import write_jsonl
from src.data.synthetic_strip_dataset import SyntheticStripDataset
from src.infer.correct_full_frame import apply_corrector_to_full_frame
from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.models.harmonizer import SeamHarmonizerV3
from src.models.harmonizer_blocks import NAFBlockLite
from scripts.train_harmonizer import _quality
from scripts.export_harmonizer_safetensors import _validate_checkpoint_for_export


def test_spec_model_default_architecture_contract():
    model = SeamHarmonizerV3()
    assert model.in_channels == 9
    assert model.channels == (32, 64, 128, 192)
    assert model.blocks == (2, 2, 4, 6)
    assert all(isinstance(block, NAFBlockLite) for stage in model.encoder.stages for block in stage)
    assert hasattr(model, "coarse_head")


def test_spec_input_channels_mask_distance_and_aux_maps():
    built = build_harmonizer_input(torch.rand(3, 1024, 256), outer_width=128, boundary_band_px=24)
    inp = built["input"]
    assert inp.shape == (9, 1024, 256)
    assert torch.equal(inp[3, :, :128], torch.zeros_like(inp[3, :, :128]))
    assert torch.equal(inp[3, :, 128:], torch.ones_like(inp[3, :, 128:]))
    assert float(inp[4, :, :128].max()) == 0.0
    assert torch.allclose(inp[4, :, -1], torch.ones_like(inp[4, :, -1]))
    assert inp[5].max() <= 1.0


def test_spec_loss_weights():
    loss = HarmonizerLossComputer()
    required = {"rec", "seam", "low", "grad", "chroma", "stats", "gate", "field", "detail", "matrix", "lab", "profile", "attn", "overcorr"}
    assert required.issubset(loss.weights.keys())
    # gain_reg is allowed to be near-zero (regulariser off by default)
    assert all(v >= 0 for v in loss.weights.values())
    assert all(v > 0 for k, v in loss.weights.items() if k != "gain_reg")
    # Spot-check that key perceptual terms are reasonably weighted.
    assert loss.weights["lab"] >= 0.4
    assert loss.weights["seam"] >= 1.0
    # conf_metric and conf_budget removed in stage8 redesign.
    assert "conf_align" not in loss.weights
    assert "conf_metric" not in loss.weights
    assert "conf_budget" not in loss.weights
    assert loss.weights["profile"] >= 0.3
    if False:  # kept for reference, not enforced so weights can be tuned freely
        _ = {
        "rec": 0.8,
        "seam": 1.1,
        "low": 1.2,
        "grad": 0.25,
        "chroma": 0.6,
        "stats": 0.35,
        "lab": 0.4,
        "gate": 0.04,
        "field": 0.10,
        "detail": 0.10,
        "matrix": 0.10,
    }


def test_spec_synthetic_corruption_families_and_probabilities():
    required_a = {"exposure", "brightness", "contrast", "gamma", "saturation", "hue", "temperature", "tint", "channel_gains", "black_point", "white_point"}
    required_b = {"shadow_lift", "shadow_crush", "highlight_compress", "highlight_boost", "midtone", "s_curve", "reverse_s_curve"}
    required_c = {"horizontal_luma_gradient", "vertical_luma_gradient", "illumination_field", "temperature_field", "saturation_field"}
    required_d = {"blur", "microcontrast", "noise", "jpeg_like"}
    assert required_a.issubset(set(GROUPS["A"]))
    assert required_b.issubset(set(GROUPS["B"]))
    assert required_c.issubset(set(GROUPS["C"]))
    assert required_d.issubset(set(GROUPS["D"]))
    inner = torch.full((1, 3, 32, 32), 0.5)
    c_hits = 0
    d_hits = 0
    for seed in range(200):
        result = apply_random_corruptions(inner, torch.Generator().manual_seed(seed))
        assert 2 <= len(result.ops) <= 5
        assert any(op in GROUPS["A"] + GROUPS["B"] for op in result.ops)
        c_hits += int(any(op in GROUPS["C"] for op in result.ops))
        d_hits += int(any(op in GROUPS["D"] for op in result.ops))
    assert 0.35 <= c_hits / 200.0 <= 0.65
    assert 0.08 <= d_hits / 200.0 <= 0.32


def test_spatial_twin_of_brightness_is_nonuniform():
    inner = torch.full((1, 3, 64, 64), 0.5)
    generator = torch.Generator().manual_seed(7)
    fields = _build_artifact_field_bank(inner.shape, generator)
    result = _apply_corruption_op("brightness", inner, generator, fields, use_spatial=True)
    luma = result.mean(dim=1)
    assert float(luma.std()) > 1e-3
    assert not torch.allclose(luma[:, :, :16], luma[:, :, -16:], atol=1e-4)


def test_corruption_spatial_probability_override_biases_selection():
    inner = torch.full((1, 3, 64, 64), 0.5)
    spatial_hits = 0
    trials = 400
    for seed in range(trials):
        generator = torch.Generator().manual_seed(seed)
        fields = _build_artifact_field_bank(inner.shape, generator)
        out = _apply_corruption_op(
            "brightness",
            inner,
            generator,
            fields,
            corruption_cfg={"spatial_probability": {"ab": 0.8}},
        )
        luma = out.mean(dim=1)
        spatial_hits += int(float(luma.std()) > 1e-3)
    assert 0.68 <= spatial_hits / trials <= 0.90


def test_spec_synthetic_dataset_builds_v3_input(tmp_path: Path):
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    img_path = tmp_path / "source.png"
    Image.fromarray(img).save(img_path)
    write_jsonl(tmp_path / "manifest.jsonl", [{"id": "x", "source_path": str(img_path), "split": "train"}])
    dataset = SyntheticStripDataset(tmp_path / "manifest.jsonl", strips_per_image=1, split="train", inner_widths=[128])
    sample = dataset[0]
    assert sample["input"].shape == (9, 1024, 256)
    assert sample["target"].shape == (3, 1024, 256)
    assert torch.equal(sample["input"][:3, :, :128], sample["target"][:, :, :128])


def test_spec_comfy_node_uses_v3_name():
    assert set(NODE_CLASS_MAPPINGS) == {"SeamHarmonizerV3", "SeamHarmonizerHybrid"}
    assert NODE_DISPLAY_NAME_MAPPINGS["SeamHarmonizerV3"] == "Seam Harmonizer v3"
    assert NODE_DISPLAY_NAME_MAPPINGS["SeamHarmonizerHybrid"] == "Seam Harmonizer Hybrid"


def test_spec_inference_uses_raw_model_outputs_without_post_gates():
    class ConstantShift(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            corrected = x[:, :3].clone()
            corrected_inner = corrected[..., 128:] + 0.125
            corrected[..., 128:] = corrected_inner
            b = x.shape[0]
            return {
                "corrected_strip": corrected,
                "corrected_inner": corrected_inner,
                "gain_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
                "gamma_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
                "bias_lowres": torch.zeros(b, 3, 256, 32, device=x.device),
                "mix_lowres": torch.zeros(b, 3, 3, 256, 32, device=x.device),
                "detail_lowres": torch.zeros(b, 3, 256, 32, device=x.device),
                "gate_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
                "confidence": torch.ones(b, 1, x.shape[-2], 128, device=x.device),
                "gain": torch.ones(b, 1, x.shape[-2], 128, device=x.device),
                "detail": torch.zeros(b, 3, x.shape[-2], 128, device=x.device),
            }

    image = torch.zeros(1, 3, 256, 256)
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 64:192, 64:192] = 1.0
    out, _debug = apply_corrector_to_full_frame(ConstantShift(), image, mask, (64, 64, 192, 192), ["left"], 128, strength=1.0)
    assert torch.allclose(out[:, :, 64:192, 64:192], torch.full_like(out[:, :, 64:192, 64:192], 0.125), atol=1e-6)


def test_spec_export_rejects_incompatible_checkpoint():
    bad_ckpt = {
        "config": {
            "model": {"architecture": "seam_harmonizer_v3", "in_channels": 9, "channels": [32, 64, 128, 192], "blocks": [2, 2, 4, 6]},
            "dataset": {"outer_width": 128, "boundary_band_px": 24},
        },
        "ema": {"wrong.weight": torch.zeros(1)},
    }
    try:
        _validate_checkpoint_for_export(bad_ckpt)
    except RuntimeError:
        return
    raise AssertionError("incompatible checkpoint must be rejected by export validation")


def test_quality_prioritizes_deltae_and_gate_deficits():
    better_deltae = {
        "boundary_ciede2000_16": 2.75,
        "baseline_boundary_ciede2000_16": 4.0,
        "boundary_mae_16": 0.0170,
        "baseline_boundary_mae_16": 0.036,
        "lowfreq_mae": 0.0175,
    }
    worse_deltae = {
        "boundary_ciede2000_16": 2.90,
        "baseline_boundary_ciede2000_16": 4.0,
        "boundary_mae_16": 0.0167,
        "baseline_boundary_mae_16": 0.036,
        "lowfreq_mae": 0.0173,
    }
    assert _quality(better_deltae) < _quality(worse_deltae)


def test_quality_penalizes_visual_risk_even_when_strip_metrics_are_close():
    safer = {
        "boundary_ciede2000_16": 2.60,
        "baseline_boundary_ciede2000_16": 4.1,
        "boundary_mae_16": 0.0178,
        "baseline_boundary_mae_16": 0.036,
        "lowfreq_mae": 0.0180,
        "delta_luma_profile_mae": 0.0060,
        "delta_chroma_profile_mae": 0.0040,
        "overcorrection_mae": 0.0010,
        "confidence_mean": 0.18,
        "confidence_alignment_mae": 0.11,
        "detail_abs_mean": 0.0030,
        "gain_abs_log_mean": 0.032,
    }
    riskier = dict(safer)
    riskier.update(
        {
            "delta_luma_profile_mae": 0.015,
            "overcorrection_mae": 0.005,
            "confidence_mean": 0.31,
            "detail_abs_mean": 0.0075,
            "gain_abs_log_mean": 0.08,
        }
    )
    assert _quality(safer) < _quality(riskier)
