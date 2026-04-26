from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.data.corruptions import GROUPS, apply_random_corruptions
from src.data.harmonizer_input import build_harmonizer_input
from src.data.manifest import write_jsonl
from src.data.synthetic_strip_dataset import SyntheticStripDataset
from src.infer.correct_full_frame import _structural_strength_scale
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
    assert loss.weights == {
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
    assert set(NODE_CLASS_MAPPINGS) == {"SeamHarmonizerV3"}
    assert NODE_DISPLAY_NAME_MAPPINGS["SeamHarmonizerV3"] == "Seam Harmonizer v3"


def test_spec_inference_structural_gate_thresholds():
    matching = torch.zeros(3, 64, 256)
    band = torch.rand(3, 64, 8)
    matching[:, :, 120:128] = torch.flip(band, dims=(-1,))
    matching[:, :, 128:136] = band
    scale, score = _structural_strength_scale(matching, outer_width=128, band_px=8)
    assert scale == 1.0
    assert score > 0.5


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
