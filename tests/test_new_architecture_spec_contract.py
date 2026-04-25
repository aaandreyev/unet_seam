from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.data.corruptions import GROUPS, apply_random_corruptions
from src.data.manifest import write_jsonl
from src.data.strip_geometry import make_distance_to_seam, make_inner_mask
from src.data.synthetic_strip_dataset import SyntheticStripDataset
from src.infer.correct_full_frame import _structural_strength_scale
from src.losses.harmonizer_losses import HarmonizerLossComputer, seam_weight
from src.models.harmonizer import SeamHarmonizerV1
from src.models.harmonizer_blocks import NAFBlockLite


def test_spec_model_default_architecture_contract():
    model = SeamHarmonizerV1()
    assert model.in_channels == 5
    assert model.channels == (32, 64, 128, 192)
    assert model.blocks == (2, 2, 4, 6)
    assert model.num_knots == 16
    assert model.alpha == 0.20
    assert all(isinstance(block, NAFBlockLite) for stage in model.encoder.stages for block in stage)
    assert not any("residual_head" in name or "rgb_head" in name for name, _ in model.named_modules())


def test_spec_input_channels_mask_and_distance_ramp():
    mask = make_inner_mask(1024, 256, 128)
    distance = make_distance_to_seam(1024, 256, 128)
    assert torch.equal(mask[..., :128], torch.zeros_like(mask[..., :128]))
    assert torch.equal(mask[..., 128:], torch.ones_like(mask[..., 128:]))
    assert float(distance[..., :128].max()) == 0.0
    assert torch.allclose(distance[..., 128], torch.zeros_like(distance[..., 128]))
    assert torch.allclose(distance[..., -1], torch.ones_like(distance[..., -1]))


def test_spec_loss_weights_and_seam_weight_shape():
    loss = HarmonizerLossComputer()
    assert loss.weights == {
        "rec": 1.0,
        "seam": 2.0,
        "low": 0.75,
        "grad": 0.25,
        "curve_smooth": 0.02,
        "curve_id": 0.01,
        "tv": 0.02,
        "mean": 0.01,
    }
    weights = seam_weight(4, 128, 12.0, torch.device("cpu"), torch.float32)
    assert weights.shape == (1, 1, 4, 128)
    assert torch.all(weights[..., :32] > 0)
    assert float(weights[..., 32:].max()) == 0.0
    assert torch.all(weights[..., 1:32] <= weights[..., :31])


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
        assert float(result.image.min()) >= 0.0
        assert float(result.image.max()) <= 1.0
        c_hits += int(any(op in GROUPS["C"] for op in result.ops))
        d_hits += int(any(op in GROUPS["D"] for op in result.ops))
    assert 0.20 <= c_hits / 200.0 <= 0.50
    assert 0.12 <= d_hits / 200.0 <= 0.40


def test_spec_synthetic_dataset_corrupts_only_inner_half(tmp_path: Path):
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    img_path = tmp_path / "source.png"
    Image.fromarray(img).save(img_path)
    write_jsonl(tmp_path / "manifest.jsonl", [{"id": "x", "source_path": str(img_path), "split": "train"}])
    dataset = SyntheticStripDataset(tmp_path / "manifest.jsonl", strips_per_image=1, split="train", inner_widths=[128])
    sample = dataset[0]
    assert sample["input"].shape == (5, 1024, 256)
    assert sample["target"].shape == (3, 1024, 256)
    assert torch.equal(sample["input"][:3, :, :128], sample["target"][:, :, :128])


def test_spec_comfy_node_uses_harmonizer_name_without_legacy_alias():
    assert set(NODE_CLASS_MAPPINGS) == {"SeamHarmonizerV1"}
    assert NODE_DISPLAY_NAME_MAPPINGS["SeamHarmonizerV1"] == "Seam Harmonizer v1"


def test_spec_inference_structural_gate_thresholds():
    matching = torch.zeros(3, 64, 256)
    band = torch.rand(3, 64, 8)
    matching[:, :, 120:128] = torch.flip(band, dims=(-1,))
    matching[:, :, 128:136] = band
    scale, score = _structural_strength_scale(matching, outer_width=128, band_px=8)
    assert scale == 1.0
    assert score > 0.5
    mismatched = torch.zeros(3, 64, 256)
    mismatched[:, :, 128:136:2] = 1.0
    scale, score = _structural_strength_scale(mismatched, outer_width=128, band_px=8)
    assert scale == 0.0
    assert score < 0.35
