import torch

from src.infer.merge_bands import build_side_weight_map, merge_side_deltas


def test_side_weight_map_is_non_negative():
    mask = torch.ones(1, 1, 16, 16)
    weight = build_side_weight_map(mask, "left")
    assert weight.min() >= 0
    assert weight.max() <= 1


def test_one_side_only_delta_merge():
    mask = torch.ones(1, 1, 8, 8)
    deltas = {"left": torch.ones(1, 3, 8, 8)}
    merged, _ = merge_side_deltas(deltas, mask)
    assert torch.allclose(merged, torch.ones_like(merged))


def test_corner_fusion_does_not_amplify_delta():
    mask = torch.ones(1, 1, 8, 8)
    side_deltas = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    merged, _ = merge_side_deltas(side_deltas, mask)
    assert float(merged.abs().max()) <= 0.2 + 1e-6
