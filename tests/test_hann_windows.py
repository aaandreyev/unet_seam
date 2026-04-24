import torch

from src.infer.merge_residuals import build_side_weight_map


def test_hann_weight_map_is_non_negative():
    mask = torch.ones(1, 1, 16, 16)
    weight = build_side_weight_map(mask, "left")
    assert weight.min() >= 0
    assert weight.max() <= 1
