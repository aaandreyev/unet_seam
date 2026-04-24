import torch

from src.infer.merge_residuals import merge_side_residuals


def test_one_side_only():
    mask = torch.ones(1, 1, 8, 8)
    residual = {"left": torch.ones(1, 3, 8, 8)}
    merged, _ = merge_side_residuals(residual, mask)
    assert torch.allclose(merged, torch.ones_like(merged))


def test_no_amplification():
    mask = torch.ones(1, 1, 8, 8)
    side_residuals = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    merged, _ = merge_side_residuals(side_residuals, mask)
    assert float(merged.abs().max()) <= 0.2 + 1e-6
