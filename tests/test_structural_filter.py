import torch

from src.data.structural_filter import gradient_cosine_similarity, keep_structurally_matched_strip


def test_gradient_cosine_identical_is_high():
    x = torch.rand(2, 3, 32, 32)
    score = gradient_cosine_similarity(x, x)
    assert torch.all(score > 0.99)


def test_structural_filter_rejects_flat_vs_edged_band():
    generated = torch.zeros(3, 64, 256)
    target = torch.zeros(3, 64, 256)
    target[:, :, 128:160:2] = 1.0
    assert not keep_structurally_matched_strip(generated, target, outer_width=128, band_px=32, threshold=0.6)
