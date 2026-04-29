import torch

from src.infer.cv_mask_harmonize import harmonize_by_mask_torch


def test_cv_harmonize_reduces_inside_outside_luma_gap():
    image = torch.full((1, 64, 64, 3), 0.7, dtype=torch.float32)
    image[:, 16:48, 16:48] = 0.3
    mask = torch.zeros(1, 64, 64, dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0
    before_gap = abs(float(image[:, 16:48, 16:48].mean()) - float(image[:, :16, :16].mean()))
    out = harmonize_by_mask_torch(image, mask, mode="inside", strip_width=8, blur_sigma=8.0, falloff=24)
    after_gap = abs(float(out[:, 16:48, 16:48, :3].mean()) - float(out[:, :16, :16, :3].mean()))
    assert after_gap < before_gap


def test_cv_harmonize_respects_protect_mask():
    image = torch.full((1, 64, 64, 3), 0.7, dtype=torch.float32)
    image[:, 16:48, 16:48] = 0.3
    mask = torch.zeros(1, 64, 64, dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0
    protect = torch.zeros(1, 64, 64, dtype=torch.float32)
    protect[:, 24:40, 24:40] = 1.0
    out = harmonize_by_mask_torch(image, mask, mode="inside", strip_width=8, blur_sigma=8.0, falloff=24, protect_mask=protect)
    assert torch.allclose(out[:, 24:40, 24:40, :3], image[:, 24:40, 24:40, :3], atol=5e-3)
