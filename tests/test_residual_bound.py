import torch

from src.models.resunet import SeamResUNet


def test_residual_is_bounded():
    model = SeamResUNet()
    x = torch.rand(1, 5, 1024, 256)
    y = model(x)
    assert y.shape == (1, 3, 1024, 256)
    assert float(y.detach().abs().max()) <= 0.3 + 1e-6
