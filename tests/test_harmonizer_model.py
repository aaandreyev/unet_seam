import torch

from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.models.harmonizer import SeamHarmonizerV1, apply_monotonic_curves, monotonic_knots, reconstruct_corrected_strip


def test_monotonic_knots_are_bounded_and_ordered():
    raw = torch.randn(4, 3, 15)
    knots = monotonic_knots(raw)
    assert knots.shape == (4, 3, 16)
    assert torch.allclose(knots[..., 0], torch.zeros_like(knots[..., 0]))
    assert torch.allclose(knots[..., -1], torch.ones_like(knots[..., -1]))
    assert torch.all(knots[..., 1:] >= knots[..., :-1])


def test_zero_initialized_harmonizer_is_identity_like():
    model = SeamHarmonizerV1(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1), num_knots=16)
    x = torch.rand(1, 5, 128, 256)
    with torch.inference_mode():
        out = model(x)
    assert out["corrected_strip"].shape == (1, 3, 128, 256)
    assert out["curves"].shape == (1, 3, 16)
    assert out["shading_lowres"].shape == (1, 1, 256, 32)
    assert torch.all(out["curves"][..., 1:] >= out["curves"][..., :-1])
    assert float((out["corrected_strip"][..., :128] - x[:, :3, :, :128]).abs().max()) == 0.0
    assert float((out["corrected_inner"] - x[:, :3, :, 128:]).abs().max()) < 1e-3


def test_reconstruct_corrected_strip_keeps_outer_exact():
    strip = torch.rand(2, 3, 32, 256)
    raw = torch.zeros(2, 3, 15)
    shading = torch.zeros(2, 1, 8, 4)
    out = reconstruct_corrected_strip(strip, monotonic_knots(raw), shading)
    assert torch.equal(out["corrected_strip"][..., :128], strip[..., :128])
    assert torch.allclose(out["corrected_inner"], strip[..., 128:], atol=1e-3)


def test_apply_monotonic_curves_shape_and_range():
    inner = torch.rand(2, 3, 16, 32)
    knots = monotonic_knots(torch.randn(2, 3, 15))
    out = apply_monotonic_curves(inner, knots)
    assert out.shape == inner.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_harmonizer_loss_is_finite():
    model = SeamHarmonizerV1(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1), num_knots=16)
    x = torch.rand(1, 5, 128, 256)
    target = x[:, :3].clone()
    with torch.inference_mode():
        out = model(x)
    losses = HarmonizerLossComputer()(out, target)
    assert "total" in losses
    assert all(torch.isfinite(value).all() for value in losses.values())
