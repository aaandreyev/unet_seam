import torch

from src.losses.harmonizer_losses import HarmonizerLossComputer
from src.models.harmonizer import SeamHarmonizerV3, reconstruct_corrected_strip


def test_zero_initialized_harmonizer_is_identity_like():
    model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    x = torch.rand(1, 9, 128, 256)
    with torch.inference_mode():
        out = model(x)
    assert out["corrected_strip"].shape == (1, 3, 128, 256)
    assert out["corrected_inner"].shape == (1, 3, 128, 128)
    assert out["gain_lowres"].shape[-2:] == (32, 32)
    assert out["mix_lowres"].shape[1:3] == (3, 3)
    assert float((out["corrected_strip"][..., :128] - x[:, :3, :, :128]).abs().max()) == 0.0
    assert float((out["corrected_inner"] - x[:, :3, :, 128:]).abs().max()) < 0.03


def test_reconstruct_corrected_strip_keeps_outer_exact():
    strip = torch.rand(2, 3, 32, 256)
    zeros_1 = torch.zeros(2, 1, 8, 16)
    zeros_3 = torch.zeros(2, 3, 8, 16)
    mix = torch.zeros(2, 3, 3, 8, 16)
    out = reconstruct_corrected_strip(
        strip,
        {
            "gain_lowres": zeros_1,
            "gamma_lowres": zeros_1,
            "bias_lowres": zeros_3,
            "mix_lowres": mix,
            "detail_lowres": zeros_3,
            "gate_lowres": zeros_1,
        },
    )
    assert torch.equal(out["corrected_strip"][..., :128], strip[..., :128])
    assert out["confidence"].shape == (2, 1, 32, 128)


def test_harmonizer_loss_is_finite():
    model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    x = torch.rand(1, 9, 128, 256)
    target = x[:, :3].clone()
    batch = {
        "target": target,
        "boundary_band_mask": torch.ones(1, 1, 128, 256),
        "decay_mask": torch.ones(1, 1, 128, 256),
    }
    with torch.inference_mode():
        out = model(x)
    losses = HarmonizerLossComputer()(out, batch)
    assert "total" in losses
    assert all(torch.isfinite(value).all() for value in losses.values())
