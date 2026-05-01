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


def test_attention_head_present_and_does_not_affect_reconstruction():
    model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    x = torch.rand(1, 9, 128, 256)
    with torch.inference_mode():
        out = model(x)
    assert "attention_lowres" in out
    assert out["attention_lowres"].shape[0] == 1
    assert out["attention_lowres"].shape[1] == 1
    assert out["attention_lowres"].shape[-2:] == (32, 32)
    assert torch.isfinite(out["attention_lowres"]).all()
    assert float(out["attention_lowres"].min()) >= 0.0
    assert float(out["attention_lowres"].max()) <= 1.0


def test_legacy_state_dict_loads_with_strict_false():
    """Old checkpoints predate attention_head; loader must accept missing keys."""
    legacy_model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    legacy_state = {k: v for k, v in legacy_model.state_dict().items() if not k.startswith("attention_head")}
    new_model = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    new_pre = {k: v.clone() for k, v in new_model.state_dict().items() if k.startswith("attention_head")}
    result = new_model.load_state_dict(legacy_state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    assert all(k.startswith("attention_head") for k in missing), f"unexpected missing: {missing}"
    assert unexpected == [], f"unexpected keys present: {unexpected}"
    new_post = {k: v for k, v in new_model.state_dict().items() if k.startswith("attention_head")}
    for k in new_pre:
        assert torch.equal(new_pre[k], new_post[k]), f"{k} should be untouched after legacy load"


def test_legacy_state_dict_load_preserves_corrected_output():
    """Loading a legacy state_dict (no attention_head) must give identical corrected_strip
    as the source model — attention_head must not influence reconstruction."""
    src = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    for p in src.parameters():
        if p.requires_grad and p.numel() > 1:
            p.data.uniform_(-0.05, 0.05)
    legacy_state = {k: v for k, v in src.state_dict().items() if not k.startswith("attention_head")}
    dst = SeamHarmonizerV3(channels=(8, 12, 16, 24), blocks=(1, 1, 1, 1))
    dst.load_state_dict(legacy_state, strict=False)
    x = torch.rand(1, 9, 128, 256)
    with torch.inference_mode():
        out_src = src(x)
        out_dst = dst(x)
    assert torch.allclose(out_src["corrected_strip"], out_dst["corrected_strip"], atol=1e-6)
    assert torch.allclose(out_src["corrected_inner"], out_dst["corrected_inner"], atol=1e-6)


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
    assert "l_attn" in losses
    assert "l_conf_metric" not in losses
    assert "l_conf_budget" not in losses
    assert all(torch.isfinite(value).all() for value in losses.values())
