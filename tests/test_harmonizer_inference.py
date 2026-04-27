import torch

from src.infer.correct_full_frame import (
    _build_profile_guard,
    _build_safety_gate,
    _canonical_model_input,
    _inner_taper,
    apply_corrector_to_full_frame,
)


class AddInnerModel(torch.nn.Module):
    def __init__(self, delta: float = 0.1) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.delta = delta

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        corrected = x[:, :3].clone()
        corrected_inner = (corrected[..., 128:] + self.delta).clamp(0.0, 1.0)
        corrected[..., 128:] = corrected_inner
        b = x.shape[0]
        return {
            "corrected_strip": corrected,
            "corrected_inner": corrected_inner,
            "gain_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
            "gamma_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
            "bias_lowres": torch.zeros(b, 3, 256, 32, device=x.device),
            "mix_lowres": torch.zeros(b, 3, 3, 256, 32, device=x.device),
            "detail_lowres": torch.zeros(b, 3, 256, 32, device=x.device),
            "gate_lowres": torch.zeros(b, 1, 256, 32, device=x.device),
            "confidence": torch.zeros(b, 1, x.shape[-2], 128, device=x.device),
            "gain": torch.ones(b, 1, x.shape[-2], 128, device=x.device),
            "detail": torch.zeros(b, 3, x.shape[-2], 128, device=x.device),
        }


def test_canonical_model_input_channels():
    strips = torch.rand(2, 3, 16, 256)
    model_in = _canonical_model_input(strips, 128)
    assert model_in.shape == (2, 9, 16, 256)
    assert torch.equal(model_in[:, 3, :, :128], torch.zeros_like(model_in[:, 3, :, :128]))
    assert torch.equal(model_in[:, 3, :, 128:], torch.ones_like(model_in[:, 3, :, 128:]))
    assert float(model_in[:, 4, :, :128].max()) == 0.0
    assert torch.allclose(model_in[:, 4, :, -1], torch.ones_like(model_in[:, 4, :, -1]))


def test_inner_taper_is_strongest_at_seam_and_zero_at_inner_edge():
    taper = _inner_taper(8, 128, torch.device("cpu"), torch.float32)
    assert torch.allclose(taper[..., 0], torch.ones_like(taper[..., 0]))
    assert torch.allclose(taper[..., -1], torch.full_like(taper[..., -1], 0.15), atol=1e-6)
    assert torch.all(taper[..., 1:] <= taper[..., :-1] + 1e-6)


def test_harmonizer_full_frame_keeps_outside_mask_exact():
    image = torch.rand(1, 3, 1024, 1024)
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[:, :, 256:768, 256:768] = 1.0
    bbox = (256, 256, 768, 768)
    out, debug = apply_corrector_to_full_frame(AddInnerModel(), image, mask, bbox, ["left", "right", "top", "bottom"], 128, strength=1.0)
    assert debug["architecture"] == "seam_harmonizer_v3"
    assert "side_confidences" in debug
    assert torch.equal(out * (1.0 - mask), image * (1.0 - mask))


def test_harmonizer_accepts_strength_above_one():
    image = torch.rand(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 128:384, 128:384] = 1.0
    bbox = (128, 128, 384, 384)
    out, _ = apply_corrector_to_full_frame(AddInnerModel(), image, mask, bbox, ["left", "right"], 128, strength=5.0)
    assert out.shape == image.shape


def test_safety_gate_suppresses_worsening_rows():
    strip = torch.full((1, 3, 8, 256), 0.5)
    strip[..., 128:152] = 0.5
    corrected = strip.clone()
    corrected[..., 128:152] = 0.9
    safety, before_err, after_err = _build_safety_gate(strip, corrected, outer_width=128, inner_width=128, band_px=24)
    assert float(before_err.mean()) < float(after_err.mean())
    assert float(safety[..., :24].mean()) < 0.5


def test_safety_gate_stays_open_when_correction_improves():
    strip = torch.full((1, 3, 8, 256), 0.5)
    strip[..., 128:152] = 0.8
    corrected = strip.clone()
    corrected[..., 128:152] = 0.55
    safety, before_err, after_err = _build_safety_gate(strip, corrected, outer_width=128, inner_width=128, band_px=24)
    assert float(after_err.mean()) < float(before_err.mean())
    assert float(safety[..., :24].mean()) > 0.9


def test_profile_guard_suppresses_long_smooth_stripe():
    merged = torch.zeros(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 16:112, 16:112] = 1.0
    bbox = (16, 16, 112, 112)
    merged[:, :, 16:112, 16:28] = 0.06
    guard, side_guards, stats = _build_profile_guard(merged, mask, bbox, band_px=12)
    assert float(guard[:, :, 16:112, 16:28].mean()) < 0.8
    assert float(side_guards["left"][:, :, 16:112, 16:28].mean()) < 0.8
    assert stats["left"]["profile_signal_mean"] > 0.02


def test_profile_guard_preserves_localized_patch():
    merged = torch.zeros(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 16:112, 16:112] = 1.0
    bbox = (16, 16, 112, 112)
    merged[:, :, 48:56, 16:28] = 0.06
    guard, _, _ = _build_profile_guard(merged, mask, bbox, band_px=12)
    assert float(guard[:, :, 48:56, 16:28].mean()) > 0.88
