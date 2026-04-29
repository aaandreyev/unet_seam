import torch

from src.infer.correct_full_frame import _canonical_model_input, apply_corrector_to_full_frame


class AddInnerModel(torch.nn.Module):
    def __init__(self, delta: float = 0.1, confidence: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.delta = delta
        self.confidence = confidence

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        corrected = x[:, :3].clone()
        corrected_inner = corrected[..., 128:] + self.delta
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
            "confidence": torch.full((b, 1, x.shape[-2], 128), self.confidence, device=x.device),
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


def test_harmonizer_full_frame_keeps_outside_mask_exact():
    image = torch.rand(1, 3, 1024, 1024)
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[:, :, 256:768, 256:768] = 1.0
    bbox = (256, 256, 768, 768)
    out, debug = apply_corrector_to_full_frame(AddInnerModel(), image, mask, bbox, ["left", "right", "top", "bottom"], 128, strength=1.0)
    assert debug["architecture"] == "seam_harmonizer_v3"
    assert "side_confidences" in debug
    assert torch.equal(out * (1.0 - mask), image * (1.0 - mask))


def test_harmonizer_applies_raw_delta_without_post_gating():
    image = torch.zeros(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 128:384, 128:384] = 1.0
    bbox = (128, 128, 384, 384)
    out, _ = apply_corrector_to_full_frame(AddInnerModel(delta=0.1, confidence=1.0), image, mask, bbox, ["left"], 128, strength=1.0)
    changed = out[:, :, 128:384, 128:256]
    assert torch.allclose(changed, torch.full_like(changed, 0.1), atol=1e-6)


def test_harmonizer_accepts_unbounded_strength_values():
    image = torch.zeros(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 128:384, 128:384] = 1.0
    bbox = (128, 128, 384, 384)
    out, _ = apply_corrector_to_full_frame(AddInnerModel(delta=0.1, confidence=1.0), image, mask, bbox, ["left"], 128, strength=12.0)
    assert out.shape == image.shape
    assert float(out[:, :, 128:384, 128:256].mean()) > 1.0


def test_zero_confidence_suppresses_side_contribution():
    image = torch.zeros(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 128:384, 128:384] = 1.0
    bbox = (128, 128, 384, 384)
    out, _ = apply_corrector_to_full_frame(AddInnerModel(delta=0.1, confidence=0.0), image, mask, bbox, ["left"], 128, strength=1.0)
    assert torch.allclose(out, image)
