import torch

from comfy_node.seam_harmonize_hybrid_node import SeamHarmonizerHybridNode


def test_hybrid_node_auto_prefers_blend_for_rectangular_inside(monkeypatch):
    called = {"ml": 0, "cv": 0}

    def fake_load_model(path: str, device: str = "cpu"):
        return object(), {}

    def fake_apply(model, image, mask, bbox, sides, inner_width, strength):
        called["ml"] += 1
        return image + 0.1 * mask, {"per_side": {}}

    def fake_cv(image, mask, **kwargs):
        called["cv"] += 1
        return image

    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.load_model", fake_load_model)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.apply_corrector_to_full_frame", fake_apply)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.harmonize_by_mask_torch", fake_cv)

    node = SeamHarmonizerHybridNode()
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    mask[:, 32:96, 32:96] = 1.0
    out, = node.run(
        image,
        mask,
        "dummy.safetensors",
        "auto",
        "inside",
        128,
        1.0,
        8,
        20.0,
        64,
        1.0,
        1.0,
        1.0,
        0.5,
        0,
        640,
        True,
        True,
        True,
        True,
        False,
    )
    assert out.shape == image.shape
    assert called["ml"] == 1
    assert called["cv"] == 1


def test_hybrid_node_auto_falls_back_to_cv_for_nonrectangular_mask(monkeypatch):
    called = {"ml": 0, "cv": 0}

    def fake_load_model(path: str, device: str = "cpu"):
        return object(), {}

    def fake_apply(model, image, mask, bbox, sides, inner_width, strength):
        called["ml"] += 1
        return image, {"per_side": {}}

    def fake_cv(image, mask, **kwargs):
        called["cv"] += 1
        return image + 0.2

    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.load_model", fake_load_model)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.apply_corrector_to_full_frame", fake_apply)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.harmonize_by_mask_torch", fake_cv)

    node = SeamHarmonizerHybridNode()
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    mask[:, 32:96, 32:64] = 1.0
    mask[:, 64:96, 64:96] = 1.0
    out, = node.run(
        image,
        mask,
        "dummy.safetensors",
        "auto",
        "inside",
        128,
        1.0,
        8,
        20.0,
        64,
        1.0,
        1.0,
        1.0,
        0.5,
        0,
        640,
        True,
        True,
        True,
        True,
        False,
    )
    assert float(out.mean()) > 0.1
    assert called["ml"] == 0
    assert called["cv"] == 1


def test_hybrid_node_blend_respects_protect_mask(monkeypatch):
    def fake_load_model(path: str, device: str = "cpu"):
        return object(), {}

    def fake_apply(model, image, mask, bbox, sides, inner_width, strength):
        return image + 0.4 * mask, {"per_side": {}}

    def fake_cv(image, mask, **kwargs):
        return image + 0.2 * mask.unsqueeze(-1)

    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.load_model", fake_load_model)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.apply_corrector_to_full_frame", fake_apply)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.harmonize_by_mask_torch", fake_cv)

    node = SeamHarmonizerHybridNode()
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    mask[:, 32:96, 32:96] = 1.0
    protect = torch.zeros(1, 128, 128)
    protect[:, 40:56, 40:56] = 1.0
    out, = node.run(
        image,
        mask,
        "dummy.safetensors",
        "blend",
        "inside",
        128,
        1.0,
        8,
        20.0,
        64,
        1.0,
        1.0,
        1.0,
        0.5,
        0,
        640,
        True,
        True,
        True,
        True,
        False,
        PROTECT_MASK=protect,
    )
    assert torch.allclose(out[:, 40:56, 40:56], image[:, 40:56, 40:56], atol=1e-6)


def test_hybrid_node_blend_injects_cv_residual_stronger_near_seam(monkeypatch):
    def fake_load_model(path: str, device: str = "cpu"):
        return object(), {}

    def fake_apply(model, image, mask, bbox, sides, inner_width, strength):
        return image + 0.4 * mask, {"per_side": {}}

    def fake_cv(image, mask, **kwargs):
        return image + 0.2 * mask.unsqueeze(-1)

    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.load_model", fake_load_model)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.apply_corrector_to_full_frame", fake_apply)
    monkeypatch.setattr("comfy_node.seam_harmonize_hybrid_node.harmonize_by_mask_torch", fake_cv)

    node = SeamHarmonizerHybridNode()
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    mask[:, 16:112, 16:112] = 1.0
    out, = node.run(
        image,
        mask,
        "dummy.safetensors",
        "blend",
        "inside",
        128,
        1.0,
        8,
        20.0,
        64,
        1.0,
        1.0,
        1.0,
        0.5,
        0,
        640,
        True,
        True,
        True,
        True,
        False,
    )
    seam_band = float(out[:, 16:32, 16:112].mean())
    center_band = float(out[:, 56:72, 56:72].mean())
    assert seam_band > center_band
