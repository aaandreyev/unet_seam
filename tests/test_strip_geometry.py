import torch

from src.data.strip_geometry import canonicalize_strip, decanonicalize_strip, make_distance_to_seam, make_inner_mask, validate_roundtrip


def test_canonicalize_roundtrip_all_sides():
    strip = torch.arange(3 * 1024 * 256, dtype=torch.float32).view(3, 1024, 256)
    for side in ("left", "right", "top", "bottom"):
        assert validate_roundtrip(strip, side)
        assert torch.equal(decanonicalize_strip(canonicalize_strip(strip, side), side), strip)


def test_mask_and_distance_shapes():
    mask = make_inner_mask(1024, 256, 128)
    distance = make_distance_to_seam(1024, 256, 128)
    assert mask.shape == (1, 1, 1024, 256)
    assert distance.shape == (1, 1, 1024, 256)
    assert mask[..., :128].sum() == 0
    assert mask[..., 128:].min() == 1
