import torch

from src.data.strip_geometry import (
    StripSpec,
    canonicalize_strip,
    decanonicalize_strip,
    extract_side_strip,
    make_distance_to_seam,
    make_inner_mask,
    validate_roundtrip,
)


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


def test_extract_side_strip_aligns_to_bbox_top_left_not_center():
    """Centering a 1024-tile on the bbox left a dead band along each long side; origin = top-left of bbox."""
    h, w = 2000, 2000
    img = torch.zeros(3, h, w)
    x0, y0, x1, y1 = 100, 500, 1600, 1600
    spec = StripSpec(strip_height=1024)
    _, meta_l = extract_side_strip(img, (x0, y0, x1, y1), "left", spec)
    assert meta_l["y_start"] == 500
    _, meta_t = extract_side_strip(img, (x0, y0, x1, y1), "top", spec)
    assert meta_t["x_start"] == 100
