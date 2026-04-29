import torch
import pytest

from src.infer.merge_bands import build_seam_local_weight_map, build_side_weight_map, merge_side_deltas


def test_side_weight_map_is_non_negative():
    mask = torch.ones(1, 1, 16, 16)
    weight = build_side_weight_map(mask, "left")
    assert weight.min() >= 0
    assert weight.max() <= 1


def test_one_side_only_delta_merge():
    mask = torch.ones(1, 1, 8, 8)
    deltas = {"left": torch.ones(1, 3, 8, 8)}
    merged, _ = merge_side_deltas(deltas, mask)
    assert torch.allclose(merged, torch.ones_like(merged))


def test_corner_fusion_does_not_amplify_delta():
    mask = torch.ones(1, 1, 8, 8)
    side_deltas = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    merged, _ = merge_side_deltas(side_deltas, mask)
    assert float(merged.abs().max()) <= 0.2 + 1e-6


def test_confidence_weighting_prefers_higher_confidence_side():
    mask = torch.ones(1, 1, 8, 8)
    side_deltas = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    side_confidences = {
        "left": torch.full((1, 1, 8, 8), 0.9),
        "top": torch.full((1, 1, 8, 8), 0.1),
    }
    merged, weights = merge_side_deltas(side_deltas, mask, side_confidences=side_confidences)
    assert float(weights["left"][..., 0, 0]) > float(weights["top"][..., 0, 0])
    assert float(merged[..., 0, 0].mean()) > 0.15


def test_seam_local_left_weight_stronger_near_bbox_left_seam():
    """Weights must follow the mask seam (bbox), not the image border.
    Check is done at the midpoint row (y=35) which is well away from corners."""
    h, w = 64, 64
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, 20:50, 20:50] = 1.0
    bbox = (20, 20, 50, 50)
    wl = build_seam_local_weight_map(mask, bbox, "left", inner_width=8)
    # Row 35 is the vertical midpoint — far from both y0=20 and y1=50 corners
    assert float(wl[0, 0, 35, 20]) > float(wl[0, 0, 35, 27])  # seam at x=20, decay into interior


def test_merge_seam_resolves_strong_corner_disagreement_with_winner():
    """Large disagreement at the seam midpoint (away from corners) must pick the winner side."""
    h, w = 64, 64
    # Use a bbox large enough that the midpoint row is well clear of corners
    bbox = (10, 10, 54, 54)
    mask = _bbox_mask(h, w, *bbox)
    iw = 8
    # Only left band has a delta; left wins at the seam-midpoint pixel (y=32, x=10)
    left_delta = torch.zeros(1, 3, h, w)
    left_delta[:, :, 10:54, 10:18] = 0.2
    top_delta = torch.zeros(1, 3, h, w)
    top_delta[:, :, 10:18, 10:54] = -0.2
    merged, _ = merge_side_deltas(
        {"left": left_delta, "top": top_delta},
        mask,
        bbox=bbox,
        inner_width=iw,
        corner_disagreement_threshold=0.05,
    )
    # At seam midpoint (y=32, x=10): top weight is 0 (far from top seam), left has all the weight
    # → merged must be close to left_delta = +0.2
    assert float(merged[0, 0, 32, 10].abs()) > 0.15


# ── Corner tapering tests ─────────────────────────────────────────────────────

def _bbox_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    m = torch.zeros(1, 1, h, w)
    m[:, :, y0:y1, x0:x1] = 1.0
    return m


@pytest.mark.parametrize("side,corner_px,mid_px", [
    # bbox=(20,20,50,50), inner_width=32 → corner_taper_px = max(8,min(32,32//4))=8
    # corner pixel is the bbox corner; mid pixel is seam midpoint (far from both corners)
    ("left",   (20, 20), (35, 20)),
    ("right",  (20, 49), (35, 49)),
    ("top",    (20, 20), (20, 35)),
    ("bottom", (49, 20), (49, 35)),
])
def test_corner_weight_is_zero_at_bbox_corner(side, corner_px, mid_px):
    """Weight at the bbox corner pixel must be zero after corner tapering."""
    h, w = 70, 70
    bbox = (20, 20, 50, 50)
    mask = _bbox_mask(h, w, *bbox)
    iw = 32  # inner_width=32 → cpx = max(8,min(32,8)) = 8
    wmap = build_seam_local_weight_map(mask, bbox, side, inner_width=iw)
    cy, cx = corner_px
    assert float(wmap[0, 0, cy, cx]) == pytest.approx(0.0, abs=1e-5), (
        f"{side}: corner weight at ({cy},{cx}) should be 0"
    )


@pytest.mark.parametrize("side,corner_px,mid_px", [
    ("left",   (20, 20), (35, 20)),
    ("right",  (20, 49), (35, 49)),
    ("top",    (20, 20), (20, 35)),
    ("bottom", (49, 20), (49, 35)),
])
def test_seam_midpoint_weight_is_positive(side, corner_px, mid_px):
    """Weight at the midpoint of a seam (far from corners) must be positive."""
    h, w = 70, 70
    bbox = (20, 20, 50, 50)
    mask = _bbox_mask(h, w, *bbox)
    iw = 32
    wmap = build_seam_local_weight_map(mask, bbox, side, inner_width=iw)
    my, mx = mid_px
    assert float(wmap[0, 0, my, mx]) > 0.5, (
        f"{side}: midpoint weight at ({my},{mx}) should be > 0.5"
    )


def test_corner_weight_less_than_midpoint_for_all_sides():
    """Corner weight must be strictly less than midpoint weight for every side.
    inner_width=32 → cpx=8; corner is zero, midpoint (at the bbox vertical/horizontal centre)
    is near the seam and far from both ends so corner-taper = 1.0 there."""
    h, w = 200, 200
    bbox = (50, 50, 150, 150)
    mask = _bbox_mask(h, w, *bbox)
    iw = 32
    cases = {
        "left":   {"corner": (50, 50),  "mid": (100, 50)},
        "right":  {"corner": (50, 149), "mid": (100, 149)},
        "top":    {"corner": (50, 50),  "mid": (50, 100)},
        "bottom": {"corner": (149, 50), "mid": (149, 100)},
    }
    for side, pts in cases.items():
        wmap = build_seam_local_weight_map(mask, bbox, side, inner_width=iw)
        cy, cx = pts["corner"]
        my, mx = pts["mid"]
        w_corner = float(wmap[0, 0, cy, cx])
        w_mid = float(wmap[0, 0, my, mx])
        assert w_corner < w_mid, f"{side}: corner weight {w_corner:.4f} >= midpoint weight {w_mid:.4f}"


def test_corner_tapering_kills_blowup_at_high_strength():
    """
    With opposing corner deltas and strength=10, the merged result at the
    actual bbox corner must stay close to zero because corner tapering drives
    both side weights to zero there.
    """
    h, w = 200, 200
    x0, y0, x1, y1 = 50, 50, 150, 150
    mask = _bbox_mask(h, w, x0, y0, x1, y1)
    bbox = (x0, y0, x1, y1)
    iw = 20

    # Opposing deltas: left says +0.1, top says -0.1 everywhere in mask
    left_delta = torch.zeros(1, 3, h, w)
    left_delta[:, :, y0:y1, x0:x0+iw] = 0.1
    top_delta = torch.zeros(1, 3, h, w)
    top_delta[:, :, y0:y0+iw, x0:x1] = -0.1

    merged, _ = merge_side_deltas(
        {"left": left_delta, "top": top_delta},
        mask,
        bbox=bbox,
        inner_width=iw,
        corner_disagreement_threshold=0.03,
    )

    # Actual corner pixel — both weights are 0 → merged must be ~0
    corner_val = float(merged[0, :, y0, x0].abs().max())
    assert corner_val < 1e-4, f"corner blowup: merged={corner_val:.6f} at ({y0},{x0})"

    # Scale by strength=10 and check the corner pixel is still safe
    corrected_corner = float((merged * 10.0)[0, :, y0, x0].abs().max())
    assert corrected_corner < 1e-3, f"corner blowup at strength=10: {corrected_corner:.6f}"


def test_merged_delta_bounded_by_max_side_delta_everywhere():
    """Merged result must never exceed the maximum delta magnitude of any individual side."""
    h, w = 64, 64
    bbox = (10, 10, 54, 54)
    mask = _bbox_mask(h, w, *bbox)
    iw = 10

    left_delta = torch.zeros(1, 3, h, w)
    left_delta[:, :, 10:54, 10:20] = 0.3
    top_delta = torch.zeros(1, 3, h, w)
    top_delta[:, :, 10:20, 10:54] = 0.2

    merged, _ = merge_side_deltas(
        {"left": left_delta, "top": top_delta},
        mask,
        bbox=bbox,
        inner_width=iw,
    )
    assert float(merged.abs().max()) <= 0.3 + 1e-5


def test_perpendicular_sides_do_not_overlap_at_corner():
    """
    After corner tapering, the two perpendicular weight maps must both be zero
    at the exact bbox corner pixel. Away from corners each side should dominate
    on its own stretch of the seam.

    inner_width=32 → cpx=8.  The taper only covers the first/last 8 px along
    the seam, so pixels ≥8 px from the corner already have full weight.
    """
    h, w = 100, 100
    bbox = (20, 20, 80, 80)
    mask = _bbox_mask(h, w, *bbox)
    iw = 32  # cpx = max(8, min(32, 32//4)) = 8
    x0, y0 = 20, 20

    wl = build_seam_local_weight_map(mask, bbox, "left", inner_width=iw)
    wt = build_seam_local_weight_map(mask, bbox, "top", inner_width=iw)

    # Exact corner → both zero
    assert float(wl[0, 0, y0, x0]) < 1e-4
    assert float(wt[0, 0, y0, x0]) < 1e-4

    # Midpoint of left seam (y=50, x=20): left weight is high, top weight is 0
    assert float(wl[0, 0, 50, x0]) > 0.5
    assert float(wt[0, 0, 50, x0]) < 1e-4  # far from top seam (y=20)

    # Midpoint of top seam (y=20, x=50): top weight is high, left weight is 0
    assert float(wl[0, 0, y0, 50]) < 1e-4  # far from left seam (x=20)
    assert float(wt[0, 0, y0, 50]) > 0.5
