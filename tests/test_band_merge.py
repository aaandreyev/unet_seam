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


def test_one_side_seam_merge_applies_inner_falloff():
    h, w = 48, 48
    bbox = (8, 8, 40, 40)
    mask = _bbox_mask(h, w, *bbox)
    delta = torch.zeros(1, 3, h, w)
    delta[:, :, 8:40, 8:24] = 1.0
    merged, weights = merge_side_deltas(
        {"left": delta},
        mask,
        bbox=bbox,
        inner_width=16,
        blend_falloff_px=4,
    )
    row = 24
    assert float(weights["left"][0, 0, row, 8]) > 0.99
    assert float(weights["left"][0, 0, row, 10]) < float(weights["left"][0, 0, row, 8])
    assert float(weights["left"][0, 0, row, 12]) < 0.3
    assert float(merged[0, :, row, 23].mean()) < 0.3


def test_corner_fusion_does_not_amplify_delta():
    mask = torch.ones(1, 1, 8, 8)
    side_deltas = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    merged, _ = merge_side_deltas(side_deltas, mask)
    assert float(merged.abs().max()) <= 0.2 + 1e-6


def test_confidence_weighting_scales_merged_result():
    """Confidence attenuates effective delta: result lies between conf-scaled contributions.

    New behaviour (stage9 fix): confidence is applied to the delta BEFORE spatial blending,
    not to the spatial weight. This makes single-side and multi-side consistent — half
    confidence produces half correction in both cases.

    For left (delta=0.2, conf=0.9) and top (delta=0.1, conf=0.1) with equal spatial weights:
    eff_left=0.18, eff_top=0.01 → merged ≈ (0.18+0.01)/2 = 0.095 (between the two).
    """
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
    # Spatial weights are now equal (pure spatial, no confidence) for a simple uniform mask.
    assert float(weights["left"][..., 0, 0]) == pytest.approx(float(weights["top"][..., 0, 0]), abs=1e-5)
    # Merged result is between the effective deltas (conf_top*d_top=0.01, conf_left*d_left=0.18)
    merged_val = float(merged[..., 0, 0].mean())
    assert merged_val > 0.01, "Merged must be above low-confidence side contribution"
    assert merged_val < 0.18, "Merged must be below high-confidence-side contribution (spatial avg)"


def test_zero_confidence_zeroes_effective_delta_not_spatial_weight():
    """Zero confidence zeroes the side's effective delta; spatial weight still exists.

    The returned weights dict contains SPATIAL weights (pure geometry, no confidence).
    A zero-confidence side still has a non-zero spatial weight map — it just contributes
    a zero effective delta, so it has no effect on the merged result.
    """
    mask = torch.ones(1, 1, 8, 8)
    side_deltas = {
        "left": torch.full((1, 3, 8, 8), 0.2),
        "top": torch.full((1, 3, 8, 8), 0.1),
    }
    side_confidences = {
        "left": torch.zeros(1, 1, 8, 8),
        "top": torch.ones(1, 1, 8, 8),
    }
    merged, weights = merge_side_deltas(side_deltas, mask, side_confidences=side_confidences)
    # Spatial weight for the zero-confidence side still exists (pure geometry)
    assert float(weights["left"].max()) > 0, "Spatial weight is independent of confidence"
    # Zero-confidence left side has zero effective delta → result driven by top alone
    assert float(merged[..., 0, 0].mean()) < 0.12


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


def test_short_blend_falloff_decays_immediately_and_faster_than_full_width():
    h, w = 64, 64
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, 16:48, 16:48] = 1.0
    bbox = (16, 16, 48, 48)
    full_width = build_seam_local_weight_map(mask, bbox, "left", inner_width=16)
    short_falloff = build_seam_local_weight_map(mask, bbox, "left", inner_width=16, blend_falloff_px=4)
    row = 32
    assert float(short_falloff[0, 0, row, 16]) == pytest.approx(1.0, abs=1e-6)
    assert float(short_falloff[0, 0, row, 17]) < 1.0
    assert float(short_falloff[0, 0, row, 20]) < 1e-6
    assert float(short_falloff[0, 0, row, 18]) < float(full_width[0, 0, row, 18])


def test_inner_falloff_sets_immediate_fade_length_from_seam():
    """inner_falloff_px controls fade length starting at the seam, with no 1.0 plateau."""
    h, w = 128, 256
    iw, falloff = 96, 32
    x0, y0, x1, y1 = 64, 16, 160, 112
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, y0:y1, x0:x1] = 1.0
    bbox = (x0, y0, x1, y1)
    wmap = build_seam_local_weight_map(mask, bbox, "left", inner_width=iw, blend_falloff_px=falloff)
    row = (y0 + y1) // 2
    assert float(wmap[0, 0, row, x0]) > 0.99
    assert float(wmap[0, 0, row, x0 + 1]) < 1.0
    # Taper zone: weight at 50% into the taper should be between 0 and 1.
    x_mid_taper = x0 + falloff // 2
    assert 0.0 < float(wmap[0, 0, row, x_mid_taper]) < 1.0
    # Fade end: around falloff pixels from the seam, weight should be near zero.
    assert float(wmap[0, 0, row, x0 + falloff]) < 0.05
    # Beyond falloff (still inside mask): weight should remain 0.
    if x0 + falloff + 1 < x1:
        assert float(wmap[0, 0, row, x0 + falloff + 1]) == pytest.approx(0.0, abs=1e-5)


def test_inner_falloff_zero_gives_full_inner_width_gradient():
    """blend_falloff_px=None falls back to inner_width, including single-side seam merges."""
    h, w = 64, 128
    x0, y0, x1, y1 = 32, 8, 96, 56
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, y0:y1, x0:x1] = 1.0
    bbox = (x0, y0, x1, y1)
    iw = 48
    wmap_default = build_seam_local_weight_map(mask, bbox, "left", inner_width=iw, blend_falloff_px=None)
    row = (y0 + y1) // 2
    assert float(wmap_default[0, 0, row, x0]) > 0.95
    assert float(wmap_default[0, 0, row, x0 + iw // 2]) < float(wmap_default[0, 0, row, x0])


def test_one_side_seam_merge_uses_gradient_when_falloff_is_none():
    h, w = 48, 48
    bbox = (8, 8, 40, 40)
    mask = _bbox_mask(h, w, *bbox)
    delta = torch.zeros(1, 3, h, w)
    delta[:, :, 8:40, 8:24] = 1.0
    merged, weights = merge_side_deltas(
        {"left": delta},
        mask,
        bbox=bbox,
        inner_width=16,
        blend_falloff_px=None,
    )
    row = 24
    assert float(weights["left"][0, 0, row, 8]) > 0.99
    assert float(weights["left"][0, 0, row, 16]) < float(weights["left"][0, 0, row, 8])
    assert float(merged[0, :, row, 23].mean()) < 0.1


def test_merge_seam_blends_corner_disagreement_without_amplification():
    """Without arbitration heuristics, merge stays a bounded weighted blend."""
    h, w = 64, 64
    bbox = (10, 10, 54, 54)
    mask = _bbox_mask(h, w, *bbox)
    iw = 8
    left_delta = torch.zeros(1, 3, h, w)
    left_delta[:, :, 10:54, 10:18] = 0.2
    top_delta = torch.zeros(1, 3, h, w)
    top_delta[:, :, 10:18, 10:54] = -0.2
    merged, _ = merge_side_deltas(
        {"left": left_delta, "top": top_delta},
        mask,
        bbox=bbox,
        inner_width=iw,
    )
    assert float(merged.abs().max()) <= 0.2 + 1e-6


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
