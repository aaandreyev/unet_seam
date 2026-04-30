"""Unit tests for visual bench triggers and grid composition (no I/O)."""
from __future__ import annotations

import numpy as np
import torch

from scripts.visual_bench_harmonizer import _grid_image, _to_uint8, _triggers


def test_to_uint8_clamps_and_scales():
    t = torch.tensor([[-0.5, 0.0, 0.5], [1.0, 1.5, 0.25]])
    arr = _to_uint8(t)
    assert arr.dtype == np.uint8
    assert arr.min() == 0
    assert arr.max() == 255


def test_grid_image_concatenates_panels():
    a = np.zeros((4, 5, 3), dtype=np.uint8)
    b = np.full((4, 5, 3), 200, dtype=np.uint8)
    g = _grid_image([a, b], gap=2)
    assert g.shape == (4, 5 + 2 + 5, 3)


def test_grid_image_promotes_2d_panels_to_rgb():
    a = np.zeros((4, 5, 3), dtype=np.uint8)
    grayscale = np.full((4, 5), 128, dtype=np.uint8)
    g = _grid_image([a, grayscale], gap=1)
    assert g.shape == (4, 11, 3)


def test_triggers_zero_for_identity_strip():
    """If corrected == input on the inner half, all three triggers should be ~0."""
    h, w = 32, 256
    outer_width = 128
    strip = torch.rand(3, h, w)
    inner = strip[..., outer_width:].clone()
    triggers = _triggers(strip, inner, outer_width, boundary_band_px=24)
    assert triggers["halo_band"] < 1e-6
    assert triggers["outside_drift"] < 1e-6
    # dark_line is a luma diff between band and rest of inner — non-zero in general
    # for a random image, but bounded.
    assert abs(triggers["dark_line"]) < 1.0


def test_triggers_detect_dark_band_at_seam():
    h, w = 32, 256
    outer_width = 128
    strip = torch.full((3, h, w), 0.7)
    strip[..., outer_width:outer_width + 8] = 0.1  # dark band right at the seam
    inner = strip[..., outer_width:].clone()
    triggers = _triggers(strip, inner, outer_width, boundary_band_px=24)
    assert triggers["dark_line"] < -0.05, f"expected negative dark_line, got {triggers['dark_line']}"
