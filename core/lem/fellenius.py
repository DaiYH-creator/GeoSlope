# -*- coding: utf-8 -*-
"""Fellenius (Ordinary) Method — simplest LEM, no iteration needed."""

from __future__ import annotations
import numpy as np
from typing import Optional
from .slip_surface import SlipSurface, slice_mass


def fellenius_fos(
    surface: SlipSurface,
    ground: np.ndarray,
    c: float, phi: float, gamma: float,
    n_slices: int = 30,
) -> Optional[float]:
    """Fellenius (ordinary method of slices).

    FoS = Σ[c·b/cosα + W·cosα·tanφ] / ΣW·sinα

    Conservative relative to Bishop. No iteration needed.
    """
    mid_x, widths, heights, alphas = slice_mass(surface, ground, n_slices)
    if mid_x is None:
        return None

    tan_phi = np.tan(np.radians(phi))
    cos_a, sin_a = np.cos(alphas), np.sin(alphas)
    W = gamma * widths * heights

    # Base length of each slice: b/cosα = widths/cosα
    base_len = widths / np.where(np.abs(cos_a) < 1e-8, 1e-8, cos_a)

    numerator = np.sum(c * base_len + W * cos_a * tan_phi)
    denominator = np.sum(W * sin_a)

    if denominator < 1e-12:
        return 999.0
    return float(numerator / denominator)
