# -*- coding: utf-8 -*-
"""Janbu's Simplified Method with implicit FoS iteration.

Uses the rigorous force equilibrium formulation with dX=0 assumption.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from .slip_surface import SlipSurface, slice_mass


def janbu_fos(
    surface: SlipSurface,
    ground: np.ndarray,
    c: float, phi: float, gamma: float,
    n_slices: int = 30,
    max_iter: int = 50, tol: float = 1e-4,
) -> Optional[float]:
    """Janbu simplified — implicit force equilibrium.

    For each slice (with dX=0):
        N = (W - c·b·sinα/F) / (cosα + tanφ·sinα/F)
        T = (c·b + N·tanφ) / F

    Force equilibrium:
        Σ T·cosα = Σ N·sinα
    """
    mid_x, widths, heights, alphas = slice_mass(surface, ground, n_slices)
    if mid_x is None:
        return None

    tan_phi = np.tan(np.radians(phi))
    cos_a, sin_a = np.cos(alphas), np.sin(alphas)
    W = gamma * widths * heights

    # d/L for correction factor f0
    L = mid_x[-1] - mid_x[0]
    d = float(np.max(heights))
    dl = d / L if L > 0 else 0

    if phi > 2.0:
        f0 = 1.0 + 0.31 + 0.31 * dl
    else:
        f0 = 1.0 + 0.5 * (dl - 1.4 * dl**2)
    f0 = max(1.0, min(f0, 1.5))

    fos = 1.0
    for _ in range(max_iter):
        # Normal force per slice
        denom = cos_a + tan_phi * sin_a / fos
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        N = (W - c * widths * sin_a / fos) / denom
        N = np.maximum(N, 0.0)

        # Shear resistance
        T = (c * widths + N * tan_phi) / fos

        # Force equilibrium: ΣT·cosα = ΣN·sinα
        sum_T_cos = np.sum(T * cos_a)
        sum_N_sin = np.sum(N * sin_a)

        if sum_N_sin < 1e-12:
            return 999.0
        fos_new = float(sum_T_cos / sum_N_sin)
        fos_new *= f0  # Apply correction factor to FoS

        # Damped update to prevent oscillation
        fos = (fos + fos_new) / 2.0

        if abs(fos_new - fos) < tol:
            return fos_new

    return fos
