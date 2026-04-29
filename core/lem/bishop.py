# -*- coding: utf-8 -*-
"""Bishop's Simplified Method for slope stability."""

from __future__ import annotations
import numpy as np
from typing import Optional
from .slip_surface import SlipSurface, slice_mass


def bishop_fos(
    surface: SlipSurface,
    ground: np.ndarray,
    c: float, phi: float, gamma: float,
    n_slices: int = 30,
    max_iter: int = 50, tol: float = 1e-4,
) -> Optional[float]:
    """Bishop simplified FoS.

    FoS = Σ[cb + W·tan(φ)]/m_α / ΣW·sin(α)
    m_α = cos(α) + sin(α)·tan(φ)/FoS
    """
    mid_x, widths, heights, alphas = slice_mass(surface, ground, n_slices)
    if mid_x is None:
        return None

    phi_rad = np.radians(phi)
    tan_phi = np.tan(phi_rad)
    W = gamma * widths * heights
    sum_W_sin = float(np.sum(W * np.sin(alphas)))
    if sum_W_sin < 1e-12:
        return 999.0

    fos = 1.0
    for _ in range(max_iter):
        m_alpha = np.cos(alphas) + np.sin(alphas) * tan_phi / fos
        m_alpha = np.where(np.abs(m_alpha) < 1e-8, 1e-8, m_alpha)
        numerator = np.sum((c * widths + W * tan_phi) / m_alpha)
        fos_new = float(numerator / sum_W_sin)
        if abs(fos_new - fos) < tol:
            return fos_new
        fos = fos_new
    return fos
