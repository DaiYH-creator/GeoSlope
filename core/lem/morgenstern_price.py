# -*- coding: utf-8 -*-
"""Morgenstern-Price Method — GLE formulation with half-sine f(x).

Iterates on (FoS, lambda) to satisfy both force and moment equilibrium.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .slip_surface import SlipSurface, slice_mass


def morgenstern_price_fos(
    surface: SlipSurface,
    ground: np.ndarray,
    c: float, phi: float, gamma: float,
    n_slices: int = 30,
    max_iter: int = 80, tol: float = 1e-4,
) -> Optional[Tuple[float, float]]:
    """GLE M-P method. Returns (FoS, lambda)."""
    mid_x, widths, heights, alphas = slice_mass(surface, ground, n_slices)
    if mid_x is None or len(mid_x) < 3:
        return None

    tan_phi = np.tan(np.radians(phi))
    cos_a, sin_a = np.cos(alphas), np.sin(alphas)
    W = gamma * widths * heights
    n = len(mid_x)

    # f(x) at slice edges: half-sine
    f_edge = np.sin(np.pi * np.linspace(0, 1, n + 1))

    lam = 0.0
    fos = 1.0
    fos_best = fos

    for _outer in range(max_iter):
        fos_prev = fos

        # Forward recursion for interslice forces E
        E = np.zeros(n + 1)
        for i in range(n):
            bi, wi, cai, sai = widths[i], W[i], cos_a[i], sin_a[i]
            fi, fi1 = f_edge[i], f_edge[i+1]

            # Solve slice equilibrium iteratively
            Ni = wi * cai
            for _ in range(15):
                Ti = (c * bi + Ni * tan_phi) / fos
                dX = wi - Ti * sai - Ni * cai
                denom = lam * fi1
                if abs(denom) < 1e-12:
                    denom = 1e-12
                E[i+1] = (lam * fi * E[i] + dX) / denom
                if E[i+1] < 0:
                    E[i+1] = 0.0
                dE = E[i+1] - E[i]
                Ni = (dE + Ti * cai) / sai if abs(sai) > 1e-10 else wi / cai
                Ni = max(Ni, 0.0)

        # Force equilibrium FoS
        sum_r = np.sum(c * widths + W * cos_a * tan_phi)
        sum_d = np.sum(W * sin_a)
        fos_f = sum_r / sum_d if sum_d > 1e-12 else 999.0

        # Moment equilibrium FoS
        sum_mr = np.sum((c * widths + W * cos_a * tan_phi) * surface.r)
        sum_md = np.sum(W * (mid_x - surface.xc))
        fos_m = sum_mr / sum_md if abs(sum_md) > 1e-10 else 999.0

        fos = (fos_f + fos_m) / 2.0

        # Update lambda
        lam += 0.05 * (fos_m - fos_f) / max(fos, 0.1)
        lam = max(-0.5, min(lam, 0.5))

        if abs(fos_f - fos_m) < 0.01 * fos:
            fos_best = fos
            break

        if abs(fos - fos_prev) < tol:
            fos_best = fos
            break

    return (fos_best, lam)
