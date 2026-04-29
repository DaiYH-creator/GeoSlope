# -*- coding: utf-8 -*-
"""SRF — Strength Reduction Factor method for slope stability."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict

from ..mesh import Mesh
from ..material import MohrCoulomb
from .eplastic import solve_elastic_plastic, EPResult


@dataclass
class SRFResult:
    factor_of_safety: float = 1.0
    displacements: list = field(default_factory=list)
    stresses: list = field(default_factory=list)
    reduction_factors: list = field(default_factory=list)
    max_displacements: list = field(default_factory=list)
    plastic_counts: list = field(default_factory=list)
    converged: bool = True
    message: str = ""


def strength_reduction(
    mesh: Mesh,
    materials: Dict[int, MohrCoulomb],
    fixities: Dict[int, list] | None = None,
    thickness: float = 1.0,
    plane_strain: bool = True,
    fos_min: float = 0.5,
    fos_max: float = 5.0,
    tol: float = 0.01,
    max_iter: int = 20,
    max_ep_iter: int = 50,
    slope_height: float = 10.0,
) -> SRFResult:
    """SRF analysis with incremental search + bisection.

    Uses incremental scan to find initial [stable, failed] bounds,
    then bisection for precision.
    """
    result = SRFResult()

    # Phase 1: incremental scan to find failure point
    fos_scan = np.arange(fos_min, fos_max + 0.2, 0.2)
    stable_hi = fos_min

    for f in fos_scan:
        sol = _solve_at_factor(
            f, mesh, materials, fixities, thickness,
            plane_strain, max_ep_iter, slope_height,
        )
        if sol.converged:
            stable_hi = f
        else:
            break
    else:
        # All tested factors are stable — extend
        hi = fos_max * 2.0
        while hi < 100.0:
            sol = _solve_at_factor(
                hi, mesh, materials, fixities, thickness,
                plane_strain, max_ep_iter, slope_height,
            )
            if sol.converged:
                stable_hi = hi
                hi *= 2.0
            else:
                break
        else:
            result.factor_of_safety = hi
            result.message = f"FoS > {hi:.1f}"
            return result

    # Found failure boundary: stable_hi is last stable, hi is first failed
    lo = max(fos_min, stable_hi - 0.1)
    failed_found = False
    for f in np.arange(lo + 0.05, fos_max * 2, 0.05):
        sol = _solve_at_factor(
            f, mesh, materials, fixities, thickness,
            plane_strain, max_ep_iter, slope_height,
        )
        if not sol.converged:
            hi = f
            lo = f - 0.05
            failed_found = True
            break

    if not failed_found:
        result.factor_of_safety = stable_hi
        result.message = f"FoS > {stable_hi:.1f}"
        return result

    # Phase 2: bisection
    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        sol = _solve_at_factor(
            mid, mesh, materials, fixities, thickness,
            plane_strain, max_ep_iter, slope_height,
        )

        result.reduction_factors.append(mid)
        md = np.abs(sol.displacements).max() if len(sol.displacements) > 0 else 0
        result.max_displacements.append(md)
        result.plastic_counts.append(len(sol.plastic_elements))

        if sol.converged:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    result.factor_of_safety = lo
    result.message = f"FoS = {lo:.3f} ({i+1} bisect)"
    return result


def _solve_at_factor(
    factor: float,
    mesh: Mesh,
    materials: Dict[int, MohrCoulomb],
    fixities=None,
    thickness=1.0,
    plane_strain=True,
    max_ep_iter=50,
    slope_height=10.0,
) -> EPResult:
    reduced = {mid: mat.reduced(factor) for mid, mat in materials.items()}
    return solve_elastic_plastic(
        mesh, reduced, fixities=fixities, thickness=thickness,
        plane_strain=plane_strain, max_ep_iter=max_ep_iter,
        slope_height=slope_height,
    )
