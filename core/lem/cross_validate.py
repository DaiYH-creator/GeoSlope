# -*- coding: utf-8 -*-
"""Cross-validation: compare LEM results with FEM Strength Reduction."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

from .slip_surface import (
    build_ground_profile, search_critical_circle, SlipSurface,
)


@dataclass
class CrossValidationResult:
    """Results from multiple methods compared."""
    slope_height: float = 10.0
    slope_angle: float = 45.0
    crest_width: float = 10.0
    toe_width: float = 10.0
    c: float = 15.0
    phi: float = 25.0
    gamma: float = 20.0
    # FEM
    fem_fos: Optional[float] = None
    # LEM
    fellenius_fos: Optional[float] = None
    bishop_fos: Optional[float] = None
    janbu_fos: Optional[float] = None
    mp_fos: Optional[float] = None
    mp_lambda: Optional[float] = None
    # Surfaces
    bishop_surface: Optional[SlipSurface] = None
    janbu_surface: Optional[SlipSurface] = None
    # Stats
    lem_mean: float = 0.0
    lem_std: float = 0.0
    fem_lem_diff_pct: float = 0.0
    methods_agree: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Slope Stability — Cross-Validation Report",
            "=" * 60,
            f"  Geometry: H={self.slope_height:.1f}m  β={self.slope_angle:.0f}°"
            f"  toe={self.toe_width:.0f}m  crest={self.crest_width:.0f}m",
            f"  Material: c={self.c:.1f}kPa  φ={self.phi:.0f}°  γ={self.gamma:.1f}kN/m³",
            "-" * 60,
        ]
        if self.fem_fos is not None:
            lines.append(f"  FEM (SRF):          FoS = {self.fem_fos:.3f}")
        lines.append("  --- Limit Equilibrium Methods ---")
        if self.fellenius_fos is not None:
            lines.append(f"  Fellenius (Ordinary): FoS = {self.fellenius_fos:.3f}")
        if self.bishop_fos is not None:
            lines.append(f"  Bishop Simplified:    FoS = {self.bishop_fos:.3f}")
        if self.janbu_fos is not None:
            lines.append(f"  Janbu Simplified:     FoS = {self.janbu_fos:.3f}")
        if self.mp_fos is not None:
            lines.append(
                f"  Morgenstern-Price:    FoS = {self.mp_fos:.3f}  (λ={self.mp_lambda:.3f})"
            )
        lines.append("-" * 60)

        # LEM consensus
        fos_list = [v for v in [self.bishop_fos, self.janbu_fos, self.mp_fos] if v is not None and v < 999]
        if fos_list:
            lines.append(f"  LEM Consensus (3 methods): {np.mean(fos_list):.3f} ± {np.std(fos_list):.3f}")
        if self.fem_fos is not None and fos_list:
            diff = (self.fem_fos - np.mean(fos_list)) / np.mean(fos_list) * 100
            lines.append(f"  FEM/LEM discrepancy: {diff:+.1f}%")
            if abs(diff) < 10:
                lines.append("  Status: METHODS AGREE — high confidence")
            elif abs(diff) < 25:
                lines.append("  Status: MODERATE discrepancy — review inputs")
            else:
                lines.append("  Status: LARGE discrepancy — check FEM boundary conditions & mesh")
        lines.append("=" * 60)
        return "\n".join(lines)


def cross_validate(
    slope_height: float = 10.0,
    slope_angle: float = 45.0,
    crest_width: float = 10.0,
    toe_width: float = 10.0,
    c: float = 15.0,
    phi: float = 25.0,
    gamma: float = 20.0,
    fem_fos: Optional[float] = None,
    grid_n: int = 30,
) -> CrossValidationResult:
    """Run all LEM methods and compare with FEM."""
    result = CrossValidationResult(
        slope_height=slope_height,
        slope_angle=slope_angle,
        crest_width=crest_width,
        toe_width=toe_width,
        c=c, phi=phi, gamma=gamma,
        fem_fos=fem_fos,
    )

    ground = build_ground_profile(slope_height, slope_angle, crest_width, toe_width)

    # Bishop — full search for critical surface
    bishop = search_critical_circle(
        ground, slope_height, slope_angle, toe_width, crest_width,
        c, phi, gamma, method="bishop",
        n_tangents=grid_n // 5, n_entries=grid_n,
    )
    if bishop.fos < 999:
        result.bishop_fos = bishop.fos
        result.bishop_surface = bishop

    # Janbu — full search
    janbu = search_critical_circle(
        ground, slope_height, slope_angle, toe_width, crest_width,
        c, phi, gamma, method="janbu",
        n_tangents=grid_n // 5, n_entries=grid_n,
    )
    if janbu.fos < 999:
        result.janbu_fos = janbu.fos
        result.janbu_surface = janbu

    # Fellenius — quick evaluation on Bishop's surface
    if result.bishop_surface is not None:
        from .fellenius import fellenius_fos
        result.fellenius_fos = fellenius_fos(
            result.bishop_surface, ground, c, phi, gamma,
        )

    # M-P — on Bishop's surface
    if result.bishop_surface is not None:
        from .morgenstern_price import morgenstern_price_fos
        mp_result = morgenstern_price_fos(
            result.bishop_surface, ground, c, phi, gamma, n_slices=30,
        )
        if mp_result is not None:
            result.mp_fos, result.mp_lambda = mp_result

    return result
