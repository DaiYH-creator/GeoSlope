# -*- coding: utf-8 -*-
"""Material models — Mohr-Coulomb and derived parameters."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MohrCoulomb:
    """Mohr-Coulomb material model.

    Parameters
    ----------
    gamma : float  — unit weight (kN/m³)
    E : float      — Young's modulus (kPa)
    nu : float     — Poisson ratio
    c : float      — cohesion (kPa)
    phi : float    — friction angle (degrees)
    psi : float    — dilatancy angle (degrees, typically 0 for conservative)
    name : str     — material label
    """
    gamma: float = 20.0
    E: float = 10000.0
    nu: float = 0.35
    c: float = 10.0
    phi: float = 30.0
    psi: float = 0.0
    name: str = "soil"

    @property
    def phi_rad(self) -> float:
        return np.radians(self.phi)

    @property
    def psi_rad(self) -> float:
        return np.radians(self.psi)

    def elastic_matrix(self, plane_strain: bool = True) -> np.ndarray:
        """Elastic constitutive matrix D (3x3).

        For plane strain:
            D = E / ((1+nu)(1-2nu)) *
                [[1-nu,  nu,    0      ],
                 [nu,    1-nu,  0      ],
                 [0,     0,    (1-2nu)/2]]
        """
        if plane_strain:
            factor = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            D = np.array([
                [1.0 - self.nu, self.nu, 0.0],
                [self.nu, 1.0 - self.nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * self.nu) / 2.0]
            ])
        else:
            factor = self.E / (1.0 - self.nu ** 2)
            D = np.array([
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - self.nu) / 2.0]
            ])
        return factor * D

    def reduced(self, factor: float) -> MohrCoulomb:
        """Return a copy with c and tan(phi) reduced by factor (for SRF)."""
        new_phi = np.degrees(np.arctan(np.tan(self.phi_rad) / factor))
        return MohrCoulomb(
            gamma=self.gamma,
            E=self.E,
            nu=self.nu,
            c=self.c / factor,
            phi=new_phi,
            psi=self.psi,
            name=f"{self.name}_SRF={factor:.2f}"
        )

    def yield_function(self, sigma: np.ndarray) -> float:
        """Mohr-Coulomb yield function f(σ).

        f = (σ1 - σ3) - (σ1 + σ3)*sin(φ) - 2*c*cos(φ)
        f > 0 → plastic (failure)

        Uses plane-strain out-of-plane stress: σz = ν(σx + σy)
        """
        s1, s3 = _principal_stresses_ps(sigma, self.nu)
        sp = self.phi_rad
        return (s1 - s3) - (s1 + s3) * np.sin(sp) - 2.0 * self.c * np.cos(sp)


def _principal_stresses_ps(
    sigma: np.ndarray, nu: float = 0.3
) -> tuple:
    """Major and minor principal stresses (geomechanics, compression positive).

    For plane strain, the out-of-plane stress is σz = ν(σx + σy).
    Returns (σ1, σ3) — the maximum and minimum among the three.
    """
    sx, sy, txy = sigma[0], sigma[1], sigma[2]
    # Continuum → geomechanics (compression positive)
    sx_g = -sx
    sy_g = -sy
    txy_g = -txy
    # In-plane principal stresses
    center = (sx_g + sy_g) / 2.0
    radius = np.sqrt(((sx_g - sy_g) / 2.0) ** 2 + txy_g ** 2)
    s_in_1 = center + radius
    s_in_2 = center - radius
    # Out-of-plane: σz = ν(σx + σy) in continuum → ν(sx_g + sy_g) in geomech
    s_out = nu * (sx_g + sy_g)
    # Sort descending: σ1 ≥ σ2 ≥ σ3
    all_s = np.array([s_in_1, s_in_2, s_out])
    all_s.sort()
    return all_s[2], all_s[0]  # σ1, σ3
