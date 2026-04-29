# -*- coding: utf-8 -*-
"""FEM element routines — Constant Strain Triangle (CST)."""

from __future__ import annotations
import numpy as np
from ..material import MohrCoulomb


def element_stiffness_cst(
    coords: np.ndarray,
    D: np.ndarray,
    thickness: float = 1.0
) -> np.ndarray:
    """Compute 6x6 stiffness matrix for a 3-node CST element.

    Parameters
    ----------
    coords    : (3, 2) — [[x1,y1],[x2,y2],[x3,y3]] in CCW order
    D         : (3, 3) — constitutive matrix
    thickness : float  — out-of-plane thickness (plane strain)

    Returns
    -------
    k : (6, 6) stiffness matrix, DOF order [u1,v1,u2,v2,u3,v3]
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Area
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if area < 1e-15:
        return np.zeros((6, 6))

    # B matrix coefficients
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    # B matrix (3 x 6)
    B = np.array([
        [b1, 0,  b2, 0,  b3, 0],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3]
    ]) / (2.0 * area)

    # k = t * area * B^T * D * B
    k = thickness * area * (B.T @ D @ B)
    return k


def element_body_force_cst(
    coords: np.ndarray,
    gamma: float,
    thickness: float = 1.0
) -> np.ndarray:
    """Compute 6-element body force vector (gravity in -y direction).

    Parameters
    ----------
    coords    : (3, 2) element node coordinates
    gamma     : float  unit weight (kN/m³)
    thickness : float

    Returns
    -------
    f : (6,) force vector [f_x1, f_y1, f_x2, f_y2, f_x3, f_y3]
    """
    area = 0.5 * abs(
        coords[0, 0] * (coords[1, 1] - coords[2, 1]) +
        coords[1, 0] * (coords[2, 1] - coords[0, 1]) +
        coords[2, 0] * (coords[0, 1] - coords[1, 1])
    )
    if area < 1e-15:
        return np.zeros(6)

    weight = gamma * area * thickness / 3.0
    f = np.zeros(6)
    f[1] = -weight  # y-dof node 1
    f[3] = -weight  # y-dof node 2
    f[5] = -weight  # y-dof node 3
    return f


def element_stresses_cst(
    coords: np.ndarray,
    u_elem: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """Compute stress vector for CST element.

    Returns σ = [σx, σy, τxy] (3,). Constant over element.
    """
    x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if area < 1e-15:
        return np.zeros(3)
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    B = np.array([
        [b1, 0,  b2, 0,  b3, 0],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3]
    ]) / (2.0 * area)
    return D @ B @ u_elem
