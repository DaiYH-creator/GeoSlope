# -*- coding: utf-8 -*-
"""Test: simple homogeneous slope with SRF analysis."""

import sys
sys.path.insert(0, "c:/Users/15274/WorkBuddy/Claw/GeoSlope")

import numpy as np
from core.geometry import SlopeGeometry, Point, Polygon
from core.material import MohrCoulomb
from core.mesh import generate_mesh
from core.fem.srf import strength_reduction


def test_simple_slope():
    slope = SlopeGeometry()
    slope.build_slope_example(
        slope_height=10.0,
        slope_angle=45.0,
        crest_width=10.0,
        toe_width=10.0,
    )

    mat = MohrCoulomb(
        gamma=20.0, E=50000.0, nu=0.30, c=15.0, phi=25.0, name="clay",
    )
    materials = {0: mat}

    mesh = generate_mesh(slope, refine_factor=0.5)
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"Boundary nodes: {len(mesh.boundary_nodes)}")

    # Smart boundary conditions: bottom fixed, sides roller, top free
    fixities = {}
    ymin, ymax = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
    xmin, xmax = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
    tol = 1e-3 * (xmax - xmin)

    for bn in mesh.boundary_nodes:
        x, y = mesh.nodes[bn]
        if abs(y - ymin) < tol:
            fixities[bn] = [1, 1]           # bottom: fully fixed
        elif abs(x - xmin) < tol:
            fixities[bn] = [1, 0]           # left side: fix x
        elif abs(x - xmax) < tol:
            fixities[bn] = [1, 0]           # right side: fix x
        # top and slope face: free

    print(f"Fixities: {len(fixities)} nodes constrained")

    result = strength_reduction(
        mesh=mesh, materials=materials, fixities=fixities,
        thickness=1.0, fos_min=0.5, fos_max=5.0, tol=0.01, max_iter=20,
    )

    print(f"\n--- SRF Result ---")
    print(f"Factor of Safety: {result.factor_of_safety:.3f}")
    print(f"Message: {result.message}")
    print(f"Reduction factors tested: {[f'{rf:.3f}' for rf in result.reduction_factors]}")
    print(f"Max displacements (m): {[f'{md:.6f}' for md in result.max_displacements]}")

    return result


if __name__ == "__main__":
    test_simple_slope()
