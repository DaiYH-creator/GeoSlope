# -*- coding: utf-8 -*-
"""Comprehensive demo: mesh → FEM → SRF → visualization."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.geometry import SlopeGeometry
from core.material import MohrCoulomb
from core.mesh import generate_mesh
from core.fem.srf import strength_reduction
from core.fem.eplastic import solve_elastic_plastic
from visualization.plot_2d import (
    plot_mesh, plot_deformed, plot_stress, plot_plastic_zone,
    plot_srf_convergence, plot_comprehensive
)

# --- Build geometry ---
slope = SlopeGeometry()
slope.build_slope_example(slope_height=10, slope_angle=45, crest_width=10, toe_width=10)

mat = MohrCoulomb(gamma=20, E=50000, nu=0.3, c=15, phi=25, name="clay")

mesh = generate_mesh(slope, refine_factor=0.5)

# --- Boundary conditions ---
fixities = {}
ymin, ymax = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
xmin, xmax = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
tol = 1e-3 * (xmax - xmin)
for bn in mesh.boundary_nodes:
    x, y = mesh.nodes[bn]
    if abs(y - ymin) < tol:
        fixities[bn] = [1, 1]
    elif abs(x - xmin) < tol or abs(x - xmax) < tol:
        fixities[bn] = [1, 0]

# --- SRF analysis ---
print("Running SRF...")
srf_result = strength_reduction(
    mesh, {0: mat}, fixities=fixities,
    fos_min=0.2, fos_max=5.0, tol=0.01, max_iter=20,
)

fos = srf_result.factor_of_safety
print(f"FoS = {fos:.3f}")

# --- Solve at FoS for best visualization ---
print("Solving at FoS...")
ep_sol = solve_elastic_plastic(
    mesh, {0: mat.reduced(fos)}, fixities=fixities, max_ep_iter=30,
)

# --- Generate visualizations ---
output_dir = "c:/Users/15274/WorkBuddy/Claw/GeoSlope/examples"
plot_comprehensive(mesh, ep_sol, fos, save_dir=output_dir)

# SRF convergence
plot_srf_convergence(
    srf_result.reduction_factors,
    srf_result.max_displacements,
    srf_result.plastic_counts,
    fos,
    save_path=f"{output_dir}/07_srf_convergence.png",
)

print(f"\nVisualizations saved to: {output_dir}/")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        print(f"  {f}")
