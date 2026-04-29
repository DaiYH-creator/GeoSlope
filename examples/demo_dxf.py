# -*- coding: utf-8 -*-
"""Demo: create DXF → import → mesh → FEM → visualize."""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezdxf
import numpy as np
from core.geometry import SlopeGeometry, Point
from core.material import MohrCoulomb
from core.mesh import generate_mesh
from core.fem.srf import strength_reduction
from core.fem.eplastic import solve_elastic_plastic
from core.io.dxf_reader import read_dxf
from visualization.plot_2d import plot_comprehensive, plot_srf_convergence

# --- Create sample DXF ---
def create_sample_dxf(path: str):
    """Create a slope geometry DXF for testing."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Outer boundary: 30x15 rectangle
    pts = [(0, 0), (30, 0), (30, 15), (0, 15)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': 'BOUNDARY'})

    doc.saveas(path)
    print(f"Sample DXF saved: {path}")

# --- Main ---
dxf_path = os.path.join(tempfile.gettempdir(), "slope_sample.dxf")
create_sample_dxf(dxf_path)

# Import
slope = read_dxf(dxf_path, material_map={'BOUNDARY': 'clay'})
print(f"Imported: {len(slope.polygons)} polygons")

# Material
mat = MohrCoulomb(gamma=20, E=50000, nu=0.3, c=15, phi=25, name="clay")
mesh = generate_mesh(slope, refine_factor=0.5)

# Boundary conditions
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

# SRF
print("Running SRF...")
srf_result = strength_reduction(
    mesh, {0: mat}, fixities=fixities, fos_min=0.2, fos_max=5.0, tol=0.01, max_iter=20,
)
fos = srf_result.factor_of_safety
print(f"FoS = {fos:.3f}")

# Solve at FoS
ep_sol = solve_elastic_plastic(
    mesh, {0: mat.reduced(fos)}, fixities=fixities, max_ep_iter=30,
)

# Visualize
output_dir = os.path.join(os.path.dirname(__file__), "dxf_demo")
os.makedirs(output_dir, exist_ok=True)
plot_comprehensive(mesh, ep_sol, fos, save_dir=output_dir)

print(f"\nDone. Output: {output_dir}/")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")
