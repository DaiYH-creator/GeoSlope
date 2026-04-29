# -*- coding: utf-8 -*-
"""Test: LEM cross-validation against FEM SRF."""

import sys
sys.path.insert(0, "c:/Users/15274/WorkBuddy/Claw/GeoSlope")

import numpy as np
from core.geometry import SlopeGeometry
from core.material import MohrCoulomb
from core.mesh import generate_mesh
from core.fem.srf import strength_reduction
from core.lem.cross_validate import cross_validate


def main():
    # --- FEM ---
    slope = SlopeGeometry()
    slope.build_slope_example(10.0, 45.0, 10.0, 10.0)
    mat = MohrCoulomb(gamma=20.0, E=50000.0, nu=0.30, c=15.0, phi=25.0, name="clay")
    materials = {0: mat}
    mesh = generate_mesh(slope, refine_factor=0.5)

    ymin = mesh.nodes[:, 1].min()
    xmin, xmax = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
    tol_bc = 1e-3 * (xmax - xmin)

    fixities = {}
    for bn in mesh.boundary_nodes:
        x, y = mesh.nodes[bn]
        if abs(y - ymin) < tol_bc:
            fixities[bn] = [1, 1]
        elif abs(x - xmin) < tol_bc:
            fixities[bn] = [1, 0]
        elif abs(x - xmax) < tol_bc:
            fixities[bn] = [1, 0]

    print("Running FEM SRF...")
    srf = strength_reduction(
        mesh=mesh, materials=materials, fixities=fixities,
        thickness=1.0, fos_min=0.5, fos_max=5.0, tol=0.01,
    )
    fem_fos = srf.factor_of_safety

    # --- LEM ---
    print("Running LEM cross-validation...")
    cv = cross_validate(
        slope_height=10.0, slope_angle=45.0,
        crest_width=10.0, toe_width=10.0,
        c=15.0, phi=25.0, gamma=20.0,
        fem_fos=fem_fos, grid_n=35,
    )

    print(cv.summary())

    # --- Plot ---
    print("\nGenerating visualization...")
    _plot(cv)
    print("Done → cv_output.png")


def _plot(cv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ground = np.array([
        [0, 0], [cv.toe_width, 0],
        [cv.toe_width + cv.slope_height / np.tan(np.radians(cv.slope_angle)), cv.slope_height],
        [cv.toe_width + cv.slope_height / np.tan(np.radians(cv.slope_angle)) + cv.crest_width, cv.slope_height],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: slip surfaces
    ax1.set_title("Critical Slip Surfaces", fontsize=12, fontweight="bold")
    ax1.plot(ground[:, 0], ground[:, 1], "k-", lw=2, label="Ground surface")
    ax1.fill_between(ground[:, 0], ground[:, 1], 0, alpha=0.08, color="gray")

    colors = {"Bishop": "#2196F3", "Janbu": "#4CAF50"}
    items = [
        ("Bishop", cv.bishop_surface, cv.bishop_fos),
        ("Janbu", cv.janbu_surface, cv.janbu_fos),
    ]
    for name, surf, fos in items:
        if surf is not None and fos is not None and fos < 999:
            theta = np.linspace(-np.pi/2.5, np.pi/2.5, 200)
            x_arc = surf.xc + surf.r * np.cos(theta)
            y_arc = surf.yc + surf.r * np.sin(theta)
            ax1.plot(x_arc, y_arc, color=colors[name], lw=1.8,
                     label=f"{name}: {fos:.3f}")
            ax1.plot(surf.xc, surf.yc, "+", color=colors[name], ms=8)

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right: FoS comparison
    ax2.set_title("FoS Comparison", fontsize=12, fontweight="bold")
    methods = []
    values = []
    bar_colors = []
    data = [
        ("FEM\n(SRF)", cv.fem_fos, "#E53935"),
        ("Fellenius", cv.fellenius_fos, "#9E9E9E"),
        ("Bishop", cv.bishop_fos, "#2196F3"),
        ("Janbu", cv.janbu_fos, "#4CAF50"),
        ("M-P", cv.mp_fos, "#FF9800"),
    ]
    for name, val, color in data:
        if val is not None and val < 999:
            methods.append(name)
            values.append(val)
            bar_colors.append(color)

    bars = ax2.bar(methods, values, color=bar_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Factor of Safety")
    ax2.set_ylim(0, max(values) * 1.2 if values else 3)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add consensus annotation
    lem_vals = [v for v in [cv.bishop_fos, cv.janbu_fos, cv.mp_fos] if v is not None and v < 999]
    if lem_vals and cv.fem_fos:
        lem_m = np.mean(lem_vals)
        diff = (cv.fem_fos - lem_m) / lem_m * 100
        ax2.axhline(y=lem_m, color="gray", linestyle="--", alpha=0.5, lw=1)
        ax2.text(len(methods)-1, lem_m, f" LEM mean={lem_m:.2f}", fontsize=8,
                 va="bottom", color="gray")

    plt.tight_layout()
    plt.savefig("c:/Users/15274/WorkBuddy/Claw/GeoSlope/cv_output.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
