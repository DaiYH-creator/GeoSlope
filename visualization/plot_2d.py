# -*- coding: utf-8 -*-
"""2D visualization for GeoSlope FEM results."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon as MPolygon
from typing import Optional

try:
    from ..core.mesh import Mesh
    from ..core.fem.eplastic import EPResult
except ImportError:
    from core.mesh import Mesh
    from core.fem.eplastic import EPResult


# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_mesh(
    mesh: Mesh,
    title: str = "有限元网格",
    show_nodes: bool = False,
    figsize=(12, 8),
    save_path: Optional[str] = None,
):
    """Plot the FE mesh."""
    fig, ax = plt.subplots(figsize=figsize)
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    ax.triplot(tri, color='black', lw=0.5, alpha=0.6)

    if show_nodes:
        ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 'r.', ms=2)

    # Highlight boundary
    for bn in mesh.boundary_nodes:
        ax.plot(mesh.nodes[bn, 0], mesh.nodes[bn, 1], 'bo', ms=3)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_deformed(
    mesh: Mesh,
    disp: np.ndarray,
    scale: float = 50.0,
    title: str = "变形图（放大 ×50）",
    figsize=(12, 8),
    save_path: Optional[str] = None,
):
    """Plot deformed mesh overlaid on original."""
    fig, ax = plt.subplots(figsize=figsize)

    # Original mesh (ghost)
    tri_orig = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    ax.triplot(tri_orig, color='gray', lw=0.3, alpha=0.4)

    # Deformed nodes
    disp_2d = disp.reshape(-1, 2)
    deformed_nodes = mesh.nodes + disp_2d * scale

    tri_def = mtri.Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1], mesh.elements)
    ax.triplot(tri_def, color='red', lw=0.8, alpha=0.8)

    # Displacement vectors
    step = max(1, mesh.n_nodes // 200)
    ax.quiver(
        mesh.nodes[::step, 0], mesh.nodes[::step, 1],
        disp_2d[::step, 0] * scale, disp_2d[::step, 1] * scale,
        angles='xy', scale_units='xy', scale=1, color='blue',
        width=0.002, alpha=0.7
    )

    ax.legend(['原始网格', '变形网格', '位移矢量'], loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_stress(
    mesh: Mesh,
    stresses: np.ndarray,
    component: str = "sigma_y",
    title: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    figsize=(12, 8),
    save_path: Optional[str] = None,
):
    """Plot stress contour.

    Parameters
    ----------
    component : "sigma_x", "sigma_y", "tau_xy", "sigma_vm" (von Mises)
    """
    idx = {"sigma_x": 0, "sigma_y": 1, "tau_xy": 2}[component] if component != "sigma_vm" else -1

    if idx == -1:
        sx, sy, txy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
        vals = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    else:
        vals = stresses[:, idx]

    if title is None:
        labels = {"sigma_x": "σ_x", "sigma_y": "σ_y", "tau_xy": "τ_xy", "sigma_vm": "von Mises"}
        title = f"应力分布 — {labels.get(component, component)} (kPa)"

    fig, ax = plt.subplots(figsize=figsize)
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    # Element-based color plot
    tcf = ax.tripcolor(tri, facecolors=vals, cmap=cmap, alpha=0.9, shading='flat')
    cbar = fig.colorbar(tcf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('kPa')

    ax.triplot(tri, color='black', lw=0.2, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_plastic_zone(
    mesh: Mesh,
    plastic_elements: set,
    title: str = "塑性区分布",
    figsize=(12, 8),
    save_path: Optional[str] = None,
):
    """Highlight plastic (yielded) elements in red."""
    fig, ax = plt.subplots(figsize=figsize)

    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    # Use numeric values: 1=plastic, 0=elastic
    vals = np.zeros(mesh.n_elements, dtype=float)
    for e in plastic_elements:
        vals[e] = 1.0

    tcf = ax.tripcolor(tri, facecolors=vals, cmap='Reds', alpha=0.8,
                       vmin=0, vmax=1, shading='flat')
    ax.triplot(tri, color='black', lw=0.2, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"{title} ({len(plastic_elements)}/{mesh.n_elements} 单元)")

    # Legend
    from matplotlib.patches import Patch
    ax.legend([
        Patch(facecolor='red', alpha=0.8),
        Patch(facecolor='lightgray', alpha=0.8),
    ], ['塑性区', '弹性区'], loc='upper right', fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_srf_convergence(
    factors: list,
    max_disps: list,
    plastic_counts: list,
    fos: float,
    title: str = "强度折减收敛曲线",
    figsize=(14, 5),
    save_path: Optional[str] = None,
):
    """Plot SRF convergence: FoS vs displacement and plastic count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Displacement
    ax1.plot(factors, max_disps, 'b-o', ms=5)
    ax1.axvline(fos, color='r', ls='--', lw=1.5, label=f'FoS = {fos:.3f}')
    ax1.set_xlabel('折减系数')
    ax1.set_ylabel('最大位移 (m)')
    ax1.set_title('位移-折减系数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plastic count
    ax2.plot(factors, plastic_counts, 'r-s', ms=5)
    ax2.axvline(fos, color='b', ls='--', lw=1.5, label=f'FoS = {fos:.3f}')
    ax2.set_xlabel('折减系数')
    ax2.set_ylabel('塑性单元数')
    ax2.set_title('塑性区-折减系数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_comprehensive(
    mesh: Mesh,
    result: EPResult,
    fos: float,
    save_dir: str = ".",
):
    """Generate comprehensive visualization suite."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    disp_2d = result.displacements.reshape(-1, 2)

    plot_mesh(mesh, title="有限元网格",
              save_path=os.path.join(save_dir, "01_mesh.png"))
    plot_deformed(mesh, result.displacements, scale=50,
                  title=f"变形图（FoS={fos:.2f}）",
                  save_path=os.path.join(save_dir, "02_deformed.png"))
    plot_stress(mesh, result.stresses, "sigma_y",
                title=f"竖向应力 σ_y (kPa) — FoS={fos:.2f}",
                save_path=os.path.join(save_dir, "03_sigma_y.png"))
    plot_stress(mesh, result.stresses, "tau_xy",
                title=f"剪应力 τ_xy (kPa) — FoS={fos:.2f}",
                save_path=os.path.join(save_dir, "04_tau_xy.png"))
    plot_stress(mesh, result.stresses, "sigma_vm",
                title=f"von Mises 应力 (kPa) — FoS={fos:.2f}",
                save_path=os.path.join(save_dir, "05_von_mises.png"))
    plot_plastic_zone(mesh, result.plastic_elements,
                      title=f"塑性区 — FoS={fos:.2f}",
                      save_path=os.path.join(save_dir, "06_plastic.png"))
