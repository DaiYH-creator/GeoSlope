# -*- coding: utf-8 -*-
"""Iterative elastic-plastic solver for slope stability.

Damage-based approach:
- Solve linear elastic
- Check Mohr-Coulomb yield
- Reduce stiffness of yielded elements gradually
- Re-solve until convergence or divergence (collapse)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict

from ..mesh import Mesh
from ..material import MohrCoulomb


@dataclass
class EPResult:
    displacements: np.ndarray
    stresses: np.ndarray
    plastic_elements: set
    converged: bool = True
    iterations: int = 0
    max_disp_growth: float = 1.0
    message: str = ""


def solve_elastic_plastic(
    mesh: Mesh,
    materials: Dict[int, MohrCoulomb],
    fixities: Dict[int, list] | None = None,
    thickness: float = 1.0,
    plane_strain: bool = True,
    max_ep_iter: int = 50,
    damage_factor: float = 0.05,
    convergence_tol: float = 0.01,
    disp_divergence_limit: float = 10.0,
    disp_failure_ratio: float = 0.02,
    slope_height: float = 10.0,
) -> EPResult:
    """Elastic-plastic FEM with damage-based softening.

    Parameters
    ----------
    disp_divergence_limit : displacement growth ratio to declare collapse
    disp_failure_ratio   : if max_disp > disp_failure_ratio * slope_height → failed
    """
    n_elem = mesh.n_elements
    result = EPResult(
        displacements=np.array([]),
        stresses=np.array([]),
        plastic_elements=set(),
    )

    D0 = {
        mid: mat.elastic_matrix(plane_strain=plane_strain)
        for mid, mat in materials.items()
    }
    elem_coords = np.array([mesh.nodes[elem] for elem in mesh.elements])

    # Gradual damage: start at 1.0, decrease toward damage_factor
    damage = np.ones(n_elem)
    first_max_disp = None
    prev_plastic_count = 0

    for iteration in range(max_ep_iter):
        from scipy.sparse import lil_matrix
        n_dof = mesh.n_nodes * 2
        K = lil_matrix((n_dof, n_dof))
        F = np.zeros(n_dof)

        for e_idx in range(n_elem):
            elem = mesh.elements[e_idx]
            mid = mesh.elem_materials[e_idx]
            mat = materials.get(mid)
            if mat is None:
                continue

            coords = elem_coords[e_idx]
            D = D0[mid] * damage[e_idx]

            from .element import element_stiffness_cst, element_body_force_cst
            ke = element_stiffness_cst(coords, D, thickness)
            fe = element_body_force_cst(coords, mat.gamma, thickness)

            dofs = []
            for node in elem:
                dofs.extend([node * 2, node * 2 + 1])
            for i_local, i_global in enumerate(dofs):
                F[i_global] += fe[i_local]
                for j_local, j_global in enumerate(dofs):
                    K[i_global, j_global] += ke[i_local, j_local]

        # Boundary conditions
        if fixities is None:
            fixities = {}
            for bn in mesh.boundary_nodes:
                fixities[bn] = [1, 1]
        for node_id, flags in fixities.items():
            for direction, fixed in enumerate(flags):
                if fixed:
                    dof = node_id * 2 + direction
                    K[dof, :] = 0
                    K[:, dof] = 0
                    K[dof, dof] = 1.0
                    F[dof] = 0.0

        from scipy.sparse.linalg import spsolve
        K = K.tocsr()
        try:
            u = spsolve(K, F)
        except Exception:
            result.converged = False
            result.message = "Linear solve failed"
            return result

        # Stresses
        stresses = np.zeros((n_elem, 3))
        for e_idx in range(n_elem):
            elem = mesh.elements[e_idx]
            mid = mesh.elem_materials[e_idx]
            if mid not in D0:
                continue
            coords = elem_coords[e_idx]
            dofs = []
            for node in elem:
                dofs.extend([node * 2, node * 2 + 1])
            u_elem = u[dofs]
            D = D0[mid] * damage[e_idx]
            from .element import element_stresses_cst
            stresses[e_idx] = element_stresses_cst(coords, u_elem, D)

        # Yield check
        plastic_now = set()
        for e_idx in range(n_elem):
            mid = mesh.elem_materials[e_idx]
            mat = materials.get(mid)
            if mat is None:
                continue
            if mat.yield_function(stresses[e_idx]) > 0:
                plastic_now.add(e_idx)

        # Gradual damage application
        for e_idx in range(n_elem):
            if e_idx in plastic_now:
                # Reduce toward damage_factor gradually
                target = damage_factor
                rate = 0.5  # damage rate per iteration
                damage[e_idx] = damage[e_idx] * (1 - rate) + target * rate

        # Displacement checks
        max_disp = np.abs(u).max()
        if first_max_disp is None:
            first_max_disp = max(max_disp, 1e-12)
        disp_ratio = max_disp / first_max_disp

        # Failure: displacement exceeds fraction of slope height
        disp_limit = disp_failure_ratio * slope_height
        if max_disp > disp_limit:
            result.converged = False
            result.displacements = u
            result.stresses = stresses
            result.plastic_elements = plastic_now
            result.iterations = iteration + 1
            result.max_disp_growth = disp_ratio
            result.message = (
                f"Failed iter {iteration+1}: "
                f"disp={max_disp:.3f}m > {disp_limit:.3f}m limit, "
                f"{len(plastic_now)} plastic"
            )
            return result

        # Divergence: displacement growing rapidly
        if disp_ratio > disp_divergence_limit:
            result.converged = False
            result.displacements = u
            result.stresses = stresses
            result.plastic_elements = plastic_now
            result.iterations = iteration + 1
            result.max_disp_growth = disp_ratio
            result.message = (
                f"Diverged iter {iteration+1}: "
                f"disp x{disp_ratio:.1f}, {len(plastic_now)} plastic"
            )
            return result

        # Convergence: plastic zone AND displacement stabilized
        disp_stable = iteration > 3 and (
            iteration == 0 or max_disp < first_max_disp * 1.5
        )
        plastic_stable = len(plastic_now) == prev_plastic_count and iteration > 3
        if plastic_stable and disp_stable:
            result.converged = True
            result.displacements = u
            result.stresses = stresses
            result.plastic_elements = plastic_now
            result.iterations = iteration + 1
            result.max_disp_growth = disp_ratio
            result.message = (
                f"Converged iter {iteration+1}, {len(plastic_now)} plastic, "
                f"disp={max_disp:.4f}m"
            )
            return result

        prev_plastic_count = len(plastic_now)

    # Max iterations — treat as converged only if displacement is bounded
    if max_disp <= disp_limit and disp_ratio < disp_divergence_limit:
        result.converged = True
        result.message = f"Max iterations ({max_ep_iter})"
    else:
        result.converged = False
        result.message = f"Max iter — DISPLACEMENT TOO LARGE"
    result.displacements = u
    result.stresses = stresses
    result.plastic_elements = plastic_now
    result.iterations = max_ep_iter
    result.max_disp_growth = disp_ratio
    return result
