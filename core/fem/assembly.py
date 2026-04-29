# -*- coding: utf-8 -*-
"""Global stiffness matrix and force vector assembly."""

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Dict

from ..mesh import Mesh
from ..material import MohrCoulomb
from .element import element_stiffness_cst, element_body_force_cst, element_stresses_cst


class Assembler:
    """Assemble global K, F and solve for displacements."""

    def __init__(self, mesh: Mesh, materials: Dict[int, MohrCoulomb]):
        self.mesh = mesh
        self.materials = materials
        self.K: csr_matrix | None = None
        self.F: np.ndarray | None = None

    def assemble(self,
                 fixities: Dict[int, List[int]] | None = None,
                 thickness: float = 1.0,
                 plane_strain: bool = True) -> None:
        """Assemble global stiffness matrix and force vector.

        Parameters
        ----------
        fixities     : {node_id: [dof_flags]}, dof_flags eg [1,1] fully fixed
                       If None, all boundary nodes are fixed in both directions.
        thickness    : float
        plane_strain : bool
        """
        n_dof = self.mesh.n_nodes * 2
        self.K = lil_matrix((n_dof, n_dof))
        self.F = np.zeros(n_dof)

        # Pre-compute D matrices for all materials
        D_cache = {
            mid: mat.elastic_matrix(plane_strain=plane_strain)
            for mid, mat in self.materials.items()
        }

        for e_idx in range(self.mesh.n_elements):
            elem = self.mesh.elements[e_idx]
            mid = self.mesh.elem_materials[e_idx]
            mat = self.materials.get(mid)
            if mat is None:
                continue

            coords = self.mesh.nodes[elem]  # (3, 2)
            D = D_cache[mid]

            # Element stiffness
            ke = element_stiffness_cst(coords, D, thickness)

            # Element body force
            fe = element_body_force_cst(coords, mat.gamma, thickness)

            # Assemble into global
            dofs = []
            for node in elem:
                dofs.extend([node * 2, node * 2 + 1])  # u, v

            for i_local, i_global in enumerate(dofs):
                self.F[i_global] += fe[i_local]
                for j_local, j_global in enumerate(dofs):
                    self.K[i_global, j_global] += ke[i_local, j_local]

        # Apply boundary conditions
        if fixities is None:
            fixities = {}
            for bn in self.mesh.boundary_nodes:
                fixities[bn] = [1, 1]  # fix both u and v

        self._apply_fixities(fixities)

    def _apply_fixities(self, fixities: Dict[int, List[int]]) -> None:
        """Apply fixed boundary conditions (zero displacement)."""
        for node_id, flags in fixities.items():
            for direction, fixed in enumerate(flags):
                if fixed:
                    dof = node_id * 2 + direction
                    # Zero the row and column, set diagonal to 1, force to 0
                    self.K[dof, :] = 0
                    self.K[:, dof] = 0
                    self.K[dof, dof] = 1.0
                    self.F[dof] = 0.0

    def solve(self) -> np.ndarray:
        """Solve K*u = F for displacements."""
        self.K = self.K.tocsr()
        u = spsolve(self.K, self.F)
        return u

    def compute_stresses(self, u: np.ndarray,
                         plane_strain: bool = True) -> np.ndarray:
        """Compute element stresses from displacements.

        Returns (n_elem, 3) array of [σx, σy, τxy].
        """
        n_elem = self.mesh.n_elements
        stresses = np.zeros((n_elem, 3))

        D_cache = {
            mid: mat.elastic_matrix(plane_strain=plane_strain)
            for mid, mat in self.materials.items()
        }

        for e_idx in range(n_elem):
            elem = self.mesh.elements[e_idx]
            mid = self.mesh.elem_materials[e_idx]
            if mid not in D_cache:
                continue
            coords = self.mesh.nodes[elem]
            dofs = []
            for node in elem:
                dofs.extend([node * 2, node * 2 + 1])
            u_elem = u[dofs]
            stresses[e_idx] = element_stresses_cst(coords, u_elem, D_cache[mid])

        return stresses
