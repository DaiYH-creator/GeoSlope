# -*- coding: utf-8 -*-
"""FEM Solver — thin wrapper around Assembler."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict

from ..mesh import Mesh
from ..material import MohrCoulomb
from .assembly import Assembler


@dataclass
class FEMResult:
    displacements: np.ndarray    # (n_dof,)
    stresses: np.ndarray         # (n_elem, 3)
    converged: bool = True
    iterations: int = 0
    message: str = ""


class Solver:
    """Linear-elastic FEM solver for slope stability."""

    def __init__(self, mesh: Mesh, materials: Dict[int, MohrCoulomb]):
        self.mesh = mesh
        self.materials = materials
        self.assembler = Assembler(mesh, materials)

    def solve(self,
              fixities: Dict[int, list] | None = None,
              thickness: float = 1.0,
              plane_strain: bool = True) -> FEMResult:
        """Assemble and solve linear-elastic FEM."""
        self.assembler.assemble(
            fixities=fixities,
            thickness=thickness,
            plane_strain=plane_strain
        )
        u = self.assembler.solve()
        stresses = self.assembler.compute_stresses(u, plane_strain=plane_strain)
        return FEMResult(
            displacements=u,
            stresses=stresses,
            converged=True,
        )
