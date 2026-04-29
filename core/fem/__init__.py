# -*- coding: utf-8 -*-
"""FEM package — element, assembly, solver, strength reduction."""

from .element import element_stiffness_cst, element_body_force_cst
from .assembly import Assembler
from .solver import Solver
from .srf import strength_reduction
