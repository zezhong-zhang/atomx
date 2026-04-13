"""Dirac equation solver using B-spline basis expansion."""

from atomx.dirac.matrix import build_dirac_matrices
from atomx.dirac.solver import solve_dirac, DiracResult
from atomx.dirac.hydrogenic import hydrogenic_dirac_energy
