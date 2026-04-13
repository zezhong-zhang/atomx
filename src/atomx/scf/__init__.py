"""Self-consistent field solvers for multi-electron atoms.

Implements Dirac-Fock (DF) with direct Hartree and non-local exchange
in the B-spline basis, following the DBSR_HF approach.
"""

from atomx.scf.solver import DFSolver, SCFResult
