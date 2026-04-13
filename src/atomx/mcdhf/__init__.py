"""Multi-Configuration Dirac-Hartree-Fock (MCDHF).

Average-level MCDHF: shared orbitals optimized for weighted average
of configurations, then CI diagonalization for state energies.
"""

from atomx.mcdhf.config import CSF, make_csf_list
from atomx.mcdhf.matrix import build_ci_matrix
from atomx.mcdhf.solver import MCDHFSolver, MCDHFResult
