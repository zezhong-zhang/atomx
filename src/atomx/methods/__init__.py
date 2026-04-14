"""Theory-level method objects for AtomX.

Each method encapsulates the physics choices for a calculation,
ordered by the pedagogical progression of atomic theory:

1. HydrogenicSchrodinger — exact non-relativistic hydrogen-like
2. HydrogenicDirac — exact relativistic hydrogen-like (Sommerfeld)
3. HartreeFock — non-relativistic many-electron (LS-coupling)
4. RelativisticHartreeFock — perturbative scalar+SO corrections (Breit-Pauli)
5. DiracFock — full relativistic Dirac-Fock (jj-coupling)
6. MCDHF — multi-configuration with electron correlation

Usage:
    from atomx import Atom, DiracFock, Exchange
    result = Atom(Z=10).solve(DiracFock(breit=True))
"""

from atomx.methods.base import Method
from atomx.methods.hydrogenic import HydrogenicSchrodinger, HydrogenicDirac
from atomx.methods.hartree_fock import HartreeFock
from atomx.methods.relativistic_hf import RelativisticHartreeFock
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.mcdhf import MCDHF

__all__ = [
    "Method",
    "HydrogenicSchrodinger",
    "HydrogenicDirac",
    "HartreeFock",
    "RelativisticHartreeFock",
    "DiracFock",
    "MCDHF",
]
