"""Theory-level method objects for AtomX.

Each method encapsulates the physics choices for a calculation,
ordered by the pedagogical progression of atomic theory:

1. NonRelativisticHydrogenic — exact Schrodinger hydrogen-like (Bohr formula)
2. RelativisticHydrogenic — exact Dirac hydrogen-like (Sommerfeld fine structure)
3. HartreeFock — non-relativistic many-electron SCF (LS-coupling)
4. BreitPauli — HF + perturbative relativistic corrections (mass-velocity, Darwin, SO)
5. DiracFock — full four-component Dirac-Fock (jj-coupling)
6. MCDHF — multi-configuration with electron correlation

Usage:
    from atomx import Atom, DiracFock, Exchange
    result = Atom(Z=10).solve(DiracFock(breit=True))
"""

from atomx.methods.base import Method
from atomx.methods.hydrogenic import (
    NonRelativisticHydrogenic,
    RelativisticHydrogenic,
    # Backward-compatible aliases
    HydrogenicSchrodinger,
    HydrogenicDirac,
)
from atomx.methods.hartree_fock import HartreeFock
from atomx.methods.relativistic_hf import (
    BreitPauli,
    # Backward-compatible alias
    RelativisticHartreeFock,
)
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.mcdhf import MCDHF

__all__ = [
    "Method",
    # Preferred names (student-friendly, physics-standard)
    "NonRelativisticHydrogenic",
    "RelativisticHydrogenic",
    "HartreeFock",
    "BreitPauli",
    "DiracFock",
    "MCDHF",
    # Backward-compatible aliases
    "HydrogenicSchrodinger",
    "HydrogenicDirac",
    "RelativisticHartreeFock",
]
