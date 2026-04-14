"""Theory-level method objects for AtomX.

Pedagogical progression:

1. HydrogenicSchrodinger — exact non-relativistic hydrogen-like
2. HydrogenicDirac — exact relativistic hydrogen-like (Sommerfeld)
3. HartreeFock — non-relativistic many-electron SCF (LS-coupling)
4. BreitPauli — HF + perturbative relativistic corrections
5. DiracFock — full four-component Dirac-Fock (jj-coupling)
6. MCDHF — multi-configuration with electron correlation
"""

from atomx.methods.base import Method
from atomx.methods.hydrogenic import HydrogenicSchrodinger, HydrogenicDirac
from atomx.methods.hartree_fock import HartreeFock
from atomx.methods.relativistic_hf import BreitPauli
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.mcdhf import MCDHF

# Alias
RelativisticHartreeFock = BreitPauli

__all__ = [
    "Method",
    "HydrogenicSchrodinger",
    "HydrogenicDirac",
    "HartreeFock",
    "BreitPauli",
    "RelativisticHartreeFock",
    "DiracFock",
    "MCDHF",
]
