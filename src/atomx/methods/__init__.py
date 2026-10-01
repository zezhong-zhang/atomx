"""Theory-level method objects for AtomX.

Pedagogical progression:

1. HydrogenicSchrodinger — exact non-relativistic hydrogen-like
2. HydrogenicDirac — exact relativistic hydrogen-like (Sommerfeld)
3. HartreeFockSlater — non-relativistic local Slater exchange (LS coupling)
4. BreitPauli — HF–Slater + hydrogenic perturbative relativistic shifts
5. DiracFock — full four-component Dirac-Fock (jj-coupling)
6. MCDHF — multi-configuration with electron correlation
"""

from atomx.methods.base import Method
from atomx.methods.breit_pauli import BreitPauli
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.hartree_fock_slater import HartreeFockSlater
from atomx.methods.hydrogenic import HydrogenicDirac, HydrogenicSchrodinger
from atomx.methods.mbpt import MBPT2Result, mbpt2_energy
from atomx.methods.mcdhf import MCDHF

__all__ = [
    "Method",
    "HydrogenicSchrodinger",
    "HydrogenicDirac",
    "HartreeFockSlater",
    "BreitPauli",
    "DiracFock",
    "MCDHF",
    "MBPT2Result",
    "mbpt2_energy",
]
