"""Type definitions for AtomX: enums, dataclasses, and protocols.

Provides type-safe alternatives to string-based parameters throughout the API.
All enum values match legacy string values for backward compatibility:

    # These are equivalent:
    Atom(Z=10, exchange="hf")
    Atom(Z=10, exchange=Exchange.HARTREE_FOCK)
"""

from enum import Enum


class NuclearModel(Enum):
    """Nuclear charge distribution model."""
    POINT = "point"
    FERMI = "fermi"
    UNIFORM = "uniform"


class Exchange(Enum):
    """Exchange potential approximation."""
    HARTREE_FOCK = "hf"
    SLATER = "slater"
    THOMAS_FERMI_DIRAC_AMALDI = "tfda"
    GREEN_SELLIN_ZACHOR = "gsz"


class Correlation(Enum):
    """Correlation excitation level for MCDHF."""
    SINGLES = "s"
    SINGLES_DOUBLES = "sd"


class Multipole(Enum):
    """Electromagnetic multipole type."""
    E1 = "E1"
    M1 = "M1"
    E2 = "E2"
    M2 = "M2"


class Gauge(Enum):
    """Gauge for transition matrix elements."""
    LENGTH = "length"
    VELOCITY = "velocity"


class EnergyUnit(Enum):
    """Energy unit for output."""
    HARTREE = "hartree"
    EV = "eV"
    CM1 = "cm-1"
    RYDBERG = "Ry"


class CouplingScheme(Enum):
    """Angular momentum coupling scheme.

    LS (Russell-Saunders): good for light atoms where spin-orbit << Coulomb.
    JJ: natural for relativistic (Dirac) calculations.
    INTERMEDIATE: CI mixing of LS terms via spin-orbit (Breit-Pauli).
    """
    LS = "LS"
    JJ = "jj"
    INTERMEDIATE = "ic"


class TheoryLevel(Enum):
    """Pedagogical ordering of atomic theory levels.

    Each level adds physical effects on top of the previous one,
    following the historical development of atomic theory:

    0. HYDROGENIC: Exact hydrogen-like solutions (Bohr/Schrodinger/Dirac)
    1. HARTREE: Mean-field electron-electron repulsion, no exchange
    2. HARTREE_FOCK: + exchange (non-relativistic, LS-coupling)
    3. HARTREE_FOCK_REL: + perturbative relativistic corrections
       (mass-velocity, Darwin, spin-orbit via Breit-Pauli Hamiltonian)
    4. DIRAC_FOCK_SLATER: Full Dirac equation + local Slater exchange
    5. DIRAC_HARTREE_FOCK: Full Dirac equation + non-local HF exchange
    6. DHF_BREIT: + Breit interaction (magnetic + retardation)
    7. DHF_QED: + QED corrections (vacuum polarization + self-energy)
    8. MCDHF: Multi-configuration (electron correlation via CI)
    """
    HYDROGENIC = 0
    HARTREE = 1
    HARTREE_FOCK = 2
    HARTREE_FOCK_REL = 3
    DIRAC_FOCK_SLATER = 4
    DIRAC_HARTREE_FOCK = 5
    DHF_BREIT = 6
    DHF_QED = 7
    MCDHF = 8
