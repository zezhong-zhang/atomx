"""Type definitions for AtomX: enums, dataclasses, and protocols.

Provides type-safe alternatives to string-based parameters throughout the API.
All enum values match the canonical string spellings:

    # These are equivalent:
    Atom(10, exchange="hf")                       # the canonical spelling
    Atom(10, exchange=Exchange.HARTREE_FOCK)      # enum member (types.Exchange)

``Exchange`` and ``Correlation`` below are the *method* API's value objects
(``DiracFock(exchange=Exchange.SLATER)``, ``MCDHF(correlation=...)``); the
``Atom`` constructor's own value objects live in :mod:`atomx.atom_inputs`.
"""

from enum import Enum


class NuclearModel(Enum):
    """Nuclear charge distribution model.

    POINT: Bare Coulomb potential V(r) = -Z/r.  Singular at origin.
    FERMI: Fermi two-parameter distribution (default, DBSR_HF convention).
           Realistic for medium-to-heavy nuclei.
    UNIFORM: Uniform charged sphere.  Simplest finite-nucleus model.
    """
    POINT = "point"
    FERMI = "fermi"
    UNIFORM = "uniform"


class Exchange(Enum):
    """Electron exchange potential approximation.

    HARTREE_FOCK: Non-local exchange via Slater integrals (Y^k expansion).
                  Exact within the single-determinant approximation.
    SLATER: Local density approximation V_x ~ -alpha * rho^{1/3} with
            alpha = 2/3 (original Hartree-Fock-Slater).  Fast but
            approximate.
    FAC_GU: Same local form, but with Gu's empirically-tuned
            alpha_xc(Z, N) — see ``atomx.physics.local_exchange.get_hxs``
            and ``atomx.numerics.local_dfs``.  Reproduces FAC 1.1.5's
            default local-exchange SCF to double-precision in the
            alpha_xc fit.
    """
    HARTREE_FOCK = "hf"
    SLATER = "slater"
    FAC_GU = "gu"


class StartingPotential(Enum):
    """Approximate potential for SCF initial guess.

    THOMAS_FERMI: Thomas-Fermi-Dirac-Amaldi statistical model.
    GREEN_SELLIN_ZACHOR: Parametric effective potential with Z-dependent parameters.
    """
    THOMAS_FERMI = "tfda"
    GREEN_SELLIN_ZACHOR = "gsz"


class Correlation(Enum):
    """Electron correlation level for multi-configuration calculations.

    SINGLES: Include single excitations from reference configuration.
    SINGLES_DOUBLES: Include single and double excitations (more complete).
    """
    SINGLES = "s"
    SINGLES_DOUBLES = "sd"


class Multipole(Enum):
    """Electromagnetic multipole type for radiative transitions.

    E1: Electric dipole (dominant for most transitions).
    M1: Magnetic dipole (forbidden transitions, weaker).
    E2: Electric quadrupole.
    M2: Magnetic quadrupole.
    """
    E1 = "E1"
    M1 = "M1"
    E2 = "E2"
    M2 = "M2"


class Gauge(Enum):
    """Gauge for radiative transition matrix elements.

    LENGTH: Babushkin gauge, uses <r> operator.  More physical.
    VELOCITY: Coulomb gauge, uses <nabla> operator.  Gauge-independent check.
    """
    LENGTH = "length"
    VELOCITY = "velocity"


class EnergyUnit(Enum):
    """Energy unit for output display."""
    HARTREE = "hartree"
    EV = "eV"
    CM1 = "cm-1"
    RYDBERG = "Ry"


class CouplingScheme(Enum):
    """Angular momentum coupling scheme.

    LS: Russell-Saunders coupling.  L and S are good quantum numbers.
        Appropriate when electron-electron repulsion >> spin-orbit coupling.
        Natural for light atoms (Z < ~30).

    JJ: jj-coupling.  Individual j_i = l_i + s_i are good quantum numbers.
        Natural for the Dirac equation and heavy atoms where spin-orbit
        is comparable to or exceeds electron-electron repulsion.

    INTERMEDIATE: Configuration interaction mixes LS terms via spin-orbit.
                  Used in Breit-Pauli calculations and for medium-Z atoms
                  where neither pure LS nor pure jj coupling is appropriate.
    """
    LS = "LS"
    JJ = "jj"
    INTERMEDIATE = "ic"


class RelativisticCorrection(Enum):
    """First-order relativistic correction terms (Breit-Pauli Hamiltonian).

    MASS_VELOCITY: -p^4/(8 m^3 c^2).  Relativistic correction to kinetic
                   energy.  Shifts ALL orbital energies (largest for inner shells).

    DARWIN: (pi Z alpha^2 / 2) delta^3(r).  Contact interaction from
            electron zitterbewegung.  Only affects s-orbitals (l=0).

    SPIN_ORBIT: (alpha^2 / 2)(1/r)(dV/dr) L*S.  Lifts l-degeneracy by
                splitting shells with l>0 into j = l +/- 1/2 subshells.
    """
    MASS_VELOCITY = "mass_velocity"
    DARWIN = "darwin"
    SPIN_ORBIT = "spin_orbit"


class TheoryLevel(Enum):
    """Pedagogical ordering of atomic structure theory levels.

    Each level builds on the previous, following the historical development
    of atomic theory and adding progressively smaller physical effects:

    HYDROGENIC (0):
        Exact single-electron solutions.  Non-relativistic: Bohr/Schrodinger
        with E_n = -Z^2/(2n^2).  Relativistic: Dirac/Sommerfeld with fine
        structure splitting.

    HARTREE (1):
        Mean-field electron-electron repulsion (Coulomb only).  Electrons
        move in an averaged potential from all other electrons.  No exchange.

    HARTREE_FOCK (2):
        Non-relativistic self-consistent field with exchange.  The Pauli
        exclusion principle is enforced via antisymmetric wavefunctions.
        Uses LS-coupling (Russell-Saunders).

    BREIT_PAULI (3):
        HF + first-order perturbative relativistic corrections:
        mass-velocity, Darwin contact term, and spin-orbit coupling.
        Bridge between non-relativistic and fully relativistic methods.

    DIRAC_FOCK_SLATER (4):
        Full four-component Dirac equation with local Slater exchange.
        All relativistic effects treated variationally (not perturbatively).

    DIRAC_HARTREE_FOCK (5):
        Full Dirac equation with exact (non-local) Hartree-Fock exchange.
        The most accurate single-configuration relativistic method.

    DIRAC_FOCK_BREIT (6):
        DHF + Breit interaction: magnetic interaction between electrons
        and retardation of the Coulomb interaction.

    DIRAC_FOCK_QED (7):
        DHF + Breit + QED corrections: vacuum polarization (Uehling
        potential) and electron self-energy (Lamb shift).

    MCDHF (8):
        Multi-configuration Dirac-Hartree-Fock.  Recovers electron
        correlation by mixing multiple configuration state functions.
    """
    HYDROGENIC = 0
    HARTREE = 1
    HARTREE_FOCK = 2
    BREIT_PAULI = 3
    DIRAC_FOCK_SLATER = 4
    DIRAC_HARTREE_FOCK = 5
    DIRAC_FOCK_BREIT = 6
    DIRAC_FOCK_QED = 7
    MCDHF = 8
