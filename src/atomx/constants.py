"""Physical constants, unit conversions, and element data.

Default values follow CODATA 2022 (published 2024) — the most recent
recommendation from the Committee on Data for Science and Technology.
CODATA 2018 values are also retained for historical reproducibility.

The SI defining constants (c, h, e, k_B, N_A) are exact by definition
since the 2019 SI redefinition and therefore unchanged across CODATA
2018 ↔ 2022.  Measured constants (m_e, m_p, α, a_0, m_u, R_∞) are
updated — the ~1 part-per-billion shifts matter only for precision
atomic-clock work but are included here for consistency.

Internal units
--------------
atomx computes entirely in Hartree atomic units (ℏ = m_e = e =
4πε₀ = 1).  The SI values below appear only at the API boundary for
unit conversion via :mod:`atomx.units` (pint-based).

References
----------
E. Tiesinga *et al.*, *Rev. Mod. Phys.* **93**, 025010 (2021) —
    CODATA 2018.
P. J. Mohr *et al.*, CODATA recommended values of the fundamental
    physical constants: 2022 (NIST, 2024), arXiv:2409.00899.
"""

from math import pi, sqrt
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# CODATA physical constants
# ---------------------------------------------------------------------------
# Defined (exact) constants:  c, h, e, k_B, N_A  — same across 2018/2022.
# Measured constants (masses, α) are updated.

CODATA = {
    "2018": {
        "_c": 299_792_458.0,            # speed of light, m/s (exact)
        "_mu0": 1.25663706212e-6,       # vacuum permeability, N/A^2 (measured since 2019 SI)
        "_Grav": 6.67430e-11,           # gravitational constant
        "_hplanck": 6.62607015e-34,     # Planck constant, J s (exact)
        "_e": 1.602176634e-19,          # elementary charge, C (exact)
        "_me": 9.1093837015e-31,        # electron mass, kg
        "_mp": 1.67262192369e-27,       # proton mass, kg
        "_Nav": 6.02214076e23,          # Avogadro number (exact)
        "_k": 1.380649e-23,             # Boltzmann, J/K (exact)
        "_amu": 1.66053906660e-27,      # atomic mass unit, kg
        # Derived (for reference)
        "_alpha": 7.2973525693e-3,      # fine-structure constant
        "_a0": 5.29177210903e-11,       # Bohr radius, m
        "_Ha_J": 4.3597447222071e-18,   # Hartree, J
        "_mp_over_me": 1836.15267343,   # proton-electron mass ratio
    },
    "2022": {
        # Defined (exact) constants — unchanged
        "_c": 299_792_458.0,            # speed of light, m/s (exact)
        "_mu0": 1.25663706127e-6,       # vacuum permeability, N/A^2 (was 1.25663706212e-6)
        "_Grav": 6.67430e-11,           # unchanged from 2018
        "_hplanck": 6.62607015e-34,     # exact
        "_e": 1.602176634e-19,          # exact
        "_Nav": 6.02214076e23,          # exact
        "_k": 1.380649e-23,             # exact
        # Measured constants — updated CODATA 2022 (NIST 2024)
        "_me": 9.1093837139e-31,        # electron mass, kg (was 9.1093837015e-31)
        "_mp": 1.67262192595e-27,       # proton mass, kg (was 1.67262192369e-27)
        "_amu": 1.66053906892e-27,      # atomic mass unit, kg (was 1.66053906660e-27)
        # Derived
        "_alpha": 7.2973525643e-3,      # α (was 7.2973525693e-3)
        "_a0": 5.29177210544e-11,       # Bohr radius (was 5.29177210903e-11)
        "_Ha_J": 4.3597447222060e-18,   # Hartree in J (was 4.3597447222071e-18)
        "_mp_over_me": 1836.152673426,  # (was 1836.15267343)
    },
}

#: Default CODATA version used by atomx (change-once module variable;
#: swap to ``"2018"`` to reproduce older atomx numbers exactly).
DEFAULT_CODATA_VERSION = "2022"


class Units:
    """Physical unit conversion factors, derived lazily from CODATA constants.

    Usage::

        from atomx.constants import units
        energy_eV = energy_hartree * units.Hartree   # Ha -> eV
        length_bohr = length_ang / units.Bohr        # Ang -> Bohr

    Defaults to CODATA 2022; pass ``codata_version="2018"`` for older
    reproducibility.
    """
    def __init__(self, codata_version: str = DEFAULT_CODATA_VERSION) -> None:
        self._units = None
        self._codata_version = codata_version

    def __getattr__(self, name):
        if self._units is None:
            self._units = self._create_units()
        val = self._units.get(name)
        if val is None:
            raise AttributeError(f"Unknown unit: {name}")
        return val

    def _create_units(self) -> Dict[str, float]:
        u = dict(CODATA[self._codata_version])

        u["_eps0"] = 1 / u["_mu0"] / u["_c"] ** 2
        u["_hbar"] = u["_hplanck"] / (2 * pi)

        # Length (base: Angstrom)
        u["Ang"] = u["Angstrom"] = 1.0
        u["nm"] = 10.0
        u["Bohr"] = (4e10 * pi * u["_eps0"] * u["_hbar"] ** 2
                     / u["_me"] / u["_e"] ** 2)

        # Energy (base: eV)
        u["eV"] = 1.0
        u["Hartree"] = (u["_me"] * u["_e"] ** 3
                        / 16 / pi ** 2 / u["_eps0"] ** 2 / u["_hbar"] ** 2)
        u["Ha"] = u["Hartree"]
        u["Rydberg"] = 0.5 * u["Hartree"]
        u["Ry"] = u["Rydberg"]
        u["kJ"] = 1000.0 / u["_e"]
        u["kcal"] = 4.184 * u["kJ"]
        u["invcm"] = 100 * u["_c"] * u["_hplanck"] / u["_e"]  # cm^-1

        # Time
        u["second"] = 1e10 * sqrt(u["_e"] / u["_amu"])
        u["fs"] = 1e-15 * u["second"]

        # Fine structure constant
        u["alpha"] = (u["_e"] ** 2 / (4 * pi * u["_eps0"])
                      / u["_hbar"] / u["_c"])

        # Boltzmann
        u["kB"] = u["_k"] / u["_e"]  # eV/K

        return u


units = Units(DEFAULT_CODATA_VERSION)


# ---------------------------------------------------------------------------
# Atomic units (direct constants for internal use)
# ---------------------------------------------------------------------------
# These are pulled from ``CODATA[DEFAULT_CODATA_VERSION]`` so that the
# module-level constants and the :class:`Units` table stay consistent.
# Users who need bit-for-bit CODATA 2018 reproduction can set
# ``atomx.constants.DEFAULT_CODATA_VERSION = "2018"`` before importing
# downstream modules, or freeze at import time via
# ``ALPHA_2018 = CODATA["2018"]["_alpha"]`` etc.

_active = CODATA[DEFAULT_CODATA_VERSION]

FINE_STRUCTURE_CONSTANT = _active["_alpha"]
ALPHA = FINE_STRUCTURE_CONSTANT
ALPHA_SQ = ALPHA * ALPHA

SPEED_OF_LIGHT = 1.0 / ALPHA  # c in atomic units (~137.036)
C = SPEED_OF_LIGHT
C_SQ = C * C

# Energy conversions from Hartree (derived from CODATA)
HARTREE_J = _active["_Ha_J"]
HARTREE_EV = HARTREE_J / _active["_e"]                     # 1 Ha / (1 eV)
RYDBERG_EV = 0.5 * HARTREE_EV                              # 1 Ry = Ha/2 in eV (~13.6057)
HARTREE_CM1 = HARTREE_J / (100.0 * _active["_hplanck"] * _active["_c"])  # 1 Ha / cm^-1

# Length
BOHR_RADIUS_M = _active["_a0"]
BOHR_RADIUS_FM = BOHR_RADIUS_M * 1e15

# Mass
ELECTRON_MASS_KG = _active["_me"]
PROTON_MASS_KG = _active["_mp"]
PROTON_ELECTRON_MASS_RATIO = _active["_mp_over_me"]

# Nuclear
R0_FM = 1.2  # nuclear radius parameter, fm

# Spectroscopy

# Atomic unit of time tau_0 = hbar/E_h, derived from the active CODATA
# table (a hard-coded 2018 value survived here until 2026-07; see
# tests/test_book_ch01_contracts.py::test_au_time_derived_from_active_table).
AU_TIME_S = _active["_hplanck"] / (2.0 * pi) / _active["_Ha_J"]
AU_RATE_PER_S = 1.0 / AU_TIME_S
HARTREE_INV_ANGSTROM = ALPHA / (2.0 * np.pi * BOHR_RADIUS_M * 1e10)
BOLTZMANN_EV = 8.617_333_262e-5


def nuclear_rms_radius_fm(A: int) -> float:
    """RMS nuclear charge radius from mass number A (in fm)."""
    return 0.836 * A ** (1.0 / 3.0) + 0.570


def nuclear_rms_radius_au(A: int) -> float:
    """RMS nuclear charge radius in Bohr radii."""
    return nuclear_rms_radius_fm(A) / (BOHR_RADIUS_M * 1e15)


# ---------------------------------------------------------------------------
# Element data (adapted from zespec/ase)
# ---------------------------------------------------------------------------

chemical_symbols = [
    # 0
    "X",
    # 1
    "H", "He",
    # 2
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    # 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    # 6
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # 7
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}


def element_symbol(Z: int) -> str:
    """Element symbol from atomic number Z."""
    if 0 < Z < len(chemical_symbols):
        return chemical_symbols[Z]
    return f"Z{Z}"
