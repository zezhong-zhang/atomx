"""Physical constants, unit conversions, and element data.

Adapted from zespec/ase conventions:
- CODATA 2018 fundamental constants
- Units class with lazy derivation of all conversion factors
- chemical_symbols and atomic_numbers for element lookup

Internal units: Hartree atomic units (hbar = m_e = e = 4*pi*eps_0 = 1).
"""

from math import pi, sqrt
from typing import Dict


# ---------------------------------------------------------------------------
# CODATA physical constants
# ---------------------------------------------------------------------------

CODATA = {
    "2018": {
        "_c": 299_792_458.0,           # speed of light, m/s (exact)
        "_mu0": 4.0e-7 * pi,          # permeability of vacuum
        "_Grav": 6.67430e-11,          # gravitational constant
        "_hplanck": 6.62607015e-34,    # Planck constant, J s (exact)
        "_e": 1.602176634e-19,         # elementary charge, C (exact)
        "_me": 9.1093837015e-31,       # electron mass, kg
        "_mp": 1.67262192369e-27,      # proton mass, kg
        "_Nav": 6.02214076e23,         # Avogadro number (exact)
        "_k": 1.380649e-23,            # Boltzmann constant, J/K (exact)
        "_amu": 1.66053906660e-27,     # atomic mass unit, kg
    },
}


class Units:
    """Physical unit conversion factors, derived lazily from CODATA constants.

    Usage::

        from atomx.constants import units
        energy_eV = energy_hartree * units.Hartree   # Ha -> eV
        length_bohr = length_ang / units.Bohr        # Ang -> Bohr
    """
    def __init__(self, codata_version: str = "2018") -> None:
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


units = Units("2018")


# ---------------------------------------------------------------------------
# Atomic units (direct constants for internal use)
# ---------------------------------------------------------------------------

FINE_STRUCTURE_CONSTANT = 7.297_352_5693e-3
ALPHA = FINE_STRUCTURE_CONSTANT
ALPHA_SQ = ALPHA * ALPHA

SPEED_OF_LIGHT = 1.0 / ALPHA  # c in atomic units (~137.036)
C = SPEED_OF_LIGHT
C_SQ = C * C

# Energy conversions from Hartree
HARTREE_EV = 27.211_386_245_988
HARTREE_CM1 = 219_474.631_363_20
HARTREE_J = 4.359_744_722_2071e-18

# Length
BOHR_RADIUS_M = 5.291_772_109_03e-11
BOHR_RADIUS_FM = BOHR_RADIUS_M * 1e15

# Mass
ELECTRON_MASS_KG = 9.109_383_7015e-31
PROTON_MASS_KG = 1.672_621_923_69e-27
PROTON_ELECTRON_MASS_RATIO = 1836.152_673_43

# Nuclear
R0_FM = 1.2  # nuclear radius parameter, fm

# Spectroscopy
import numpy as np
AU_TIME_S = 2.418_884_326_5857e-17
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
