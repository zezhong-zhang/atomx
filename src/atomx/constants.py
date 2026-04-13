"""CODATA 2018 fundamental physical constants in atomic units.

All values from NIST CODATA 2018 recommended values.
Internal units: Hartree atomic units (hbar = m_e = e = 4*pi*eps_0 = 1).
"""

import numpy as np

# --- Exact defining constants ---
SPEED_OF_LIGHT_SI = 299_792_458.0  # m/s (exact)

# --- CODATA 2018 recommended values ---
FINE_STRUCTURE_CONSTANT = 7.297_352_5693e-3  # alpha
FINE_STRUCTURE_CONSTANT_INV = 1.0 / FINE_STRUCTURE_CONSTANT  # ~137.036

# Speed of light in atomic units: c = 1/alpha
SPEED_OF_LIGHT = FINE_STRUCTURE_CONSTANT_INV  # ~137.036 a.u.

# Energy conversions
HARTREE_EV = 27.211_386_245_988  # 1 Hartree in eV
HARTREE_CM1 = 219_474.631_363_20  # 1 Hartree in cm^-1
HARTREE_J = 4.359_744_722_2071e-18  # 1 Hartree in Joules

# Length
BOHR_RADIUS_M = 5.291_772_109_03e-11  # Bohr radius in meters
BOHR_RADIUS_FM = BOHR_RADIUS_M * 1e15  # Bohr radius in femtometers

# Mass
ELECTRON_MASS_KG = 9.109_383_7015e-31
PROTON_MASS_KG = 1.672_621_923_69e-27
PROTON_ELECTRON_MASS_RATIO = 1836.152_673_43

# Derived constants useful for Dirac calculations
ALPHA = FINE_STRUCTURE_CONSTANT
ALPHA_SQ = ALPHA * ALPHA
C = SPEED_OF_LIGHT
C_SQ = C * C

# Nuclear radius parameter: R = R0 * A^(1/3) in fm
R0_FM = 1.2  # fm, standard nuclear radius parameter


def nuclear_rms_radius_fm(A: int) -> float:
    """RMS nuclear charge radius from mass number A (in fm).

    Uses the empirical formula: R_rms = 0.836 * A^(1/3) + 0.570 fm
    from Angeli & Marinova (2013).
    """
    return 0.836 * A ** (1.0 / 3.0) + 0.570


def nuclear_rms_radius_au(A: int) -> float:
    """RMS nuclear charge radius in Bohr radii."""
    return nuclear_rms_radius_fm(A) / (BOHR_RADIUS_M * 1e15)


# --- Transition & spectroscopy constants ---
AU_TIME_S = 2.418_884_326_5857e-17  # atomic time unit (hbar/E_h) in seconds
AU_RATE_PER_S = 1.0 / AU_TIME_S     # 1/t_au in s^-1 (for converting A-coefficients)

# Wavelength conversion: lambda(A) = 2*pi*a_0 / (alpha * E_hartree)
# = (2*pi * BOHR_RADIUS_M * 1e10) / (ALPHA * E_hartree)
HARTREE_INV_ANGSTROM = ALPHA / (2.0 * np.pi * BOHR_RADIUS_M * 1e10)  # E(Ha) -> 1/lambda(A)

# Boltzmann constant
BOLTZMANN_EV = 8.617_333_262e-5  # eV/K
