"""Constants for the shooting solver, single-sourced from :mod:`atomx.constants`.

Historically these mirrored Salvat's RADIAL package (radial.f) for bit-for-bit
DHFS reproduction.  They are now derived from the central CODATA-2022 table so
the whole library carries one value per physical constant; the RADIAL literals
are retained in comments for provenance.  All differences are <~1e-8 relative
and physically negligible.
"""


from atomx.constants import BOHR_RADIUS_M, C_SQ, HARTREE_EV, SPEED_OF_LIGHT

# Speed of light in atomic units (= 1/alpha).  RADIAL:269 SL=137.035999139D0.
SL = SPEED_OF_LIGHT

# Bohr radius in cm.  RADIAL:271 A0B=5.2917721067D-9.
A0B = BOHR_RADIUS_M * 100.0

# Hartree energy in eV.  RADIAL:273 HREV=27.21138602D0.
HREV = HARTREE_EV

# Electron rest energy m_e c^2 in eV (= c^2 a.u. x Hartree/eV).
# RADIAL:275 REV=510.9989461D3.
REV = C_SQ * HARTREE_EV

# Derived constants
PI = 3.1415926535897932
FOURPI = 4.0 * PI
CSLATE = 0.75 / (PI * PI)  # Slater exchange prefactor
FR1O3 = 1.0 / 3.0
