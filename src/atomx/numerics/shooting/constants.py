"""Physical constants matching Salvat's RADIAL package (MODULE CONSTANTS).

These values are used exclusively in the shooting solver to ensure
exact numerical reproduction of RADIAL/DHFS results.
"""

import numba as nb

# Speed of light in atomic units (= 1/alpha).
# radial.f line 269: SL=137.035999139D0
SL = 137.035999139

# Bohr radius in cm.
# radial.f line 271: A0B=5.2917721067D-9
A0B = 5.2917721067e-9

# Hartree energy in eV.
# radial.f line 273: HREV=27.21138602D0
HREV = 27.21138602

# Electron rest energy in eV.
# radial.f line 275: REV=510.9989461D3
REV = 510998.9461

# Derived constants
PI = 3.1415926535897932
FOURPI = 4.0 * PI
CSLATE = 0.75 / (PI * PI)  # Slater exchange prefactor
FR1O3 = 1.0 / 3.0
