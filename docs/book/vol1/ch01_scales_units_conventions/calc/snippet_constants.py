"""Calculate-section code block: the constants module.

Printed verbatim in the chapter; output pasted from this run.
"""
from atomx import codata_version
from atomx.constants import (ALPHA, BOHR_RADIUS_M, HARTREE_CM1,
                             HARTREE_EV, SPEED_OF_LIGHT)

print(f"CODATA vintage : {codata_version()}")
print(f"alpha          : {ALPHA!r}")
print(f"c (a.u.)       : {SPEED_OF_LIGHT!r}")
print(f"E_h in eV      : {HARTREE_EV!r}")
print(f"E_h in cm^-1   : {HARTREE_CM1!r}")
print(f"a_0 in m       : {BOHR_RADIUS_M!r}")
