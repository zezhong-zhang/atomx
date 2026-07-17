"""Reconstruct the two mixed-vintage defects found while writing Ch. 1.

Defect 1 (fixed 2026-07-16): atomx's legacy ASE-style ``Units`` table
derived its conversion factors from mu_0 = 4 pi x 10^-7 N/A^2 — exact
before the 2019 SI redefinition, but a *measured* constant after it.
This script recomputes the pre-fix derived Hartree and compares it with
both the pre-fix behaviour and the tabulated CODATA value.

Defect 2 (fixed 2026-07-16): AU_TIME_S was hard-coded to the CODATA 2018
derivation of hbar/E_h inside a module whose docstring pins CODATA 2022.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/vintage_audit.py
"""

from math import pi

from atomx import constants as c

t = c.CODATA["2022"]
h, cc, e, me = t["_hplanck"], t["_c"], t["_e"], t["_me"]
hbar = h / (2.0 * pi)

# --- Defect 1: pre-2019 "exact" mu_0 fed into derived conversion factors
for label, mu0 in [("pre-2019 exact", 4.0e-7 * pi),
                   ("CODATA 2022   ", t["_mu0"])]:
    eps0 = 1.0 / (mu0 * cc**2)
    Ha_eV = me * e**3 / (16 * pi**2 * eps0**2 * hbar**2)  # legacy formula
    alpha = e**2 / (4 * pi * eps0) / hbar / cc
    print(f"mu_0 {label}: derived Ha = {Ha_eV:.12f} eV "
          f"(rel vs HARTREE_EV {Ha_eV / c.HARTREE_EV - 1:+.2e}), "
          f"derived alpha rel {alpha / c.ALPHA - 1:+.2e}")

from atomx.constants import units  # post-fix legacy table
print(f"post-fix units.Hartree = {units.Hartree!r} eV "
      f"(rel vs HARTREE_EV {units.Hartree / c.HARTREE_EV - 1:+.2e})")

# --- Defect 2: hard-coded 2018-vintage atomic time unit
tau_2018 = c.CODATA["2018"]["_hplanck"] / (2 * pi) / c.CODATA["2018"]["_Ha_J"]
tau_2022 = t["_hplanck"] / (2 * pi) / t["_Ha_J"]
old_hardcode = 2.4188843265857e-17          # value in constants.py pre-fix
nist_2022 = 2.4188843265864e-17             # NIST concise, u_r = 1.1e-12
print(f"\nhbar/E_h (2018 table)   = {tau_2018!r}")
print(f"hbar/E_h (2022 table)   = {tau_2022!r}")
print(f"pre-fix hard-coded      = {old_hardcode!r} "
      f"(matches 2018 derivation to {abs(old_hardcode/tau_2018-1):.1e})")
print(f"post-fix AU_TIME_S      = {c.AU_TIME_S!r} "
      f"(rel vs NIST 2022 {abs(c.AU_TIME_S/nist_2022-1):.1e})")
print(f"2018 vs 2022 shift      = {abs(tau_2018/tau_2022-1):.1e} relative")
