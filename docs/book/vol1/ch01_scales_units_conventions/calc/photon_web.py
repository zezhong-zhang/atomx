"""One photon, six numbers: the Fe XVII 3C line at 15.014 Angstrom.

Reference wavelength: Bernitt et al., Nature 492, 225 (2012), Fe XVII 3C.

Route A converts with explicit CODATA-2022 constants from atomx.constants
(h, c, e are exact SI definitions; only the step into Hartree uses a
measured constant, E_h in joules).  Route B uses the pint layer
(atomx.units) with the spectroscopy context.  The script prints both and
their relative disagreement, which the chapter quotes.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/photon_web.py
"""

from atomx import constants as c
from atomx.units import ureg

LAMBDA_A = 15.014  # Angstrom, Bernitt 2012 (3C)

t = c.CODATA["2022"]
h, cc, e = t["_hplanck"], t["_c"], t["_e"]  # exact SI definitions

lam_m = LAMBDA_A * 1e-10
E_J = h * cc / lam_m
E_eV = E_J / e
E_Ha = E_J / c.HARTREE_J
E_Ry = E_Ha * 2.0
sigma_cm1 = 1.0 / (lam_m * 100.0)
nu_THz = cc / lam_m / 1e12

print("Route A — explicit constants (CODATA 2022 table)")
print(f"  lambda   = {LAMBDA_A} Angstrom")
print(f"  E        = {E_eV!r} eV")
print(f"  E        = {E_Ry!r} Ry")
print(f"  E        = {E_Ha!r} Ha")
print(f"  sigma    = {sigma_cm1!r} cm^-1")
print(f"  nu       = {nu_THz!r} THz")
print(f"  hc/e     = {h * cc / e * 1e10!r} eV*Angstrom")

q = LAMBDA_A * ureg.angstrom
routes = {
    "eV": q.to("eV", "spectroscopy").magnitude,
    "hartree": q.to("hartree", "spectroscopy").magnitude,
    "cm^-1": q.to("cm^-1", "spectroscopy").magnitude,
    "THz": q.to("THz", "spectroscopy").magnitude,
}
print("\nRoute B — pint spectroscopy context")
for unit, val in routes.items():
    print(f"  {unit:8s} = {val!r}")

print("\nCross-route relative disagreement")
print(f"  eV    : {abs(routes['eV'] / E_eV - 1):.2e}")
print(f"  Ha    : {abs(routes['hartree'] / E_Ha - 1):.2e}")
print(f"  cm^-1 : {abs(routes['cm^-1'] / sigma_cm1 - 1):.2e}")
print(f"  THz   : {abs(routes['THz'] / nu_THz - 1):.2e}")
