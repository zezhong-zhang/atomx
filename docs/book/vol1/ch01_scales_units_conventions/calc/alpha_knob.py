"""Turn the alpha knob: perturb alpha -> alpha(1 + 1e-8) and watch.

Uses the Sommerfeld formula in the exact algebraic form atomx implements
(atomx/methods/hydrogenic.py), verified bit-for-bit against
HydrogenicDirac at the unperturbed alpha.  Because a 1e-8 perturbation
moves some observables by less than their float64 cancellation noise
(the c^2 prefactor amplifies rounding by ~1.9e4), every derivative is
also evaluated in 40-digit arithmetic (mpmath); the table reports both,
and the chapter quotes the digits each computation supports.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/alpha_knob.py
"""

import mpmath as mp
import numpy as np

from atomx import Atom, HydrogenicDirac
from atomx.constants import ALPHA, CODATA, HARTREE_CM1

mp.mp.dps = 40
EPS = 1.0e-8  # relative perturbation dalpha/alpha


def sommerfeld(n, kappa, Z, alpha, m=np):
    """Dirac point-nucleus eigenvalue, atomx's algebraic form."""
    za = Z * alpha
    gamma = m.sqrt(kappa**2 - za**2)
    n_r = n - abs(kappa)
    return (1.0 / alpha**2) * (
        1.0 / m.sqrt(1.0 + (za / (n_r + gamma))**2) - 1.0)


# Guard: unperturbed float64 formula must reproduce HydrogenicDirac.
for Z in (1, 26, 92):
    r = Atom(Z=Z, config="1s1").solve(HydrogenicDirac())
    assert sommerfeld(1, -1, Z, ALPHA) == r.orbital_energies["1s1/2"], Z
print("guard: formula == HydrogenicDirac at Z = 1, 26, 92 (bit-for-bit)")

a_mp = mp.mpf("7.2973525643e-3")  # == ALPHA (CODATA 2022)
a1_64, a1_mp = ALPHA * (1.0 + EPS), a_mp * (1 + mp.mpf(f"{EPS:.0e}"))

print(f"\nrelative shift of eps(1s1/2) under dalpha/alpha = {EPS:.0e}")
print("Z    float64        40-digit       leading order (Z*alpha)^2/2 * 1e-8")
for Z in (1, 26, 92):
    d64 = sommerfeld(1, -1, Z, a1_64) / sommerfeld(1, -1, Z, ALPHA) - 1.0
    dmp = sommerfeld(1, -1, Z, a1_mp, mp) / sommerfeld(1, -1, Z, a_mp, mp) - 1
    print(f"{Z:<4d} {d64:<14.3e} {mp.nstr(dmp, 5):<14} "
          f"{(Z * ALPHA)**2 / 2 * EPS:.3e}")

print("\nH 2p fine-structure interval eps(2p3/2) - eps(2p1/2):")
fs64 = sommerfeld(2, -2, 1, ALPHA) - sommerfeld(2, 1, 1, ALPHA)
fs_mp = sommerfeld(2, -2, 1, a_mp, mp) - sommerfeld(2, 1, 1, a_mp, mp)
print(f"  float64 : {fs64:.10e} Eh")
print(f"  40-digit: {mp.nstr(fs_mp, 11)} Eh"
      f" = {mp.nstr(fs_mp * HARTREE_CM1, 7)} cm^-1")
d64 = (sommerfeld(2, -2, 1, a1_64) - sommerfeld(2, 1, 1, a1_64)) / fs64 - 1.0
dmp = (sommerfeld(2, -2, 1, a1_mp, mp)
       - sommerfeld(2, 1, 1, a1_mp, mp)) / fs_mp - 1
print(f"  relative shift, float64 : {d64:.4e}   <- WRONG SIGN (noise)")
print(f"  relative shift, 40-digit: {mp.nstr(dmp, 5)}"
      f"   (prediction 2 * dalpha/alpha = {2 * EPS:.1e})")

# The SI boundary: E_h = alpha^2 m_e c^2, with m_e, c, e held fixed.
t = CODATA["2022"]
Eh_eV = ALPHA**2 * t["_me"] * t["_c"]**2 / t["_e"]
shift = (a1_64 / ALPHA)**2 - 1.0
print(f"\nE_h from alpha^2 m_e c^2 = {Eh_eV:.9f} eV")
print(f"relative shift of E_h(eV): {shift:.4e}  (= 2 * dalpha/alpha)")
print("=> in eV every nonrelativistic level moves by 2e-8;"
      " in Eh none of them move at all.")
