"""Reduced mass: hydrogen, deuterium, positronium from one formula.

The Bohr formula with nuclear mass M replaces m_e by
mu = m_e M / (m_e + M).  In atomic units (m_e = 1) every energy simply
scales by mu/m_e and every length by m_e/mu.  This script computes the
Lyman-alpha wavelength for infinite nuclear mass, H, D, and positronium,
and the numbers quoted in Exercise 1.3.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/reduced_mass.py
"""

from atomx.constants import HARTREE_CM1, HARTREE_EV, PROTON_ELECTRON_MASS_RATIO

R_INF_CM1 = HARTREE_CM1 / 2.0        # Rydberg constant, cm^-1
MP = PROTON_ELECTRON_MASS_RATIO      # m_p/m_e, CODATA 2022
MD = 3670.482967655                  # m_d/m_e, CODATA 2022: 3670.482 967 655(63)

cases = [
    ("infinite mass", float("inf")),
    ("hydrogen", MP),
    ("deuterium", MD),
    ("positronium", 1.0),            # "nucleus" is a positron, M = m_e
]

print(f"{'system':<14} {'mu/m_e':<14} {'E(1s) / Eh':<14} "
      f"{'E(1s) / eV':<12} {'Ly-alpha / Angstrom':<12}")
for name, M in cases:
    mu = 1.0 if M == float("inf") else M / (1.0 + M)
    E1s = -0.5 * mu                              # Eh
    sigma = mu * R_INF_CM1 * (1.0 - 0.25)        # 1s-2p, cm^-1
    lam_A = 1e8 / sigma
    print(f"{name:<14} {mu:<14.9f} {E1s:<14.9f} "
          f"{E1s * HARTREE_EV:<12.6f} {lam_A:<12.4f}")

lam_inf = 1e8 / (R_INF_CM1 * 0.75)
lam_H = lam_inf * (1.0 + 1.0 / MP)
print(f"\nH minus infinite-mass Ly-alpha: {lam_H - lam_inf:+.4f} Angstrom")
print(f"D minus H isotope shift       : "
      f"{lam_inf * (1 + 1 / MD) - lam_H:+.4f} Angstrom")
print(f"positronium / infinite-mass   : {2.0 * lam_inf / lam_inf:.1f}x")
