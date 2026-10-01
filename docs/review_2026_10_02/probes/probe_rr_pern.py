import numpy as np, re
from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering.radiative_recombination import rr_cross_section_rydberg
from atomx.plasma.rates import maxwellian_rate

h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
E = np.geomspace(0.03, 60.0, 200)
rr = rr_cross_section_rydberg(h._result, E, n_max_per_kappa=12, kappa_max=8)
T_eV = 0.8617  # 1e4 K

per_n = {}
for lbl, s in rr.sigma_by_orbital.items():
    n = int(re.match(r"(\d+)", lbl).group(1))
    per_n.setdefault(n, np.zeros_like(E))
    per_n[n] += s
print("Storey-Hummer 1e4 K: alpha_A=4.18e-13 total, alpha_1s=1.596e-13")
tot = 0.0
for n in sorted(per_n):
    a = maxwellian_rate(per_n[n], E, T_eV)
    tot += a
    print(f"  n={n:2d}  alpha_n = {a:.4e} cm3/s")
print("  TOTAL =", f"{tot:.4e}")
