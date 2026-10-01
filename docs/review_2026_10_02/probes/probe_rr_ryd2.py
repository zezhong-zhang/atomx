import numpy as np, re
from atomx import Atom
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M
from atomx.methods.hydrogenic_cross_sections import milne_rr_cross_section
from atomx.scattering.radiative_recombination import rr_cross_section_rydberg

A0_CM2 = (BOHR_RADIUS_M*100)**2
h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
Ee_eV = np.array([1.0])
Ee_au = Ee_eV / HARTREE_EV
rr = rr_cross_section_rydberg(h._result, Ee_eV, n_max_per_kappa=8, kappa_max=4)

# aggregate atomx per n
per_n = {}
for lbl, s in rr.sigma_by_orbital.items():
    n = int(re.match(r"(\d+)", lbl).group(1))
    per_n[n] = per_n.get(n, 0.0) + s[0]
print(" n |  atomx per-n  |  analytic per-n (Kramers/Stobbe, g=2n^2) | ratio")
for n in sorted(per_n):
    an = milne_rr_cross_section(1, n, Ee_au, g_recombined=2.0*n*n, g_recombining=1.0)[0]/A0_CM2
    print(f"{n:2d} | {per_n[n]:.4e} | {an:.4e} | {per_n[n]/an:8.3f}")

# per-(n,l) at n=5 detail
print("\nn=5 breakdown:")
for lbl, s in sorted(rr.sigma_by_orbital.items()):
    if lbl.startswith("5"):
        print(f"  {lbl:8s} {s[0]:.4e}")
# Kramers per (n,l) is not tabulated here; but per-shell comparison shows the scale.
