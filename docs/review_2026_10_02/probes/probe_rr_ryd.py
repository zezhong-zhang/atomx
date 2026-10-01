import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M
from atomx.methods.hydrogenic_cross_sections import milne_rr_cross_section
from atomx.scattering.radiative_recombination import rr_cross_section_rydberg, RRResult

A0_CM2 = (BOHR_RADIUS_M*100)**2
h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()

Ee_eV = np.array([0.5, 1.0, 5.0])
rr = rr_cross_section_rydberg(h._result, Ee_eV, n_max_per_kappa=12, kappa_max=6)
print("total sigma_RR (a0^2):", rr.sigma_total_a0_sq)

# analytic: n=1 Stobbe + n>=2 Kramers, g_n = 2 n^2 total per shell, g_ion=1
Ee_au = Ee_eV / HARTREE_EV
analytic = np.zeros_like(Ee_au)
for n in range(1, 13):
    analytic += milne_rr_cross_section(1, n, Ee_au, g_recombined=2.0*n*n, g_recombining=1.0) / A0_CM2
print("analytic sum n<=12 (a0^2):", analytic)
print("ratio:", rr.sigma_total_a0_sq / analytic)

# per-state breakdown at 1 eV
per = sorted(((lbl, s[1]) for lbl, s in rr.sigma_by_orbital.items()), key=lambda t: -t[1])
for lbl, s in per[:12]:
    print(f"  {lbl:10s} sigma={s:.4e}")
