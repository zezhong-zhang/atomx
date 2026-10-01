import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M
from atomx.methods.hydrogenic_cross_sections import stobbe_1s_pi_cross_section, milne_rr_cross_section
from atomx.scattering.radiative_recombination import rr_cross_section

A0_CM2 = (BOHR_RADIUS_M*100)**2
h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()

Ee_eV = np.array([1.0, 5.0, 13.6])
rr = rr_cross_section(h._result, Ee_eV, multipoles=("E1",))
Ee_au = Ee_eV / HARTREE_EV
milne = milne_rr_cross_section(1, 1, Ee_au, g_recombined=2, g_recombining=1)
print("RR atomx (a0^2):", rr.sigma_total_a0_sq)
print("RR Milne analytic (a0^2):", milne)
print("ratio:", rr.sigma_total_a0_sq / milne)

# H 1s total alpha_RR at 1e4 K ~ 4.2e-13 cm3/s (Storey & Hummer 1995, incl. all n via Rydberg sum)
from atomx.scattering.radiative_recombination import rr_cross_section_rydberg, rr_rate_coefficient
rr_ryd = rr_cross_section_rydberg(h._result, np.geomspace(0.05, 300.0, 120), n_max_per_kappa=10, kappa_max=4)
alpha = rr_rate_coefficient(rr_ryd, np.array([0.8617]))  # 1e4 K ~ 0.8617 eV
print("alpha_RR(1e4 K) Rydberg-summed:", alpha, "cm3/s  (Storey-Hummer ~4.2e-13)")
