import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_EV, BOHR_RADIUS_M
from atomx.methods.hydrogenic_cross_sections import stobbe_1s_pi_cross_section, milne_rr_cross_section
from atomx.scattering.photoionization.direct import photoionization_cross_section
from atomx.scattering.radiative_recombination import rr_cross_section

A0_CM2 = (BOHR_RADIUS_M*100)**2

h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
res = h._result
bsp = h._bsp_data
sh = res.subshells[0]
eps = res.orbital_energies[sh.label]
P, Q, _ = res.orbitals_raw[sh.kappa][0]
print("H 1s eps (Ha):", eps, "(exact -0.5)")

photon_au = np.array([0.55, 0.75, 1.0, 1.5, 2.0])
sig = photoionization_cross_section(P, Q, sh.kappa, eps, photon_au, 1, bsp)
stobbe = stobbe_1s_pi_cross_section(1, photon_au) / A0_CM2
print("photon_au:", photon_au)
print("atomx PI (a0^2):", sig)
print("Stobbe  (a0^2):", stobbe)
print("ratio:", sig/stobbe)

# RR via the public path vs analytic Milne helper
Ee_eV = np.array([1.0, 5.0, 13.6]) 
rr = rr_cross_section(h, Ee_eV, multipoles=("E1",))
Ee_au = Ee_eV / HARTREE_EV
milne = milne_rr_cross_section(1, 1, Ee_au, g_recombined=2, g_recombining=1)  # a0^2
print("RR atomx (a0^2):", rr.sigma_total_a0_sq)
print("RR Milne analytic (a0^2):", milne)
print("ratio:", rr.sigma_total_a0_sq / milne)
