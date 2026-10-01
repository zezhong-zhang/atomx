import numpy as np, time
from atomx import Atom
from atomx.scattering.photoionization.direct import photoionization_cross_section
from atomx.scattering.electron_ionization import eii_cross_section

h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
res = h._result; bsp = res._bsp_data
sh = res.subshells[0]
eps = res.orbital_energies[sh.label]
P, Q, _ = res.orbitals_raw[sh.kappa][0]

t0 = time.perf_counter()
sig = photoionization_cross_section(P, Q, sh.kappa, eps, np.array([1.0]), 1, bsp)
t1 = time.perf_counter()
sig2 = photoionization_cross_section(P, Q, sh.kappa, eps, np.array([1.1, 1.2, 1.3, 1.4, 1.5]), 1, bsp)
t2 = time.perf_counter()
print(f"F10: first PI call {t1-t0:.2f}s; 5 more energies {t2-t1:.2f}s (cache reuse -> sublinear)")
print("  sig[0] recheck:", sig[0], " new:", sig2)

# EII DW vs BEB on He (closed shell): order-of-magnitude cross-check
he = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
he.solve()
E = np.array([30.0, 60.0, 120.0])  # I(He)=24.6 eV
t0 = time.perf_counter()
r_dw = eii_cross_section(he._result, E, method="dw", kappa_max=3, n_outgoing_energies=8)
t1 = time.perf_counter()
r_beb = eii_cross_section(he._result, E, method="beb")
print(f"EII DW ({t1-t0:.1f}s):", r_dw.sigma_total_cm2)
print(f"EII BEB        :", r_beb.sigma_total_cm2)
print("ratio DW/BEB:", r_dw.sigma_total_cm2/r_beb.sigma_total_cm2)
