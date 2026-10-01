import numpy as np
from atomx import Atom
from atomx.constants import BOHR_RADIUS_M
from atomx.methods.hydrogenic_cross_sections import stobbe_1s_pi_cross_section
from atomx.scattering.photoionization.direct import photoionization_cross_section

A0 = (BOHR_RADIUS_M*100)**2
he = Atom(Z=2, config="1s1", charge=1, engine="bspline", corrections="none")
he.solve()
res = he._result; bsp = res._bsp_data
sh = res.subshells[0]
eps = res.orbital_energies[sh.label]
P, Q, _ = res.orbitals_raw[sh.kappa][0]
print("He+ 1s eps:", eps, "exact -2.0")
photon_au = np.array([2.02, 2.05, 2.1, 2.5, 3.0])
sig = photoionization_cross_section(P, Q, sh.kappa, eps, photon_au, 2, bsp)
st = stobbe_1s_pi_cross_section(2, photon_au)/A0
print("omega:", photon_au)
print("atomx:", sig)
print("Stobbe:", st)
print("ratio:", sig/st)
