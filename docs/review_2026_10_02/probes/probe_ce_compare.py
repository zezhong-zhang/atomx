import numpy as np, time
from atomx import Atom
from atomx.scattering.distorted_wave.solver import dw_excitation_cross_section
from atomx.scattering.r_matrix import r_matrix_excitation_cross_section

g = Atom(Z=2, config="1s1", charge=1, engine="bspline", corrections="none")
f = Atom(Z=2, config="2s1", charge=1, engine="bspline", corrections="none")
g.solve(); f.solve()
E = np.array([60.0])
t0=time.perf_counter()
s_dw, om_dw = dw_excitation_cross_section(g, f, E, L_max=4, kappa_max=8)
t1=time.perf_counter()
print(f"DW  e+He+ 1s->2s at 60 eV: sigma={s_dw[0]:.4e} cm2  Omega={om_dw[0]:.4f}  ({t1-t0:.1f}s)")
t0=time.perf_counter()
s_rm, om_rm = r_matrix_excitation_cross_section(g, f, E, l_c_max=2, n_basis=40)
t1=time.perf_counter()
print(f"RMT e+He+ 1s->2s at 60 eV: sigma={s_rm[0]:.4e} cm2  Omega={om_rm[0]:.4f}  ({t1-t0:.1f}s)")
