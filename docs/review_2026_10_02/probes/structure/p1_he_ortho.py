import numpy as np
from atomx import Atom

a = Atom(2, config=['1s2', '1s1 2s1'], engine='bspline', nuclear='point',
         corrections='none')
r = a.solve(tol=1e-10).result
print("converged:", r.converged)
print("inner_converged:", r.__dict__.get('_inner_converged'))
print("ground_energy: %.10f" % r.ground_energy)
print("state_energies:", r.state_energies)
print("orbital_optimization:", r.orbital_optimization)

bsp = r._bsp_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
ns = bsp["ns"]
h, S = build_dirac_matrices(ns, -1, 2, bsp, None)
o = r.orbitals_dict
v1 = np.concatenate(o[(1, -1)]); v2 = np.concatenate(o[(2, -1)])
print("<1s|1s> = %.12f  <2s|2s> = %.12f  <1s|2s> = %.3e" %
      (v1 @ S @ v1, v2 @ S @ v2, v1 @ S @ v2))
print("occupations:", r.occupations, "sum =", sum(r.occupations.values()))
