import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.methods.mbpt import _build_closed_shell_fock
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.convol import ConvolCache

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
ns_full = bsp["ns"]; nsp = bsp.get("nsp", ns_full); nsq = bsp.get("nsq", ns_full)

H1, S1 = _build_closed_shell_fock(res, 1, cache=ConvolCache(bsp), potential_mode="VN-1")
H2, S2 = build_dirac_matrices(ns_full, 1, 1, bsp, None)
print("H1 == H2 (VN-1 should equal H0 for H):", np.allclose(H1, H2), np.max(np.abs(H1-H2)))
print("S1 == S2:", np.allclose(S1, S2))
r1 = solve_dirac(H1, S1, ns_full, 1, nsp=nsp, nsq=nsq)
r2 = solve_dirac(H2, S2, ns_full, 1, nsp=nsp, nsq=nsq)
print("eig via fock H1:", np.sort(r1.eigenvalues)[:6])
print("eig via raw  H2:", np.sort(r2.eigenvalues)[:6])
print("shapes:", H1.shape, H2.shape)
