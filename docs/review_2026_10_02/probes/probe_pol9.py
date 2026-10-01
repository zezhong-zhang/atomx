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
npf = res._nuclear_potential
H1, _ = _build_closed_shell_fock(res, 1, cache=ConvolCache(bsp), potential_mode="VN-1")
H3, _ = build_dirac_matrices(ns_full, 1, 1, bsp, npf)
D = H1 - H3
print("nsp, nsq:", nsp, nsq)
print("block max: PP", np.abs(D[:nsp,:nsp]).max(), " QQ", np.abs(D[ns_full:ns_full+nsq, ns_full:ns_full+nsq]).max(),
      " PQ", np.abs(D[:nsp, ns_full:]).max())
# VN mode for comparison
Hv, _ = _build_closed_shell_fock(res, 1, cache=ConvolCache(bsp), potential_mode="VN")
Dv = Hv - H3
print("VN-H0: PP", np.abs(Dv[:nsp,:nsp]).max(), " QQ", np.abs(Dv[ns_full:ns_full+nsq, ns_full:ns_full+nsq]).max(),
      " PQ", np.abs(Dv[:nsp, ns_full:]).max())
rv = solve_dirac(H1, _, ns_full, 1, nsp=nsp, nsq=nsq)
ev = np.sort(rv.eigenvalues)
neg = ev[(ev<0)&(ev>-3e4)]
print("VN-1 H1 bound states:", np.round(neg[:6],6))
