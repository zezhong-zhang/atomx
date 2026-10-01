import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.scattering.radiative_recombination import _rr_potential_matrices
from atomx.numerics.bspline.evaluate import eval_P, eval_Q

h = Atom(Z=1, config="1s1", engine="bspline", corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
R_box = float(bsp["t"][-1])
print("B-spline box radius:", R_box, "a0")
outer = res.subshells[0]
pp, qq = _rr_potential_matrices(res, outer, None)
ns = bsp["ns"]
hmat, smat = build_dirac_matrices(ns, -1, 1, bsp)
hmat[:ns,:ns] += pp; hmat[ns:,ns:] += qq
basis = solve_dirac(hmat, smat, ns, -1, nsp=bsp.get("nsp"), nsq=bsp.get("nsq"))
gr = bsp["gr"]; grw = bsp["grw"]
print(" i | eps (Ha) | exact -1/2n^2 | <r> | 2n^2 (hydrogenic extent)")
for i, eps in enumerate(basis.eigenvalues):
    if eps >= 0: break
    n = i + 1
    P, Q, _ = basis.get_orbital(i)
    Pg, Qg = eval_P(P, bsp), eval_Q(Q, bsp)
    rm = float(np.sum(grw*(Pg**2+Qg**2)*gr))
    print(f"{i:2d} | {eps:.6f} | {-0.5/n**2:.6f} | {rm:8.2f} | {2*n*n:5d}")
