import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
npf = res._nuclear_potential
r = np.logspace(-8, 2, 11)
V = np.array([npf(x) for x in r])
for x, v in zip(r, V):
    print(f"r={x:.2e}  V_fermi={v:14.6e}  V_point={-1.0/x:14.6e}")
ns_full = bsp["ns"]; nsp = bsp.get("nsp", ns_full); nsq = bsp.get("nsq", ns_full)
for kap in (-1, 1):
    Hf, Sf = build_dirac_matrices(ns_full, kap, 1, bsp, npf)
    rf = solve_dirac(Hf, Sf, ns_full, kap, nsp=nsp, nsq=nsq)
    ev = np.sort(rf.eigenvalues)
    neg = ev[(ev < 0) & (ev > -1.5 * (1/0.0072973525693)**2)]
    print(f"kap={kap:+d} fermi bound states: {np.round(neg[:6], 6)}")
