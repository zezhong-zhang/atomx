import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac

h = Atom(1, config=["1s1"], corrections="none")
h.solve()
res = h._result
bsp = res._bsp_data
ns_full = bsp["ns"]; nsp = bsp.get("nsp", ns_full); nsq = bsp.get("nsq", ns_full)
print("ns, nsp, nsq, ks:", ns_full, nsp, nsq, bsp["ks"], "ksp/ksq:", bsp.get("ksp"), bsp.get("ksq"))
print("grid range:", bsp["gr"].min(), bsp["gr"].max())
from atomx.constants import C
for kap in (1, -2):
    H, S = build_dirac_matrices(ns_full, kap, 1, bsp, None)
    r = solve_dirac(H, S, ns_full, kap, nsp=nsp, nsq=nsq)
    ev = np.sort(r.eigenvalues)
    neg = ev[ev < 0]
    print(f"kap={kap:+d}: n_eig={len(ev)}, negative: {np.round(neg[:8], 6)}")
    print(f"   first positive: {np.round(ev[len(neg):len(neg)+5], 6)}")
