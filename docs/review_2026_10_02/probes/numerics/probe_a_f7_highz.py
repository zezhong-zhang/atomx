"""Probe A: F7 (MCDHF grid override) + default-grid high-Z point-nucleus error."""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.constants import C

Z = 92
exact = C**2 * (np.sqrt(1.0 - (Z / C) ** 2) - 1.0)
print(f"exact point-nucleus Dirac 1s (Z=92): {exact:.9f} Ha")

def solve_1s(hi, he, r_max, label):
    t, ns = make_physics_knots(ks=9, Z=Z, nuclear="point",
                               r_max=r_max, hi=hi, he=he)
    data = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
    H, S = build_dirac_matrices(ns, -1, Z, data)
    res = solve_dirac(H, S, ns, -1, nsp=data["nsp"], nsq=data["nsq"])
    e = res.bound_energies[0]
    err_ev = (e - exact) * 27.211386245988
    print(f"{label:35s} ns={ns:4d} tmax={t[-1]:8.3f}  E={e:.9f}  err={err_ev:+9.4f} eV")
    return ns, t[-1], e

solve_1s(None, None, None, "default (hi=0.25,he=0.25,box50)")
solve_1s(0.005, 0.10, 30.0, "refined (hi=.005,he=.10,box30)")
solve_1s(0.005, 0.10, None, "refined hi/he, default box")
