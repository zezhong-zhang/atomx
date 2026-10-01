"""Probe G: B-spline eigenvalue convergence vs basis size; Z=1 book gate."""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.constants import C

def err_1s(Z, hi, he, r_max, ks=9):
    t, ns = make_physics_knots(ks=ks, Z=Z, nuclear='point', r_max=r_max, hi=hi, he=he)
    data = prepare_bspline_data(t, ns, ks, kinetic_balance=True)
    H, S = build_dirac_matrices(ns, -1, Z, data)
    res = solve_dirac(H, S, ns, -1, nsp=data["nsp"], nsq=data["nsq"])
    exact = C**2 * (np.sqrt(1.0 - (Z / C) ** 2) - 1.0)
    return ns, (res.bound_energies[0] - exact)

print("Z=92 point nucleus, refinement sweep (hi=he=h, r_max=None):")
for h in (0.25, 0.125, 0.0625, 0.03125):
    ns, err = err_1s(92, h, h, None)
    print(f"  hi=he={h:8.4f} ns={ns:4d}  err={err*27.211386245988:+10.4f} eV")

print("Z=92, order sweep at hi=he=0.125:")
for ks in (9, 11, 13):
    ns, err = err_1s(92, 0.125, 0.125, None, ks=ks)
    print(f"  ks={ks:2d} ns={ns:4d}  err={err*27.211386245988:+10.4f} eV")

print("Z=1 (book gate needs |err| < 1e-11 Ha, he=0.0625 auto):")
ns, err = err_1s(1, None, None, None)
print(f"  default physics grid ns={ns}  err={err:+.3e} Ha")

print("Z=1 with generic he=0.25:")
ns, err = err_1s(1, None, 0.25, None)
print(f"  ns={ns}  err={err:+.3e} Ha")
