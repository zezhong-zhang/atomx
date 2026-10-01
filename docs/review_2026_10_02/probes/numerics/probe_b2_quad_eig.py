"""Does the n_gauss=ks quadrature error move the Z=92 point-nucleus 1s eigenvalue?"""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.constants import C

Z = 92
exact = C**2 * (np.sqrt(1.0 - (Z / C) ** 2) - 1.0)
t, ns = make_physics_knots(ks=9, Z=Z, nuclear='point')
for ng in (9, 18, 48):
    data = prepare_bspline_data(t, ns, 9, n_gauss=ng, kinetic_balance=True)
    H, S = build_dirac_matrices(ns, -1, Z, data)
    res = solve_dirac(H, S, ns, -1, nsp=data["nsp"], nsq=data["nsq"])
    e = res.bound_energies[0]
    print(f"n_gauss={ng:2d}  E(1s)={e:.9f}  err={(e-exact)*27.211386245988:+9.4f} eV")

# Refined grid: is quadrature still relevant there?
t2, ns2 = make_physics_knots(ks=9, Z=Z, nuclear='point', hi=0.005, he=0.10, r_max=30.0)
for ng in (9, 48):
    data = prepare_bspline_data(t2, ns2, 9, n_gauss=ng, kinetic_balance=True)
    H, S = build_dirac_matrices(ns2, -1, Z, data)
    res = solve_dirac(H, S, ns2, -1, nsp=data["nsp"], nsq=data["nsq"])
    e = res.bound_energies[0]
    print(f"refined n_gauss={ng:2d}  E(1s)={e:.9f}  err={(e-exact)*27.211386245988:+9.4f} eV")
