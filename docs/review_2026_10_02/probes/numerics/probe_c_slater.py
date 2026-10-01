"""Probe C: B-spline Slater R^k (poisson vs direct) vs analytic hydrogenic values.
Also F^0(1s,2s) reference via mpmath."""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.numerics.bspline.slater import slater_rk

Z = 2
t, ns = make_physics_knots(ks=9, Z=Z, nuclear='point')
data = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
H, S = build_dirac_matrices(ns, -1, Z, data)
res = solve_dirac(H, S, ns, -1, nsp=data["nsp"], nsq=data["nsq"])
P1, Q1, E1 = res.get_orbital(0)
P2, Q2, E2 = res.get_orbital(1)
print(f"B-spline H-like Z=2: E1s={E1:.12f} (NR exact -2.0), E2s={E2:.12f} (-0.5)")

rho11 = (eval_P(P1, data)**2 + eval_Q(Q1, data)**2).ravel()
rho22 = (eval_P(P2, data)**2 + eval_Q(Q2, data)**2).ravel()
rho12 = (eval_P(P1, data)*eval_P(P2, data) + eval_Q(Q1, data)*eval_Q(Q2, data)).ravel()

F0_11_exact = 5.0 * Z / 8.0
for method in ("poisson", "direct"):
    F0_11 = slater_rk(rho11, rho11, data, 0, method=method)
    F0_12 = slater_rk(rho12, rho12, data, 0, method=method)
    print(f"{method:8s} F0(1s,1s)={F0_11:.12f} (exact {F0_11_exact:.12f}, rel {(F0_11/F0_11_exact-1):+.2e})"
          f"   F0(1s,2s)={F0_12:.12f}")

# mpmath reference for F0(1s,2s), nonrelativistic hydrogenic, charge Z
import mpmath as mp
mp.mp.dps = 40
Zmp = mp.mpf(Z)
def P1s(r): return 2*Zmp**mp.mpf('1.5')*r*mp.e**(-Zmp*r)
def P2s(r): return (Zmp**mp.mpf('1.5')/(2*mp.sqrt(2)))*(2-Zmp*r)*r*mp.e**(-Zmp*r/2)
rho = lambda r: P1s(r)*P2s(r)
def Y0_over_r(r):
    inner = mp.quad(lambda s: rho(s), [0, r])
    outer = mp.quad(lambda s: rho(s)/s, [r, mp.inf])
    return inner/r + outer
F0_ref = mp.quad(lambda r: rho(r)*Y0_over_r(r), [0, mp.inf])
print("mpmath NR reference F0(1s,2s) =", mp.nstr(F0_ref, 20))
# note: B-spline orbitals are relativistic (point nucleus), so expect O((Z/c)^2) differences
