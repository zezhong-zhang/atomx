"""Probe D: shooting SLAG6 Slater integral vs analytic; shooting H-like spectrum."""
import numpy as np
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.coulomb import _slater_integral_slag6, _slater_integral

Z = 2.0
# Salvat grid for a hydrogenic ion: r2 ~ small, box 75
r, difr, np_ = sgrid(75.0, 1.0e-6, 0.5, 2000)
# NR hydrogenic 1s density P^2
P1 = 2 * Z**1.5 * r * np.exp(-Z * r)
f1 = P1 * P1
F0_slag6 = _slater_integral_slag6(f1, f1, 0, r, difr, np_)
F0_ode   = _slater_integral(f1, f1, 0, r, difr, np_)
exact = 5 * Z / 8
print(f"shooting grid np={np_}")
print(f"F0(1s,1s) SLAG6 = {F0_slag6:.12f}  rel {(F0_slag6/exact-1):+.3e}")
print(f"F0(1s,1s) ODE   = {F0_ode:.12f}  rel {(F0_ode/exact-1):+.3e}")

# 1s-2s cross
P2 = (Z**1.5/(2*np.sqrt(2))) * (2 - Z*r) * r * np.exp(-Z*r/2)
f12 = P1 * P2
ref = 0.043895747599451303155
F0x_slag6 = _slater_integral_slag6(f12, f12, 0, r, difr, np_)
F0x_ode   = _slater_integral(f12, f12, 0, r, difr, np_)
print(f"F0(1s,2s) SLAG6 = {F0x_slag6:.12f}  rel {(F0x_slag6/ref-1):+.3e}")
print(f"F0(1s,2s) ODE   = {F0x_ode:.12f}  rel {(F0x_ode/ref-1):+.3e}")

# F^2(2p,2p) hydrogenic = 1125/... use analytic: F2(2p,2p) = (1/Z)*? compute numerically high precision instead
# Hydrogenic 2p: P = (1/(2*sqrt(6))) Z^(5/2) r^2 e^(-Zr/2)
P2p = (1/(2*np.sqrt(6))) * Z**2.5 * r**2 * np.exp(-Z*r/2)
f2p = P2p*P2p
import mpmath as mp
mp.mp.dps = 40
Zmp = mp.mpf(2)
def P2p_mp(rr): return (1/(2*mp.sqrt(6))) * Zmp**mp.mpf('2.5') * rr**2 * mp.e**(-Zmp*rr/2)
rho = lambda rr: P2p_mp(rr)**2
def Y2_over_r(rr):
    inner = mp.quad(lambda s: rho(s)*s**2, [0, rr])
    outer = mp.quad(lambda s: rho(s)/s**3, [rr, mp.inf])
    return inner/rr**3 + rr**2*outer
F2_ref = mp.quad(lambda rr: rho(rr)*Y2_over_r(rr), [0, mp.inf])
print("mpmath F2(2p,2p) ref =", mp.nstr(F2_ref, 18))
F2_slag6 = _slater_integral_slag6(f2p, f2p, 2, r, difr, np_)
F2_ode   = _slater_integral(f2p, f2p, 2, r, difr, np_)
print(f"F2(2p,2p) SLAG6 = {F2_slag6:.12f}  rel {(F2_slag6/float(F2_ref)-1):+.3e}")
print(f"F2(2p,2p) ODE   = {F2_ode:.12f}  rel {(F2_ode/float(F2_ref)-1):+.3e}")
