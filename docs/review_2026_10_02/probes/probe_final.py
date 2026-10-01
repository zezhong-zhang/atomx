import sys
import numpy as np
sys.path.insert(0, "/data/code/atomx/src")
from atomx.angular.coefficients import reduced_C
from atomx.angular.recoupling import ls_jj_transformation

# reduced_C at k=0
for kap in (-1, 1, -2, 2, -3):
    j = abs(kap) - 0.5
    print(f"reduced_C(k={0}, kap={kap:+d},{kap:+d}) = {reduced_C(kap, kap, 0):.6f}  (-sqrt(2j+1)={-np.sqrt(2*j+1):.6f})")

# LS-jj transformation unitarity for a few blocks
for l1, l2, J2 in [(1, 1, 4), (1, 1, 2), (2, 1, 3), (1, 1, 0), (2, 2, 4)]:
    T = ls_jj_transformation(l1, l2, J2)
    if T.size == 0:
        print(f"l1={l1},l2={l2},2J={J2}: empty")
        continue
    dev = np.abs(T @ T.T - np.eye(T.shape[0])).max()
    print(f"l1={l1},l2={l2},2J={J2}: T shape {T.shape}, |TT^T-I|={dev:.2e}")

# spinor density integrates to 1
from atomx.angular.spinor_harmonics import spinor_density
th = np.linspace(0, np.pi, 400)
ph = np.linspace(0, 2*np.pi, 400)
TH, PH = np.meshgrid(th, ph, indexing="ij")
w = np.sin(TH)
for kap, mu in [(-1, 0.5), (1, 0.5), (-2, 0.5), (-2, 1.5), (2, 1.5)]:
    d = spinor_density(kap, mu, TH, PH)
    integ = np.trapezoid(np.trapezoid(d * w, ph, axis=1), th)
    print(f"spinor_density kap={kap:+d} mu={mu}: integral = {integ:.8f} (expect 1)")

# recoil operator decomposition for H 1s
from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.numerics.radial_workspace import eval_orbital_pair
from atomx.constants import C
P, Q, bsp, E = hydrogenic_dirac_wavefunction(1, -1, 1, ns=80, ks=9)
p, q = eval_orbital_pair(P, Q, bsp)
r, w_ = bsp["gr"], bsp["grw"]
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
dp = eval_orbital_on_grid(P, bsp["bsp_p_d"], bsp["nv"], bsp.get("ksp", bsp["ks"]), bsp.get("nsp", bsp["ns"]))
dq = eval_orbital_on_grid(Q, bsp["bsp_q_d"], bsp["nv"], bsp.get("ksq", bsp["ks"]), bsp.get("nsq", bsp["ns"]))
p2_nr = 0.5 * np.sum(w_ * (dp*dp + dq*dq))   # kappa=-1: l(l+1)=0, lt(lt+1)=2
lt_term = 0.5 * np.sum(w_ * (2 * q * q) / r**2)
breit1 = 0.5 * np.sum(w_ * (-2.0 / C * (q * dp + q * dp) / r))   # Z=1
breit2 = 0.5 * np.sum(w_ * (-(1.0 / C) * (-1 - 1) * (q * p + p * q) / r**2))
print(f"H 1s: p2/2={p2_nr:.8f} lt(lt+1)Q^2/2r^2={lt_term:.8e} breit1={breit1:.8e} breit2={breit2:.8e}")
print(f"  sum NR parts={p2_nr+lt_term:.8f}  +breit={p2_nr+lt_term+breit1+breit2:.8f}  -E={-E:.8f}")
