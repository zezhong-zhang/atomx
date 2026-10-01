"""Convention discriminator for the Ne 2s2s KLL channel.

Replicates _coarse_auger_rate's loop with switchable conventions:
  A: repo current      — sum of 2*pi*W^2*R_k^2 per k (direct + exch separately)
  B: interference      — 2*pi*|sum_k W_k R_k(direct) - sum_k W_k R_k(exch)|^2
  C: angular=wa*wb     — no sqrt in the angular factor (amplitude A)
  D: B + occupation prefactor n_a*(n_b-delta)/(2j+1)-style
Target XATOM 1s-2s2s = 8.70666e-4 au = 3.599e13 s^-1.
"""
import re
import sys

import numpy as np

sys.path.insert(0, "/data/code/atomx")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_shooting,
)
from atomx.numerics.yk import compute_yk
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.angular.coefficients import exchange_k_range, exchange_lambda
from atomx.constants import AU_RATE_PER_S

subs = [Subshell(n=1, kappa=-1, occupation=1.0),
        Subshell(n=2, kappa=-1, occupation=2.0),
        Subshell(n=2, kappa=1, occupation=2.0),
        Subshell(n=2, kappa=-2, occupation=4.0)]
dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7,
               starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
      in r.orbitals_by_subshell().items()}
eps = {}
for lab, e in r.orbital_energies.items():
    m = re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
    n = int(m.group(1)); ll = "spdf".index(m.group(2))
    j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
    eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
bsp = dhf.bsp_data
gr = bsp["gr"]; grw = bsp["grw"]

Pa, Qa = od[(2, -1)]   # source a = 2s
Pb, Qb = od[(2, -1)]   # source b = 2s
Pc, Qc = od[(1, -1)]   # hole c = 1s
Pa_g, Qa_g = eval_P(Pa, bsp), eval_Q(Qa, bsp)
Pb_g, Qb_g = eval_P(Pb, bsp), eval_Q(Qb, bsp)
Pc_g, Qc_g = eval_P(Pc, bsp), eval_Q(Qc, bsp)

E_ha = 815.26 / 27.2114
sum_sq = 0.0
amp_direct = 0.0
amp_exch = 0.0
for kappa_eps in range(-3, 4):
    if kappa_eps == 0:
        continue
    r_eps, P_r, Q_r, _ = solve_dirac_continuum_shooting(
        E_ha, kappa_eps, lambda rr: -10.0 / np.maximum(rr, 1e-30),
        r_max=float(bsp["t"][-1]), n_grid=400, Z_residual=10.0,
        match_phase=False)
    Pe = np.interp(gr.ravel(), r_eps, P_r).reshape(gr.shape)
    Qe = np.interp(gr.ravel(), r_eps, Q_r).reshape(gr.shape)
    rho1 = Pa_g * Pc_g + Qa_g * Qc_g
    rho2 = Pb_g * Pe + Qb_g * Qe
    y2 = compute_yk(rho2, gr, grw, bsp["nv"], gr.shape[1], 0)
    R0 = float(np.sum(grw * rho1 * y2))
    wa = exchange_lambda(-1, -1, 0)          # 2s -> 1s, k=0
    wb = exchange_lambda(-1, kappa_eps, 0)   # 2s -> eps
    if wa <= 0 or wb <= 0:
        continue
    W = float(np.sqrt(wa * wb))
    sum_sq += 2 * np.pi * W**2 * R0**2
    amp_direct += W * R0
    amp_exch += W * R0                        # identical orbitals: same

print(f"XATOM target           = {8.70666e-4:.4e} au")
print(f"A repo (sum of squares)= {sum_sq:.4e} au  "
      f"ratio {sum_sq/8.70666e-4:.2f}")
b_val = 2 * np.pi * (amp_direct - amp_exch) ** 2
print(f"B interference         = {b_val:.4e} au  (identical: ~0)")
# exchange with SAME k gives destructive for like electrons -> try +
b_plus = 2 * np.pi * (amp_direct + amp_exch) ** 2
print(f"B+ (constructive)      = {b_plus:.4e} au  ratio "
      f"{b_plus/8.70666e-4:.2f}")
# C: angular = wa*wb
c_val = 2 * np.pi * (wa * wb) ** 2 * R0**2 if R0 else float("nan")
print(f"W factors for k=0: wa={wa:.4f} wb={wb:.4f}")
print(f"kappa_eps contributions: s1/2 and d3/2 only (E1-like selection"
      f" absent here — all kappa tried)")
