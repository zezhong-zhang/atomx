"""Continuous pole tracking along the 2-STO path.

zeta1 = 1.2 fixed; zeta2 swept 1.2 -> 2.4.  At zeta2 = zeta1 the
target IS the single-STO one (tracker seeds at the known closed pole
~ -0.689).  The tracker follows the NEAREST eigenvalue (closed-weight
tie-break) instead of max-closed — immune to the jump-fork trap.
Also reports the n=2 threshold e_2s# so pole-vs-threshold is readable.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
sys.path.insert(0, "/tmp/atomx-cimbpt/scripts")

from rmatrix_2sto_target import pole_2sto  # noqa: F401  (import check)
from rmatrix_contracted_zeta import contracted_targets
from rmatrix_2sto_target import two_sto_target
from scipy.interpolate import interp1d
from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from atomx.scattering.r_matrix.channel_table import (Channel,
                                                     TargetState)
import scipy.linalg as sla


def full_spectrum(z1, z2, a=25.0, n_basis=60):
    r_fine = np.linspace(1e-6, a + 5.0, 4000)
    P1s, _, P2p, e1s, e2s_h, e2p = contracted_targets(1.2, r_fine)
    P2s = two_sto_target(z1, z2, r_fine, 0.5, P1s=P1s)
    dP = np.gradient(P2s, r_fine)
    e2s = (0.5 * np.trapezoid(dP ** 2, r_fine)
           - 2.0 * np.trapezoid(P2s ** 2 / r_fine, r_fine))
    fs = [interp1d(r_fine, P, fill_value=0.0, bounds_error=False)
          for P in (P1s, P2s, P2p)]
    ts = [TargetState(label=l, n_electrons=1, l=ll, energy_au=e, P=f,
                      n=2, Z=2.0)
          for l, ll, e, f in [("1s", 0, e1s, fs[0]),
                              ("2s#", 0, e2s, fs[1]),
                              ("2p#", 1, e2p, fs[2])]]
    chans = [Channel(index=i, target=t, l_c=l, threshold_au=t.energy_au)
             for i, (t, l) in enumerate(zip(ts, (0, 0, 1)))]
    sol = assemble_cc_inner_region(chans, L_total=0, S_total=0.0, Z=2.0,
                                   a_au=a, n_basis=n_basis,
                                   k_spline_order=6, n_gauss=12)
    ev = sol.eigenvalues_au
    H, S = sol.H_au, sol.S_au
    n_eff = sol.n_basis_effective
    _, V = sla.eigh(H, S)
    comp = np.zeros((len(ev), 3))
    for i in range(3):
        sl = slice(i * n_eff, (i + 1) * n_eff)
        comp[:, i] = np.einsum("an,ab,bn->n", V[sl], S[sl, sl], V[sl])
    return ev, comp, e2s


if __name__ == "__main__":
    print(f"{'z2':>5} {'e_2s#':>8} {'tracked pole':>12} "
          f"{'comp(1s/2s/2p)':>18}  below-threshold?")
    prev = -0.6885   # single-STO pole at zeta=1.2 (known seed)
    for z2 in (1.3, 1.45, 1.6, 1.8, 2.0, 2.2, 2.4):
        ev, comp, e2s = full_spectrum(1.2, z2)
        mask = (ev > -0.95) & (ev < -0.45)
        sel = np.where(mask)[0]
        if True:
            # nearest to previous energy, closed-weight tie-break
            n0 = min(sel, key=lambda n: (abs(ev[n] - prev),
                                         -(comp[n, 1] + comp[n, 2])))
        prev = ev[n0]
        c = comp[n0]
        below = "YES" if ev[n0] < e2s else "no"
        print(f"{z2:5.2f} {e2s:+8.4f} {ev[n0]:+12.5f}  "
              f"({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  {below} "
              f"(gap {ev[n0]-e2s:+.3f})", flush=True)
