"""v3 step 3: chi_j pair channel — the missing 0.09 Ha.

The (phi, eps s) pseudo-state channel adds the pair-relaxation span
(the (phi, phi) dielectronic function is inside its target x spline
product space).  phi must be orthogonal to BOTH l=0 targets (1s and
the contracted 2s-bar) — by construction: g = r(1 + b r + c r^2)
exp(-zeta2 r) with (b, c) solving the two overlap conditions.
Scan zeta2 at fixed zeta(2s-bar) and watch the closed 1Se pole.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rmatrix_contracted_zeta import contracted_targets
from scipy.interpolate import interp1d

from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from atomx.scattering.r_matrix.channel_table import Channel, TargetState


def phi_pair(zeta2, r, P1s, P2s):
    """r(1 + b r + c r^2) exp(-zeta2 r), orthogonal to 1s and 2s-bar."""
    e = np.exp(-zeta2 * r)
    r1 = r * e
    r2 = r ** 2 * e
    r3 = r ** 3 * e
    # overlap conditions: <r1> + b<r2> + c<r3> = 0 vs each target
    A = np.array([[np.trapezoid(t * r2, r), np.trapezoid(t * r3, r)]
                  for t in (P1s, P2s)])
    rhs = -np.array([np.trapezoid(t * r1, r) for t in (P1s, P2s)])
    b, c = np.linalg.solve(A, rhs)
    P = r1 + b * r2 + c * r3
    for t in (P1s, P2s):
        assert abs(np.trapezoid(P * t, r)) < 1e-3
    return P / np.sqrt(np.trapezoid(P ** 2, r))


def pole_4ch(zeta, zeta2, a=25.0, n_basis=60):
    r_fine = np.linspace(1e-6, a + 5.0, 4000)
    P1s, P2s, P2p, e1s, e2s, e2p = contracted_targets(zeta, r_fine)
    Pphi = phi_pair(zeta2, r_fine, P1s, P2s)
    dP = np.gradient(Pphi, r_fine)
    e_phi = (0.5 * np.trapezoid(dP ** 2, r_fine)
             + np.trapezoid(Pphi ** 2 * (-2.0 / r_fine), r_fine))
    fs = [interp1d(r_fine, P, fill_value=0.0, bounds_error=False)
          for P in (P1s, P2s, P2p, Pphi)]
    ts = [TargetState(label=l, n_electrons=1, l=ll, energy_au=e, P=f,
                      n=2, Z=2.0)
          for l, ll, e, f in [("1s", 0, e1s, fs[0]),
                              ("2s#", 0, e2s, fs[1]),
                              ("2p#", 1, e2p, fs[2]),
                              ("phi", 0, e_phi, fs[3])]]
    chans = [Channel(index=i, target=t, l_c=l, threshold_au=t.energy_au)
             for i, (t, l) in enumerate(zip(ts, (0, 0, 1, 0), strict=False))]
    sol = assemble_cc_inner_region(chans, L_total=0, S_total=0.0, Z=2.0,
                                   a_au=a, n_basis=n_basis,
                                   k_spline_order=6, n_gauss=12)
    import scipy.linalg as sla
    ev = sol.eigenvalues_au
    H, S = sol.H_au, sol.S_au
    n_eff = sol.n_basis_effective
    _, V = sla.eigh(H, S)
    nch = 4
    comp = np.zeros((len(ev), nch))
    for i in range(nch):
        sl = slice(i * n_eff, (i + 1) * n_eff)
        Vi = V[sl, :]
        comp[:, i] = np.einsum("an,ab,bn->n", Vi, S[sl, sl], Vi)
    mask = (ev > -0.95) & (ev < -0.50)
    sel = np.where(mask)[0]
    ranked = sorted(sel, key=lambda n: -(comp[n, 1:].sum()))
    top = [(ev[n], comp[n]) for n in ranked[:2]]
    return top, e_phi


if __name__ == "__main__":
    zeta = 1.2
    print(f"4-channel 1Se (zeta_2s={zeta}); scan zeta2 of the pair "
          f"channel phi:", flush=True)
    print(f"{'zeta2':>6} {'e_phi':>8}  top closed-like poles "
          f"(E, comp 1s/2s#/2p#/phi)", flush=True)
    for zeta2 in (1.2, 1.3, 1.4, 1.5, 1.6, 1.75):
        try:
            top, e_phi = pole_4ch(zeta, zeta2)
            s = "; ".join(
                f"{e:+.4f}(" + "/".join(f"{x:.2f}" for x in c) + ")"
                for e, c in top)
            print(f"{zeta2:6.2f} {e_phi:+8.4f}  {s}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{zeta2:6.2f}  FAILED: {exc}", flush=True)
