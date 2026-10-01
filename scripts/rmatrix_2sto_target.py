"""2-STO rich target (Burke's route, R-matrix finishing piece).

2s-bar = N r (a e^{-z1 r} + b e^{-z2 r}), orthogonalised to He+ 1s
(one linear condition fixes b/a) and normalised.  Free: (z1, z2).
Coarse 2D scan for the deepest closed 1Se pole -> Burke -0.77816.
Structurally immune to both earlier exclusions (no extra channel: the
span genuinely widens through the richer target).
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
sys.path.insert(0, "/tmp/atomx-cimbpt/scripts")

from scipy.interpolate import interp1d

from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from atomx.scattering.r_matrix.channel_table import (Channel,
                                                     TargetState)
from rmatrix_contracted_zeta import contracted_targets


def two_sto_target(z1, z2, r, mix=0.5, P1s=None):
    """r(a e^{-z1 r} + b e^{-z2 r}) orthogonal to P1s; ratio a/b from
    mix (before orthogonalisation), then one linear constraint."""
    e1, e2 = np.exp(-z1 * r), np.exp(-z2 * r)
    f1, f2 = r * e1, r * e2
    a0, b0 = mix, 1.0 - mix
    ov1 = np.trapezoid(P1s * f1, r)
    ov2 = np.trapezoid(P1s * f2, r)
    # project (a0, b0) onto the constraint line a*ov1 + b*ov2 = 0
    t = (a0 * ov1 + b0 * ov2) / (ov1 ** 2 + ov2 ** 2)
    a, b = a0 - t * ov1, b0 - t * ov2
    P = a * f1 + b * f2
    assert abs(np.trapezoid(P * P1s, r)) < 1e-3
    return P / np.sqrt(np.trapezoid(P ** 2, r))


def pole_2sto(z1, z2, mix, zeta_p=1.2, a=25.0, n_basis=60):
    r_fine = np.linspace(1e-6, a + 5.0, 4000)
    P1s, _P2s, P2p, e1s, e2s, e2p = contracted_targets(zeta_p, r_fine)
    P2s = two_sto_target(z1, z2, r_fine, mix, P1s=P1s)
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
    import scipy.linalg as sla
    ev = sol.eigenvalues_au
    H, S = sol.H_au, sol.S_au
    n_eff = sol.n_basis_effective
    _, V = sla.eigh(H, S)
    comp = np.zeros((len(ev), 3))
    for i in range(3):
        sl = slice(i * n_eff, (i + 1) * n_eff)
        comp[:, i] = np.einsum("an,ab,bn->n", V[sl], S[sl, sl], V[sl])
    mask = (ev > -0.95) & (ev < -0.50)
    sel = np.where(mask)[0]
    if not len(sel):
        return None, None
    n0 = max(sel, key=lambda n: comp[n, 1] + comp[n, 2])
    return ev[n0], comp[n0]


if __name__ == "__main__":
    print("2-STO rich 2s-bar target; Burke 1Se = -0.77816 "
          "(single-STO best was -0.689)")
    best = (None, None)
    for mix in (0.35, 0.5, 0.65):
        for z1, z2 in [(0.9, 1.5), (0.9, 1.9), (1.1, 1.6), (1.1, 2.0),
                       (1.3, 1.8), (1.5, 2.2)]:
            pole, comp = pole_2sto(z1, z2, mix)
            if pole is None:
                continue
            tag = ""
            if best[0] is None or pole < best[0]:
                best = (pole, (z1, z2, mix))
                tag = "  <-- deepest"
            print(f"mix={mix:.2f} z=({z1},{z2}): pole={pole:+.5f} "
                  f"comp=({comp[0]:.2f},{comp[1]:.2f},{comp[2]:.2f})"
                  f"{tag}", flush=True)
    print(f"\nbest: {best[0]:+.5f} at (z1,z2,mix)={best[1]} "
          f"[Burke -0.77816, single-STO -0.689]")
