"""Non-orthogonal chi_j pair channel: the R-matrix finishing piece.

4-channel 1Se: (1s,es), (2s-bar(zeta),es), (2p-bar(zeta),ep) + the
NON-ORTHOGONAL pair pseudo-state phi(zeta2) = N r exp(-zeta2 r)
(overlaps both 1s and 2s-bar — the S cross-blocks handle it).  The
(phi, phi) dielectronic function is inside the phi-channel span.
Scan zeta2 for the closed pole -> Burke -0.77816.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
sys.path.insert(0, "/tmp/atomx-cimbpt/scripts")

from scipy.interpolate import interp1d

from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region
from rmatrix_contracted_zeta import contracted_targets


def pole_with_chi(zeta, zeta2, a=25.0, n_basis=60):
    r_fine = np.linspace(1e-6, a + 5.0, 4000)
    P1s, P2s, P2p, e1s, e2s, e2p = contracted_targets(zeta, r_fine)
    Pphi = zeta2 ** 1.5 * r_fine * np.exp(-zeta2 * r_fine)
    Pphi = Pphi / np.sqrt(np.trapezoid(Pphi ** 2, r_fine))
    dphi = np.gradient(Pphi, r_fine)
    e_phi = (0.5 * np.trapezoid(dphi ** 2, r_fine)
             - 2.0 * np.trapezoid(Pphi ** 2 / r_fine, r_fine))
    fs = [interp1d(r_fine, P, fill_value=0.0, bounds_error=False)
          for P in (P1s, P2s, P2p)]
    fphi = interp1d(r_fine, Pphi, fill_value=0.0, bounds_error=False)
    from atomx.scattering.r_matrix.channel_table import (
        Channel, TargetState)
    ts = [TargetState(label=l, n_electrons=1, l=ll, energy_au=e, P=f,
                      n=2, Z=2.0)
          for l, ll, e, f in [("1s", 0, e1s, fs[0]),
                              ("2s#", 0, e2s, fs[1]),
                              ("2p#", 1, e2p, fs[2])]]
    chans = [Channel(index=i, target=t, l_c=l, threshold_au=t.energy_au)
             for i, (t, l) in enumerate(zip(ts, (0, 0, 1)))]
    sol = assemble_cc_inner_region(
        chans, L_total=0, S_total=0.0, Z=2.0, a_au=a, n_basis=n_basis,
        k_spline_order=6, n_gauss=12,
        pair_targets=[("phi", fphi, e_phi)])
    import scipy.linalg as sla
    ev = sol.eigenvalues_au
    H, S = sol.H_au, sol.S_au
    n_eff = sol.n_basis_effective
    _, V = sla.eigh(H, S)
    nch = 4
    comp = np.zeros((len(ev), nch))
    for i in range(nch):
        sl = slice(i * n_eff, (i + 1) * n_eff)
        comp[:, i] = np.einsum("an,ab,bn->n", V[sl], S[sl, sl],
                               V[sl])
    mask = (ev > -0.95) & (ev < -0.50)
    sel = np.where(mask)[0]
    if not len(sel):
        return None, None
    ranked = sorted(sel, key=lambda n: -(comp[n, 1] + comp[n, 3]))
    n0 = ranked[0]
    return ev[n0], (comp[n0, 0], comp[n0, 1], comp[n0, 2], comp[n0, 3])


if __name__ == "__main__":
    print("4-channel + NON-ORTHOGONAL chi(zeta2); Burke 1Se = -0.77816")
    print(f"{'zeta2':>6} {'pole':>9}  comp(1s/2s#/2p#/phi)")
    for zeta2 in (1.0, 1.2, 1.35, 1.5, 1.65, 1.8):
        pole, comp = pole_with_chi(1.2, zeta2)
        if pole is None:
            print(f"{zeta2:6.2f}  no state in window")
            continue
        c = "/".join(f"{x:.2f}" for x in comp)
        print(f"{zeta2:6.2f} {pole:+9.5f}  {c}", flush=True)
