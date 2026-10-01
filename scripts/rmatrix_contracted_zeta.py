"""v3 step 1-2: Burke-style contracted n=2 target orbitals + zeta sweep.

2s-bar = N r exp(-zeta r)  (Schmidt-orthogonalised to the 1s target)
2p-bar = N r^2 exp(-zeta r)
Target pseudo-state energies = <bar| H(He+) |bar> (zeta-dependent — the
n=2 threshold MOVES with zeta, which is part of Burke's construction).
Track the CLOSED-channel-dominated 1Se eigenvalue vs zeta; target
Burke -0.77816 Ha.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.scattering.r_matrix import heplus_target_states
from atomx.scattering.r_matrix.channel_table import Channel, TargetState
from atomx.scattering.r_matrix.cc_inner import assemble_cc_inner_region


def contracted_targets(zeta, r):
    """Return (P1s, P2s_bar, P2p_bar) samples + He+ energies."""
    # exact He+ 1s
    P1s = 2.0 * np.sqrt(2.0) * np.exp(-2.0 * r)
    e1s = -2.0
    # contracted 2s-bar, orthogonal to He+ 1s BY CONSTRUCTION:
    # g = r(1 + b r) exp(-zeta r) with b = -(zeta+2)/2 solving
    # <g | 1s> = 0 exactly (Schmidt after the fact breaks P(0)=0 and
    # log-diverges <1/r>)
    b = -(zeta + 2.0) / 2.0
    P2s = r * (1.0 + b * r) * np.exp(-zeta * r)
    ov = np.trapezoid(P1s * P2s, r)
    assert abs(ov) < 1e-3, ov  # trapezoid/truncation residual
    P2s = P2s / np.sqrt(np.trapezoid(P2s ** 2, r))
    # STO 2p-bar
    P2p = np.sqrt(zeta ** 7 / 120.0) * r ** 2 * np.exp(-zeta * r)
    # energies on the same grid: <T + -2/r + l(l+1)/2r^2>
    def energy(P, l):
        dP = np.gradient(P, r)
        T = 0.5 * np.trapezoid(dP ** 2, r) if hasattr(np, "trapezoid") \
            else 0.5 * np.trapz(dP ** 2, r)
        V = np.trapezoid(P ** 2 * (-2.0 / r), r) if hasattr(np,
                                                            "trapezoid") \
            else np.trapz(P ** 2 * (-2.0 / r), r)
        C = 0.5 * l * (l + 1) * (np.trapezoid(P ** 2 / r ** 2, r)
                                 if hasattr(np, "trapezoid")
                                 else np.trapz(P ** 2 / r ** 2, r))
        return T + V + C
    return P1s, P2s, P2p, e1s, energy(P2s, 0), energy(P2p, 1)


def closed_pole(zeta, a=25.0, n_basis=60):
    # fine radial grid for the analytic target samples (cc_inner samples
    # them at ITS quadrature nodes via the callable — pass callables)
    from scipy.interpolate import interp1d
    r_fine = np.linspace(1e-6, a + 5.0, 4000)
    P1s, P2s, P2p, e1s, e2s, e2p = contracted_targets(zeta, r_fine)
    f1 = interp1d(r_fine, P1s, fill_value=0.0, bounds_error=False)
    f2 = interp1d(r_fine, P2s, fill_value=0.0, bounds_error=False)
    f3 = interp1d(r_fine, P2p, fill_value=0.0, bounds_error=False)
    t1 = TargetState(label="1s", n_electrons=1, l=0, energy_au=e1s,
                     P=f1, n=1, Z=2.0)
    t2 = TargetState(label="2s#", n_electrons=1, l=0, energy_au=e2s,
                     P=f2, n=2, Z=2.0)
    t3 = TargetState(label="2p#", n_electrons=1, l=1, energy_au=e2p,
                     P=f3, n=2, Z=2.0)
    chans = [Channel(index=i, target=t, l_c=l, threshold_au=t.energy_au)
             for i, (t, l) in enumerate([(t1, 0), (t2, 0), (t3, 1)])]
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
        Vi = V[sl, :]
        comp[:, i] = np.einsum("an,ab,bn->n", Vi, S[sl, sl], Vi)
    mask = (ev > -0.95) & (ev < -0.50)
    sel = np.where(mask)[0]
    if not len(sel):
        return None, None, e2s, e2p, []
    ranked = sorted(sel, key=lambda n: -(comp[n, 1] + comp[n, 2]))
    ncl = ranked[0]
    top = [(ev[n], comp[n, 0], comp[n, 1], comp[n, 2])
           for n in ranked[:3]]
    return ev[ncl], (comp[ncl, 0], comp[ncl, 1], comp[ncl, 2]), \
        e2s, e2p, top


if __name__ == "__main__":
    print(f"{'zeta':>5} {'e_2s#':>9}  top closed-like states "
          f"(E, comp open/2s/2p)", flush=True)
    for zeta in np.arange(1.00, 1.451, 0.05):
        pole, comp, e2s, e2p, top = closed_pole(zeta)
        tops = "; ".join(f"{e:+.4f}({c0:.2f}/{c1:.2f}/{c2:.2f})"
                         for e, c0, c1, c2 in top)
        print(f"{zeta:5.2f} {e2s:+9.4f}  {tops}", flush=True)
