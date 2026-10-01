"""Ab initio Burke-1971 check: can the multi-channel inner region
actually PRODUCE the He doubly-excited poles from the computed
eigenphase, without empirical poles?  (The existing test is a
model/fitter regression on tabulated poles, not an ab initio gate.)
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.scattering.r_matrix import (compute_multichannel,
                                       heplus_target_states)
from atomx.scattering.r_matrix.resonance import fit_breit_wigner

BURKE1971 = {
    "1Se(2s2)":  (-0.77816, 0.00500),
    "1Po(2s2p)": (-0.69312, 0.00139),
    "1De(2p2)":  (-0.70220, 0.00264),
}

if __name__ == "__main__":
    targets = list(heplus_target_states(Z=2.0).values())
    scans = {
        "1Se(2s2)": dict(L_total=0, S_total=0, parity=+1, l_c_max=2,
                         energies=np.linspace(-0.95, -0.55, 800)),
        "1Po(2s2p)": dict(L_total=1, S_total=0, parity=-1, l_c_max=2,
                          energies=np.linspace(-0.85, -0.55, 800)),
        "1De(2p2)": dict(L_total=2, S_total=0, parity=+1, l_c_max=3,
                         energies=np.linspace(-0.85, -0.55, 800)),
    }
    for name, kw in scans.items():
        E_b, G_b = BURKE1971[name]
        res = compute_multichannel(
            targets, L_total=kw["L_total"], S_total=kw["S_total"],
            parity=kw["parity"], energies_au=kw["energies"], Z=2.0,
            a_au=25.0, n_basis=60, k_spline_order=6, n_gauss=12,
            l_c_max=kw["l_c_max"], include_l2_pseudostates=True)
        delta = res.eigenphase_sum
        E = kw["energies"]
        d_un = np.unwrap(delta) if delta.ndim == 1 else \
            np.unwrap(delta[:, 0])
        found = [fit_breit_wigner(E, d_un, E_guess=E_b,
                                  Gamma_guess=G_b,
                                  window_au=max(0.03, 8 * G_b))]
        print(f"\n{name}: Burke E_r={E_b:.5f}, Gamma={G_b:.5f} Ha",
              flush=True)
        for f in (found if isinstance(found, list) else [found]):
            try:
                dE = f.E_r_au - E_b
                dG = f.Gamma_au - G_b
                print(f"  ab initio: E_r={f.E_r_au:.5f} ({dE:+.5f}, "
                      f"{dE*27.211:+.2f} eV), Gamma={f.Gamma_au:.5f} "
                      f"({dG:+.5f})", flush=True)
            except AttributeError:
                print("  extractor returned:", f, flush=True)
        # eigenphase structure: total pi-rise across the window
        print(f"  eigenphase range in window: "
              f"{d_un.min():.2f}..{d_un.max():.2f} rad "
              f"(a pi-rise resonance needs ~3.14)", flush=True)
