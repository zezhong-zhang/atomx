"""Ne TDHF with canonical V^N bound and box-continuum virtual states.

The virtual complement retains exchange and native-metric orthogonality.
Response and oscillator-strength sums still require radial, box, energy,
and partial-wave convergence; TDHF need not decrease the uncoupled response.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.rpa import (
    dense_continuum_virtuals,
    dipole_polarizability,
    reduced_dipole,
    solve_rpa,
    trk_sum,
)

OCC = [(1, -1), (2, -1), (2, 1), (2, -2)]


def main(e_cap=8.0, ns=40):
    subs = [
        Subshell(n=n, kappa=k, occupation=float(2 * k if k > 0 else -2 * k))
        for n, k in OCC
    ]
    dhf = DFSolver(Z=10, subshells=subs, ns=ns, ks=7, starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
    import re as _re

    eps = {}
    for lab, e in r.orbital_energies.items():
        m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    occ_keys = list(od)

    t0 = time.time()
    vod, veps = dense_continuum_virtuals(r, e_cap=e_cap)
    od.update(vod)
    eps.update(veps)
    virt = list(vod)
    print(f"V^N virtual complement built in {time.time() - t0:.1f}s", flush=True)
    print(f"\ntotal virtuals: {len(virt)}", flush=True)
    t0 = time.time()
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt, twoJ=2, exc_parity=-1)
    print(
        f"RPA solved in {time.time() - t0:.0f}s: {len(res.ph_pairs)} ph pairs",
        flush=True,
    )
    trk = trk_sum(res, od, dhf.bsp_data, eps)
    a_rpa = dipole_polarizability(res, od, dhf.bsp_data)
    d = np.array([reduced_dipole(od, dhf.bsp_data, p, h) for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    a_df = float(np.sum(2.0 / 3.0 * d**2 / de))
    print(f"\nTRK = {trk:.2f} / 10   (was 0.08 with 15 bound virtuals)")
    print(
        f"alpha_DF = {a_df:.3f}   alpha_RPA = {a_rpa:.3f}   "
        f"(experiment 2.669, RPA/DF = {a_rpa / a_df:.3f})"
    )
    print(
        f"lowest pole = {res.omega_au[0] * 27.2114:.2f} eV "
        f"(first allowed excitation: 16.85 eV)"
    )

    return {
        "ns": ns,
        "energy_ceiling_Ha": e_cap,
        "virtuals": len(virt),
        "TRK": trk,
        "alpha_DF": a_df,
        "alpha_RPA": float(a_rpa),
    }


if __name__ == "__main__":
    main()
