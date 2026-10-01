"""RPA/TDHF first physics gate: Ne dipole polarizability.

DF-only (B=0, no coupling) vs full RPA: RPA must SCREEN (reduce) the
DF value; experiment alpha(Ne) = 2.669 a0^3.  TRK sum rule as sanity.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.rpa import (dipole_polarizability, reduced_dipole,
                               solve_rpa)

if __name__ == "__main__":
    occ = [(1, -1), (2, -1), (2, 1), (2, -2)]     # [He] 2s2 2p6 shells
    virt = [(3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
            (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, 3), (4, -4),
            (5, -1), (5, 1), (5, -2)]
    subs = [Subshell(n=n, kappa=k, occupation=float(
        2 * k if k > 0 else -2 * k)) for n, k in occ]
    subs += [Subshell(n=n, kappa=k, occupation=1e-6) for n, k in virt]
    t0 = time.time()
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        import re as _re
        m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    print(f"DF solved in {time.time()-t0:.0f}s; orbitals={len(od)}",
          flush=True)

    occ_keys = [k for k in occ if k in od]
    virt_keys = [k for k in virt if k in od and k in eps]
    t0 = time.time()
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt_keys,
                    twoJ=2, exc_parity=-1)
    print(f"RPA block solved in {time.time()-t0:.0f}s: "
          f"{len(res.ph_pairs)} ph pairs, {len(res.omega_au)} poles",
          flush=True)
    print("lowest poles (eV):",
          [f"{w*27.2114:.2f}" for w in res.omega_au[:5]], flush=True)

    d = np.array([reduced_dipole(od, dhf.bsp_data, p, h)
                  for p, h in res.ph_pairs])
    # DF-only sum-over-ph (no coupling): alpha_df = 2/3 sum |d|^2/de
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    alpha_df = float(np.sum(2.0 / 3.0 * d ** 2 / de))
    alpha_rpa = dipole_polarizability(res, od, dhf.bsp_data)
    from atomx.methods.rpa import trk_sum
    trk = trk_sum(res, od, dhf.bsp_data, eps)
    print(f"\nTRK sum S = {trk:.2f} (complete basis would give ~N=10; "
          f"the deficit quantifies the frozen-virtual completeness)",
          flush=True)
    print(f"\nalpha_DF  = {alpha_df:.3f} a0^3", flush=True)
    print(f"alpha_RPA = {alpha_rpa:.3f} a0^3", flush=True)
    print(f"experiment = 2.669 a0^3; RPA/DF = "
          f"{alpha_rpa/alpha_df:.3f} (screening must be < 1)",
          flush=True)
