"""Virtual-basis density experiment: does the response completeness
deficit (TRK 0.08/10) close with a DENSE pseudostate virtual set?

Sweep n_max of the virtual set (all kappa s,p,d,f up to n_max):
watch TRK -> N=10, alpha_DF / alpha_RPA -> experiment 2.669 a0^3,
lowest pole -> 16.85 eV (first allowed 2p^5 3s).
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.rpa import (dipole_polarizability, solve_rpa,
                               trk_sum)

OCC = [(1, -1), (2, -1), (2, 1), (2, -2)]


def virtuals_up_to(nmax):
    out = []
    for n in range(3, nmax + 1):
        for l, kappas in [(0, (-1,)), (1, (1, -2)),
                          (2, (2, -3)), (3, (3, -4))]:
            if n <= l:
                continue
            out += [(n, k) for k in kappas]
    return out


def run(nmax, ns=40):
    virt = virtuals_up_to(nmax)
    subs = [Subshell(n=n, kappa=k,
                     occupation=float(2 * k if k > 0 else -2 * k))
            for n, k in OCC]
    subs += [Subshell(n=n, kappa=k, occupation=1e-6) for n, k in virt]
    t0 = time.time()
    dhf = DFSolver(Z=10, subshells=subs, ns=ns, ks=7,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        m = re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    occ_keys = [k for k in OCC if k in od]
    virt_keys = [k for k in virt if k in od and k in eps]
    t1 = time.time()
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt_keys,
                    twoJ=2, exc_parity=-1)
    t2 = time.time()
    trk = trk_sum(res, od, dhf.bsp_data, eps)
    a_rpa = dipole_polarizability(res, od, dhf.bsp_data)
    from atomx.methods.rpa import reduced_dipole
    d = np.array([reduced_dipole(od, dhf.bsp_data, p, h)
                  for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    a_df = float(np.sum(2.0 / 3.0 * d ** 2 / de))
    print(f"nmax={nmax:2d}: virt={len(virt_keys):3d} "
          f"pairs={len(res.ph_pairs):3d}  TRK={trk:6.2f}/10  "
          f"a_DF={a_df:6.3f}  a_RPA={a_rpa:6.3f}  "
          f"pole1={res.omega_au[0]*27.2114:5.2f} eV  "
          f"[DF {t1-t0:.0f}s + RPA {t2-t1:.0f}s]", flush=True)


if __name__ == "__main__":
    for nmax in (5, 6, 7):
        try:
            run(nmax)
        except Exception as exc:  # noqa: BLE001
            print(f"nmax={nmax}: FAILED {exc}", flush=True)
