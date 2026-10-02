"""atomx vs XATOM (xraypac Rev96) on Ne: photoionization @2keV + KLL Auger.

XATOM reference (HFS nonrelativistic config-average):
  sigma(Mb) @ 2000 eV: 1s 3.91847e-2, 2s 1.88919e-3, 2p 4.90806e-4
  Auger (au), 1s hole: -2s2s 8.70666e-4 (KE 815.3 eV),
    -2s2p 2.65878e-3 (837.5), -2p2p 6.15151e-3 (859.7); total 9.68095e-3
atomx: closed-shell Ne DF; per-subshell V^(N-1) continuum; relativistic
per-kappa channels summed for the 2p comparison.  1 a0^2 = 28.002852 Mb.
"""
import re
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.numerics.bspline.continuum import solve_continuum
from atomx.scattering.photoionization.direct import (
    photoionization_cross_section)
from atomx.scattering.radiative_recombination import (
    _rr_potential_matrices)

A02_MB = 28.002852
AU_S = 4.134137e16
X_SIG = {"1s": 3.91847e-2, "2s": 1.88919e-3, "2p": 4.90806e-4}
X_AUG = {"2s2s": 8.70666e-4, "2s2p": 2.65878e-3,
         "2p2p": 6.15151e-3}
X_AUG_TOT = 9.68095e-3


def solve_ne(neutral=True):
    occ = [(1, -1, 1.0 if not neutral else 2.0),
           (2, -1, 2.0), (2, 1, 2.0), (2, -2, 4.0)]
    subs = [Subshell(n=n, kappa=k, occupation=o) for n, k, o in occ]
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7,
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
    return r, od, eps


def sigma_benchmark():
    r, od, eps = solve_ne(neutral=True)
    photon = np.array([2000.0 / 27.2114])
    rows = []
    for shell, keys in [("1s", [(1, -1)]), ("2s", [(2, -1)]),
                        ("2p", [(2, 1), (2, -2)])]:
        tot = 0.0
        for key in keys:
            sh = next(s for s in r.subshells
                      if (s.n, s.kappa) == key)
            v_pp, v_qq = _rr_potential_matrices(r, sh, None)
            P, Q = od[key]
            s = photoionization_cross_section(
                P, Q, key[1], eps[key], photon, 10, r._bsp_data,
                V_scf_PP=v_pp, V_scf_QQ=v_qq)
            tot += float(s[0])
        rows.append((shell, tot * A02_MB))
    print("=== photoionization @ 2000 eV (Mb) ===")
    print(f"{'shell':>5} {'atomx':>12} {'XATOM':>12} {'ratio':>7}")
    for shell, mb in rows:
        print(f"{shell:>5} {mb:12.4e} {X_SIG[shell]:12.4e} "
              f"{mb / X_SIG[shell]:7.3f}")


def auger_benchmark():
    r, od, eps = solve_ne(neutral=False)
    from atomx.scattering.autoionization import auger_rate_csf_average
    # _coarse_auger_rate reads result.orbitals_dict — populate it from
    # the DF solve (the Atom-engine path provides it; raw DFSolver
    # results do not carry it)
    r.orbitals_dict = od
    lab = {(1, -1): "1s1/2", (2, -1): "2s1/2", (2, 1): "2p1/2",
           (2, -2): "2p3/2"}
    chans = {"2s2s": ([[(2, -1), (2, -1)]], 815.26),
             "2s2p": ([[(2, -1), (2, 1)], [(2, -1), (2, -2)]], 837.50),
             "2p2p": ([[(2, 1), (2, 1)], [(2, 1), (2, -2)],
                      [(2, -2), (2, -2)]], 859.74)}
    print("\n=== 1s-hole KLL Auger (au) ===")
    print(f"{'chan':>9} {'atomx':>12} {'XATOM':>12} {'ratio':>7}")
    tot = 0.0
    for name, spec in chans.items():
        pairs, ke = spec
        rate = 0.0
        for s1, s2 in pairs:
            out = auger_rate_csf_average(
                r, hole_label="1s1/2", source_label=lab[s1],
                source2_label=lab[s2], resonance_energy_eV=ke)
            rate += out["Gamma_per_s"]
        rate_au = rate / AU_S
        tot += rate_au
        print(f"{name:>9} {rate_au:12.4e} {X_AUG[name]:12.4e} "
              f"{rate_au / X_AUG[name]:7.3f}")
    print(f"{'total':>9} {tot:12.4e} {X_AUG_TOT:12.4e} "
          f"{tot / X_AUG_TOT:7.3f}")


if __name__ == "__main__":
    sigma_benchmark()
    auger_benchmark()
