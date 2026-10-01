"""Dense continuum pseudostate virtuals: the virtual-orbital fix.

The DF solver only yields BOUND subshells (its guard correctly refuses
5d/6f in Ne) — the response needs the box-discretised CONTINUUM.
Wire the existing machinery: _rr_potential_matrices (local SCF
potential from the converged DF) + _continuum_spectrum (positive-
energy B-spline pseudostates per kappa) -> dense virtual set for RPA.
Watch TRK climb 0.08 -> N and alpha -> experiment.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.rpa import (dipole_polarizability, reduced_dipole,
                               solve_rpa, trk_sum)
from atomx.numerics.bspline.continuum import _continuum_spectrum
from atomx.scattering.radiative_recombination import \
    _rr_potential_matrices

OCC = [(1, -1), (2, -1), (2, 1), (2, -2)]


def main(e_cap=8.0):
    subs = [Subshell(n=n, kappa=k,
                     occupation=float(2 * k if k > 0 else -2 * k))
            for n, k in OCC]
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
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
    from atomx.scattering.radiative_recombination import (
        _residual_hartree_potential)
    from atomx.numerics.bspline.integrals import _assemble_integral
    # capture_label matching NO subshell => no exclusion => the FULL
    # V^N direct field (virtuals see all N electrons)
    potential, _residual = _residual_hartree_potential(r, "__none__")
    data = r._bsp_data
    extra = potential(data["gr"]) + r.Z / data["gr"]
    weights = data["grw"] * extra
    v_pp, v_qq = [
        _assemble_integral(data["nv"], data.get("ks" + c, data["ks"]),
                           data.get("ks" + c, data["ks"]), data["ns"],
                           data["bsp_" + c], data["bsp_" + c], weights)
        for c in ("p", "q")]
    print(f"SCF potential matrices built in {time.time()-t0:.0f}s",
          flush=True)

    virt = []
    for kap in (-1, 1, -2, 2, -3):
        ens, vecs, keep_p, keep_q = _continuum_spectrum(
            kap, 10, dhf.bsp_data, v_pp, v_qq)
        ns = dhf.bsp_data["ns"]
        cnt = 0
        for i, e in enumerate(ens):
            if e > e_cap:
                continue
            P = np.zeros(ns)
            Q = np.zeros(ns)
            for ii, k in enumerate(keep_p):
                P[k] = vecs[ii, i]
            for ii, k in enumerate(keep_q):
                Q[k - ns] = vecs[len(keep_p) + ii, i]
            key = (100 + cnt, kap)
            od[key] = (P, Q)
            eps[key] = e
            virt.append(key)
            cnt += 1
        print(f"  kappa={kap:+d}: {cnt} pseudostates < {e_cap} Ha",
              flush=True)

    print(f"\ntotal virtuals: {len(virt)}", flush=True)
    t0 = time.time()
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt,
                    twoJ=2, exc_parity=-1)
    print(f"RPA solved in {time.time()-t0:.0f}s: "
          f"{len(res.ph_pairs)} ph pairs", flush=True)
    trk = trk_sum(res, od, dhf.bsp_data, eps)
    a_rpa = dipole_polarizability(res, od, dhf.bsp_data)
    d = np.array([reduced_dipole(od, dhf.bsp_data, p, h)
                  for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    a_df = float(np.sum(2.0 / 3.0 * d ** 2 / de))
    print(f"\nTRK = {trk:.2f} / 10   (was 0.08 with 15 bound virtuals)")
    print(f"alpha_DF = {a_df:.3f}   alpha_RPA = {a_rpa:.3f}   "
          f"(experiment 2.669, RPA/DF = {a_rpa/a_df:.3f})")
    print(f"lowest pole = {res.omega_au[0]*27.2114:.2f} eV "
          f"(first allowed excitation: 16.85 eV)")


if __name__ == "__main__":
    main()
