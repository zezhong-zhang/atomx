"""F^k precision: the series-transferability experiment (2026-10-01).

Fit (s2, s4, s6) Slater-integral scalings on Pr3+ 4f2 NIST levels, then
APPLY THE SAME scalings unchanged to Eu3+ 4f6 and predict 5D0.
Transfer works => a 3-parameter series calibration (R4) suffices;
fails => per-ion Cowan fits (R0) stay.

Scaling = two_body_delta: element' = sum_k (1+s_k) coef_k R^k.
"""
import sys
from collections import Counter

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.angular.csf_hamiltonian import two_body_angular, twoj
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.excitations import generate_sd_excitations
from atomx.physics.csf import sparse_ci as SC
from atomx.physics.csf.sparse_ci import patterns_from_csfs
import atomx.physics.csf.matrix as MAT

CM = 219474.63
Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def _cap(k):
    return 2 * k if k > 0 else -2 * k


def build(Z, nval):
    """Frozen [Xe] statistical-4f DF workspace for Z with 4f^nval."""
    subs = [Subshell(n=n, kappa=k, occupation=float(_cap(k)))
            for n, k in Xe]
    subs.append(Subshell(n=4, kappa=3, occupation=nval * 6 / 14))
    subs.append(Subshell(n=4, kappa=-4, occupation=nval * 8 / 14))
    dhf = DFSolver(Z=Z, subshells=subs, ns=50, ks=9,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    REF = ("1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f" + str(nval))
    cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                                   max_excitations=0)
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
    return od, dhf.bsp_data, pats


def make_slater_delta(od, bsp, scale):
    """two_body_delta: additive antisymmetrised Slater scaling.

    The assembler forms anti = v(ab,cd) - v(ab,dc) + delta(ab,cd), so
    the CORRECT scaling increment is
        delta = sum_k s_k [term_k(ab,cd) - term_k(ab,dc)]
    (scaling the raw element only in the first slot would break the
    antisymmetrisation).
    """
    from functools import lru_cache
    s2, s4, s6 = scale
    scale_of = {2: s2, 4: s4, 6: s6}

    def delta(ia, ib, ic, id_, spinors):
        a, b, c, d = (spinors[ia], spinors[ib], spinors[ic],
                      spinors[id_])
        t1 = two_body_angular(a[1], a[2], b[1], b[2],
                              c[1], c[2], d[1], d[2])
        t2 = two_body_angular(a[1], a[2], b[1], b[2],
                              d[1], d[2], c[1], c[2])
        if not t1 and not t2:
            return 0.0
        Pa, Qa = od[(a[0], a[1])]
        Pc, Qc = od[(c[0], c[1])]
        Pb, Qb = od[(b[0], b[1])]
        Pd, Qd = od[(d[0], d[1])]
        total = 0.0
        for k, coef in t1:
            sk = scale_of.get(k)
            if sk:
                total += sk * coef * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb,
                                             Pd, Qd, bsp, k)
        for k, coef in t2:
            sk = scale_of.get(k)
            if sk:
                total -= sk * coef * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb,
                                             Pd, Qd, bsp, k)
        return total
    return delta


def levels_with_scale(od, bsp, Z, pats, scale, twoJmax=6):
    """J-ladder with Slater scaling applied (reference path, one spinor
    table per sector — small spaces, dense diagonalisation)."""
    out = []
    sector_cache = {}
    for twoJ in range(0, 2 * twoJmax + 2, 2):
        masks, spinors = SC.enumerate_determinants(pats, od, twoJ)
        w = w_hi = None
        if scale == (0.0, 0.0, 0.0):
            w_hi, _ = SC.sparse_ci_spectrum(od, bsp, Z, 50, pats,
                                            twoJ + 2, n_states=10 ** 7)
            w, _ = SC.sparse_ci_spectrum(od, bsp, Z, 50, pats, twoJ,
                                         n_states=10 ** 7)
        if scale != (0.0, 0.0, 0.0):

            def sector_eval(m2):
                if m2 not in sector_cache:
                    delta = make_slater_delta(od, bsp, scale)
                    msk, spn = SC.enumerate_determinants(pats, od, m2)

                    def dd(ia, ib, ic, id_):
                        return delta(ia, ib, ic, id_, spn)

                    H = SC.assemble_sparse_hamiltonian(msk, spn, od,
                                                       bsp, Z, 50,
                                                       two_body_delta=dd)
                    sector_cache[m2] = np.linalg.eigvalsh(H.toarray())
                return sector_cache[m2]

            w = sector_eval(twoJ)
            w_hi = sector_eval(twoJ + 2)
        # multiset subtract
        pure = sorted(w.tolist() + [-x for x in w_hi.tolist()])
        # proper subtraction: remove w_hi entries from w
        wl = list(w)
        for x in w_hi:
            j = min(range(len(wl)), key=lambda i: abs(wl[i] - x))
            wl.pop(j)
        out.append((twoJ / 2, [x * CM for x in wl]))
    return out


NIST_PR = {0: [21390.0, 50000.0], 1: [22012.0], 2: [4997.0, 17334.0,
                                                   23160.0],
           3: [6415.0], 4: [0.0, 6853.0, 9925.0], 5: [2152.0],
           6: [4389.0, 22215.0]}


def pr_residuals(scale, od, bsp, pats, cache={}):
    key = tuple(round(s, 8) for s in scale)
    if key in cache:
        return cache[key]
    lv = levels_with_scale(od, bsp, 59, pats, scale)
    allE = sorted(e for J, es in lv for e in es)
    E0 = allE[0]                      # 3H4 (global ground), not J=0's
    res = []
    for J, es in lv:
        for e, ref in zip(sorted(es), sorted(NIST_PR[J])):
            res.append((e - E0) - ref)
    cache[key] = res
    return res


if __name__ == "__main__":
    print("building Pr3+ workspace...", flush=True)
    od_pr, bsp_pr, pats_pr = build(59, 2)
    print("fitting (s2,s4,s6) on Pr3+ 12 NIST levels...", flush=True)
    r0 = pr_residuals((0.0, 0.0, 0.0), od_pr, bsp_pr, pats_pr)
    print("  bare RMS:", f"{np.sqrt(np.mean(np.square(r0))):.0f} cm^-1",
          flush=True)
    # level energies are analytic (nearly linear) in the scalings ->
    # one numerical Jacobian + linear solve converges in a couple of
    # refinement rounds
    base = np.array(pr_residuals((0.0, 0.0, 0.0), od_pr, bsp_pr, pats_pr))
    h = 0.02
    Jm = np.column_stack([
        (np.array(pr_residuals(tuple(e), od_pr, bsp_pr, pats_pr)) - base)
        / h for e in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
    step, *_ = np.linalg.lstsq(Jm, -base, rcond=None)
    x = np.array([0.0, 0.0, 0.0])
    for _ in range(2):
        r1 = np.array(pr_residuals(tuple(x + step), od_pr, bsp_pr,
                                   pats_pr))
        if np.sqrt(np.mean(r1 ** 2)) < \
                np.sqrt(np.mean((np.array(pr_residuals(tuple(x), od_pr,
                                                       bsp_pr, pats_pr))
                                 - 0) ** 2)):
            x = x + step
        Jm = np.column_stack([
            (np.array(pr_residuals(tuple(x + e2), od_pr, bsp_pr, pats_pr))
             - np.array(pr_residuals(tuple(x), od_pr, bsp_pr, pats_pr)))
            / h for e2 in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
        step, *_ = np.linalg.lstsq(Jm, -r1, rcond=None)
    s2, s4, s6 = x
    rf = pr_residuals(tuple(fit.x), od_pr, bsp_pr, pats_pr)
    print(f"  fitted s = ({s2:+.4f}, {s4:+.4f}, {s6:+.4f})", flush=True)
    print(f"  fitted RMS: {np.sqrt(np.mean(np.square(rf))):.0f} cm^-1",
          flush=True)

    print("\nTRANSFER to Eu3+ 4f6 with the SAME scalings...", flush=True)
    od_eu, bsp_eu, pats_eu = build(63, 6)
    lv0 = levels_with_scale(od_eu, bsp_eu, 63, pats_eu, (0, 0, 0))
    lv1 = levels_with_scale(od_eu, bsp_eu, 63, pats_eu,
                            (s2, s4, s6))
    for tag, lv in [("bare", lv0), ("Pr-scaled", lv1)]:
        e7f0 = min(lv[0][1]) if lv[0][1] else None
        allL = sorted((e, J) for J, es in lv for e in es)
        e_7f0 = allL[0][0]
        # 5D0: lowest J=0 level above ~15000
        e_5d0 = min(e for e, J in allL if J == 0 and e - e_7f0 > 10000)
        print(f"  {tag}: 5D0 = {e_5d0 - e_7f0:9.1f} cm^-1 "
              f"(NIST 17270, gap {e_5d0 - e_7f0 - 17270:+.0f})", flush=True)
    print("\nVERDICT: transferability " +
          ("WORKS" if abs((e_5d0 - e_7f0) - 17270) < 500 else
           "PARTIAL" if abs((e_5d0 - e_7f0) - 17270) < 1500 else
           "FAILS"), flush=True)
