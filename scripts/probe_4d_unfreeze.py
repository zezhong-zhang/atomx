"""4d-unfreeze probe (root-cause route, stage 1).

Pr3+: P = 4d10 4f2 vs P+Q with Q = single-4d-hole 4d->4f sectors
(4d9 4f3).  Fixed DF workspace (statistical 4f2, both runs same
orbitals).  Induced shifts on the 12 NIST-anchored levels -> fit the
EFFECTIVE s_k^(4d): how much of the fitted (-0.27, -0.08, -0.12) does
the 4d channel actually deliver?  (All orders within P+Q, leading 2nd.)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomx.methods.csf import slater_scaling as SS

CM = 219474.63
Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def cap(k):
    return 2 * k if k > 0 else -2 * k


# closed core WITHOUT 4d (4d added explicitly per pattern; a
# duplicated (n,kappa) in a pattern is silently unsatisfiable)
CLOSED = tuple(sorted((n, k, cap(k)) for n, k in Xe
                      if (n, k) not in [(4, 2), (4, -3)]))
# 4d: kappa=+2 -> d3/2 (cap 4); kappa=-3 -> d5/2 (cap 6)
D32, D52 = (4, 2), (4, -3)
F52, F72 = (4, 3), (4, -4)


VIRTUAL_EVEN = [(5, 2), (5, -3), (6, -1)]   # 5d3/2, 5d5/2, 6s1/2


def patterns_pr():
    """P = 4d10 4f2.  Q = SAME-PARITY single- and double-4d-hole
    sectors (a single 4d->4f excitation is PARITY-ODD and never
    couples in a parity-conserving CI — measured nnz=0):
      4d8 4f4      (double 4d->4f, the dominant semi-core channel)
      4d9 4f2 v    (4d->v with v even-l: 5d, 6s)
    """
    P = [CLOSED + (D32 + (4,), D52 + (6,), F52 + (a,),
                   F72 + (2 - a,)) for a in range(3)]
    Q = []
    # 4d8 4f4
    for q32 in range(2, 5):                 # d3/2 occ 2..4
        q52 = 8 - q32
        if not (0 <= q52 <= 6):
            continue
        for a in range(5):
            pat = CLOSED + (D32 + (q32,), D52 + (q52,), F52 + (a,),
                            F72 + (4 - a,))
            Q.append(pat)
    # 4d9 4f2 v
    for hd in (D32, D52):
        for a in range(3):
            for v in VIRTUAL_EVEN:
                pat = list(CLOSED) + [D32 + (4,), D52 + (6,),
                                      F52 + (a,), F72 + (2 - a,),
                                      v + (1,)]
                pat = [(n, k, q - 1 if (n, k) == hd else q)
                       for n, k, q in pat]
                Q.append(tuple(sorted(pat)))
    return P, Q


NIST_PR = {0: [21390.0], 1: [22012.0], 2: [4997.0, 17334.0, 23160.0],
           3: [6415.0], 4: [0.0, 6853.0, 9925.0], 5: [2152.0],
           6: [4389.0, 22215.0]}

if __name__ == "__main__":
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.physics.csf import sparse_ci as SC

    subs = [Subshell(n=n, kappa=k, occupation=float(cap(k)))
            for n, k in Xe]
    subs.append(Subshell(n=4, kappa=3, occupation=2 * 6 / 14))
    subs.append(Subshell(n=4, kappa=-4, occupation=2 * 8 / 14))
    for nk in [(5, 2), (5, -3), (6, -1)]:     # even-l correlating virtuals
        subs.append(Subshell(n=nk[0], kappa=nk[1], occupation=1e-6))
    dhf = DFSolver(Z=59, subshells=subs, ns=40, ks=7,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    bsp = dhf.bsp_data
    P, Q = patterns_pr()

    from collections import Counter
    for tag, pats in (("P", P), ("P+Q", P + Q)):
        cnt = Counter()
        for m2 in range(0, 15, 2):
            masks, _ = SC.enumerate_determinants(pats, od, m2)
            cnt[m2] = len(masks)
        print(f"{tag}: dets per M = {dict(cnt)}", flush=True)

    lvP = SS.free_ion_levels(od, bsp, 59, 40, P)
    lvPQ = SS.free_ion_levels(od, bsp, 59, 40, P + Q)

    def ladder_to_res(lv):
        allE = sorted(e for es in lv.values() for e in es)
        E0 = allE[0]
        return {(J, i): (sorted(es)[i] - E0)
                for J, es in lv.items() for i in range(len(es))}, E0

    rP, e0P = ladder_to_res(lvP)
    rPQ, e0PQ = ladder_to_res(lvPQ)
    print(f"\nabsolute ground shift from 4d sectors: "
          f"{(e0PQ - e0P):+.1f} cm^-1 (common-mode)", flush=True)
    print("\ninduced shifts on NIST-anchored levels (cm^-1):", flush=True)
    shifts = {}
    for J in sorted(NIST_PR):
        for i, ref in enumerate(NIST_PR[J]):
            d = rPQ[(J, i)] - rP[(J, i)]
            shifts[(J, i)] = d
            print(f"  J={J} #{i} (ref {ref:6.0f}): {d:+8.1f}", flush=True)

    # fit effective s_k^(4d) to the induced shifts via the slater
    # Jacobian of the P-space levels
    def pr_res(s):
        lv = SS.free_ion_levels(od, bsp, 59, 40, P, scale=s)
        allE = sorted(e for es in lv.values() for e in es)
        E0 = allE[0]
        return [(sorted(lv[J])[i] - E0) - ref for J in sorted(NIST_PR)
                for i, ref in enumerate(NIST_PR[J])]

    r0 = np.array(pr_res((0.0, 0.0, 0.0)))
    h = 0.02
    Jm = np.column_stack([(np.array(pr_res(e)) - r0) / h
                          for e in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
    target = np.array([shifts[(J, i)] for J in sorted(NIST_PR)
                       for i, _ in enumerate(NIST_PR[J])])
    s_4d, *_ = np.linalg.lstsq(Jm, target, rcond=None)
    resid = Jm @ s_4d - target
    print(f"\nEFFECTIVE s_k^(4d) = ({s_4d[0]:+.4f}, {s_4d[1]:+.4f}, "
          f"{s_4d[2]:+.4f})", flush=True)
    print("fitted total (needed): (-0.2726, -0.0804, -0.1193)", flush=True)
    print(f"s^(4d) fraction of total: "
          f"{s_4d[0]/-0.2726*100:.0f}% / {s_4d[1]/-0.0804*100:.0f}% / "
          f"{s_4d[2]/-0.1193*100:.0f}%  "
          f"(fit resid RMS {np.sqrt(np.mean(resid**2)):.0f} cm^-1)",
          flush=True)
