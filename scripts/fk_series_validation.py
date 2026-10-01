"""F^k series-transfer validation round 2 (2026-10-01).

(1) Refit Pr3+ WITHOUT the guessed 1S0 reference (robustness).
(2) PREDICT Nd3+ 4f3 with the SAME (s2,s4,s6): 4I multiplet (SO
    control) + 4F3/2 (the F^k-sensitive probe, Wyart 2006 free-ion
    11698.6; 4I11/2 = 1897.07).
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
exec(open("/tmp/atomx-cimbpt/scripts/fk_transfer_experiment.py").read()
     .split('if __name__')[0])

# --- (1) Pr refit without 1S0 -------------------------------------------
NIST_PR_CLEAN = {0: [21390.0], 1: [22012.0], 2: [4997.0, 17334.0,
                                                 23160.0],
                 3: [6415.0], 4: [0.0, 6853.0, 9925.0], 5: [2152.0],
                 6: [4389.0, 22215.0]}
globals()["NIST_PR"] = NIST_PR_CLEAN

od, bsp, pats = build(59, 2)
h = 0.02
x = np.zeros(3)
for it in range(6):
    r0 = np.array(pr_residuals(tuple(x), od, bsp, pats))
    Jm = np.column_stack([
        (np.array(pr_residuals(tuple(x + e), od, bsp, pats)) - r0) / h
        for e in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
    step, *_ = np.linalg.lstsq(Jm, -r0, rcond=None)
    r1 = np.array(pr_residuals(tuple(x + step), od, bsp, pats))
    if np.sqrt(np.mean(r1 ** 2)) >= np.sqrt(np.mean(r0 ** 2)) - 1:
        break
    x = x + step
s2, s4, s6 = x
print(f"(1) Pr refit (no 1S0): (s2,s4,s6) = ({s2:+.4f}, {s4:+.4f}, "
      f"{s6:+.4f}), RMS = {np.sqrt(np.mean(r0 ** 2)):.0f} cm^-1",
      flush=True)

# --- (2) Nd3+ prediction, same scalings ---------------------------------
NIST_ND = {9: {"4I9/2": 0.0}, 11: {"4I11/2": 1897.0},
           13: {"4I13/2": 3900.0}, 15: {"4I15/2": 5850.0},
           3: {"4F3/2": 11699.0}}
print("\n(2) Nd3+ 4f3 with the SAME scalings (prediction, no Nd fit):",
      flush=True)
od_nd, bsp_nd, pats_nd = build(60, 3)
for tag, sc in [("bare", (0.0, 0.0, 0.0)), ("Pr-scaled", (s2, s4, s6))]:
    lv = {}
    for twoJ in (3, 9, 11, 13, 15, 17):
        masks, spinors = SC.enumerate_determinants(pats_nd, od_nd, twoJ)
        if not masks:
            continue
        if sc == (0.0, 0.0, 0.0):
            w, _ = SC.sparse_ci_spectrum(od_nd, bsp_nd, 60, 50, pats_nd,
                                         twoJ, n_states=10 ** 7)
            w_hi, _ = SC.sparse_ci_spectrum(od_nd, bsp_nd, 60, 50,
                                            pats_nd, twoJ + 2,
                                            n_states=10 ** 7)
        else:
            delta = make_slater_delta(od_nd, bsp_nd, 60, sc)
            cache = {}
            def ev(m2):
                if m2 not in cache:
                    msk, spn = SC.enumerate_determinants(pats_nd, od_nd,
                                                         m2)
                    if not msk:
                        cache[m2] = np.array([])
                        return cache[m2]
                    dd = lambda a, b, c, d: delta(a, b, c, d, spn)
                    H = SC.assemble_sparse_hamiltonian(msk, spn, od_nd,
                                                       bsp_nd, 60, 50,
                                                       two_body_delta=dd)
                    cache[m2] = np.linalg.eigvalsh(H.toarray())
                return cache[m2]
            w, w_hi = ev(twoJ), ev(twoJ + 2)
        wl = list(w)
        for xv in w_hi:
            j = min(range(len(wl)), key=lambda i: abs(wl[i] - xv))
            wl.pop(j)
        lv[twoJ / 2] = sorted(x * CM for x in wl)
    e0 = lv[4.5][0]                       # 4I9/2 ground
    print(f"  [{tag}]", flush=True)
    for J, d in NIST_ND.items():
        term, ref = next(iter(d.items()))
        e = lv[J / 2][0] - e0
        pct = (e - ref) / ref * 100 if ref > 0 else 0.0
        print(f"    {term:8s}: {e:9.1f} vs {ref:9.1f}  "
              f"({pct:+6.1f}%)", flush=True)
