"""Eu3+ Tier-A anchor calibration (F^k-class gaps only).

Fit (s2,s4,s6) on the 5D0/5D1/5D2 inter-multiplet gaps (vs 7F0):
F^k-class observables only — fine-structure (zeta-class) residuals
must stay OUT of the fit (measured: mixing them takes s6=-0.64 and
doubles the RMS).  Series-consistency gate: the fitted s must land
within scatter of the Pr source fit (-0.273, -0.080, -0.119).

References: 5D0 = 17270 (free ion), 5D1 = 19025 +- 20, 5D2 = 21500 +- 50
(Carnall aquo-ion series IV; free-ion within the uncertainty).
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
sys.path.insert(0, "/tmp/atomx-cimbpt/tests")

from atomx.methods.csf.slater_scaling import fit_slater_scaling
from test_slater_scaling import _build

REFS = [(0.0, 17270.0), (1.0, 19025.0), (2.0, 21500.0)]
NIST_7F = [0.0, 370.0, 1041.0, 1890.0, 2862.0, 3807.0, 4940.0]


def extract(lv):
    allL = sorted((e, J) for J, es in lv.items() for e in es)
    e0 = allL[0][0]
    out = {}
    for J, ref in REFS:
        thr = 15000 if J != 0.0 else 15000
        e = min(e for e, j in allL if j == J and e - e0 > thr)
        out[J] = e - e0
    return out, e0


if __name__ == "__main__":
    od, bsp, pats = _build(63, 6, ns=50)
    from atomx.methods.csf.slater_scaling import free_ion_levels

    def res(s):
        lv = free_ion_levels(od, bsp, 63, 50, pats, scale=s)
        gaps, _ = extract(lv)
        return [gaps[J] - ref for J, ref in REFS]

    x, hist = fit_slater_scaling(res, tol=0.05, max_iter=10)
    s2, s4, s6 = x
    print(f"Eu Tier-A fit (5D0/1/2 gaps): s = ({s2:+.4f}, {s4:+.4f}, "
          f"{s6:+.4f})", flush=True)
    print("Pr series source:            s = (-0.2726, -0.0804, -0.1193)",
          flush=True)
    dev = [abs(a - b) for a, b in zip(x, (-0.2726, -0.0804, -0.1193))]
    print(f"series scatter: {dev[0]:.3f} / {dev[1]:.3f} / {dev[2]:.3f}"
          f"  -> {'CONSISTENT' if max(dev) < 0.08 else 'DRIFT — flag'}",
          flush=True)

    print("\ncalibrated anchor ladder vs NIST:", flush=True)
    lv = free_ion_levels(od, bsp, 63, 50, pats, scale=tuple(x))
    gaps, _ = extract(lv)
    for J, ref in REFS:
        print(f"  5D{J:.0f}: {gaps[J]:9.1f} vs {ref:9.1f} "
              f"({gaps[J]-ref:+7.1f})", flush=True)
    lad, used = [], set()
    for e, J in sorted((e, J) for J, es in lv.items() for e in es):
        if len(lad) >= 7 or J in used:
            continue
        lad.append(e)
        used.add(J)
    e0 = lad[0]
    print("  7F fine structure (NOT fitted — stays at zeta level):",
          flush=True)
    for j in range(1, 7):
        pct = (lad[j] - e0 - NIST_7F[j]) / NIST_7F[j] * 100
        print(f"    7F{j}: {lad[j]-e0:9.1f} vs {NIST_7F[j]:8.1f} "
              f"({pct:+5.1f}%)", flush=True)
