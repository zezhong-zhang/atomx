"""Sm3+ 4f^5 series transfer validation (half-integer J first case).

PREDICTION with the Pr-fit (s2,s4,s6)=(-0.2726,-0.0804,-0.1193)
unchanged: the 6H_J fine structure + inter-multiplet gaps.
Refs (Carnall aquo, +-tolerance): 6H5/2=0, 6H7/2~1080, 6H9/2~2290,
6H11/2~3600 (approximate — the gates use loose bands).
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")
sys.path.insert(0, "/tmp/atomx-cimbpt/tests")

from atomx.methods.csf.slater_scaling import free_ion_levels
from test_slater_scaling import _build

REFS = {2.5: (0.0, "6H5/2"), 3.5: (1080.0, "6H7/2"),
        4.5: (2290.0, "6H9/2"), 5.5: (3600.0, "6H11/2")}
S_PR = (-0.2726, -0.0804, -0.1193)

if __name__ == "__main__":
    od, bsp, pats = _build(62, 5, ns=50)
    print(f"Sm3+ 4f5 workspace; patterns={len(pats)}", flush=True)
    for tag, s in [("bare", None), ("Pr-scaled", S_PR)]:
        lv = free_ion_levels(od, bsp, 62, 50, pats, scale=s)
        allL = sorted((e, J) for J, es in lv.items() for e in es)
        e0 = allL[0][0]
        lad, used = [], set()
        for e, J in allL:
            if len(lad) >= 6 or J in used:
                continue
            lad.append((J, e - e0))
            used.add(J)
        print(f"\n[{tag}] lowest-J ladder:", flush=True)
        for J, e in lad:
            ref = REFS.get(J, (None, ""))[0]
            pct = (e - ref) / ref * 100 if ref else float("nan")
            print(f"  J={J:3.1f}: {e:9.1f} cm^-1"
                  + (f"  ref~{ref:7.0f} ({pct:+6.1f}%)" if ref else ""),
                  flush=True)
    # half-integer sector integrity: J values all half-integers
    js = sorted({J for J, _ in lad})
    assert all(abs(2 * J - round(2 * J)) == 1.0 for J in js), js
    print("\nHALF-INTEGER SECTOR INTEGRITY PASS", flush=True)
