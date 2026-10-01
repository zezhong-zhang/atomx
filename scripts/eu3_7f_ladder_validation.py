"""Eu3+ 4f^6 7F_J ground-multiplet ladder vs NIST (tier-1 lightweight).

The 5D0 anchor residual (+3100 cm^-1) lives in F^k accuracy; the 7F_J
SPACINGS are spin-orbit-dominated and far less F^k-sensitive.  This run
separates "machinery" from "exchange-integral precision" on the SAME
ion the anchor problem lives on.  Single-configuration CI (4f6 only).

NIST/literature free-ion 7F_J (cm^-1): 0, 370, 1041, 1890, 2862,
3807, 4940 (literature-memory values, % level comparison only).
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.excitations import generate_sd_excitations
from atomx.physics.csf.sparse_ci import (patterns_from_csfs,
                                         sparse_j_levels)

CM = 219474.63
NIST_7F = [0.0, 370.0, 1041.0, 1890.0, 2862.0, 3807.0, 4940.0]

Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def _cap(k):
    return 2 * k if k > 0 else -2 * k


subs = [Subshell(n=n, kappa=k, occupation=float(_cap(k))) for n, k in Xe]
subs.append(Subshell(n=4, kappa=3, occupation=6 * 6 / 14))
subs.append(Subshell(n=4, kappa=-4, occupation=6 * 8 / 14))
dhf = DFSolver(Z=63, subshells=subs, ns=50, ks=9,
               starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
      in r.orbitals_by_subshell().items()}
bsp = dhf.bsp_data
print("DF solved (Eu3+ [Xe]4f6)", flush=True)

REF = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f6"
cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                               max_excitations=0)
pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
print(f"4f6 patterns: {len(pats)}", flush=True)

levels = []
for twoJ in range(0, 14, 2):        # J = 0..6; full spectra (dense)
    w, n_j, n_hi = sparse_j_levels(od, bsp, 63, 50, pats, twoJ,
                                   n_states=10 ** 7)
    print(f"  J={twoJ/2}: M-sector dets {n_j} (J+1: {n_hi}), "
          f"pure-J levels: {len(w)}", flush=True)
    for e in w:
        levels.append((twoJ / 2, e * CM))

levels.sort(key=lambda t: (t[1], t[0]))
E0 = levels[0][1]
print("\nlowest 4f6 levels (rel. cm^-1):", flush=True)
for J, e in levels[:12]:
    print(f"  J={J:.0f}  {e - E0:10.1f}", flush=True)

# the 7F ladder = the 7 lowest levels with J = 0..6 each appearing once
lad = []
used = set()
for J, e in levels:
    if len(lad) >= 7:
        break
    if J in used:
        continue
    lad.append((J, e - E0))
    used.add(J)
print("\n7F_J ladder vs NIST:", flush=True)
pcts = []
for (J, e), ref in zip(lad, NIST_7F):
    pct = (e - ref) / ref * 100 if ref > 0 else float("nan")
    if ref > 0:
        pcts.append(pct)
    print(f"  7F{J:.0f}: {e:9.1f} vs {ref:9.1f}  "
          f"({pct:+6.1f}%)" if ref > 0 else
          f"  7F{J:.0f}: {e:9.1f} (reference)", flush=True)
print(f"\nmean |deviation| (J>=1): {np.mean(np.abs(pcts)):.1f}%",
      flush=True)
print("MACHINERY VERDICT: 7F spacings track NIST at the % level"
      if np.mean(np.abs(pcts)) < 15 else "LARGE deviation — investigate",
      flush=True)
