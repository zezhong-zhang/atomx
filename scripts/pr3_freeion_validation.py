"""Pr3+ 4f^2 free-ion multiplets vs NIST (Sugar 1965 / Carnall).

Tier-2 lightweight validation: two f electrons = the first real
multiplet test (e-e + spin-orbit).  Single-configuration CI (4f2 only,
no SD excitations), frozen [Xe]-core DF with statistical 4f occupation.

NIST free-ion (cm^-1): 3H4=0, 3H5=2152, 3H6=4389, 3F2=4997, 3F3=6415,
3F4=6853, 1G4=9925, 1D2=17334, 3P0=21390, 3P1=22012, 1I6=22215,
3P2=23160.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.excitations import generate_sd_excitations
from atomx.physics.csf.sparse_ci import (enumerate_determinants,
                                         patterns_from_csfs, sparse_j_levels)

CM = 219474.63
NIST = {"3H4": 0.0, "3H5": 2152.0, "3H6": 4389.0, "3F2": 4997.0,
        "3F3": 6415.0, "3F4": 6853.0, "1G4": 9925.0, "1D2": 17334.0,
        "3P0": 21390.0, "3P1": 22012.0, "1I6": 22215.0, "3P2": 23160.0}

Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def _cap(k):
    return 2 * k if k > 0 else -2 * k


subs = [Subshell(n=n, kappa=k, occupation=float(_cap(k))) for n, k in Xe]
subs.append(Subshell(n=4, kappa=3, occupation=2 * 6 / 14))
subs.append(Subshell(n=4, kappa=-4, occupation=2 * 8 / 14))
dhf = DFSolver(Z=59, subshells=subs, ns=50, ks=9,
               starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
      in r.orbitals_by_subshell().items()}
bsp = dhf.bsp_data
print("DF solved", flush=True)

REF = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f2"
cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                               max_excitations=0)
pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
print(f"4f2 patterns: {len(pats)}", flush=True)

levels = []
for twoJ in range(0, 14, 2):        # J = 0..6
    w, n, _ = sparse_j_levels(od, bsp, 59, 50, pats, twoJ, n_states=30)
    for e in w:
        levels.append((twoJ / 2, e * CM))
    print(f"  J={twoJ/2}: {n} dets, levels:",
          [f"{e*CM:.0f}" for e in w[:6]], flush=True)

levels.sort(key=lambda t: t[1])
E0 = levels[0][1]
print("\nfree-ion ladder (rel. cm^-1) with J:")
for J, e in levels[:20]:
    print(f"  J={J:.0f}  {e - E0:10.1f}", flush=True)

# assignment: expected per-J multiplicities in 4f2:
# J=0: 2 (3P0, 1S0), J=1: 1 (3P1), J=2: 3 (3F2,1D2,3P2),
# J=3: 1 (3F3), J=4: 3 (3H4,3F4,1G4), J=5: 1 (3H5), J=6: 2 (3H6,1I6)
from collections import Counter
cnt = Counter(J for J, _ in levels)
print("\nper-J level counts:", dict(sorted(cnt.items())), flush=True)
expect = {0: 2, 1: 1, 2: 3, 3: 1, 4: 3, 5: 1, 6: 2}
assert all(cnt[J] == n for J, n in expect.items()), (cnt, expect)
print("MULTIPLICITY GATE PASS (4f2 term structure complete)", flush=True)

# NIST comparison: match by energy order within each J
nist_by_J = {}
for term, e in NIST.items():
    J = int(term.split(("P" if "P" in term else
                        "D" if "D" in term else
                        "F" if "F" in term else
                        "G" if "G" in term else
                        "H" if "H" in term else "I"))[-1])
    nist_by_J.setdefault(J, []).append((e, term))
print("\nNIST comparison (term: calc vs nist, %):", flush=True)
rows = []
for J in sorted(nist_by_J):
    mine = sorted(e for j, e in levels if j == J)
    ref = sorted(nist_by_J[J])
    for (re_, term), ec in zip(ref, mine):
        pct = (ec - E0 - re_) / re_ * 100 if re_ > 0 else float("nan")
        rows.append((re_, term, ec - E0, pct))
        print(f"  {term}: {ec - E0:9.1f} vs {re_:9.1f}  "
              f"({pct:+6.1f}%)", flush=True)
