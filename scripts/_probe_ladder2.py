"""Instrument delta_ladder internals step by step on Be (2s2s diagonal)."""
import re

import numpy as np

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt_sigma2 import build_sigma2_ladder


def kap(label):
    m = re.fullmatch(r"(\d+)([spdfgh])(\d+)/(\d+)", label)
    n = int(m.group(1))
    ll = "spdfgh".index(m.group(2))
    num = int(m.group(3))
    den = int(m.group(4))
    j2 = int(round(2 * num / den))
    return n, (-(ll + 1) if j2 == 2 * ll + 1 else ll)


def keys_of(shells):
    out = []
    for v in shells:
        n = int(v[0])
        ll = "spdfgh".index(v[1])
        for j2 in (2 * ll - 1, 2 * ll + 1):
            if j2 <= 0:
                continue
            out.append((n, -(ll + 1) if j2 == 2 * ll + 1 else ll))
    return out


extra = [(2, 1), (2, -2), (3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
         (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, -4),
         (5, -1), (5, 1), (5, -2), (5, 2), (5, -3), (5, -4)]
dhf = DFSolver(Z=4, subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                                Subshell(n=2, kappa=-1, occupation=2.0)] +
               [Subshell(n=nk[0], kappa=nk[1], occupation=1e-6)
                for nk in extra],
               ns=50, ks=9, starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
eps = {kap(l): e for l, e in r.orbital_energies.items()}

VAL = [k for k in keys_of(["2s", "2p", "3s", "3p", "3d"]) if k in od]
Q = [k for k in keys_of(["4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f"])
     if k in od and k in eps]
print("VAL:", sorted(set(VAL)))
print("Q:", sorted(set(Q)))

lad = build_sigma2_ladder(od, dhf.bsp_data, 4, 50, virt_keys=Q,
                           valence_keys=VAL, eps=eps)
up, dn = lad.index[(2, -1, 1)], lad.index[(2, -1, -1)]
print("delta_ladder(2s2s) =", lad(up, dn, up, dn))

# --- manual mirror of delta_ladder internals (cannot reach closures, so
# rebuild the same tables here with the module's exposed spinors) ---
import atomx.physics.csf.matrix as MAT
from atomx.angular.csf_hamiltonian import two_body_angular

spinors = lad.spinors
virt_idx = [i for i, s in enumerate(spinors) if s[:2] in set(Q)]
val_idx = [i for i, s in enumerate(spinors) if s[:2] in set(VAL)]
print("virt spinors:", len(virt_idx), " val spinors:", len(val_idx))

vvp = {}
for mi in virt_idx:
    for ni in virt_idx:
        if mi == ni:
            continue
        b = spinors[mi][2] + spinors[ni][2]
        vvp.setdefault(b, []).append((mi, ni))
print("bucket sizes:", {b: len(v) for b, v in sorted(vvp.items())})


def g2(ia, ib, im, in_):
    a, b, m, n_ = spinors[ia], spinors[ib], spinors[im], spinors[in_]
    terms = two_body_angular(a[1], a[2], b[1], b[2], m[1], m[2], n_[1], n_[2])
    if not terms:
        return 0.0
    Pa, Qa = od[(a[0], a[1])]
    Pb, Qb = od[(b[0], b[1])]
    Pm, Qm = od[(m[0], m[1])]
    Pn, Qn = od[(n_[0], n_[1])]
    return sum(c * MAT._rk(Pa, Qa, Pm, Qm, Pb, Qb, Pn, Qn,
                         dhf.bsp_data, k) for k, c in terms)

pairs0 = vvp.get(0, [])
row = np.array([g2(up, dn, mi, ni) for (mi, ni) in pairs0])
print("manual G1 row norm (2s2s over bucket 0):", np.linalg.norm(row))
print("nonzero entries:", int((row != 0).sum()), "/", len(row))

e_ab = eps[(2, -1)] * 2
emn = np.array([eps[spinors[mi][:2]] + eps[spinors[ni][:2]]
               for (mi, ni) in pairs0])
ok = (np.abs(e_ab - emn) > 0.02)
inv = np.where(ok, 0.5 * 2.0 / (e_ab - emn), 0.0)
d_manual = float(np.sum(row * row * inv))  # G2t row == G1 row for diagonal
print("manual d (direct part):", d_manual)
