"""Probe 3: where does the ladder element lose 3 orders of magnitude?
Compare radial Slater integrals and angular coefficients for 2s2s->4s4s."""

import re

import numpy as np

import atomx.physics.csf.matrix as MAT
from atomx.angular.csf_hamiltonian import two_body_angular
from atomx.methods.dirac_fock import DFSolver, Subshell

extra = [
    (2, 1),
    (2, -2),
    (3, -1),
    (3, 1),
    (3, -2),
    (3, 2),
    (3, -3),
    (4, -1),
    (4, 1),
    (4, -2),
    (4, 2),
    (4, -3),
    (4, -4),
    (5, -1),
    (5, 1),
    (5, -2),
    (5, 2),
    (5, -3),
    (5, -4),
]
dhf = DFSolver(
    Z=4,
    subshells=[
        Subshell(n=1, kappa=-1, occupation=2.0),
        Subshell(n=2, kappa=-1, occupation=2.0),
    ]
    + [Subshell(n=nk[0], kappa=nk[1], occupation=1e-6) for nk in extra],
    ns=50,
    ks=9,
    starting_potential="tfda",
)
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
lab2key = {}
for lab, _e in r.orbital_energies.items():
    m = re.fullmatch(r"(\d+)([spdfgh])(\d+)/(\d+)", lab)
    n = int(m.group(1))
    ll = "spdfgh".index(m.group(2))
    j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
    lab2key[lab] = (n, -(ll + 1) if j2 == 2 * ll + 1 else ll)
print("eps (Ha):", {l: round(e, 4) for l, e in sorted(r.orbital_energies.items())})

P2, Q2 = od[(2, -1)]
P4, Q4 = od[(4, -1)]
P3, Q3 = od[(3, -1)]
P2p, Q2p = od[(2, 1)]
# R^0(2s4s; 2s4s): pure electrostatic overlap of the pair densities
r0_2424 = MAT._rk(P2, Q2, P4, Q4, P2, Q2, P4, Q4, dhf.bsp_data, 0)
r0_2323 = MAT._rk(P2, Q2, P3, Q3, P2, Q2, P3, Q3, dhf.bsp_data, 0)
r0_2p2p = MAT._rk(P2p, Q2p, P2p, Q2p, P2p, Q2p, P2p, Q2p, dhf.bsp_data, 0)
print(
    f"R0(2s4s;2s4s) = {r0_2424:.3e}   R0(2s3s;2s3s) = {r0_2323:.3e}   R0(2p2p;2p2p) = {r0_2p2p:.3e}"
)
# norms of the virtual radial functions (a collapsed orbital would show it)
for key in [(2, -1), (3, -1), (4, -1), (5, -1)]:
    P, Q = od[key]
    print(f"  |P({key})|_2 = {np.linalg.norm(P):.4f}  max|P| = {np.max(np.abs(P)):.4f}")
# angular coefficient for the exact diagonal spinor tuple
terms = two_body_angular(-1, 1, -1, -1, -1, 1, -1, -1)
print("angular terms (2s+2s- -> 2s+2s-):", terms)
terms_q = two_body_angular(-1, 1, -1, -1, -1, 1, -1, -1)
t44 = two_body_angular(-1, 1, -1, -1, -1, 1, -1, -1)
# and for the 4s4s target
tg = two_body_angular(-1, 1, -1, -1, -1, 1, -1, -1)
print("R^0 pairing sanity: g2(2s2s->4s4s) should be ~ coef * R0(2s4s;2s4s)")
# full m-scheme element into 4s4s with each allowed intermediate m2 combo
for ma, mb in [(1, -1)]:
    for mm, mn in [(1, -1)]:
        t = two_body_angular(-1, ma, -1, mb, -1, mm, -1, mn)
        if not t:
            continue
        val = sum(
            c * MAT._rk(P2, Q2, P4, Q4, P2, Q2, P4, Q4, dhf.bsp_data, k) for k, c in t
        )
        print(f"  <2s({ma})2s({mb})|V|4s({mm})4s({mn})> terms={t} value={val:.3e}")
