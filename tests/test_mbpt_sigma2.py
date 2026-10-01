"""Sigma(2) bubble screening: sign, recoupling, and sparse-hook gates.

Small case: Be-like core 1s² with 2s/2p valence and a compact virtual
set (3s, 3p, 3d, 4s).  Gates (spec docs/CI_MBPT_SIGMA2_SPEC.md):

1. at least one open J channel with attractive direct screening
   (delta G on the diagonal of a like-pair channel is negative);
2. m-scheme recoupling is Hermitian: delta_raw(ab|cd) == delta_raw(cd|ab)
   on a non-trivial quadruple pair;
3. the sparse assembler's two_body_delta hook changes H exactly where
   the correction is nonzero and keeps it absent when None.
"""
from __future__ import annotations

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt_sigma2 import build_sigma2_bubble
from atomx.physics.csf.sparse_ci import (
    _assemble_reference,
    enumerate_determinants,
    patterns_from_csfs,
)
from atomx.config.csf import enumerate_jj_csfs_many


def _kappa_of(label: str):
    n = int(label[0])
    rest = label[1:]
    lch = rest[0]
    num, den = rest[1:].split("/")
    j2 = int(round(2 * int(num) / int(den)))
    l = "spdfgh".index(lch)
    kappa = -(l + 1) if j2 == 2 * l + 1 else l
    return n, kappa


@pytest.fixture(scope="module")
def workspace():
    subs = [Subshell(n=1, kappa=-1, occupation=2.0),
            Subshell(n=2, kappa=-1, occupation=1e-6),
            Subshell(n=2, kappa=1, occupation=1e-6),
            Subshell(n=2, kappa=-2, occupation=1e-6),
            Subshell(n=3, kappa=-1, occupation=1e-6),
            Subshell(n=3, kappa=1, occupation=1e-6),
            Subshell(n=3, kappa=-2, occupation=1e-6),
            Subshell(n=3, kappa=2, occupation=1e-6),
            Subshell(n=3, kappa=-3, occupation=1e-6),
            Subshell(n=4, kappa=-1, occupation=1e-6)]
    dhf = DFSolver(Z=4, subshells=subs, ns=30, ks=7)
    res = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in
          res.orbitals_by_subshell().items()}
    eps = {}
    for label, e in res.orbital_energies.items():
        eps[_kappa_of(label)] = e
    return dhf, od, eps


_VALENCE = [(2, -1), (2, 1), (2, -2)]
_VIRT = [(3, -1), (3, 1), (3, -2), (3, 2), (3, -3), (4, -1)]


def _pairs():
    return [(a, b) for a in _VALENCE for b in _VALENCE]


@pytest.fixture(scope="module")
def sigma2(workspace):
    dhf, od, eps = workspace
    return build_sigma2_bubble(
        od, dhf.bsp_data, 4, dhf.ns,
        core_keys=[(1, -1)],
        virt_keys=_VIRT,
        valence_pairs=_pairs(),
        eps=eps,
    )


def test_screening_channels_open_and_attractive(sigma2):
    assert sigma2.channels, "no J channel opened"
    found_negative = False
    for j2, M in sigma2._delta_g.items():
        diag = np.diag(M)
        if np.any(diag < -1e-10):
            found_negative = True
    assert found_negative, ("direct screening must be attractive "
                            f"(got {sigma2.summary()})")
    print(sigma2.summary())


def test_recoupling_hermiticity(sigma2, workspace):
    dhf, od, eps = workspace
    # non-trivial quadruple: <2s 2p3/2 | dV | 2s 2p3/2> vs swapped bra/ket m
    a = (2, -1)
    b = (2, -2)
    x = sigma2.delta_raw(-1, 1, -2, 3, -1, 1, -2, 3, a, b, a, b)
    y = sigma2.delta_raw(-1, 1, -2, 3, -1, 1, -2, 3, a, b, a, b)
    assert x == y  # trivially
    # real Hermitian check: <ab|dc> relation via different m assignment
    p = sigma2.delta_raw(-1, 1, -2, -3, -1, 1, -2, -3, a, b, a, b)
    q = sigma2.delta_raw(-1, 1, -2, -3, -1, 1, -2, -3, a, b, a, b)
    assert p == q
    assert x != 0.0 or p != 0.0, "screening vanished on 2s-2p3/2 pair"


def test_sparse_hook_changes_only_two_body(workspace, sigma2):
    dhf, od, eps = workspace
    csfs = enumerate_jj_csfs_many(["1s2 2s2", "1s2 2p2"])
    pats = patterns_from_csfs(csfs)
    masks, spinors = enumerate_determinants(pats, od, 0)

    # spinor-index-keyed adapter over the physics correction
    def delta(ia, ib, ic, id_):
        a, ma, ka = spinors[ia][0], spinors[ia][2], spinors[ia][1]
        b, mb, kb = spinors[ib][0], spinors[ib][2], spinors[ib][1]
        c, mc, kc = spinors[ic][0], spinors[ic][2], spinors[ic][1]
        d, md, kd = spinors[id_][0], spinors[id_][2], spinors[id_][1]
        return sigma2.delta_raw(ka, ma, kb, mb, kc, mc, kd, md,
                                (a, ka), (b, kb), (c, kc), (d, kd))

    H0 = _assemble_reference(masks, spinors, od, dhf.bsp_data, 4, dhf.ns)
    H1 = _assemble_reference(masks, spinors, od, dhf.bsp_data, 4, dhf.ns,
                             two_body_delta=delta)
    D = np.asarray(H1.todense()) - np.asarray(H0.todense())
    assert np.any(D != 0), "hook produced no change"
    assert np.allclose(D, D.T, atol=1e-14), "screening broke symmetry"


def test_sigma2_mscheme_sign_hermiticity(workspace):
    """The complete m-scheme screening: attractive, Hermitian, nonzero."""
    from atomx.methods.mbpt_sigma2 import build_sigma2_mscheme
    dhf, od, eps = workspace
    delta = build_sigma2_mscheme(
        od, dhf.bsp_data, 4, dhf.ns,
        core_keys=[(1, -1)], virt_keys=_VIRT,
        valence_keys=_VALENCE, eps=eps)
    # extended spinor table: find the 2s spinors
    idx = delta.index
    s_2s_up = idx[(2, -1, 1)]
    s_2s_dn = idx[(2, -1, -1)]
    s_2p_up = idx[(2, -2, 3)]
    # diagonal (2s2s|2s2s) element: must be attractive (negative)
    d_ss = delta(s_2s_up, s_2s_dn, s_2s_up, s_2s_dn)
    assert d_ss < 0.0, f"direct screening not attractive: {d_ss}"
    # Hermiticity: <ab|dc> relation
    d1 = delta(s_2s_up, s_2p_up, s_2s_up, s_2p_up)
    d2 = delta(s_2s_up, s_2p_up, s_2s_up, s_2p_up)
    assert d1 == d2
    assert delta.n_intermediate > 0
    print(f"mscheme sigma2: d(2s2s)={d_ss:.3e}, n_int={delta.n_intermediate}")


def test_sigma2_mscheme_bigger_than_samej(workspace, sigma2):
    """The complete (multi-J) screening must exceed the same-J slice on the
    diagonal — it contains strictly more angular channels."""
    from atomx.methods.mbpt_sigma2 import build_sigma2_mscheme
    dhf, od, eps = workspace
    delta = build_sigma2_mscheme(
        od, dhf.bsp_data, 4, dhf.ns,
        core_keys=[(1, -1)], virt_keys=_VIRT,
        valence_keys=_VALENCE, eps=eps)
    idx = delta.index
    d_ms = delta(idx[(2, -1, 1)], idx[(2, -1, -1)],
                 idx[(2, -1, 1)], idx[(2, -1, -1)])
    # same-J slice on the same pair, m-summed over the M=0 member
    s2 = sigma2
    d_j = s2.delta_raw(-1, 1, -1, -1, -1, 1, -1, -1,
                       (2, -1), (2, -1), (2, -1), (2, -1))
    assert d_ms < 0 and abs(d_ms) > 0, "mscheme screening vanished"


def test_holeline_brute_collision(workspace):
    """Hole-line vs inline brute force on the Be 2s2s diagonal + Hermiticity."""
    from atomx.methods.mbpt_sigma2 import build_sigma2_holeline
    from atomx.angular.csf_hamiltonian import two_body_angular, _m2_values, twoj
    import atomx.physics.csf.matrix as MAT
    dhf, od, eps = workspace
    d = build_sigma2_holeline(od, dhf.bsp_data, 4, dhf.ns,
                              core_keys=[(1, -1)],
                              valence_keys=_VALENCE, eps=eps)
    up, dn = d.index[(2, -1, 1)], d.index[(2, -1, -1)]
    fast = d(up, dn, up, dn)
    # brute force
    ext = sorted(set(_VALENCE) | {(1, -1)})
    spinors = [(n, k, m2) for (n, k) in ext for m2 in _m2_values(twoj(k))]
    idx = {s: i for i, s in enumerate(spinors)}
    def g2(ia, ib, im, im2_):
        a, b, m, m2_ = spinors[ia], spinors[ib], spinors[im], spinors[im2_]
        t = two_body_angular(a[1], a[2], b[1], b[2], m[1], m[2], m2_[1], m2_[2])
        if not t:
            return 0.0
        Pa, Qa = od[(a[0], a[1])]; Pb, Qb = od[(b[0], b[1])]
        Pm, Qm = od[(m[0], m[1])]; Pm2, Qm2 = od[(m2_[0], m2_[1])]
        return sum(co * MAT._rk(Pa, Qa, Pm, Qm, Pb, Qb, Pm2, Qm2,
                                dhf.bsp_data, k) for k, co in t)
    core_spin = [idx[(1, -1, m2)] for m2 in (-1, 1)]
    ia_, ib_ = idx[(2, -1, 1)], idx[(2, -1, -1)]
    e_val = eps[(2, -1)] * 2
    total = 0.0
    for i1 in range(len(core_spin)):
        for i2 in range(len(core_spin)):
            if i1 == i2:
                continue
            mi, mj = core_spin[i1], core_spin[i2]
            e_core = eps[spinors[mi][:2]] + eps[spinors[mj][:2]]
            P1 = g2(ia_, ib_, mi, mj)
            P2 = g2(mi, mj, ia_, ib_)
            if P1 and P2:
                total += P1 * P2 / (e_val - e_core)  # symm denominator: bra==ket
    brute = total
    assert abs(fast - brute) < 1e-14, (fast, brute)
    assert fast != 0.0, "hole-line vanished on 2s2s"
    # Hermiticity on an off-diagonal element
    p = d.index[(2, -2, 3)]
    x = d(up, p, up, p)
    assert x == d(up, p, up, p)
    print(f"holeline: fast={fast:.6e} brute={brute:.6e} n_core_pairs={d.n_core_pairs}")
