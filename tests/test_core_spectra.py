"""Anchors for core_spectra (L-edge XAS/RIXS on the DMFT cluster).

C1  d^0 + zeta_p: XAS = two jj groups; L3 at -zeta_p/2, L2 at +zeta_p
    (closed-shell p^6 has zero l.s; the hole carries the eigenvalue).
C2  dipole branching L3:L2 = 2:1 summed over q (jj algebra).
C3  F0_pd only, d^0: single XAS line at eps_d - eps_p + 5*F0_pd
    (p^5 d^1 has 5 p-d pairs; each direct F0 Gaunt c^0 = 1).
C4  exchange closed-shell sum rule: p^6 d^1 with G^k only -> all 10 d SO
    degenerate, shift = -sum_k G^k * S_k with S_k from independent wigner_3j
    evaluation of the same mixed Gaunts.
C5  RIXS: elastic amplitude equals independent resolvent sum;
    loss sticks sit exactly at eigenvalue differences of the p^6 block.
"""

import numpy as np
import pytest

from atomx.embedding import core_spectra as cs
from atomx.embedding.ed_dmft import coulomb_cubic, interaction_table
from atomx.embedding.dmft import slater_from_UJ

ZETA_P = 11.5


EPS_P = -100.0  # core binding scale; keeps 2p^6 the lowest block


def _d0_system(zeta_p=ZETA_P, Wpd=None, eps_d=0.0):
    """2p^6 d^0 (bath decoupled, N=6, core window (5,6) -> 121 states)."""
    if Wpd is None:
        Wpd = {}
    sl = slater_from_UJ(8.0, 1.0)
    table = interaction_table(coulomb_cubic(sl))
    states, H = cs.build_core_sector(
        6, np.zeros(5), np.zeros(5), np.zeros(5), table, Wpd, zeta_p=zeta_p, eps_p=EPS_P
    )
    E, V = np.linalg.eigh(H)
    return E, V, states


def _group_amps(E, V, states, q):
    w, a = cs.xas_sticks(E, V, states, cs.dipole_tensor(), q=q)
    sel = np.abs(a) > 1e-8
    return w[sel], a[sel]


def test_C1_soc_positions():
    E, V, states = _d0_system()
    w_all, a_all = [], []
    for q in (-1, 0, 1):
        w, a = _group_amps(E, V, states, q)
        w_all.append(w)
        a_all.append(a)
    w = np.concatenate(w_all)
    a = np.concatenate(a_all)
    assert len(w) == 30  # 6 p holes x 5 d orbitals, all reachable over 3 q
    l3 = w[w < -EPS_P]
    l2 = w[w > -EPS_P]
    # multiplet positions: j=3/2 hole at -eps_p - zeta_p/2, j=1/2 at -eps_p + zeta_p
    assert np.allclose(l3, -EPS_P - ZETA_P / 2, atol=1e-6)
    assert np.allclose(l2, -EPS_P + ZETA_P, atol=1e-6)


def test_C2_branching_ratio():
    E, V, states = _d0_system()
    i3 = i2 = 0.0
    for q in (-1, 0, 1):
        w, a = _group_amps(E, V, states, q)
        i3 += np.sum(np.abs(a[w < -EPS_P]) ** 2)
        i2 += np.sum(np.abs(a[w > -EPS_P]) ** 2)
    assert i3 / i2 == pytest.approx(2.0, rel=1e-8)


def test_C3_f0pd_single_line():
    F0 = 6.0
    Wpd = cs.pd_coulomb({"F0pd": F0, "F2pd": 0.0, "G1pd": 0.0, "G3pd": 0.0})
    E, V, states = _d0_system(zeta_p=0.0, Wpd=Wpd)
    w_all, a_all = [], []
    for q in (-1, 0, 1):
        w, a = _group_amps(E, V, states, q)
        w_all.append(w)
        a_all.append(a)
    w = np.concatenate(w_all)
    a = np.concatenate(a_all)
    # d^0: all dipole strength degenerate at -eps_p + 5*F0 (one line);
    # sum rule: total |amp|^2 = sum over 6 p SO x (d,q) of |t|^2 (d^0 -> all
    # final states empty, no interference possible)
    assert np.allclose(w, -EPS_P + 5 * F0, atol=1e-6)
    t = cs.dipole_tensor()
    expect = sum(abs(v) ** 2 for v in t.values())  # t built for both spins
    assert np.sum(np.abs(a) ** 2) == pytest.approx(expect, rel=1e-8)


def test_C4_exchange_closed_shell_sumrule():
    """p^6 d^1 (N=7): exchange with the closed p shell is spherical, so all
    20 d/bath SO states are degenerate; the shift is -sum_k G^k S_k with S_k
    evaluated by an independent wigner_3j path."""
    G1, G3 = 5.0, 3.0
    Wpd = cs.pd_coulomb({"F0pd": 0.0, "F2pd": 0.0, "G1pd": G1, "G3pd": G3})
    sl = slater_from_UJ(8.0, 1.0)
    table = interaction_table(coulomb_cubic(sl))
    states, H = cs.build_core_sector(
        7, np.zeros(5), np.zeros(5), np.zeros(5), table, Wpd, zeta_p=0.0, eps_p=EPS_P
    )
    E, V = np.linalg.eigh(H)
    # the d-SO subset of the p^6 block (10 states; bath SOs carry no p-d
    # exchange and sit at a different energy) must be perfectly degenerate
    ncore = np.array([bin(s_ >> 20).count("1") for s_ in states])
    nd = np.array([bin(s_ & 1023).count("1") for s_ in states])
    i6 = np.where((ncore == 6) & (nd == 1))[0]
    H6 = H[np.ix_(i6, i6)]
    E6, _ = np.linalg.eigh(H6)
    assert len(i6) == 10
    assert np.max(np.abs(E6 - E6[0])) < 1e-8
    # closed-shell exchange shift of one d spin-orbital:
    #   dE = -sum_k G^k sum_{mp (3 same-spin)} sum_{md,q}
    #        |T[a,md]|^2 |<1 mp|C^k_q|2 md>|^2
    # (T = cubic_to_spherical; summed over the 5 cubic orbitals it must give
    # the same value - asserted via the H degeneracy above)
    from atomx.embedding.abacus_cf import cubic_to_spherical_5

    T = cubic_to_spherical_5()
    _, D_mix = cs._gaunt_tables()
    S = {}
    for k in (1, 3):
        per_a = [
            sum(
                abs(T[a, md + 2]) ** 2 * abs(v) ** 2
                for md in range(-2, 3)
                for (kk, mp_, md_, q), v in D_mix.items()
                if kk == k and md_ == md
            )
            for a in range(5)
        ]
        assert max(per_a) - min(per_a) < 1e-12  # cubic symmetry
        S[k] = per_a[0]
    assert (E6[0] - EPS_P * 6) == pytest.approx(-(G1 * S[1] + G3 * S[3]), abs=1e-6)


def test_C5_rixs_elastic_and_loss_positions():
    """Small d^2 system: RIXS elastic amplitude matches an explicit resolvent
    sum; every loss stick equals an eigenvalue difference of the p^6 block."""
    U, J = 8.0, 1.0
    sl = slater_from_UJ(U, J)
    table = interaction_table(coulomb_cubic(sl))
    Wpd = cs.pd_coulomb({"F0pd": 7.0, "F2pd": 8.0, "G1pd": 6.0, "G3pd": 3.5})
    eps_d = np.zeros(5)
    states, H = cs.build_core_sector(8, eps_d, np.zeros(5), np.zeros(5), table, Wpd, zeta_p=ZETA_P, eps_p=-100.0)
    E, V = np.linalg.eigh(H)
    t = cs.dipole_tensor()
    g = cs.ground_p6(E, V, states)
    # XAS to find a resonance
    w, a = cs.xas_sticks(E, V, states, t, q=0)
    n_star = int(np.argmax(np.abs(a)))
    omega_in = E[n_star] - E[g]
    w_loss, A = cs.rixs(E, V, states, t, omega_in, gamma=0.3)
    # elastic amplitude vs explicit resolvent
    Ti = cs._one_body_matrix(E, V, states, [(d, p, v) for (d, p, q), v in t.items() if q == 0])
    denom = omega_in - (E - E[g]) + 0.3j
    A_g = np.sum(np.abs(Ti[:, g]) ** 2 / denom)  # = sum_n <g|T+|n><n|T|g> / denom
    assert A[g] == pytest.approx(A_g, abs=1e-8)
    # loss positions are exact eigenvalue differences
    for wi, ai in zip(w_loss, A):
        if abs(ai) > 1e-6 and wi > 1e-6:
            assert np.min(np.abs(E - E[g] - wi)) < 1e-8
