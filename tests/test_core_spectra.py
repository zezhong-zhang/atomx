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
from atomx.embedding.dmft import slater_from_UJ
from atomx.embedding.ed_dmft import coulomb_cubic, interaction_table

ZETA_P = 11.5


EPS_P = -100.0  # core binding scale; keeps 2p^6 the lowest block


def _d0_system(zeta_p=ZETA_P, Wpd=None, eps_d=0.0):
    """2p^6 d^0 (bath decoupled, N=6, core window (5,6) -> 121 states)."""
    if Wpd is None:
        Wpd = {}
    sl = slater_from_UJ(8.0, 1.0)
    table = interaction_table(coulomb_cubic(sl))
    states, H = cs.build_core_sector(
        6, np.zeros(5), np.zeros(5), np.zeros(5), table, Wpd, zeta_p=zeta_p,
        eps_p=EPS_P, subtract_core_mf=False,  # probe the raw Wpd tensor
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
        7, np.zeros(5), np.zeros(5), np.zeros(5), table, Wpd, zeta_p=0.0,
        eps_p=EPS_P, subtract_core_mf=False,  # probe the raw Wpd tensor
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


def test_E1_multipole_L1_is_dipole():
    """multipole_tensor(1, radial) must equal dipole_tensor() scaled by radial."""
    r = 0.37
    t1 = cs.multipole_tensor(1, radial=r)
    td = cs.dipole_tensor()
    assert set(t1) == set(td)
    for k in td:
        assert t1[k] == pytest.approx(r * td[k], rel=1e-12)


def test_E2a_quadrupole_is_parity_forbidden():
    """2p -> 3d quadrupole (L=2) is parity-forbidden: 1+2+2 = 5 odd kills the
    (1 2 2; 0 0 0) 3j, so the L=2 tensor is EMPTY.  (A vacuous sum-rule test
    here once passed 0 == 0 -- the emptiness must be pinned explicitly.)"""
    assert cs.multipole_tensor(2, radial=1.0) == {}


def test_E2b_octupole_closed_shell_sumrule():
    """L=3 on the closed-shell d^0 reference: F0_pd pins every stick to the
    same line (multiplet-independent), and the total weight is the L=3 Gaunt
    sum rule with both spins.  Truth values from sympy wigner_3j with the
    C^k normalization -- an independent external angular source."""
    from sympy import N
    from sympy.physics.wigner import wigner_3j

    F0, R3 = 6.0, 0.21
    Wpd = cs.pd_coulomb({"F0pd": F0, "F2pd": 0.0, "G1pd": 0.0, "G3pd": 0.0})
    E, V, states = _d0_system(zeta_p=0.0, Wpd=Wpd)
    t = cs.multipole_tensor(3, radial=R3)
    assert len(t) > 0
    w_all, a_all = [], []
    for q in range(-3, 4):
        w, a = cs.xas_sticks(E, V, states, t, q=q)
        w_all.append(w)
        a_all.append(a)
    w = np.concatenate(w_all)
    a = np.concatenate(a_all)
    sel = np.abs(a) > 1e-8
    w, a = w[sel], a[sel]
    assert np.allclose(w, -EPS_P + 5 * F0, atol=1e-6)  # 5 p-d pairs, c^0 = 1
    # truth: <1 mp|C^3_q|2 md> = (-1)^mp sqrt(3*7*5/4pi) (1 3 2;000)(1 3 2;-mp q md)
    # truth: <1 mp|C^3_q|2 md> with C^k = sqrt(4pi/(2k+1)) Y_kq:
    #   = sqrt(4pi/7) * (-1)^mp sqrt(3*7*5/4pi) (1 3 2;000) (1 3 2;-mp q3j md)
    # (the two prefactors collapse to sqrt(15))
    pref = float(np.sqrt(4 * np.pi / 7.0) * np.sqrt(3 * 7 * 5 / (4 * np.pi)))
    g000 = float(N(wigner_3j(1, 3, 2, 0, 0, 0)))
    expect = 0.0
    for mp in (-1, 0, 1):
        for md in range(-2, 3):
            q3j = mp - md  # 3j m-conservation: -mp + q3j + md = 0
            g = ((-1) ** mp) * pref * g000 * float(
                N(wigner_3j(1, 3, 2, -mp, q3j, md)))
            expect += 2 * R3**2 * g**2  # both spins
    assert np.sum(np.abs(a) ** 2) == pytest.approx(expect, rel=1e-8)


def test_E3_merge_multipoles_interference():
    """merge_multipoles keeps coherent amplitudes: same tensor merged with
    itself at phase i gives 4x the weight; dipole+quadrupole at relative
    phase i on real Gaunt amplitudes gives the incoherent sum (the EELS
    i^L (2L+1) convention)."""
    F0 = 5.0
    Wpd = cs.pd_coulomb({"F0pd": F0, "F2pd": 0.0, "G1pd": 0.0, "G3pd": 0.0})
    E, V, states = _d0_system(zeta_p=0.0, Wpd=Wpd)
    t1 = cs.multipole_tensor(1, radial=0.5)
    t3 = cs.multipole_tensor(3, radial=0.7)

    def total(t):
        return sum(
            np.sum(np.abs(cs.xas_sticks(E, V, states, t, q=q)[1]) ** 2)
            for q in (-1, 0, 1)
        )

    i1, i3 = total(t1), total(t3)
    m_same = cs.merge_multipoles((1j, t1), (1j, t1))
    assert total(m_same) == pytest.approx(4 * i1, rel=1e-10)
    m_cross = cs.merge_multipoles((1j, t1), (-1.0, t3))
    # Gaunt amplitudes are real here: |i a1 - a3|^2 = a1^2 + a3^2
    assert total(m_cross) == pytest.approx(i1 + i3, rel=1e-10)


def test_E5a_soc_cubic_basis_triplet_degeneracy():
    """d^8 ^3A2g ground in Oh is EXACTLY 3-fold degenerate (S=1 x orbital
    singlet) for any zeta_d: build_core_sector must apply zeta_d * l.s in
    the CUBIC d-orbital basis (bits 2a+s), not the spherical (m,s) order of
    _soc_matrix. A basis-mixed SOC lifts the triplet (caught here)."""
    import itertools
    table = interaction_table(coulomb_cubic(slater_from_UJ(8.0, 1.0)))

    # reuse the d-only CF+U 45-state model: N=8 in the 10 d spin-orbitals,
    # eg = (0,3) at +0.51 eV, t2g = (1,2,4) at -0.34 eV (10Dq = 0.85).
    from atomx.embedding.core_spectra import ND
    states = sorted(
        sum(1 << i for i in c) for c in itertools.combinations(range(ND), 8)
    )
    idx = {s: n for n, s in enumerate(states)}
    cf5 = np.zeros(5)
    cf5[[0, 3]] = 0.51
    cf5[[1, 2, 4]] = -0.34

    def build(zeta_d):
        H = np.zeros((len(states), len(states)), complex)
        for n, st in enumerate(states):
            for a in range(5):
                for sp in (0, 1):
                    if (st >> (2 * a + sp)) & 1:
                        H[n, n] += cf5[a]
            for st2, amp in table[st]:
                H[idx[st2], n] += amp
        # zeta_d through the SAME code path as build_core_sector
        T5 = cs.cubic_to_spherical_5()
        U2 = np.kron(T5, np.eye(2))
        from atomx.embedding.impurity import _soc_matrix
        ls10 = U2 @ _soc_matrix(zeta_d, l=2) @ U2.conj().T
        for n, st in enumerate(states):
            for i in range(ND):
                if not (st >> i) & 1:
                    continue
                for j in range(ND):
                    if i == j or (st >> j) & 1:
                        continue
                    v = ls10[j, i]
                    if abs(v) < 1e-14:
                        continue
                    s1 = st ^ (1 << i)
                    g = -1 if bin(st & ((1 << i) - 1)).count("1") % 2 else 1
                    s2 = s1 | (1 << j)
                    g2 = -1 if bin(s1 & ((1 << j) - 1)).count("1") % 2 else 1
                    H[idx[s2], n] += v * g * g2
        return np.linalg.eigvalsh(H)

    e = build(0.33)  # deliberately large zeta to expose any basis mixing
    gaps = np.diff(e[:4])
    assert gaps[0] < 1e-10 and gaps[1] < 1e-10, (
        f"3A2g triplet lifted by zeta_d path: e[:4] = {e[:4]}"
    )
    assert gaps[2] > 0.1, f"expected a gap to the spin-orbit excitations: {e[:4]}"


def test_E6_core_mf_normal_ordering_ground_conf():
    """R4 anchor: the p^6 ground block of the XAS cluster must reproduce the
    ed_dmft ground sector's configuration weights when the closed 2p^6
    mean-field is normal-ordered out of Wpd (double-counting fix).

    With identical eps/eb/Vb/U/J, the raw build (subtract off) double-counts
    the 2p^6 Hartree+exchange on 3d (eps_d from DFT already contains it),
    crushing the d^9 (ZRS) weight; with subtract_core_mf=True both builds
    agree. Reference numbers from build_sector (the DMFT ground sector)."""
    eps = np.full(5, -10.0)
    eb = np.full(5, 10.0)
    Vb = np.full(5, 2.0)
    PD = {"F0pd": 7.5, "F2pd": 8.23, "G1pd": 6.43, "G3pd": 3.65}
    Wpd = cs.pd_coulomb(PD)
    table = interaction_table(coulomb_cubic(slater_from_UJ(8.0, 1.0)))

    def conf_of(states, vec):
        maskd = (1 << cs.ND) - 1
        prob = np.abs(vec) ** 2
        nd = np.array([bin(s & maskd).count("1") for s in states])
        return {k: float(prob[nd == k].sum()) for k in (8, 9, 10)}

    # reference: the DMFT ground sector (no core shell at all)
    from atomx.embedding.ed_dmft import build_sector
    st0, H0 = build_sector(18, eps, eb, Vb, table)
    _, v0 = np.linalg.eigh(H0)
    ref = conf_of(st0, v0[:, 0])

    st, H = cs.build_core_sector(24, eps, eb, Vb, table, Wpd, zeta_d=0.11,
                                 eps_p=-100.0, core_window=(6, 6))
    _, v = np.linalg.eigh(H)
    fixed = conf_of(st, v[:, 0])

    st_raw, H_raw = cs.build_core_sector(24, eps, eb, Vb, table, Wpd, zeta_d=0.11,
                                         eps_p=-100.0, core_window=(6, 6),
                                         subtract_core_mf=False)
    _, v_raw = np.linalg.eigh(H_raw)
    raw = conf_of(st_raw, v_raw[:, 0])

    for k in (8, 9, 10):
        assert abs(fixed[k] - ref[k]) < 0.01, (
            f"d^{k}: fixed {fixed[k]:.4f} vs DMFT ref {ref[k]:.4f}")
    # the double-counted raw build visibly crushes the ZRS weight
    assert raw[9] < 0.3 * ref[9], (
        f"raw d^9 {raw[9]:.4f} should be far below ref {ref[9]:.4f} "
        "(double counting demo)")


def _core_bits():
    return 0b111111 << 20  # six occupied 2p spin-orbitals (p^6 selector)


def test_E7a_van_vleck_spin_half():
    """R5 anchor: Kramers doublet (M = +/-1, g=2) -> mu_eff^2 = 3, chi->0
    when frozen; T*chi plateau at 1/k_B."""
    delta = 0.01
    states = [_core_bits() | 0b01, _core_bits() | 0b10]  # d up / d down
    E = np.array([-delta / 2, delta / 2])
    V = np.eye(2)
    # delta = 0.01 eV: frozen at 0.1 K, but only ~2.6 kT at 300 K -- fit the
    # Curie plateau only on T >= 1000 K (delta/kT <= 0.12, < 0.4% Schottky
    # correction), assert the asymptote directly at 5000 K.
    r = cs.thermal_response(E, V, states, (0.1, 300.0, 1000.0, 5000.0, 10000.0))
    assert r["mu_eff2"] is not None
    assert abs(r["mu_eff2"] - 3.0) < 0.03, f"spin-1/2 mu_eff^2 = {r['mu_eff2']}"
    assert r["chi"][0.1] < 1e-3 * r["chi"][300.0], (
        f"frozen doublet should be diamagnetic: chi(0.1K)={r['chi'][0.1]}")
    KB = 8.617333262e-5
    assert abs(r["chi"][5000.0] * 5000.0 - 1.0 / KB) < 0.001 / KB


def test_E7b_van_vleck_spin_one():
    """R5 anchor: spin-1 triplet (M = +2, 0, -2) -> mu_eff^2 = 8 (spin-only
    Ni2+ value) at high T."""
    delta = 0.01
    states = [_core_bits() | 0b0101,   # up,up   Sz=+1
              _core_bits() | 0b0011,   # up,down Sz=0
              _core_bits() | 0b1010]   # down,down Sz=-1
    E = np.array([-delta, 0.0, delta])
    V = np.eye(3)
    r = cs.thermal_response(E, V, states, (300.0, 1000.0, 5000.0, 10000.0))
    assert abs(r["mu_eff2"] - 8.0) < 0.08, f"spin-1 mu_eff^2 = {r['mu_eff2']}"


def test_E7c_thermal_response_on_cluster():
    """R5 integration: thermal_response runs end-to-end on a full XAS
    cluster build; chi and mu_eff are physical.  (BR(T) flatness at 300 K
    is a property of the REAL NiO cluster -- the toy build here has
    near-equal core-hole jj mixing, so its BR is classifier-noise; the
    flatness check lives in the R5 production run on the real cluster.)"""
    eps = np.full(5, -10.0)
    eb = np.full(5, 10.0)
    Vb = np.full(5, 2.0)
    PD = {"F0pd": 7.5, "F2pd": 8.23, "G1pd": 6.43, "G3pd": 3.65}
    Wpd = cs.pd_coulomb(PD)
    table = interaction_table(coulomb_cubic(slater_from_UJ(8.0, 1.0)))
    st, H = cs.build_core_sector(24, eps, eb, Vb, table, Wpd, zeta_d=0.11,
                                 eps_p=-100.0, core_window=(5, 6))
    E, V = np.linalg.eigh(H)
    r = cs.thermal_response(E, V, st, (0.0, 300.0, 1000.0, 5000.0))
    for T in (0.0, 300.0):
        assert np.isfinite(r["BR"][T]), f"BR({T}) = {r['BR'][T]}"
    assert r["chi"][300.0] > 0.0
    assert r["chi_vv"] is not None and r["chi_vv"] >= 0.0
    assert 4.0 < r["mu_eff2"] < 16.0, f"mu_eff^2 = {r['mu_eff2']}"
