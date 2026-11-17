"""L2,3-edge XAS and RIXS from the converged ED-DMFT cluster.

The DMFT cluster (5 d + 5 bath spin-orbital pairs = 20 SO) is extended by the
2p core shell (6 SO placed on bits 20..25, ABOVE d and bath, so all d-d and
d-bath sign conventions of ed_dmft are unchanged). The full (2p + 3d + bath)
problem is diagonalized in the N0+6 electron sector. XAS final states
(2p^5 3d^{N+1}) and RIXS final states (2p^6 3d^N*) conserve total N, so the
entire spectrum lives in ONE small sector:

    ground                2p^6 d^N    bath^M   (N + M = N0)
    XAS/RIXS intermediate 2p^5 d^{N+1} bath^M
    RIXS final            2p^6 d^N*   bath^M*   (dd / shake-up / CT)

This is the cluster-ED Kramers-Heisenberg route, but the bath is the
*converged DMFT hybridization* (from eddmft_*.pkl), not a hand-guessed
two-level charge-transfer model.

Physics included
  2p SOC (zeta_p), 3d SOC (zeta_d, optional),
  d-d Slater multipoles (same table as ed_dmft),
  p-d Coulomb direct F^k_pd (k=0,2) and exchange G^k_pd (k=1,3),
  dipole T_q = sum c^+_d C^1_q c_p with radial integral factored to 1.

Index conventions (shared with ed_dmft)
  d spin-orbitals:       i = 2*a + s,  a in cubic order (dz2,dxz,dyz,dx2y2,dxy)
  bath spin-orbitals:    ND + 2*a + s
  core 2p spin-orbitals: NS + (m+1) + 3*s,  m in {-1,0,1}
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from atomx.embedding.abacus_cf import cubic_to_spherical_5
from atomx.embedding.ed_dmft import (
    ND,
    NS,
    _apply_c,
    _apply_cdag,
    _spin2,
)
from atomx.embedding.impurity import _soc_matrix

NC = 6  # 2p core spin-orbitals
NT = NS + NC  # 26


def core_index(m: int, s: int) -> int:
    """2p spin-orbital index: m in {-1,0,1} -> NS + (m+1) + 3*s."""
    return NS + (m + 1) + 3 * s


# ---------------------------------------------------------------- angular
def _gaunt_tables():
    """Quadrature Gaunt tables (same recipe as impurity._shell_k).

    D_same[l][(k, m, mp, q)] = <l m | C^k_q | l mp>,  q = m - mp
    D_mix[(k, m_p, m_d, q)]  = <1 m_p | C^k_q | 2 m_d>, q = m_p - m_d
    with C^k_q = sqrt(4pi/(2k+1)) Y_kq.
    """
    nth, nph = 120, 240
    x, w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(nph) / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W2 = w[:, None] * (2 * np.pi / nph)
    from scipy.special import sph_harm_y

    def Y(l_, m_):
        return sph_harm_y(l_, m_, TH, PH)

    D_same = {1: {}, 2: {}}
    for l in (1, 2):
        for k in range(0, 2 * l + 1):
            norm = math.sqrt(4 * math.pi / (2 * k + 1))
            for m in range(-l, l + 1):
                for mp in range(-l, l + 1):
                    q = m - mp
                    if abs(q) > k:
                        continue
                    D_same[l][(k, m, mp, q)] = np.sum(
                        W2 * np.conj(Y(l, m)) * norm * Y(k, q) * Y(l, mp)
                    )
    D_mix = {}
    for k in (1, 2, 3):
        norm = math.sqrt(4 * math.pi / (2 * k + 1))
        for mp_ in range(-1, 2):
            for md in range(-2, 3):
                q = mp_ - md
                if abs(q) > k:
                    continue
                D_mix[(k, mp_, md, q)] = np.sum(
                    W2 * np.conj(Y(1, mp_)) * norm * Y(k, q) * Y(2, md)
                )
    return D_same, D_mix


def pd_coulomb(pd: dict):
    """p-d Coulomb tensor W[(i,j,k,l)] = amplitude of c^+_i c^+_j c_l c_k.

    Direct   (k = 0,2): W[(p1,d2,p3,d4)] += F^k * conj(<p3|C^k|p1>) * <d2|C^k|d4>
    Exchange (k = 1,3): W[(p1,d2,d4,p3)] += G^k * conj(<d4|C^k|p1>) * <d2|C^k|p3>
    The slot pattern (conj on the 3rd slot, plain on the 2nd) mirrors
    coulomb_cubic exactly; d orbitals are rotated to cubic basis with
    cubic_to_spherical_5 (same matrix as the validated d-d path).
    """
    D_same, D_mix = _gaunt_tables()
    T = cubic_to_spherical_5()  # rows: cubic a, cols: spherical m+2

    def gs(l, k, m, mp):
        """<l m|C^k_{m-mp}|l mp>."""
        return D_same[l].get((k, m, mp, m - mp), 0.0 + 0.0j)

    def gm_dbra(k, md, mp):
        """<2 md|C^k_{md-mp}|1 mp>."""
        return np.conj(D_mix.get((k, mp, md, mp - md), 0.0 + 0.0j))

    # d-side cubic Gaunt keyed by q (shared-q structure mirrors coulomb_cubic):
    # Dd[(k, a, b, q)] = <d_a|C^k_q|d_b>_cubic, only md_a - md_b = q terms.
    Dd = {}
    for k in (0, 2, 4):
        for a in range(5):
            for b in range(5):
                for md1 in range(-2, 3):
                    for md2 in range(-2, 3):
                        q = md1 - md2
                        if abs(q) > k:
                            continue
                        v = gs(2, k, md1, md2)
                        if abs(v) < 1e-14:
                            continue
                        key = (k, a, b, q)
                        Dd[key] = Dd.get(key, 0.0) + T[a, md1 + 2] * np.conj(T[b, md2 + 2]) * v

    W = {}
    # ---- direct F^0_pd, F^2_pd: p leg and d leg share q
    for k, F in ((0, pd["F0pd"]), (2, pd["F2pd"])):
        if F == 0:
            continue
        for mp1 in range(-1, 2):
            for mp3 in range(-1, 2):
                for q in range(-k, k + 1):
                    cp = np.conj(gs(1, k, mp3, mp1))  # nonzero only for q = mp3-mp1
                    if abs(cp) < 1e-14:
                        continue
                    for a in range(5):
                        for b in range(5):
                            amp = F * cp * Dd.get((k, a, b, q), 0.0)
                            if abs(amp) < 1e-14:
                                continue
                            for s in (0, 1):
                                for t in (0, 1):
                                    key = (core_index(mp1, s), 2 * a + t, core_index(mp3, s), 2 * b + t)
                                    W[key] = W.get(key, 0.0) + amp
    # ---- exchange G^1_pd, G^3_pd: shared q = md_b - mp1 = md_a - mp3
    for k, G in ((1, pd["G1pd"]), (3, pd["G3pd"])):
        if G == 0:
            continue
        for mp1 in range(-1, 2):
            for mp3 in range(-1, 2):
                for q in range(-k, k + 1):
                    mb = mp1 + q
                    ma = mp3 + q
                    if not (-2 <= mb <= 2 and -2 <= ma <= 2):
                        continue
                    # truth: E_ex = sum_q <p1|C_q|d_b> * conj(<p3|C_q|d_a>)
                    gb = D_mix.get((k, mp1, mb, mp1 - mb), 0.0 + 0.0j)
                    ga = D_mix.get((k, mp3, ma, mp3 - ma), 0.0 + 0.0j)
                    if abs(gb) < 1e-14 or abs(ga) < 1e-14:
                        continue
                    for a in range(5):
                        ta = T[a, ma + 2] * np.conj(ga)
                        if abs(ta) < 1e-14:
                            continue
                        for b in range(5):
                            amp = G * np.conj(T[b, mb + 2] * gb) * ta
                            if abs(amp) < 1e-14:
                                continue
                            for s in (0, 1):
                                # exchange acts only between same-spin electrons
                                t = s
                                key = (core_index(mp1, s), 2 * a + t, 2 * b + t, core_index(mp3, s))
                                W[key] = W.get(key, 0.0) + amp
    return W


def dipole_tensor():
    """t[(d_idx, p_idx, q)] = <d md|C^1_q|p mp> with q = md - mp (radial = 1)."""
    _, D_mix = _gaunt_tables()
    t = {}
    for md in range(-2, 3):
        for mp in range(-1, 2):
            q = md - mp
            if abs(q) > 1:
                continue
            v = np.conj(D_mix[(1, mp, md, mp - md)])
            for s in (0, 1):
                t[(2 * (md + 2) + s, core_index(mp, s), q)] = v
    return t


def multipole_tensor(L, radial=1.0):
    """General electric multipole operator of the ``e^{iq.r}`` partial-wave
    expansion.

    ``t[(d_idx, p_idx, q)] = radial * <d md|C^L_q|p mp>`` with ``q = md - mp``
    (same convention as :func:`dipole_tensor`, which is exactly
    ``multipole_tensor(1, 1.0)``).  ``radial`` is the q-dependent radial
    integral ``<3d|j_L(qr)|2p>`` (atomic units; for L=1, small-q limit
    ``R_1(q) -> q * <r> / 3`` in the length gauge).

    Selection rules carried by the Gaunt tables: ``|md - mp| <= L`` and the
    3j triangle (2 L 1) **with the parity condition 1 + L + 2 even** — the
    (l1 l2 l3; 0 0 0) 3j kills every even L for 2p -> 3d.  Concretely:
    L=1 (dipole) and L=3 (octupole) are the only nonzero channels at the
    L-edge; L=2 (quadrupole) is parity-forbidden and yields an EMPTY tensor
    (pinned by test_E2a).  L=3 enters e^{iq.r} with radial j_3(qr) ~ q^3,
    so it is the leading non-dipole correction to the EELS cross section.
    """
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    _, D_mix = _gaunt_tables()
    t = {}
    for md in range(-2, 3):
        for mp in range(-1, 2):
            q = md - mp
            if abs(q) > L:
                continue
            v = D_mix.get((L, mp, md, mp - md))
            if v is None or abs(v) < 1e-14:
                continue
            for s in (0, 1):
                t[(2 * (md + 2) + s, core_index(mp, s), q)] = radial * np.conj(v)
    return t


def merge_multipoles(*weighted_tensors):
    """Merge ``(phase, tensor)`` pairs into one transition operator.

    The double-differential EELS cross section at momentum transfer q needs
    the coherent sum over multipolarities L with partial-wave phases
    ``i^L`` (and the (2L+1) j_L convention absorbed into the caller's
    radial weights).  Amplitudes sharing a ``(d_idx, p_idx, q)`` key are
    summed, so interference terms survive ``xas_sticks``.
    """
    out = {}
    for phase, t in weighted_tensors:
        for k, v in t.items():
            out[k] = out.get(k, 0.0) + complex(phase) * v
    return out


# ---------------------------------------------------------------- sector ED
def _apply4(s, i, j, l, k):
    """Amplitude of c^+_i c^+_j c_l c_k |s> on the full NT-bit state."""
    s1, g1 = _apply_c(s, k)
    if s1 is None:
        return None, 0.0
    s2, g2 = _apply_c(s1, l)
    if s2 is None:
        return None, 0.0
    s3, g3 = _apply_cdag(s2, j)
    if s3 is None:
        return None, 0.0
    s4, g4 = _apply_cdag(s3, i)
    if s4 is None:
        return None, 0.0
    return s4, g1 * g2 * g3 * g4


def build_core_sector(N, eps_d, eb, Vb, table_dd, Wpd, zeta_p=0.0, zeta_d=0.0,
                      eps_p=0.0, core_window=(5, 6), subtract_core_mf=True):
    """Hamiltonian of the N-electron (2p + 3d + bath) sector.

    eps_d/eb/Vb: (5,) or (5,2) as in ed_dmft; eps_p is the core on-site
    reference energy. core_window: keep only determinants whose 2p occupancy
    falls in this range. The Hamiltonian conserves n_core (SOC and p-d
    Coulomb move one p electron at a time), so the restriction to (5,6) is
    exact: p^6 holds the ground state, p^5 the XAS/RIXS intermediates.
    Returns (states, H).

    subtract_core_mf: the eps_d passed in comes from DFT and already contains
    the closed 2p^6 shell's Coulomb mean field on 3d. Wpd would add that same
    field a second time (its p^6 expectation). Normal-ordering Wpd w.r.t. the
    p^6 core removes the double counting; R4 cross-check: without it the
    ground-state d^9 (ZRS) weight collapses 11% -> 0.4% vs the ed_dmft
    ground sector with identical parameters.
    """
    eps_d, eb, Vb = _spin2(eps_d), _spin2(eb), _spin2(Vb)
    if subtract_core_mf and Wpd:
        vmf = np.zeros(ND)
        for t in range(ND):
            for p in range(NS, NT):  # the 6 core 2p spin-orbitals
                vmf[t] += Wpd.get((p, t, p, t), 0.0) - Wpd.get((p, t, t, p), 0.0)
        eps_d = eps_d - vmf.reshape(5, 2)
    lo, hi = core_window
    states = [
        s
        for s in (sum(1 << i for i in c) for c in itertools.combinations(range(NT), N))
        if lo <= bin(s >> NS).count("1") <= hi
    ]
    idx = {s: n for n, s in enumerate(states)}
    H = np.zeros((len(states), len(states)), complex)
    soc_p = _soc_matrix(zeta_p, l=1)
    if zeta_d:
        # d bits are CUBIC orbitals (2a+s) while _soc_matrix is built in the
        # spherical (m,s) order: rotate into the cubic basis, otherwise the
        # "zeta_d" term is a basis-mixed operator that lifts the 3A2g triplet
        # degeneracy (regression anchor E5a).
        T5 = cubic_to_spherical_5()  # rows cubic a, cols spherical m+2
        U2 = np.kron(T5, np.eye(2))
        soc_d = U2 @ _soc_matrix(zeta_d, l=2) @ U2.conj().T
    else:
        soc_d = None
    maskd = (1 << ND) - 1
    maskb = (1 << NS) - 1
    terms = [(i, j, k, l, v) for (i, j, k, l), v in Wpd.items() if abs(v) > 1e-12]
    for n, s in enumerate(states):
        d, rest = s & maskd, s >> ND
        b, c6 = rest & maskb, s >> NS
        H[n, n] += eps_p * bin(c6).count("1")
        for a in range(5):
            for sp in (0, 1):
                if (s >> (2 * a + sp)) & 1:
                    H[n, n] += eps_d[a, sp]
                if (s >> (ND + 2 * a + sp)) & 1:
                    H[n, n] += eb[a, sp]
        # p SOC (diagonal l_z s_z part)
        if zeta_p:
            for i in range(NC):
                if (s >> (NS + i)) & 1:
                    H[n, n] += soc_p[i, i]
        # p SOC (off-diagonal ladder part)
        for i in range(NC):
            if not ((s >> (NS + i)) & 1):
                continue
            for j in range(NC):
                if i == j or ((s >> (NS + j)) & 1):
                    continue
                v = soc_p[j, i]
                if abs(v) < 1e-14:
                    continue
                s1, g = _apply_c(s, NS + i)
                s2, g2 = _apply_cdag(s1, NS + j)
                H[idx[s2], n] += v * g * g2
        # d SOC (optional, same structure on the lowest 10 bits)
        if soc_d is not None:
            for i in range(ND):
                if not ((s >> i) & 1):
                    continue
                for j in range(ND):
                    if i == j or ((s >> j) & 1):
                        continue
                    v = soc_d[j, i]
                    if abs(v) < 1e-14:
                        continue
                    s1, g = _apply_c(s, i)
                    s2, g2 = _apply_cdag(s1, j)
                    H[idx[s2], n] += v * g * g2
        # d-d interaction: table lives on the lowest 10 bits -> no extra signs
        for d2, amp in table_dd[d]:
            H[idx[d2 | (rest << ND)], n] += amp
        # d-bath hybridization (same pattern as ed_dmft.build_sector)
        for a in range(5):
            for sp in (0, 1):
                if Vb[a, sp] == 0:
                    continue
                i, j = 2 * a + sp, ND + 2 * a + sp
                for src, dst in ((j, i), (i, j)):
                    s1, g1 = _apply_c(s, src)
                    if s1 is None:
                        continue
                    s2, g2 = _apply_cdag(s1, dst)
                    if s2 is None:
                        continue
                    H[idx[s2], n] += Vb[a, sp] * g1 * g2
        # p-d Coulomb (full-state fermion signs)
        for i, j, k, l, v in terms:
            s4, g = _apply4(s, i, j, l, k)
            if s4 is not None:
                H[idx[s4], n] += v * g
    return states, H


# ---------------------------------------------------------------- spectra
def _one_body_matrix(E, V, states, op_terms):
    """<a|O|b> in the energy eigenbasis for O = sum t c^+_i c_j.

    op_terms: list of (i, j, t). Returns dense (dim, dim) complex.
    """
    idx = {s: n for n, s in enumerate(states)}
    O = np.zeros((len(E), len(E)), complex)
    cols = V.T  # columns are eigenstates in SO-sector basis
    # apply O to each eigenstate column, then project
    Opsi = np.zeros((len(states), len(E)), complex)
    for col in range(len(E)):
        for sn, s in enumerate(states):
            c = cols[col, sn]
            if abs(c) < 1e-12:
                continue
            for i, j, t in op_terms:
                s1, g1 = _apply_c(s, j)
                if s1 is None:
                    continue
                s2, g2 = _apply_cdag(s1, i)
                if s2 is None or s2 not in idx:
                    continue  # outside core_window: separate H block, dropped
                Opsi[idx[s2], col] += t * g1 * g2 * c
    O = V.conj().T @ Opsi
    return O


def ground_p6(E, V, states):
    """Index of the lowest eigenstate living in the 2p^6 block."""
    ncore = np.array([bin(s >> NS).count("1") for s in states])
    p6 = ncore == 6
    w6 = (np.abs(V[p6, :]) ** 2).sum(axis=0)
    candidates = np.where(w6 > 0.5)[0]
    return int(candidates[np.argmin(E[candidates])])


def ground_observables(E, V, states, temperatures=(0.0,)):
    """Many-body observables of the 2p^6 ground multiplet (R4 cross-check).

    From the SAME diagonalization that yields the XAS spectrum:
      - d-shell S^2, S_z of the T=0 ground state (d^8 ^3A2g anchor: S^2 ~ 2);
      - d-shell <L.S> (unit l.s operator, cubic basis) — the Thole-van der
        Laan sum-rule numerator — Boltzmann-averaged over the p^6 multiplet
        at each requested temperature;
      - <n_d> per cubic orbital, eg/t2g totals, and d^n configuration
        weights (the d^9 weight is the ZRS ligand-hole fraction rho);
      - sum-rule branch ratio BR = (2 - r)/(1 + r), r = <L.S>/<n_h>,
        <n_h> = 10 - <n_d>.

    Returns {'E0', 'ground': {...}, 'thermal': {T_K: {...}}}.
    """
    idx = {s: n for n, s in enumerate(states)}
    maskd = (1 << ND) - 1
    ls10 = np.zeros((ND, ND), complex)
    T5 = cubic_to_spherical_5()
    U2 = np.kron(T5, np.eye(2))
    ls10 = U2 @ _soc_matrix(1.0, l=2) @ U2.conj().T  # unit l.s, cubic basis

    nd_state = np.array([bin(s & maskd).count("1") for s in states])
    cnt = np.zeros((len(states), 5))
    sz_state = np.zeros(len(states))
    ndn_state = np.zeros(len(states))
    for n, s in enumerate(states):
        for a in range(5):
            up = (s >> (2 * a)) & 1
            dn = (s >> (2 * a + 1)) & 1
            cnt[n, a] = up + dn
            sz_state[n] += 0.5 * (up - dn)
            ndn_state[n] += dn

    # L.S: one-body over the d spin-orbitals
    LSo = np.zeros((len(states), len(states)), complex)
    # S^2 = Sz^2 + Sz + S-S+ ; S-S+ = N_down - sum_ab c+_{a dn} c+_{b up} c_{a up} c_{b dn}
    S2o = np.diag(sz_state**2 + sz_state + ndn_state).astype(complex)
    for n, s in enumerate(states):
        for i in range(ND):
            if not (s >> i) & 1:
                continue
            s1, g1 = _apply_c(s, i)
            for j in range(ND):
                v = ls10[j, i]
                if i == j or (s >> j) & 1 or abs(v) < 1e-14:
                    continue
                s2, g2 = _apply_cdag(s1, j)
                LSo[idx[s2], n] += v * g1 * g2
        for a in range(5):
            for b in range(5):
                s4, g = _apply4(s, 2 * a + 1, 2 * b, 2 * a, 2 * b + 1)
                if s4 is not None:
                    S2o[idx[s4], n] -= g
    P2 = np.abs(V) ** 2  # states x eigen: per-eigenstate diagonal expectations
    LS_eig = np.real(np.diag(V.conj().T @ LSo @ V))
    S2_eig = np.real(np.diag(V.conj().T @ S2o @ V))
    nd_eig = P2.T @ nd_state
    nd_orb_eig = P2.T @ cnt
    conf_eig = {k: P2.T @ (nd_state == k) for k in (7, 8, 9, 10)}

    ncore = np.array([bin(s >> NS).count("1") for s in states])
    w6 = (np.abs(V[ncore == 6, :]) ** 2).sum(axis=0)
    p6 = np.where(w6 > 0.5)[0]
    g = p6[np.argmin(E[p6])]

    def entry(i):
        return {
            "S2": float(S2_eig[i]),
            "Sz": float(sz_state @ P2[:, i]),
            "LS": float(LS_eig[i]),
            "n_d": float(nd_eig[i]),
            "n_d_orb": [float(x) for x in nd_orb_eig[i]],
            "eg": float(nd_orb_eig[i][[0, 3]].sum()),
            "t2g": float(nd_orb_eig[i][[1, 2, 4]].sum()),
            "conf": {k: float(v[i]) for k, v in conf_eig.items()},
        }

    KB = 8.617333262e-5
    out = {"E0": float(E[g]), "ground": entry(g), "thermal": {}}
    for T in temperatures:
        if T <= 0:
            tg = entry(g)
            n_h = 10.0 - tg["n_d"]
            r = tg["LS"] / n_h if n_h > 1e-9 else 0.0
            out["thermal"][0.0] = {
                "Z": 1, "LS": tg["LS"], "n_d": tg["n_d"], "n_h": n_h,
                "r": r, "BR_sumrule": (2 - r) / (1 + r),
            }
            continue
        dE = E[p6] - E[g]
        wgt = np.exp(-dE / (KB * T))
        Z = float(wgt.sum())
        wgt /= Z
        LS_T = float(wgt @ LS_eig[p6])
        nd_T = float(wgt @ nd_eig[p6])
        n_h = 10.0 - nd_T
        r = LS_T / n_h if n_h > 1e-9 else 0.0
        out["thermal"][T] = {
            "Z": int(len(p6)), "LS": LS_T, "n_d": nd_T, "n_h": n_h,
            "r": r, "BR_sumrule": (2 - r) / (1 + r),
        }
    return out


def xas_sticks(E, V, states, t_dip, q=0):
    """XAS stick spectrum from the 2p^6 ground state.

    Returns (w, amps): transition energies w = E_f - E_g and amplitudes
    amps[f] = <f|T_q|g> (radial integral = 1).
    """
    g = ground_p6(E, V, states)
    terms = [(d, p, v) for (d, p, qq), v in t_dip.items() if qq == q]
    O = _one_body_matrix(E, V, states, terms)
    w = E - E[g]
    return w, O[:, g]


def rixs(E, V, states, t_dip, omega_in, q_in=0, q_out=0, gamma=0.3):
    """Kramers-Heisenberg RIXS at incident energy omega_in.

    A_f = sum_n <f|T^+_qout|n><n|T_qin|g> / (omega_in - (E_n - E_g) + i gamma)
    Returns (w_loss, |A_f|^2).
    """
    g = ground_p6(E, V, states)
    Ti = _one_body_matrix(E, V, states, [(d, p, v) for (d, p, qq), v in t_dip.items() if qq == q_in])
    To = _one_body_matrix(E, V, states, [(d, p, v) for (d, p, qq), v in t_dip.items() if qq == q_out])
    Tog = To.conj().T  # <f|T^+|n>
    denom = omega_in - (E - E[g]) + 1j * gamma
    A = (Tog * (Ti[:, g] / denom)[None, :]).sum(axis=1)
    return E - E[g], A


def lorentz(w, amps, eta=0.3, grid=None):
    """Lorentzian-broadened intensity from sticks (w, amps)."""
    w = np.asarray(w)
    sel = np.abs(amps) > 1e-10
    w, a = w[sel], np.asarray(amps)[sel]
    if grid is None:
        lo = w.min() - 5 * eta if len(w) else -1.0
        hi = w.max() + 5 * eta if len(w) else 1.0
        grid = np.linspace(lo, hi, 4000)
    y = np.zeros_like(grid)
    for wi, ai in zip(w, a):
        y += abs(ai) ** 2 * eta / ((grid - wi) ** 2 + eta**2) / np.pi
    return grid, y
