"""Sigma(2) two-body screening: RPA bubble in J-coupled pair space.

First CI+MBPT increment per ``docs/CI_MBPT_SIGMA2_SPEC.md``: the
core-polarisation screening of the valence-valence Coulomb interaction,

    delta G^{(J)}[(ab),(cd)] = - S^{(J)} . D^{-1} . S^{(J)T}
    S^{(J)}[(ab),(mr)] = <ab; J|v|mr; J>_A          (valence pair x core-virtual)
    D[(mr)] = eps_r - eps_m                          (floored away from zero)

evaluated per total-pair-J channel as one dense product over the
(m, r) intermediate pairs (the high-performance architecture: the only
radial work is the S-block build; the channel products are dgemms).
``V_J`` values come from :func:`atomx.methods.mbpt._pair_v_J` — the same
kernel the tested closed-shell E^(2) path uses.

Scope of this first increment (documented honestly):

* SAME-J-channel screening only — both bubble vertices carry the pair's
  total J.  The full Dzuba Sigma(2) additionally mixes channels at the
  two vertices and adds hole-line and particle-particle families; those
  are follow-ups gated on the Eu3+ anchor below.
* The screening is delivered as a RAW (un-antisymmetrised) additive
  correction delta_v(a, b, c, d) to the m-scheme two-body element,
  obtained from the channel tables by Clebsch-Gordan recoupling; the
  sparse assembler's existing antisymmetrisation (v - v2) then applies
  unchanged.

Validation gates (spec): sign — the direct screening must be attractive
(delta G_dd < 0); magnitude anchor — on Eu3+ 4f6 the 5D0-7F0 gap must
move DOWN toward NIST 17270 from the unscreened 20465; convergence —
virtual window sweeps must be stable.
"""
from __future__ import annotations

from functools import lru_cache
from math import sqrt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from atomx.angular.wigner import wigner_3j
from atomx.methods.mbpt import _pair_v_J, _twoj_from_kappa  # same kernel as E^(2)
from atomx.numerics.bspline.convol import ConvolCache

__all__ = ["Sigma2Bubble", "build_sigma2_bubble"]


def _twoj(kappa: int) -> int:
    return _twoj_from_kappa(kappa)


def _cg(j2_a, m2_a, j2_b, m2_b, j2_tot, m2_tot) -> float:
    """Clebsch-Gordan <j_a m_a, j_b m_b | J M> from the Wigner 3j.

    All quantum numbers in doubled units (2j, 2m, 2J, 2M).
    """
    # CG = (-1)^{j_a - j_b + M} sqrt(2J+1) * 3j(j_a j_b J; m_a m_b -M)
    phase = (-1.0) ** ((j2_a - j2_b + m2_tot) // 2)
    w = wigner_3j(j2_a, j2_b, j2_tot, m2_a, m2_b, -m2_tot)
    if w == 0.0:
        return 0.0
    return phase * sqrt(j2_tot + 1) * w


class Sigma2Bubble:
    """Screened-interaction corrections for one core/virtual reference.

    Holds the per-J channel products delta G^{(J)} over the valence pair
    list and answers raw m-scheme quadruple corrections by CG recoupling.
    """

    def __init__(self, delta_g: Dict[int, np.ndarray],
                 pair_index: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
                 pair_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                 n_clipped: int):
        # delta_g[2J] : (n_pairs, n_pairs) screening matrix in pair space
        self._delta_g = delta_g
        self._pair_index = pair_index
        self._pair_list = pair_list
        self.n_clipped = n_clipped
        self.n_calls = 0

    @property
    def channels(self):
        return sorted(self._delta_g)

    def summary(self) -> str:
        nz = {j2: float(np.min(M[M != 0])) if np.any(M) else 0.0
              for j2, M in self._delta_g.items()}
        return (f"Sigma2Bubble: channels={self.channels}, "
                f"pairs={len(self._pair_list)}, clipped_denoms={self.n_clipped}, "
                f"min_diag_screening={nz}")

    def delta_raw(self, ka, m2a, kb, m2b, kc, m2c, kd, m2d,
                  a: Tuple[int, int], b: Tuple[int, int],
                  c: Tuple[int, int], d: Tuple[int, int]) -> float:
        """Antisymmetrised m-scheme correction <ab|dv_A|cd>.

        ``a..d`` are (n, kappa) orbital keys; ``m2*`` doubled m values;
        ``ka..kd`` kappas.  The channel tables are already
        pair-antisymmetrised (``_pair_v_J`` returns <ab;J|v|cd;J>_A), so
        the CG recoupling lands directly on the antisymmetrised
        m-scheme element — the caller must ADD it to the assembled
        (v - v2) combination, never to the raw pieces.
        """
        key_ab = (a, b) if (a, b) in self._pair_index else None
        key_cd = (c, d) if (c, d) in self._pair_index else None
        if key_ab is None or key_cd is None:
            return 0.0
        i_ab = self._pair_index[key_ab]
        i_cd = self._pair_index[key_cd]
        j2a, j2b, j2c, j2d = _twoj(ka), _twoj(kb), _twoj(kc), _twoj(kd)
        j2_min = max(abs(j2a - j2b), abs(j2c - j2d))
        j2_max = min(j2a + j2b, j2c + j2d)
        m2 = m2a + m2b  # = m2c + m2d for m-conserving elements
        total = 0.0
        for j2 in range(j2_min, j2_max + 1, 2):
            M = self._delta_g.get(j2)
            if M is None:
                continue
            # sum over the single M fixed by the m values
            cg_bra = _cg(j2a, m2a, j2b, m2b, j2, m2)
            cg_ket = _cg(j2c, m2c, j2d, m2d, j2, m2)
            if cg_bra == 0.0 or cg_ket == 0.0:
                continue
            total += cg_bra * cg_ket * M[i_ab, i_cd]
        self.n_calls += 1
        return total


def build_sigma2_bubble(orbitals_dict, bsp_data, Z, ns,
                        core_keys: Sequence[Tuple[int, int]],
                        virt_keys: Sequence[Tuple[int, int]],
                        valence_pairs: Sequence[Tuple[Tuple[int, int],
                                                       Tuple[int, int]]],
                        eps: Dict[Tuple[int, int], float],
                        denom_floor: float = 0.05,
                        j2_max: int | None = None,
                        kappa_max_virt: int = 4):
    """Build the per-J screening matrices over ``valence_pairs``.

    Parameters
    ----------
    orbitals_dict, bsp_data, Z, ns
        The CI run's radial workspace (the same one feeding the sparse
        assembler) — the core is the closed [Xe] carried in every
        determinant, the virtuals supply the intermediate states.
    core_keys
        (n, kappa) of the closed core subshells.
    virt_keys
        (n, kappa) of the intermediate-state virtuals.
    valence_pairs
        Ordered ((n,kappa), (n,kappa)) valence pairs spanning the CI
        open shells (both orderings the CI will ask for must be present).
    eps
        Orbital energies (Hartree) for the denominator, keyed like the
        orbitals.
    denom_floor
        Minimum |eps_m - eps_r| kept (Hartree); smaller denominators are
        clipped and COUNTED — quasi-degenerate intermediates must not
        blow up the screening silently.
    """
    cache = ConvolCache(bsp_data)
    valence_pairs = list(valence_pairs)
    pair_index = {p: i for i, p in enumerate(valence_pairs)}
    channels: Dict[int, np.ndarray] = {}
    n_clipped = 0

    core = [k for k in core_keys if k in orbitals_dict]
    virt = [k for k in virt_keys if k in orbitals_dict]

    for j2 in range(0, (j2_max or 16) + 1, 2):
        rows = []
        diag = []
        # S blocks: valence-pair x (core, virt) channels
        any_pair = False
        S = np.zeros((len(valence_pairs),
                      sum(1 for m in core for _ in virt)))
        col = 0
        denom = np.zeros(S.shape[1])
        for m in core:
            Pm, Qm = orbitals_dict[m]
            for r in virt:
                Pr, Qr = orbitals_dict[r]
                d = eps[r] - eps[m]  # excitation cost; > 0 for bound core
                if abs(d) < denom_floor:
                    d = denom_floor * (1.0 if d >= 0 else -1.0)
                    n_clipped += 1
                denom[col] = d
                for i, (a, b) in enumerate(valence_pairs):
                    Pa, Qa = orbitals_dict[a]
                    Pb, Qb = orbitals_dict[b]
                    v = _pair_v_J(Pa, Qa, a[1], Pb, Qb, b[1],
                                  Pm, Qm, m[1], Pr, Qr, r[1],
                                  j2, bsp_data, cache)
                    # normalisation of coupled product pairs
                    v /= sqrt((1 + (a == b)) * (1 + (m == r)))
                    if v != 0.0:
                        any_pair = True
                    S[i, col] = v
                col += 1
        if not any_pair:
            continue
        # delta G^{(J)} = - S D^{-1} S^T
        Dinv = 1.0 / denom
        G = -(S * Dinv) @ S.T
        channels[j2] = 0.5 * (G + G.T)
    return Sigma2Bubble(channels, pair_index, valence_pairs, n_clipped)


def build_sigma2_mscheme(orbitals_dict, bsp_data, Z, ns,
                         core_keys, virt_keys, valence_keys, eps,
                         denom_floor: float = 0.05):
    """Complete second-order screening in the m-scheme (all J channels).

    The bubble in m-scheme needs no J-channel reconstruction — the
    angular-momentum sums live inside the m-scheme Coulomb elements
    themselves, so this construction includes the cross-channel (multi-J)
    physics the same-J pair-space version structurally could not see:

        d(a,b,c,d) = -sum_{m in core, r in virt}
                     [g(a,m,c,r) g(b,r,d,m) - g(a,m,d,r) g(b,r,c,m)]
                     / (eps_r - eps_m)

    with raw (un-antisymmetrised) m-scheme Coulomb elements g.

    High-performance layout (m-conservation bucketing): the element
    g(a,m,c,r) is nonzero only when m2a - m2c == m2r - m2m, so the
    (m, r) intermediates partition into buckets by that difference and
    the per-bucket G tables are dense [valence-pair x intermediate].
    Each quadruple then costs one small numba dot product per diagram
    term instead of ~600 Python element calls.
    """
    from functools import lru_cache as _lru
    from math import sqrt as _sqrt

    import numba as _nb
    import numpy as _np

    from atomx.angular.csf_hamiltonian import _m2_values, two_body_angular, twoj
    import atomx.physics.csf.matrix as MAT

    @_nb.njit(cache=True)
    def _bucket_dot(g1, g2, dinv):
        total = 0.0
        for j in range(g1.shape[0]):
            total += g1[j] * g2[j] * dinv[j]
        return total

    ext = sorted(set(valence_keys) | set(core_keys) | set(virt_keys))
    spinors = [(n, k, m2) for (n, k) in ext
               for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}

    core_idx = [index[(n, k, m2)] for (n, k) in core_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    virt_idx = [index[(n, k, m2)] for (n, k) in virt_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    val_idx = [index[(n, k, m2)] for (n, k) in valence_keys
               if (n, k) in orbitals_dict
               for m2 in _m2_values(twoj(k))]

    # intermediates, bucketed by (m2r - m2m)
    buckets = {}
    denom_all = []
    mr_flat = []
    for mi in core_idx:
        for ri in virt_idx:
            d = eps[spinors[ri][:2]] - eps[spinors[mi][:2]]
            if abs(d) < denom_floor:
                continue
            delta = spinors[ri][2] - spinors[mi][2]
            buckets.setdefault(delta, []).append(len(mr_flat))
            mr_flat.append((mi, ri))
            denom_all.append(d)

    @_lru(maxsize=None)
    def g(ia, ic, im, ir_):
        a, c, m, r_ = spinors[ia], spinors[ic], spinors[im], spinors[ir_]
        terms = two_body_angular(a[1], a[2], m[1], m[2],
                                 c[1], c[2], r_[1], r_[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pm, Qm = orbitals_dict[(m[0], m[1])]
        Pr, Qr = orbitals_dict[(r_[0], r_[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pm, Qm, Pr, Qr,
                                    bsp_data, k)
        return total

    # G1[delta]: rows = valence pairs (a, c) with m2a - m2c == delta,
    #            cols = bucket(delta) intermediates; values g(a, m, c, r).
    # G2[delta]: rows = valence pairs (b, d) with m2b - m2d == -delta
    #            (slot-transposed element g(b, d, r, m)), SAME columns —
    #            building the transposed table directly avoids the
    #            reversed-pair lookup, which cannot exist because (r, m)
    #            swaps the core/virtual slot roles.
    G1 = {}
    G2 = {}
    dinv = {}
    row_of = {}
    for delta, js in buckets.items():
        rows1 = [(ia, ic) for ia in val_idx for ic in val_idx
                 if spinors[ia][2] - spinors[ic][2] == delta]
        rows2 = [(ib, idd) for ib in val_idx for idd in val_idx
                 if spinors[ib][2] - spinors[idd][2] == -delta]
        M1 = _np.zeros((len(rows1), len(js)))
        M2 = _np.zeros((len(rows2), len(js)))
        for r_, (ia, ic) in enumerate(rows1):
            for c_, j in enumerate(js):
                mi, ri = mr_flat[j]
                M1[r_, c_] = g(ia, ic, mi, ri)
        for r_, (ib, idd) in enumerate(rows2):
            for c_, j in enumerate(js):
                mi, ri = mr_flat[j]
                M2[r_, c_] = g(ib, idd, ri, mi)
        G1[delta] = M1
        G2[delta] = M2
        dinv[delta] = 1.0 / _np.array([denom_all[j] for j in js])
        for r_, pair in enumerate(rows1):
            row_of[(delta, pair)] = ("1", r_)
        for r_, pair in enumerate(rows2):
            row_of[(delta, pair)] = ("2", r_)

    def delta(ia, ib, ic, id_):
        """Antisymmetrised m-scheme screening <ab|dV_A|cd>, spinor indices."""
        d1 = spinors[ia][2] - spinors[ic][2]   # direct bucket key
        d2 = spinors[ia][2] - spinors[id_][2]  # exchange bucket key
        total = 0.0
        ra = row_of.get((d1, (ia, ic)))
        rb = row_of.get((d1, (ib, id_)))
        if ra and rb:
            g1 = G1[d1][ra[1]]
            g2 = G2[d1][rb[1]]
            total -= _bucket_dot(g1, g2, dinv[d1])
        rc = row_of.get((d2, (ia, id_)))
        rd = row_of.get((d2, (ib, ic)))
        if rc and rd:
            g3 = G1[d2][rc[1]]
            g4 = G2[d2][rd[1]]
            total += _bucket_dot(g3, g4, dinv[d2])
        return total

    delta.n_intermediate = len(mr_flat)
    delta.spinors = spinors
    delta.index = index
    return delta


def build_sigma2_holeline(orbitals_dict, bsp_data, Z, ns,
                          core_keys, virt_keys, valence_keys, eps,
                          denom_floor: float = 0.02):
    """Second-order hole-line family, CORRECTED construction.

    The 2026-10-02 dual anchor established that the previous
    core-pair-ground-state intermediates were wrong physics (added ~9 Ha
    of wholesale core-pair correlation; see the spec's hole-line verdict).
    The valence resolvent requires intermediate states OUTSIDE the CI
    reference: a core electron m promoted to a virtual r.  Pair level:

        d_hl(a,b,c,d) = -1/2 sum_{m core, r virt}
                        [g(a,m,c,r) g(b,r,d,m) - g(a,m,d,r) g(b,r,c,m)]
                        * [1/D_ab + 1/D_cd],   D_xy = eps_x+eps_y-eps_m-eps_r

    (direct minus exchange, symmetrised denominator => Hermitian; the
    same raw m-scheme elements and (m, r) pools as the bubble builder,
    recombined at the PAIR level).  Gates, in order: brute-force
    collision AND a physics sign gate (the diagonal element must be
    NEGATIVE/attractive for valence-above-core) before any anchor run.
    """
    from functools import lru_cache as _lru
    import numba as _nb
    import numpy as _np

    from atomx.angular.csf_hamiltonian import _m2_values, two_body_angular, twoj
    import atomx.physics.csf.matrix as MAT

    @_nb.njit(cache=True)
    def _pair_dot(p1, p2, dinv):
        total = 0.0
        for j in range(p1.shape[0]):
            total += p1[j] * p2[j] * dinv[j]
        return total

    ext = sorted(set(valence_keys) | set(core_keys) | set(virt_keys))
    spinors = [(n, k, m2) for (n, k) in ext
               for m2 in _m2_values(twoj(k))]
    index = {s_: i for i, s_ in enumerate(spinors)}
    core_idx = [index[(n, k, m2)] for (n, k) in core_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    virt_idx = [index[(n, k, m2)] for (n, k) in virt_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    val_idx = [index[(n, k, m2)] for (n, k) in valence_keys
               if (n, k) in orbitals_dict
               for m2 in _m2_values(twoj(k))]

    @_lru(maxsize=None)
    def g(ia, ic, im, ir_):
        a, c, m, r_ = spinors[ia], spinors[ic], spinors[im], spinors[ir_]
        terms = two_body_angular(a[1], a[2], m[1], m[2],
                                 c[1], c[2], r_[1], r_[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pm, Qm = orbitals_dict[(m[0], m[1])]
        Pr, Qr = orbitals_dict[(r_[0], r_[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pm, Qm, Pr, Qr,
                                    bsp_data, k)
        return total

    # (m, r) intermediates bucketed by difference (m2r - m2m) exactly as
    # the bubble does — the same pools, pair-level recombination.
    buckets = {}
    denom_all = []
    mr_flat = []
    for mi in core_idx:
        for ri in virt_idx:
            d = eps[spinors[ri][:2]] - eps[spinors[mi][:2]]
            if abs(d) < denom_floor:
                continue
            delta = spinors[ri][2] - spinors[mi][2]
            buckets.setdefault(delta, []).append(len(mr_flat))
            mr_flat.append((mi, ri))
            denom_all.append(None)  # per-row below; hole-line D uses valence

    # Pair tables per difference bucket:
    #   G1[d][(a,c) rows, mr cols] = g(a, m, c, r)      (same as bubble G1)
    #   G2t[d][(b,d) rows, mr cols] = g(b, r, d, m)     (transposed slots)
    G1 = {}
    G2t = {}
    Dm = {}     # eps_m + eps_r per column
    rows1_of = {}
    rows2_of = {}
    for delta, js in buckets.items():
        rows1 = [(ia, ic) for ia in val_idx for ic in val_idx
                 if spinors[ia][2] - spinors[ic][2] == delta]
        rows2 = [(ib, idd) for ib in val_idx for idd in val_idx
                 if spinors[ib][2] - spinors[idd][2] == -delta]
        M1 = _np.zeros((len(rows1), len(js)))
        M2 = _np.zeros((len(rows2), len(js)))
        for r_, (ia, ic) in enumerate(rows1):
            for c_, j in enumerate(js):
                mi, ri = mr_flat[j]
                M1[r_, c_] = g(ia, ic, mi, ri)
        for r_, (ib, idd) in enumerate(rows2):
            for c_, j in enumerate(js):
                mi, ri = mr_flat[j]
                M2[r_, c_] = g(ib, idd, ri, mi)
        G1[delta] = M1
        G2t[delta] = M2
        Dm[delta] = _np.array([
            eps[spinors[mr_flat[j][0]][:2]] + eps[spinors[mr_flat[j][1]][:2]]
            for j in js])
        for r_, pr in enumerate(rows1):
            rows1_of[(delta, pr)] = r_
        for r_, pr in enumerate(rows2):
            rows2_of[(delta, pr)] = r_

    def delta_hl(ia, ib, ic, id_):
        """Antisymmetrised m-scheme hole-line element <ab|dV_A|cd>."""
        d1 = spinors[ia][2] - spinors[ic][2]
        d2 = spinors[ia][2] - spinors[id_][2]
        total = 0.0
        for delta, rows, table_pair in ((d1, 0, 0), (d2, 1, 0)):
            if delta not in buckets:
                continue
            js = buckets[delta]
            mr_e = Dm[delta]
            if delta == d1:
                r_ac = rows1_of.get((delta, (ia, ic)))
                r_bd = rows2_of.get((delta, (ib, id_)))
                if r_ac is None or r_bd is None:
                    continue
                e_ab = eps[spinors[ia][:2]] + eps[spinors[ib][:2]]
                e_cd = eps[spinors[ic][:2]] + eps[spinors[id_][:2]]
                dinv = _np.array([
                    0.5 * (1.0 / (e_ab - mr_e[j]) + 1.0 / (e_cd - mr_e[j]))
                    for j in range(len(js))])
                g1 = G1[delta][r_ac]
                g2 = G2t[delta][r_bd]
                total -= _pair_dot(g1, g2, dinv)
                # exchange bubble of the pair family
                r_ad = rows1_of.get((delta, (ia, id_)))
                r_bc = rows2_of.get((delta, (ib, ic)))
                if r_ad is not None and r_bc is not None:
                    g3 = G1[delta][r_ad]
                    g4 = G2t[delta][r_bc]
                    total += _pair_dot(g3, g4, dinv)
        return total

    delta_hl.spinors = spinors
    delta_hl.index = index
    delta_hl.n_intermediate = len(mr_flat)
    return delta_hl


def build_sigma1_mscheme(orbitals_dict, bsp_data, Z, ns,
                          core_keys, virt_keys, valence_keys, eps,
                          denom_floor: float = 0.05):
    """Second-order one-body self-energy (core-polarisation screening of
    the valence field) in the m-scheme — the one-body analogue of the
    bubble:

        S1(v, v') = -2 sum_{m in core, r in virt}
                    g(v, r, m, m) g(r, v', m, m) / (eps_r - eps_m)

    (factor 2 = two time orderings, same convention as the bubble).
    Delivered as an {(n,kappa) x (n,kappa) -> shift} dict suitable for the
    assembler's one-body table: the assembler only asks same-(kappa, m)
    one-body elements, and m-conservation of g forces m_v == m_r == m_v'
    automatically, so the m-sum collapses per (v, v').
    """
    from functools import lru_cache as _lru
    from atomx.angular.csf_hamiltonian import _m2_values, two_body_angular, twoj
    import atomx.physics.csf.matrix as MAT

    ext = sorted(set(valence_keys) | set(core_keys) | set(virt_keys))
    spinors = [(n, k, m2) for (n, k) in ext
               for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}
    core_idx = [index[(n, k, m2)] for (n, k) in core_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    virt_idx = [index[(n, k, m2)] for (n, k) in virt_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    val_idx = [index[(n, k, m2)] for (n, k) in valence_keys
               if (n, k) in orbitals_dict
               for m2 in _m2_values(twoj(k))]

    @_lru(maxsize=None)
    def g(ia, ic, im, ir_):
        a, c, m, r_ = spinors[ia], spinors[ic], spinors[im], spinors[ir_]
        terms = two_body_angular(a[1], a[2], m[1], m[2],
                                 c[1], c[2], r_[1], r_[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pm, Qm = orbitals_dict[(m[0], m[1])]
        Pr, Qr = orbitals_dict[(r_[0], r_[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pm, Qm, Pr, Qr,
                                    bsp_data, k)
        return total

    shifts = {}
    for iv in val_idx:
        for iv2 in val_idx:
            if spinors[iv][1] != spinors[iv2][1]:
                continue  # one-body preserves kappa
            if spinors[iv][2] != spinors[iv2][2]:
                continue  # and m
            total = 0.0
            for mi in core_idx:
                for ri in virt_idx:
                    d = eps[spinors[ri][:2]] - eps[spinors[mi][:2]]
                    if abs(d) < denom_floor:
                        continue
                    g1 = g(iv, ri, mi, mi)     # <v m|V|r m>
                    if g1 == 0.0:
                        continue
                    g2 = g(ri, iv2, mi, mi)    # <r m|V|v' m>
                    total -= 2.0 * g1 * g2 / d
            key = (spinors[iv][:2], spinors[iv2][:2])
            shifts[key] = shifts.get(key, 0.0) + total / 1.0
    # each (n,k) pair appears once per shared m value with identical
    # contribution by symmetry; average over the number of m values to
    # avoid double counting when the assembler applies it per element.
    from collections import Counter
    counts = Counter()
    for iv in val_idx:
        for iv2 in val_idx:
            if spinors[iv][1] == spinors[iv2][1] and spinors[iv][2] == spinors[iv2][2]:
                counts[(spinors[iv][:2], spinors[iv2][:2])] += 1
    for key in list(shifts):
        shifts[key] /= counts.get(key, 1)
    return shifts
