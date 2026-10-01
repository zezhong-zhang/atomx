"""Sparse M-scheme determinant CI for spaces beyond the exact-CSF cap.

Motivation (Eu3+ probe series, 2026-09-30): the full valence SD space of
``[Xe]4f⁶`` needs 60 720 determinants for the 2J=0 block alone against the
exact engine's 2 000-determinant cap — and the exact engine's cost model
(all determinant pairs, dense projection) cannot go there.  This module
keeps the same Dirac-Coulomb Hamiltonian and the same radial kernels
(``matrix._h_one_body`` / ``_norm_sq`` / ``_rk``), but works directly in
the M-scheme determinant basis with *sparse* assembly: determinants are
occupation bitmasks, the Hamiltonian couples only determinants differing
by at most two spin-orbitals (Slater–Condon), and only non-zero elements
are ever formed.  The lowest states come from ``scipy`` Lanczos
(``eigsh``), never from a dense diagonalisation.

Conventions — identical to ``benchmarks/mscheme_ci_reference.py`` (the
independent oracle the exact engine is gated against):

* one-body element ``<a|h|c> / <c|c>`` between spinors sharing
  ``(kappa, m)``, signs from ``_ops_sign``;
* two-body antisymmetrised ``v(ab,cd) - v(ab,dc)`` with BOTH integrals
  evaluated before any angular shortcut (the oracle's 2026-09-21 lesson:
  skipping the exchange integral when the direct one vanishes breaks
  rotational invariance);
* terms leaving the truncated determinant space are dropped, exactly as
  the CSF representation drops them.

What the M-scheme gives back: the spectrum of one ``2M`` sector collects
*every* ``J >= |M|`` state of the space (plus, if the pattern set mixes
parities, both parities — pass a single-parity pattern set for
parity-pure results).  The ground of the even-parity M=0 sector is the
J=0 ground state.  Resolving a full J ladder needs the M-block
subtraction (``E(M=J) minus the spectra of M=J-1, J-2, ...``) or an
angular projection — deliberately out of scope for this first engine.

This module is the "sparse" half of the selected/sparse upgrade; the
PT2-guided *selection* layer (CIPSI/SHCI-style growth of the determinant
space) composes on top of ``assemble_sparse_hamiltonian`` and is planned
as the next increment.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import numba as _NB_MOD
except ImportError:  # pragma: no cover - numba is a hard dep, guarded anyway
    _NB_MOD = None
_NB = _NB_MOD
from scipy.sparse.linalg import eigsh

from atomx.angular.csf_hamiltonian import _m2_values, _ops_sign, two_body_angular, twoj
from atomx.physics.csf.breit_ci import breit_terms
from atomx.physics.csf.matrix import _breit_radial_evaluator
import atomx.physics.csf.matrix as MAT

__all__ = [
    "enumerate_determinants",
    "assemble_sparse_hamiltonian",
    "sparse_ci_spectrum",
    "patterns_from_csfs",
]


def patterns_from_csfs(csfs) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
    """Unique occupation patterns of a CSF list, in sorted order.

    A pattern is the tuple ``((n, kappa, q), ...)`` of occupied subshells
    with integer occupations — the same shape ``csf_signature`` reduces a
    CSF to, and what the oracle's ``patterns_from_configs`` returns.
    """
    seen = set()
    for csf in csfs:
        pat = tuple(sorted((sh.n, sh.kappa, int(round(sh.occupation)))
                           for sh in csf.subshells))
        seen.add(pat)
    return tuple(sorted(seen))


def _spinor_table(orbitals_dict, patterns=None) -> List[Tuple[int, int, int]]:
    """Global spin-orbital list: (n, kappa, 2m), sorted for stable masks.

    With ``patterns``, only subshells the patterns actually reference are
    listed — orbitals that exist purely as MBPT intermediate states never
    enter a determinant, and including them both bloats the pair buckets
    and (past 128 spinors) overflows the two-word masks.
    """
    if patterns is not None:
        allowed = {(n, k) for pattern in patterns for (n, k, _q) in pattern}
        keys = sorted(k for k in orbitals_dict if k in allowed)
    else:
        keys = sorted(orbitals_dict)
    out = []
    for (n, kappa) in keys:
        for m2 in _m2_values(twoj(kappa)):
            out.append((n, kappa, m2))
    return out


def enumerate_determinants(patterns, orbitals_dict, m2_total: int = 0):
    """Occupation bitmasks of the pattern set at fixed ``2M``.

    A determinant is a bitmask over the global spin-orbital table; two
    determinants are the same physical state iff their masks are equal
    (each determinant belongs to exactly one occupation pattern).
    """
    spinors = _spinor_table(orbitals_dict, patterns)
    # (n, kappa) -> (bit offset, list of global indices)
    offset = {}
    for si, (n, kappa, _m) in enumerate(spinors):
        offset.setdefault((n, kappa), []).append(si)
    masks = []
    for pattern in patterns:
        per_shell = []
        for (n, kappa, q) in pattern:
            members = offset[(n, kappa)]
            per_shell.append((members, q))
        # recursive product of combinations with a running m-sum
        def rec(i, chosen, m_sum):
            if i == len(per_shell):
                if m_sum == m2_total:
                    mask = 0
                    for si in chosen:
                        mask |= 1 << si
                    masks.append(mask)
                return
            members, q = per_shell[i]
            if q == 0:
                rec(i + 1, chosen, m_sum)
                return
            for combo in itertools.combinations(members, q):
                rec(i + 1, chosen + list(combo),
                    m_sum + sum(spinors[si][2] for si in combo))
        rec(0, [], 0)
    masks.sort()
    return masks, spinors


def assemble_sparse_hamiltonian(masks, spinors, orbitals_dict, bsp_data, Z, ns,
                                *, method="numba", two_body_delta=None,
                                one_body_delta=None, one_body_m_delta=None,
                                include_breit=False):
    """CSR Hamiltonian over the determinant masks (Dirac-Coulomb).

    ``method="numba"`` (default): two-phase build — every distinct
    one-/two-body value is computed once in Python (phase 1), then a
    numba kernel walks the determinants doing pure bit arithmetic
    (phase 2).  ``method="python"`` keeps the reference loop; the two
    paths are asserted element-identical in the test suite.

    ``one_body_m_delta`` — optional callable ``(ia, ic) -> float`` over
    spinor indices adding an m-MIXING one-body channel (same kappa,
    different m; e.g. the crystal-field q != 0 terms).  Delivered as a
    separate pass so the base one-body value (m-blind) is not re-emitted
    for every m pair.  Must be real: complex CF parameter sets need a
    tesseral (real) representation before CI wiring.
    """
    if method == "python" or _NB is None:
        return _assemble_reference(masks, spinors, orbitals_dict,
                                   bsp_data, Z, ns,
                                   two_body_delta=two_body_delta,
                                   one_body_delta=one_body_delta,
                                   one_body_m_delta=one_body_m_delta,
                                   include_breit=include_breit)
    return _assemble_numba(masks, spinors, orbitals_dict, bsp_data, Z, ns,
                           two_body_delta=two_body_delta,
                           one_body_delta=one_body_delta,
                           one_body_m_delta=one_body_m_delta,
                           include_breit=include_breit)


def _assemble_reference(masks, spinors, orbitals_dict, bsp_data, Z, ns,
                        *, two_body_delta=None, one_body_delta=None,
                        one_body_m_delta=None, include_breit=False):
    """CSR Hamiltonian over the determinant masks (Dirac-Coulomb)."""
    index = {mask: i for i, mask in enumerate(masks)}
    n = len(masks)

    @lru_cache(maxsize=None)
    def one_body(a_key, c_key):
        Pa, Qa = orbitals_dict[a_key]
        Pc, Qc = orbitals_dict[c_key]
        h = MAT._h_one_body(Pa, Qa, Pc, Qc, a_key[1], Z, bsp_data, ns)
        return h / MAT._norm_sq(Pc, Qc, c_key[1], Z, bsp_data, ns)

    @lru_cache(maxsize=None)
    def angular(ka, m2a, kb, m2b, kc, m2c, kd, m2d):
        return two_body_angular(ka, m2a, kb, m2b, kc, m2c, kd, m2d)

    @lru_cache(maxsize=None)
    def v_element(ia, ib, ic, id_):
        a = spinors[ia]; b = spinors[ib]; c = spinors[ic]; d = spinors[id_]
        terms = angular(a[1], a[2], b[1], b[2], c[1], c[2], d[1], d[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pb, Qb = orbitals_dict[(b[0], b[1])]
        Pd, Qd = orbitals_dict[(d[0], d[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd,
                                    bsp_data, k)
        return total

    breit_evaluate = (_breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns)
                      if include_breit else None)

    if breit_evaluate is not None:
        def _breit_element(ia, ib, ic, id_):
            total = 0.0
            for key, coeff in breit_terms(ia, ib, ic, id_, 1.0, spinors):
                total += coeff * breit_evaluate(key)
            return total
    else:
        def _breit_element(ia, ib, ic, id_):
            return 0.0

    # add-pair buckets by m-sum, for the two-body loop
    pair_buckets: Dict[int, List[Tuple[int, int]]] = {}
    for ia, ib in itertools.combinations(range(len(spinors)), 2):
        pair_buckets.setdefault(spinors[ia][2] + spinors[ib][2], []).append((ia, ib))

    # one-body partners sharing (kappa, m)
    km_partners: Dict[Tuple[int, int], List[int]] = {}
    kappa_partners: Dict[int, List[int]] = {}
    for si, (_n, kappa, m2) in enumerate(spinors):
        km_partners.setdefault((kappa, m2), []).append(si)
        kappa_partners.setdefault(kappa, []).append(si)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    def emit(row, col, value):
        if value != 0.0:
            rows.append(row); cols.append(col); vals.append(value)

    diag = np.zeros(n)
    for col, mask in enumerate(masks):
        occ = [si for si in range(len(spinors)) if mask >> si & 1]
        # one-body
        for c in occ:
            nc, kc, mc = spinors[c]
            for a in km_partners[(kc, mc)]:
                na, ka, ma = spinors[a]
                h = one_body((na, ka), (nc, kc))
                if one_body_delta is not None:
                    h += one_body_delta.get(((na, ka), (nc, kc)), 0.0)
                if a == c:
                    diag[col] += h
                    continue
                sign = _ops_sign(mask, [(a, True), (c, False)])
                if not sign:
                    continue
                nmask = (mask & ~(1 << c)) | (1 << a)
                row = index.get(nmask)
                if row is not None:
                    emit(row, col, sign * h)
        # one-body m-resolved channel (crystal field): same kappa, any m
        # — m-diagonal elements (q=0) are m-dependent, so neither the
        # m-blind base value nor one_body_delta can carry them
        if one_body_m_delta is not None:
            for c in occ:
                kc = spinors[c][1]
                for a in kappa_partners[kc]:
                    h = one_body_m_delta(a, c)
                    if h == 0.0:
                        continue
                    if a == c:
                        diag[col] += h
                        continue
                    sign = _ops_sign(mask, [(a, True), (c, False)])
                    if not sign:
                        continue
                    nmask = (mask & ~(1 << c)) | (1 << a)
                    row = index.get(nmask)
                    if row is not None:
                        emit(row, col, sign * h)
        # two-body
        for c, d in itertools.combinations(occ, 2):
            m_sum = spinors[c][2] + spinors[d][2]
            for a, b in pair_buckets.get(m_sum, ()):
                v = v_element(a, b, c, d)
                v2 = v_element(a, b, d, c)
                anti = v - v2
                if include_breit:
                    anti += (_breit_element(a, b, c, d)
                             - _breit_element(a, b, d, c))
                if two_body_delta is not None:
                    anti += two_body_delta(a, b, c, d)
                if anti == 0.0:
                    continue
                sign = _ops_sign(mask, [(a, True), (b, True),
                                         (d, False), (c, False)])
                if not sign:
                    continue
                nmask = (mask & ~(1 << c) & ~(1 << d)) | (1 << a) | (1 << b)
                row = index.get(nmask)
                if row is not None:
                    emit(row, col, sign * anti)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    H = H + sp.diags(diag)
    H.sum_duplicates()
    return H


_SECTOR_CACHE: Dict[Tuple[int, int, tuple, int], tuple] = {}
_SECTOR_CACHE_MAX = 4


def sparse_ci_spectrum(orbitals_dict, bsp_data, Z, ns, patterns,
                       m2_total: int = 0, n_states: int = 10):
    """Lowest ``n_states`` energies of one ``2M`` determinant sector.

    Returns ``(eigenvalues, n_determinants)`` with eigenvalues ascending.

    Assembled sector Hamiltonians are cached (up to 4) keyed by
    ``(Z, ns, patterns, m2_total)`` — ``sparse_j_levels`` needs the
    ``2M = 2J`` sector that a spectrum call may already have built, and
    rebuilding a 60k-determinant assembly costs minutes.
    """
    key = (Z, ns, tuple(patterns), m2_total)
    if key in _SECTOR_CACHE:
        H, n = _SECTOR_CACHE[key]
    else:
        masks, spinors = enumerate_determinants(patterns, orbitals_dict,
                                                m2_total)
        n = len(masks)
        if n == 0:
            return np.array([]), 0
        H = assemble_sparse_hamiltonian(masks, spinors, orbitals_dict,
                                        bsp_data, Z, ns)
        if len(_SECTOR_CACHE) >= _SECTOR_CACHE_MAX:
            _SECTOR_CACHE.clear()
        _SECTOR_CACHE[key] = (H, n)
    if n_states >= n - 1:
        w = np.linalg.eigvalsh(H.toarray())
        return w[:max(0, n_states)], n
    # ARPACK's default ncv = max(2k+1, 20) restarts heavily for large k;
    # a wider Krylov space converges far fewer times.
    ncv = min(n - 1, max(4 * n_states + 1, 60))
    w = eigsh(H, k=n_states, which="SA", ncv=ncv,
              return_eigenvectors=False)
    return np.sort(w), n


def _multiset_subtract(low: np.ndarray, high: np.ndarray, tol: float):
    """Remove every element of ``high`` from ``low`` (nearest-match first).

    ``spec(M=J)`` contains, with multiplicity, every state of every
    manifold with ``J' >= J``; subtracting ``spec(M=J+2)`` leaves exactly
    the ``J`` manifold.  Degeneracies are handled by matching each
    high-sector level to its nearest remaining low-sector level.
    """
    remaining = list(map(float, low))
    for h in np.sort(high):
        if not remaining:
            break
        best = min(range(len(remaining)), key=lambda i: abs(remaining[i] - h))
        if abs(remaining[best] - h) <= tol:
            remaining.pop(best)
    return np.sort(np.array(remaining)) if remaining else np.array([])


def sparse_j_levels(orbitals_dict, bsp_data, Z, ns, patterns,
                    j2: int, n_states: int = 10, tol: float = 1e-8):
    """Energies of the pure ``2J`` manifold from M-sector subtraction.

    ``spec(J) = spec(M=J) − spec(M=J+2)`` as multisets: every eigenvalue of
    the ``2M = 2J`` determinant sector belongs to some manifold with
    ``J' >= J``, and the ``2M = 2J + 2`` sector contains exactly those with
    ``J' >= J + 1``.  Works for integer and half-integer ``j2`` alike —
    pass a pattern set of one parity for parity-pure manifolds.

    Returns ``(energies, n_dets_mj, n_dets_mj_plus2)``.
    """
    w_j, n_j = sparse_ci_spectrum(orbitals_dict, bsp_data, Z, ns, patterns,
                                  m2_total=j2, n_states=n_states)
    w_hi, n_hi = sparse_ci_spectrum(orbitals_dict, bsp_data, Z, ns, patterns,
                                    m2_total=j2 + 2, n_states=n_states)
    return _multiset_subtract(w_j, w_hi, tol), n_j, n_hi


# ---------------------------------------------------------------------------
# Numba phase-2 kernel: pure bit arithmetic over determinant masks.
# ---------------------------------------------------------------------------

if _NB is not None:
    @_NB.njit(cache=True)
    def _popcnt64(x):  # uint64 SWAR popcount
        m1 = np.uint64(0x5555555555555555)
        m2 = np.uint64(0x3333333333333333)
        m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
        h  = np.uint64(0x0101010101010101)
        x = x - ((x >> np.uint64(1)) & m1)
        x = (x & m2) + ((x >> np.uint64(2)) & m2)
        x = (x + (x >> np.uint64(4))) & m4
        return (x * h) >> np.uint64(56)

    @_NB.njit(cache=True)
    def _bit(lo, hi, idx):
        if idx < 64:
            return (lo >> np.uint64(idx)) & np.uint64(1)
        return (hi >> np.uint64(idx - 64)) & np.uint64(1)

    @_NB.njit(cache=True)
    def _below(lo, hi, idx):
        """popcount of bits strictly below idx (the sign parity input)."""
        if idx == 0:
            return 0
        if idx <= 64:
            return np.int64(_popcnt64(lo & ((np.uint64(1) << np.uint64(idx)) - np.uint64(1))))
        return np.int64(_popcnt64(lo)) + np.int64(_popcnt64(
            hi & ((np.uint64(1) << np.uint64(idx - 64)) - np.uint64(1))))

    @_NB.njit(cache=True)
    def _clr(lo, hi, idx):
        if idx < 64:
            return lo & ~(np.uint64(1) << np.uint64(idx)), hi
        return lo, hi & ~(np.uint64(1) << np.uint64(idx - 64))

    @_NB.njit(cache=True)
    def _set(lo, hi, idx):
        if idx < 64:
            return lo | (np.uint64(1) << np.uint64(idx)), hi
        return lo, hi | (np.uint64(1) << np.uint64(idx - 64))

    @_NB.njit(cache=True)
    def _sign_1e(lo, hi, a, c):
        """Sign of a+_a a_c |det>, the _ops_sign convention."""
        sign = 1
        if _bit(lo, hi, c) == 0:
            return 0
        if _below(lo, hi, c) & 1:
            sign = -sign
        lo, hi = _clr(lo, hi, c)
        if _bit(lo, hi, a) != 0:
            return 0
        if _below(lo, hi, a) & 1:
            sign = -sign
        return sign

    @_NB.njit(cache=True)
    def _sign_2e(lo, hi, a, b, c, d):
        """Sign of a+_a a+_b a_d a_c |det> (right-to-left: c, d, b, a)."""
        sign = 1
        if _bit(lo, hi, c) == 0:
            return 0
        if _below(lo, hi, c) & 1:
            sign = -sign
        lo, hi = _clr(lo, hi, c)
        if _bit(lo, hi, d) == 0:
            return 0
        if _below(lo, hi, d) & 1:
            sign = -sign
        lo, hi = _clr(lo, hi, d)
        if _bit(lo, hi, b) != 0:
            return 0
        if _below(lo, hi, b) & 1:
            sign = -sign
        lo, hi = _set(lo, hi, b)
        if _bit(lo, hi, a) != 0:
            return 0
        if _below(lo, hi, a) & 1:
            sign = -sign
        return sign


    @_NB.njit(cache=True)
    def _find_row(shi, slo, srow, tlo, thi):
        """Binary search of (thi, tlo) in the lexsorted det table."""
        a = np.searchsorted(shi, thi)
        b = np.searchsorted(shi, thi, side='right')
        while a < b:
            mid = (a + b) // 2
            if slo[mid] < tlo:
                a = mid + 1
            else:
                b = mid
        if a < shi.shape[0] and shi[a] == thi and slo[a] == tlo:
            return srow[a]
        return -1


    @_NB.njit(cache=True, parallel=True)
    def _assemble_kernel_parallel(mlo, mhi, occ, nspin, m2arr,
                                  h1_keys, h1_vals,
                                  pgrp, pflat, pstart, plen,
                                  kgrp, kflat, kstart, klen,
                                  h1m_keys, h1m_vals,
                                  bflat, bstart, blen, msum_shift,
                                  vb_keys, vb_vals,
                                  shi, slo, srow,
                                  rows_b, cols_b, vals_b, diag_b, chunk):
        """Block-partitioned prange twin of _assemble_kernel.

        Determinant ``col`` belongs to block ``col // chunk`` and writes
        only into that block's COO slice and diag row — no cross-block
        state, so prange is race-free.
        """
        n = mlo.shape[0]
        nblocks = rows_b.shape[0]
        counts = np.empty(nblocks, dtype=np.int64)
        for blk in _NB.prange(nblocks):
            cap = rows_b.shape[1]
            count = 0
            col_start = blk * chunk
            col_end = min(col_start + chunk, n)
            for col in range(col_start, col_end):
                lo = mlo[col]
                hi = mhi[col]
                ne = occ.shape[1]
                for ei in range(ne):
                    c = occ[col, ei]
                    g = pgrp[c]
                    for pi in range(pstart[g], pstart[g] + plen[g]):
                        a = pflat[pi]
                        key = a * nspin + c
                        lo_i, hi_i = 0, h1_keys.shape[0]
                        while lo_i < hi_i:
                            mid = (lo_i + hi_i) // 2
                            if h1_keys[mid] < key:
                                lo_i = mid + 1
                            else:
                                hi_i = mid
                        if lo_i >= h1_keys.shape[0] or h1_keys[lo_i] != key:
                            continue
                        h = h1_vals[lo_i]
                        if a == c:
                            diag_b[blk, col] += h
                            continue
                        tlo, thi = _clr(lo, hi, c)
                        tlo, thi = _set(tlo, thi, a)
                        row = _find_row(shi, slo, srow, tlo, thi)
                        if row < 0:
                            continue
                        sg = _sign_1e(lo, hi, a, c)
                        if sg == 0:
                            continue
                        if count < cap:
                            rows_b[blk, count] = row
                            cols_b[blk, count] = col
                            vals_b[blk, count] = sg * h
                            count += 1
                # ---- one-body m-mixing (crystal field q != 0) ----
                if h1m_keys.shape[0] > 0:
                    for ei in range(ne):
                        c = occ[col, ei]
                        g = kgrp[c]
                        for pi in range(kstart[g], kstart[g] + klen[g]):
                            a = kflat[pi]
                            key = a * nspin + c
                            lo_i, hi_i = 0, h1m_keys.shape[0]
                            while lo_i < hi_i:
                                mid = (lo_i + hi_i) // 2
                                if h1m_keys[mid] < key:
                                    lo_i = mid + 1
                                else:
                                    hi_i = mid
                            if (lo_i >= h1m_keys.shape[0]
                                    or h1m_keys[lo_i] != key):
                                continue
                            if a == c:
                                diag_b[blk, col] += h1m_vals[lo_i]
                                continue
                            tlo, thi = _clr(lo, hi, c)
                            tlo, thi = _set(tlo, thi, a)
                            row = _find_row(shi, slo, srow, tlo, thi)
                            if row < 0:
                                continue
                            sg = _sign_1e(lo, hi, a, c)
                            if sg == 0:
                                continue
                            if count < cap:
                                rows_b[blk, count] = row
                                cols_b[blk, count] = col
                                vals_b[blk, count] = sg * h1m_vals[lo_i]
                                count += 1
                for ei in range(ne):
                    c = occ[col, ei]
                    for fj in range(ei + 1, ne):
                        d = occ[col, fj]
                        ms = m2arr[c] + m2arr[d] + msum_shift
                        for bi in range(bstart[ms], bstart[ms] + blen[ms]):
                            a = bflat[2 * bi]
                            b = bflat[2 * bi + 1]
                            tlo, thi = _clr(lo, hi, c)
                            tlo, thi = _clr(tlo, thi, d)
                            tlo, thi = _set(tlo, thi, a)
                            tlo, thi = _set(tlo, thi, b)
                            row = _find_row(shi, slo, srow, tlo, thi)
                            if row < 0:
                                continue
                            key = ((a * nspin + b) * nspin + c) * nspin + d
                            lo_i, hi_i = 0, vb_keys.shape[0]
                            while lo_i < hi_i:
                                mid = (lo_i + hi_i) // 2
                                if vb_keys[mid] < key:
                                    lo_i = mid + 1
                                else:
                                    hi_i = mid
                            if lo_i >= vb_keys.shape[0] or vb_keys[lo_i] != key:
                                continue
                            sg = _sign_2e(lo, hi, a, b, c, d)
                            if sg == 0:
                                continue
                            if count < cap:
                                rows_b[blk, count] = row
                                cols_b[blk, count] = col
                                vals_b[blk, count] = sg * vb_vals[lo_i]
                                count += 1
            counts[blk] = count
        return counts

    @_NB.njit(cache=True)
    def _assemble_kernel(mlo, mhi, occ, nspin, m2arr,
                         h1_keys, h1_vals,
                         pgrp, pflat, pstart, plen,
                         kgrp, kflat, kstart, klen,
                         h1m_keys, h1m_vals,
                         bflat, bstart, blen, msum_shift,
                         vb_keys, vb_vals,
                         shi, slo, srow,
                         rows, cols, vals, diag):
        """Emit the COO triplets of H over all determinants (col-major)."""
        n = mlo.shape[0]
        count = 0
        cap = rows.shape[0]
        for col in range(n):
            lo = mlo[col]
            hi = mhi[col]
            ne = occ.shape[1]
            # ---- one-body ----
            for ei in range(ne):
                c = occ[col, ei]
                g = pgrp[c]
                for pi in range(pstart[g], pstart[g] + plen[g]):
                    a = pflat[pi]
                    key = a * nspin + c
                    # binary search h1
                    lo_i, hi_i = 0, h1_keys.shape[0]
                    while lo_i < hi_i:
                        mid = (lo_i + hi_i) // 2
                        if h1_keys[mid] < key:
                            lo_i = mid + 1
                        else:
                            hi_i = mid
                    if lo_i >= h1_keys.shape[0] or h1_keys[lo_i] != key:
                        continue
                    h = h1_vals[lo_i]
                    if a == c:
                        diag[col] += h
                        continue
                    tlo, thi = _clr(lo, hi, c)
                    tlo, thi = _set(tlo, thi, a)
                    row = _find_row(shi, slo, srow, tlo, thi)
                    if row < 0:
                        continue
                    sg = _sign_1e(lo, hi, a, c)
                    if sg == 0:
                        continue
                    if count < cap:
                        rows[count] = row
                        cols[count] = col
                        vals[count] = sg * h
                        count += 1
            # ---- one-body m-mixing (crystal field q != 0) ----
            if h1m_keys.shape[0] > 0:
                for ei in range(ne):
                    c = occ[col, ei]
                    g = kgrp[c]
                    for pi in range(kstart[g], kstart[g] + klen[g]):
                        a = kflat[pi]
                        key = a * nspin + c
                        lo_i, hi_i = 0, h1m_keys.shape[0]
                        while lo_i < hi_i:
                            mid = (lo_i + hi_i) // 2
                            if h1m_keys[mid] < key:
                                lo_i = mid + 1
                            else:
                                hi_i = mid
                        if lo_i >= h1m_keys.shape[0] or h1m_keys[lo_i] != key:
                            continue
                        if a == c:
                            diag[col] += h1m_vals[lo_i]
                            continue
                        tlo, thi = _clr(lo, hi, c)
                        tlo, thi = _set(tlo, thi, a)
                        row = _find_row(shi, slo, srow, tlo, thi)
                        if row < 0:
                            continue
                        sg = _sign_1e(lo, hi, a, c)
                        if sg == 0:
                            continue
                        if count < cap:
                            rows[count] = row
                            cols[count] = col
                            vals[count] = sg * h1m_vals[lo_i]
                            count += 1
            # ---- two-body ----
            for ei in range(ne):
                c = occ[col, ei]
                for fj in range(ei + 1, ne):
                    d = occ[col, fj]
                    ms = m2arr[c] + m2arr[d] + msum_shift
                    for bi in range(bstart[ms], bstart[ms] + blen[ms]):
                        a = bflat[2 * bi]
                        b = bflat[2 * bi + 1]
                        tlo, thi = _clr(lo, hi, c)
                        tlo, thi = _clr(tlo, thi, d)
                        tlo, thi = _set(tlo, thi, a)
                        tlo, thi = _set(tlo, thi, b)
                        row = _find_row(shi, slo, srow, tlo, thi)
                        if row < 0:
                            continue
                        key = ((a * nspin + b) * nspin + c) * nspin + d
                        lo_i, hi_i = 0, vb_keys.shape[0]
                        while lo_i < hi_i:
                            mid = (lo_i + hi_i) // 2
                            if vb_keys[mid] < key:
                                lo_i = mid + 1
                            else:
                                hi_i = mid
                        if lo_i >= vb_keys.shape[0] or vb_keys[lo_i] != key:
                            continue
                        sg = _sign_2e(lo, hi, a, b, c, d)
                        if sg == 0:
                            continue
                        if count < cap:
                            rows[count] = row
                            cols[count] = col
                            vals[count] = sg * vb_vals[lo_i]
                            count += 1
        return count


def _assemble_numba(masks, spinors, orbitals_dict, bsp_data, Z, ns,
                    *, two_body_delta=None, one_body_delta=None,
                    one_body_m_delta=None, include_breit=False):
    """Phase 1 (Python, once per distinct value) + phase 2 (numba)."""
    from functools import lru_cache

    n = len(masks)
    if n == 0:
        return sp.csr_matrix((0, 0))
    nspin = len(spinors)
    nelec = bin(masks[0]).count("1")

    @lru_cache(maxsize=None)
    def _one_body(a_key, c_key):
        Pa, Qa = orbitals_dict[a_key]
        Pc, Qc = orbitals_dict[c_key]
        h = MAT._h_one_body(Pa, Qa, Pc, Qc, a_key[1], Z, bsp_data, ns)
        return h / MAT._norm_sq(Pc, Qc, c_key[1], Z, bsp_data, ns)

    @lru_cache(maxsize=None)
    def _v(ia, ib, ic, id_):
        a, b, c, d = spinors[ia], spinors[ib], spinors[ic], spinors[id_]
        terms = two_body_angular(a[1], a[2], b[1], b[2], c[1], c[2], d[1], d[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pb, Qb = orbitals_dict[(b[0], b[1])]
        Pd, Qd = orbitals_dict[(d[0], d[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd,
                                    bsp_data, k)
        return total

    # ---- phase 1a: one-body table over (kappa, m) partner pairs ----
    km_groups: Dict[Tuple[int, int], List[int]] = {}
    for si, (_n, kappa, m2) in enumerate(spinors):
        km_groups.setdefault((kappa, m2), []).append(si)
    h1 = {}
    pgrp = np.empty(nspin, dtype=np.int64)
    gid = {}
    pflat: List[int] = []
    pstart: List[int] = []
    plen: List[int] = []
    for g, members in km_groups.items():
        g_id = len(pstart)
        gid[g] = g_id
        pstart.append(len(pflat))
        for si in members:
            pgrp[si] = g_id
            pflat.append(si)
        plen.append(len(members))
        for c in members:
            for a in members:
                h1[a * nspin + c] = _one_body(
                    (spinors[a][0], spinors[a][1]), (spinors[c][0], spinors[c][1]))
                if one_body_delta is not None:
                    h1[a * nspin + c] += one_body_delta.get(
                        ((spinors[a][0], spinors[a][1]),
                         (spinors[c][0], spinors[c][1])), 0.0)
    h1_keys = np.array(sorted(h1), dtype=np.int64)
    h1_vals = np.array([h1[k] for k in h1_keys], dtype=np.float64)

    # ---- phase 1a': m-mixing one-body table over kappa groups ----
    # (crystal-field q != 0 terms; empty tables => zero-cost when unused)
    k_groups: Dict[int, List[int]] = {}
    for si, (_n, kappa, _m) in enumerate(spinors):
        k_groups.setdefault(kappa, []).append(si)
    h1m: Dict[int, float] = {}
    if one_body_m_delta is not None:
        for members in k_groups.values():
            for c in members:
                for a in members:
                    v = one_body_m_delta(a, c)
                    if v != 0.0:
                        h1m[a * nspin + c] = v
    h1m_keys = np.array(sorted(h1m), dtype=np.int64)
    h1m_vals = np.array([h1m[k] for k in h1m_keys], dtype=np.float64)
    kgrp = np.zeros(nspin, dtype=np.int64)
    kflat_a: List[int] = []
    kstart_a: List[int] = []
    klen_a: List[int] = []
    for g, members in k_groups.items():
        kstart_a.append(len(kflat_a))
        kflat_a.extend(members)
        klen_a.append(len(members))
        for si in members:
            kgrp[si] = len(kstart_a) - 1
    kflat = np.array(kflat_a, dtype=np.int64)
    kstart = np.array(kstart_a, dtype=np.int64)
    klen = np.array(klen_a, dtype=np.int64)

    # ---- phase 1b: two-body value table over m-matched quadruples ----
    # A spin-orbital set in every determinant mask is frozen (closed core):
    # every target determinant must still occupy it, so removing {c, d}
    # and adding {a, b} must restore each frozen member — a frozen c (or
    # d) has to reappear as a or b (spectator terms count).  Quadruples
    # violating this can never connect two determinants of the space.
    frozen = set(range(nspin))
    for mask in masks:
        frozen &= {si for si in range(nspin) if (mask >> si) & 1}
    open_spinors = frozenset(set(range(nspin)) - frozen)

    m2arr_py = [s_[2] for s_ in spinors]
    m2min = min(m2arr_py)
    msum_shift = -2 * m2min  # bucket index >= 0
    nbuckets = 2 * (max(m2arr_py) - m2min) + 1
    buckets: Dict[int, list] = {}
    for ia, ib in itertools.combinations(range(nspin), 2):
        buckets.setdefault(m2arr_py[ia] + m2arr_py[ib] + msum_shift,
                           []).append((ia, ib))

    def _key(a, b, c, d):
        return ((a * nspin + b) * nspin + c) * nspin + d

    # Candidate (a, b) lists classified by the frozen/open status of the
    # annihilated pair: when (c, d) are both open, the created pair may be
    # any pair of OPEN spinors in the same m bucket (creations onto frozen
    # slots are Pauli-blocked); when c or d is frozen, the created pair
    # must restore it, so only pairs containing that spinor are visited.
    open_pairs: Dict[int, list] = {}
    for ms, ab_list in buckets.items():
        open_pairs[ms] = [(a, b) for (a, b) in ab_list
                          if a in open_spinors and b in open_spinors]
    pairs_with: Dict[Tuple[int, int], list] = {}
    for ms, ab_list in buckets.items():
        for (a, b) in ab_list:
            for si in (a, b):
                pairs_with.setdefault((si, ms), []).append((a, b))

    breit_evaluate = (_breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns)
                      if include_breit else None)
    if breit_evaluate is not None:
        def _breit_element(ia, ib, ic, id_):
            total = 0.0
            for key, coeff in breit_terms(ia, ib, ic, id_, 1.0, spinors):
                total += coeff * breit_evaluate(key)
            return total
    else:
        def _breit_element(ia, ib, ic, id_):
            return 0.0

    vb: Dict[int, float] = {}
    for ms, ab_list in buckets.items():
        cand_open = open_pairs[ms]
        for (c, d) in ab_list:
            c_open = c in open_spinors
            d_open = d in open_spinors
            if c_open and d_open:
                ab_cands = cand_open
            elif c_open:
                ab_cands = pairs_with.get((d, ms), [])
            elif d_open:
                ab_cands = pairs_with.get((c, ms), [])
            else:
                ab_cands = [(a, b) for (a, b) in pairs_with.get((c, ms), [])
                            if a == d or b == d]
            for (a, b) in ab_cands:
                key = _key(a, b, c, d)
                if key in vb:
                    continue
                v = _v(a, b, c, d)
                v2 = _v(a, b, d, c)
                anti = v - v2
                if anti == 0.0 and two_body_delta is None and not include_breit:
                    continue
                if include_breit:
                    anti += (_breit_element(a, b, c, d)
                             - _breit_element(a, b, d, c))
                if two_body_delta is not None:
                    anti += two_body_delta(a, b, c, d)
                if anti == 0.0:
                    continue
                vb[key] = anti

    vb_keys = np.array(sorted(vb), dtype=np.int64)
    vb_vals = np.array([vb[k] for k in vb_keys], dtype=np.float64)

    bflat: List[int] = []
    bstart = [0] * nbuckets
    blen = [0] * nbuckets
    for ms in range(nbuckets):
        pairs = buckets.get(ms, [])
        bstart[ms] = len(bflat) // 2
        for (a, b) in pairs:
            bflat.append(a)
            bflat.append(b)
        blen[ms] = len(pairs)

    # ---- determinant structures ----
    occ = np.empty((n, nelec), dtype=np.int64)
    mlo = np.empty(n, dtype=np.uint64)
    mhi = np.empty(n, dtype=np.uint64)
    for i, mask in enumerate(masks):
        bits = [si for si in range(nspin) if (mask >> si) & 1]
        for e, si in enumerate(bits):
            occ[i, e] = si
        mlo[i] = np.uint64(mask & ((1 << 64) - 1))
        mhi[i] = np.uint64(mask >> 64)
    order = np.lexsort((mlo, mhi))
    shi = mhi[order]
    slo = mlo[order]
    srow = np.asarray(order, dtype=np.int64)

    m2arr = np.array(m2arr_py, dtype=np.int64)
    pflat_a = np.array(pflat, dtype=np.int64)
    pstart_a = np.array(pstart, dtype=np.int64)
    plen_a = np.array(plen, dtype=np.int64)
    bflat_a = np.array(bflat, dtype=np.int64)
    bstart_a = np.array(bstart, dtype=np.int64)
    blen_a = np.array(blen, dtype=np.int64)

    use_parallel = n >= 2000 and _assemble_kernel_parallel is not None
    if use_parallel:
        nblocks = min(n, max(1, _NB.get_num_threads() * 4 if hasattr(_NB, "get_num_threads") else 16))
        chunk = (n + nblocks - 1) // nblocks
        nblocks = (n + chunk - 1) // chunk
        cap_per_det = 768
        while True:
            capb = max(1, chunk * cap_per_det)
            rows_b = np.zeros((nblocks, capb), dtype=np.int64)
            cols_b = np.zeros((nblocks, capb), dtype=np.int64)
            vals_b = np.zeros((nblocks, capb), dtype=np.float64)
            diag_b = np.zeros((nblocks, n))
            counts = _assemble_kernel_parallel(
                mlo, mhi, occ, nspin, m2arr,
                h1_keys, h1_vals,
                pgrp, pflat_a, pstart_a, plen_a,
                kgrp, kflat, kstart, klen,
                h1m_keys, h1m_vals,
                bflat_a, bstart_a, blen_a, msum_shift,
                vb_keys, vb_vals,
                shi, slo, srow,
                rows_b, cols_b, vals_b, diag_b, chunk)
            if int(counts.max()) < capb:
                break
            cap_per_det *= 8
        keep = int(counts.sum())
        rows = np.empty(keep, dtype=np.int64)
        cols = np.empty(keep, dtype=np.int64)
        vals = np.empty(keep, dtype=np.float64)
        pos = 0
        for blk in range(nblocks):
            c = int(counts[blk])
            rows[pos:pos + c] = rows_b[blk, :c]
            cols[pos:pos + c] = cols_b[blk, :c]
            vals[pos:pos + c] = vals_b[blk, :c]
            pos += c
        diag = diag_b.sum(axis=0)
    else:
        cap_per_det = 2048
        while True:
            cap = n * cap_per_det
            rows = np.empty(cap, dtype=np.int64)
            cols = np.empty(cap, dtype=np.int64)
            vals = np.empty(cap, dtype=np.float64)
            diag = np.zeros(n)
            count = _assemble_kernel(
                mlo, mhi, occ, nspin, m2arr,
                h1_keys, h1_vals,
                pgrp, pflat_a, pstart_a, plen_a,
                kgrp, kflat, kstart, klen,
                h1m_keys, h1m_vals,
                bflat_a, bstart_a, blen_a, msum_shift,
                vb_keys, vb_vals,
                shi, slo, srow,
                rows, cols, vals, diag)
            if count < cap:
                rows, cols, vals = rows[:count], cols[:count], vals[:count]
                break
            cap_per_det *= 8  # rare: dense row — regrow
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    H = H + sp.diags(diag)
    H.sum_duplicates()
    H.sort_indices()
    return H
