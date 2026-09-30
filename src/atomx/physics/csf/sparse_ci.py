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
from scipy.sparse.linalg import eigsh

from atomx.angular.csf_hamiltonian import _m2_values, _ops_sign, two_body_angular, twoj
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


def _spinor_table(orbitals_dict) -> List[Tuple[int, int, int]]:
    """Global spin-orbital list: (n, kappa, 2m), sorted for stable masks."""
    out = []
    for (n, kappa) in sorted(orbitals_dict):
        for m2 in _m2_values(twoj(kappa)):
            out.append((n, kappa, m2))
    return out


def enumerate_determinants(patterns, orbitals_dict, m2_total: int = 0):
    """Occupation bitmasks of the pattern set at fixed ``2M``.

    A determinant is a bitmask over the global spin-orbital table; two
    determinants are the same physical state iff their masks are equal
    (each determinant belongs to exactly one occupation pattern).
    """
    spinors = _spinor_table(orbitals_dict)
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


def assemble_sparse_hamiltonian(masks, spinors, orbitals_dict, bsp_data, Z, ns):
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

    # add-pair buckets by m-sum, for the two-body loop
    pair_buckets: Dict[int, List[Tuple[int, int]]] = {}
    for ia, ib in itertools.combinations(range(len(spinors)), 2):
        pair_buckets.setdefault(spinors[ia][2] + spinors[ib][2], []).append((ia, ib))

    # one-body partners sharing (kappa, m)
    km_partners: Dict[Tuple[int, int], List[int]] = {}
    for si, (_n, kappa, m2) in enumerate(spinors):
        km_partners.setdefault((kappa, m2), []).append(si)

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
                if v == 0.0 and v2 == 0.0:
                    continue
                sign = _ops_sign(mask, [(a, True), (b, True),
                                         (d, False), (c, False)])
                if not sign:
                    continue
                nmask = (mask & ~(1 << c) & ~(1 << d)) | (1 << a) | (1 << b)
                row = index.get(nmask)
                if row is not None:
                    emit(row, col, sign * (v - v2))
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    H = H + sp.diags(diag)
    H.sum_duplicates()
    return H


def sparse_ci_spectrum(orbitals_dict, bsp_data, Z, ns, patterns,
                       m2_total: int = 0, n_states: int = 10):
    """Lowest ``n_states`` energies of one ``2M`` determinant sector.

    Returns ``(eigenvalues, n_determinants)`` with eigenvalues ascending.
    """
    masks, spinors = enumerate_determinants(patterns, orbitals_dict, m2_total)
    H = assemble_sparse_hamiltonian(masks, spinors, orbitals_dict, bsp_data, Z, ns)
    n = len(masks)
    if n_states >= n - 1:
        w = np.linalg.eigvalsh(H.toarray())
        return w[:max(0, n_states)], n
    w = eigsh(H, k=n_states, which="SA", return_eigenvectors=False)
    return np.sort(w), n
