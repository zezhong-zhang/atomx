"""Exact m-scheme helpers for relativistic subshells.

The FAC RCFP implementation uses tabulated/recursive reduced coefficients for
``j^q`` subshells.  This module provides an independent, small-shell exact
engine that constructs antisymmetrized determinant bases and extracts
highest-weight ``J`` states numerically.  It is intended as the validation and
fallback backend for the pure-Python RCFP layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import sqrt

import numpy as np
from scipy.linalg import null_space

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.wigner import wigner_3j


@dataclass(frozen=True)
class SubshellTermState:
    """A highest-weight state of a ``j^q`` subshell."""

    j2: int
    q: int
    J2: int
    index: int
    determinants: tuple[int, ...]
    coefficients: tuple[float, ...]


def _m2_values(j2: int) -> tuple[int, ...]:
    return tuple(range(-j2, j2 + 1, 2))


def _popcount(mask: int) -> int:
    return int(mask.bit_count())


@lru_cache(maxsize=512)
def determinants_by_m(j2: int, q: int, M2: int) -> tuple[int, ...]:
    """Return determinant bitmasks for ``j^q`` with total ``2M == M2``."""
    m2 = _m2_values(j2)
    dets = []
    for occ in combinations(range(len(m2)), q):
        if sum(m2[i] for i in occ) == M2:
            mask = 0
            for i in occ:
                mask |= 1 << i
            dets.append(mask)
    return tuple(dets)


def _occupation_sign(mask: int, orbital: int) -> int:
    """Fermion sign ``(-1)**n`` with ``n`` occupied bits below ``orbital``.

    The single bitmask-sign convention shared by every determinant engine
    (this module's ``_annihilate``/``_create``,
    :mod:`atomx.angular.csf_hamiltonian`, and
    :mod:`atomx.scattering.autoionization.mscheme_ci`).  Evaluated on the
    mask *before* the toggle, it is the same for creation and annihilation.
    """
    return -1 if (_popcount(mask & ((1 << orbital) - 1)) % 2) else 1


def _annihilate(mask: int, orbital: int) -> tuple[int, int] | None:
    if not (mask & (1 << orbital)):
        return None
    return _occupation_sign(mask, orbital), mask ^ (1 << orbital)


def _create(mask: int, orbital: int) -> tuple[int, int] | None:
    if mask & (1 << orbital):
        return None
    return _occupation_sign(mask, orbital), mask | (1 << orbital)


@lru_cache(maxsize=128)
def _one_body_links(bra, ket, pairs):
    """Determinant incidence and fermion signs, independent of radial values."""
    index = {det: i for i, det in enumerate(bra)}
    partners = {}
    for pair_index, (a, b) in enumerate(pairs):
        partners.setdefault(b, []).append((a, pair_index))
    rows, columns, entries, signs = [], [], [], []
    for column, det in enumerate(ket):
        for b, destinations in partners.items():
            removed = _annihilate(det, b)
            if removed is None:
                continue
            sign_b, intermediate = removed
            for a, entry in destinations:
                created = _create(intermediate, a)
                if created is None:
                    continue
                sign_a, final = created
                row = index.get(final)
                if row is not None:
                    rows.append(row)
                    columns.append(column)
                    entries.append(entry)
                    signs.append(sign_a * sign_b)
    arrays = tuple(np.asarray(x, dtype=int) for x in (rows, columns, entries, signs))
    for array in arrays:
        array.setflags(write=False)
    return arrays


def one_body_action(dets_bra, dets_ket, weights, elements):
    """Project ``sum_ab t_ab a†_a a_b |ket>`` into a determinant basis.

    ``elements[(a,b)]`` holds orbital matrix elements in a common spin-orbital
    ordering. All amplitudes are retained, including weak interfering terms.
    The cached incidence table is independent of the operator's frequency.
    """
    weights = np.asarray(weights)
    if weights.shape != (len(dets_ket),):
        raise ValueError("one-body action requires one amplitude per ket determinant")
    values = np.asarray(list(elements.values()))
    rows, columns, entries, signs = _one_body_links(
        tuple(dets_bra), tuple(dets_ket), tuple(elements)
    )
    result = np.zeros(len(dets_bra), dtype=np.result_type(weights, values, float))
    np.add.at(result, rows, signs * values[entries] * weights[columns])
    return result


def _jplus_single_coeff(j2: int, m2: int) -> float:
    """Coefficient for ``j_+ |j,m>`` in twice-integer notation."""
    return 0.5 * sqrt((j2 - m2) * (j2 + m2 + 2))


def _jplus_apply(j2: int, mask: int) -> dict[int, float]:
    m2 = _m2_values(j2)
    out: dict[int, float] = {}
    for i, m in enumerate(m2[:-1]):
        if not (mask & (1 << i)):
            continue
        j = i + 1
        if mask & (1 << j):
            continue
        ann = _annihilate(mask, i)
        if ann is None:
            continue
        phase_a, tmp = ann
        cre = _create(tmp, j)
        if cre is None:
            continue
        phase_c, new_mask = cre
        out[new_mask] = out.get(
            new_mask, 0.0
        ) + phase_a * phase_c * _jplus_single_coeff(j2, m)
    return out


def jplus_matrix(j2: int, q: int, M2: int) -> np.ndarray:
    """Matrix of total ``J_+`` from M to M+1 subspaces."""
    ket = determinants_by_m(j2, q, M2)
    bra = determinants_by_m(j2, q, M2 + 2)
    bra_index = {mask: i for i, mask in enumerate(bra)}
    mat = np.zeros((len(bra), len(ket)))
    for col, mask in enumerate(ket):
        for new_mask, coeff in _jplus_apply(j2, mask).items():
            row = bra_index.get(new_mask)
            if row is not None:
                mat[row, col] += coeff
    return mat


@lru_cache(maxsize=512)
def highest_weight_states(j2: int, q: int) -> tuple[SubshellTermState, ...]:
    """Return all highest-weight J states for a relativistic subshell."""
    if q < 0 or q > j2 + 1:
        return tuple()
    states = []
    for J2 in range(0, q * j2 + 1):
        if (J2 + q * j2) % 2:
            continue
        dets = determinants_by_m(j2, q, J2)
        if not dets:
            continue
        jp = jplus_matrix(j2, q, J2)
        ns = null_space(jp, rcond=1e-12)
        if ns.size == 0:
            continue
        # Remove highest-weight vectors that are generated by lowering higher-J
        # states.  The null-space condition alone is already exact for highest
        # weights in the fixed-M=J subspace.
        for i in range(ns.shape[1]):
            vec = ns[:, i]
            norm = np.linalg.norm(vec)
            if norm <= 1e-12:
                continue
            vec = vec / norm
            for coeff in vec:
                if abs(coeff) > 1e-12:
                    if coeff < 0:
                        vec = -vec
                    break
            states.append(
                SubshellTermState(
                    j2=j2,
                    q=q,
                    J2=J2,
                    index=sum(1 for s in states if s.J2 == J2),
                    determinants=dets,
                    coefficients=tuple(float(x) for x in vec),
                )
            )
    return tuple(states)


def subshell_j_counts_mscheme(j2: int, q: int) -> dict[int, int]:
    """Exact ``{2J: multiplicity}`` for ``j^q`` from highest-weight states."""
    counts: dict[int, int] = {}
    for state in highest_weight_states(j2, q):
        counts[state.J2] = counts.get(state.J2, 0) + 1
    return dict(sorted(counts.items()))


def _state_vector_map(state: SubshellTermState) -> dict[int, float]:
    return {
        det: coeff
        for det, coeff in zip(state.determinants, state.coefficients, strict=False)
        if abs(coeff) > 1e-14
    }


def creation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, m2: int
) -> float:
    """Return ``<bra|a†_m|ket>`` between highest-weight subshell states."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 1:
        return 0.0
    if ket.J2 + m2 != bra.J2:
        return 0.0
    m_values = _m2_values(bra.j2)
    try:
        orbital = m_values.index(m2)
    except ValueError:
        return 0.0
    bra_vec = _state_vector_map(bra)
    total = 0.0
    for det, coeff_ket in _state_vector_map(ket).items():
        cre = _create(det, orbital)
        if cre is None:
            continue
        phase, new_det = cre
        coeff_bra = bra_vec.get(new_det, 0.0)
        total += coeff_bra * phase * coeff_ket
    return float(total)


def reduced_creation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState
) -> float:
    """Reduced matrix element ``<bra||a†||ket>``.

    Uses the Wigner-Eckart convention

    ``<Jb Mb|T^j_m|Jk Mk> = (-1)^(Jb-Mb)
       (Jb j Jk; -Mb m Mk) <Jb||T^j||Jk>``.

    Highest-weight states have ``Mb=Jb`` and ``Mk=Jk``.
    """
    if bra.j2 != ket.j2 or bra.q != ket.q + 1:
        return 0.0
    m2 = bra.J2 - ket.J2
    actual = creation_matrix_element(bra, ket, m2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, bra.j2, ket.J2, -bra.J2, m2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def coupled_one_body_tensor_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, Q2: int
) -> float:
    """Return ``<bra|[a† x atilde]^K_Q|ket>`` for one ``j^q`` subshell.

    The time-reversed annihilation operator is
    ``atilde_m = (-1)^(j-m) a_{-m}``.  This is the same elementary coupled
    operator used in RCFP ``W``-type reduced coefficients.
    """
    if bra.j2 != ket.j2 or bra.q != ket.q:
        return 0.0
    if bra.J2 != ket.J2 + Q2:
        return 0.0
    j2 = bra.j2
    m_values = _m2_values(j2)
    bra_vec = _state_vector_map(bra)
    total = 0.0
    for det, coeff_ket in _state_vector_map(ket).items():
        for m_create in m_values:
            m_tilde = Q2 - m_create
            if m_tilde not in m_values:
                continue
            cg = clebsch_gordan(j2, m_create, j2, m_tilde, K2, Q2)
            if abs(cg) < 1e-14:
                continue
            remove_m = -m_tilde
            remove_orbital = m_values.index(remove_m)
            create_orbital = m_values.index(m_create)
            ann = _annihilate(det, remove_orbital)
            if ann is None:
                continue
            phase_a, tmp = ann
            cre = _create(tmp, create_orbital)
            if cre is None:
                continue
            phase_c, new_det = cre
            tilde_phase_exp = (j2 - m_tilde) // 2
            tilde_phase = -1.0 if tilde_phase_exp % 2 else 1.0
            total += (
                bra_vec.get(new_det, 0.0)
                * phase_a
                * phase_c
                * tilde_phase
                * cg
                * coeff_ket
            )
    return float(total)


def reduced_one_body_tensor_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int
) -> float:
    """Reduced ``<bra||[a† x atilde]^K||ket>`` matrix element."""
    if bra.j2 != ket.j2 or bra.q != ket.q:
        return 0.0
    Q2 = bra.J2 - ket.J2
    if abs(Q2) > K2:
        return 0.0
    actual = coupled_one_body_tensor_matrix_element(bra, ket, K2, Q2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, K2, ket.J2, -bra.J2, Q2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def _one_body_tensor_apply(
    j2: int, q: int, vector: dict[int, float], K2: int, Q2: int
) -> dict[int, float]:
    """Apply ``[a† x atilde]^K_Q`` to a determinant-space vector."""
    m_values = _m2_values(j2)
    out: dict[int, float] = {}
    for det, amplitude in vector.items():
        if abs(amplitude) < 1e-14:
            continue
        if _popcount(det) != q:
            continue
        for m_create in m_values:
            m_tilde = Q2 - m_create
            if m_tilde not in m_values:
                continue
            cg = clebsch_gordan(j2, m_create, j2, m_tilde, K2, Q2)
            if abs(cg) < 1e-14:
                continue
            remove_m = -m_tilde
            remove_orbital = m_values.index(remove_m)
            create_orbital = m_values.index(m_create)
            ann = _annihilate(det, remove_orbital)
            if ann is None:
                continue
            phase_a, tmp = ann
            cre = _create(tmp, create_orbital)
            if cre is None:
                continue
            phase_c, new_det = cre
            tilde_phase_exp = (j2 - m_tilde) // 2
            tilde_phase = -1.0 if tilde_phase_exp % 2 else 1.0
            out[new_det] = out.get(new_det, 0.0) + (
                amplitude * phase_a * phase_c * tilde_phase * cg
            )
    return out


def _creation_apply(
    j2: int, q: int, vector: dict[int, float], m2: int
) -> dict[int, float]:
    """Apply ``a†_m`` to a determinant-space vector."""
    m_values = _m2_values(j2)
    if m2 not in m_values:
        return {}
    orbital = m_values.index(m2)
    out: dict[int, float] = {}
    for det, amplitude in vector.items():
        if abs(amplitude) < 1e-14 or _popcount(det) != q:
            continue
        cre = _create(det, orbital)
        if cre is None:
            continue
        phase, new_det = cre
        out[new_det] = out.get(new_det, 0.0) + phase * amplitude
    return out


def _annihilation_apply(
    j2: int, q: int, vector: dict[int, float], m2: int
) -> dict[int, float]:
    """Apply ``a_m`` to a determinant-space vector."""
    m_values = _m2_values(j2)
    if m2 not in m_values:
        return {}
    orbital = m_values.index(m2)
    out: dict[int, float] = {}
    for det, amplitude in vector.items():
        if abs(amplitude) < 1e-14 or _popcount(det) != q:
            continue
        ann = _annihilate(det, orbital)
        if ann is None:
            continue
        phase, new_det = ann
        out[new_det] = out.get(new_det, 0.0) + phase * amplitude
    return out


def _inner_product(left: dict[int, float], right: dict[int, float]) -> float:
    return float(sum(left.get(det, 0.0) * amp for det, amp in right.items()))


def coupled_wxw_scalar_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int
) -> float:
    """Return ``<bra|[W^K x W^K]^0_0|ket>`` for one ``j^q`` subshell."""
    if bra.j2 != ket.j2 or bra.q != ket.q or bra.J2 != ket.J2:
        return 0.0
    j2 = bra.j2
    q = bra.q
    ket_vec = _state_vector_map(ket)
    accum: dict[int, float] = {}
    for Q2 in range(-K2, K2 + 1, 2):
        cg = clebsch_gordan(K2, Q2, K2, -Q2, 0, 0)
        if abs(cg) < 1e-14:
            continue
        first = _one_body_tensor_apply(j2, q, ket_vec, K2, -Q2)
        second = _one_body_tensor_apply(j2, q, first, K2, Q2)
        for det, amplitude in second.items():
            accum[det] = accum.get(det, 0.0) + cg * amplitude
    bra_vec = _state_vector_map(bra)
    return float(sum(bra_vec.get(det, 0.0) * amp for det, amp in accum.items()))


def coupled_axw_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, R2: int, Q2: int
) -> float:
    """Return ``<bra|[A^j x W^K]^R_Q|ket>`` for one subshell.

    ``A`` is the electron creation operator.  The result changes occupation
    from ``q`` to ``q+1``.
    """
    if bra.j2 != ket.j2 or bra.q != ket.q + 1 or bra.J2 != ket.J2 + Q2:
        return 0.0
    j2 = bra.j2
    ket_vec = _state_vector_map(ket)
    accum: dict[int, float] = {}
    for m2 in _m2_values(j2):
        for q_w2 in range(-K2, K2 + 1, 2):
            if m2 + q_w2 != Q2:
                continue
            cg = clebsch_gordan(j2, m2, K2, q_w2, R2, Q2)
            if abs(cg) < 1e-14:
                continue
            first = _one_body_tensor_apply(j2, ket.q, ket_vec, K2, q_w2)
            second = _creation_apply(j2, ket.q, first, m2)
            for det, amplitude in second.items():
                accum[det] = accum.get(det, 0.0) + cg * amplitude
    return _inner_product(_state_vector_map(bra), accum)


def reduced_axw_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, R2: int
) -> float:
    """Reduced ``<bra||[A^j x W^K]^R||ket>`` matrix element."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 1:
        return 0.0
    Q2 = bra.J2 - ket.J2
    if abs(Q2) > R2:
        return 0.0
    actual = coupled_axw_matrix_element(bra, ket, K2, R2, Q2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, R2, ket.J2, -bra.J2, Q2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def coupled_wxa_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, R2: int, Q2: int
) -> float:
    """Return ``<bra|[W^K x A^j]^R_Q|ket>`` for one subshell."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 1 or bra.J2 != ket.J2 + Q2:
        return 0.0
    j2 = bra.j2
    ket_vec = _state_vector_map(ket)
    accum: dict[int, float] = {}
    for q_w2 in range(-K2, K2 + 1, 2):
        for m2 in _m2_values(j2):
            if q_w2 + m2 != Q2:
                continue
            cg = clebsch_gordan(K2, q_w2, j2, m2, R2, Q2)
            if abs(cg) < 1e-14:
                continue
            first = _creation_apply(j2, ket.q, ket_vec, m2)
            second = _one_body_tensor_apply(j2, ket.q + 1, first, K2, q_w2)
            for det, amplitude in second.items():
                accum[det] = accum.get(det, 0.0) + cg * amplitude
    return _inner_product(_state_vector_map(bra), accum)


def reduced_wxa_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, R2: int
) -> float:
    """Reduced ``<bra||[W^K x A^j]^R||ket>`` matrix element."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 1:
        return 0.0
    Q2 = bra.J2 - ket.J2
    if abs(Q2) > R2:
        return 0.0
    actual = coupled_wxa_matrix_element(bra, ket, K2, R2, Q2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, R2, ket.J2, -bra.J2, Q2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def coupled_pair_creation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, Q2: int
) -> float:
    """Return ``<bra|[A^j x A^j]^K_Q|ket>`` for one subshell."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 2 or bra.J2 != ket.J2 + Q2:
        return 0.0
    j2 = bra.j2
    ket_vec = _state_vector_map(ket)
    accum: dict[int, float] = {}
    for m1 in _m2_values(j2):
        for m2 in _m2_values(j2):
            if m1 + m2 != Q2:
                continue
            cg = clebsch_gordan(j2, m1, j2, m2, K2, Q2)
            if abs(cg) < 1e-14:
                continue
            first = _creation_apply(j2, ket.q, ket_vec, m2)
            second = _creation_apply(j2, ket.q + 1, first, m1)
            for det, amplitude in second.items():
                accum[det] = accum.get(det, 0.0) + cg * amplitude
    return _inner_product(_state_vector_map(bra), accum)


def reduced_pair_creation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int
) -> float:
    """Reduced ``<bra||[A^j x A^j]^K||ket>`` matrix element."""
    if bra.j2 != ket.j2 or bra.q != ket.q + 2:
        return 0.0
    Q2 = bra.J2 - ket.J2
    if abs(Q2) > K2:
        return 0.0
    actual = coupled_pair_creation_matrix_element(bra, ket, K2, Q2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, K2, ket.J2, -bra.J2, Q2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)


def coupled_pair_annihilation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int, Q2: int
) -> float:
    """Return ``<bra|[a x a]^K_Q|ket>`` for one subshell."""
    if bra.j2 != ket.j2 or bra.q + 2 != ket.q or bra.J2 != ket.J2 + Q2:
        return 0.0
    j2 = bra.j2
    ket_vec = _state_vector_map(ket)
    accum: dict[int, float] = {}
    for m1 in _m2_values(j2):
        for m2 in _m2_values(j2):
            if m1 + m2 != Q2:
                continue
            cg = clebsch_gordan(j2, m1, j2, m2, K2, Q2)
            if abs(cg) < 1e-14:
                continue
            first = _annihilation_apply(j2, ket.q, ket_vec, m2)
            second = _annihilation_apply(j2, ket.q - 1, first, m1)
            for det, amplitude in second.items():
                accum[det] = accum.get(det, 0.0) + cg * amplitude
    return _inner_product(_state_vector_map(bra), accum)


def reduced_pair_annihilation_matrix_element(
    bra: SubshellTermState, ket: SubshellTermState, K2: int
) -> float:
    """Reduced ``<bra||[a x a]^K||ket>`` matrix element."""
    if bra.j2 != ket.j2 or bra.q + 2 != ket.q:
        return 0.0
    Q2 = bra.J2 - ket.J2
    if abs(Q2) > K2:
        return 0.0
    actual = coupled_pair_annihilation_matrix_element(bra, ket, K2, Q2)
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(bra.J2, K2, ket.J2, -bra.J2, Q2, ket.J2)
    if abs(w3j) < 1e-14:
        return 0.0
    return float(actual / w3j)
