"""Reduced subshell coefficients backed by the exact m-scheme engine.

This is the public AtomX-facing layer for FAC-style reduced coefficients of
fractional parentage.  The implementation currently uses exact determinant
states from :mod:`atomx.angular.mscheme`; later optimized table backends can
slot in behind the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from atomx.angular.mscheme import (
    SubshellTermState,
    coupled_wxw_scalar_matrix_element,
    highest_weight_states,
    reduced_axw_matrix_element,
    reduced_creation_matrix_element,
    reduced_one_body_tensor_matrix_element,
    reduced_pair_annihilation_matrix_element,
    reduced_pair_creation_matrix_element,
    reduced_wxa_matrix_element,
)


@dataclass(frozen=True)
class ReducedSubshellState:
    """Compact label for a ``j^q`` reduced subshell state."""

    j2: int
    q: int
    J2: int
    index: int = 0


def _to_reduced_state(state: SubshellTermState) -> ReducedSubshellState:
    return ReducedSubshellState(
        j2=state.j2,
        q=state.q,
        J2=state.J2,
        index=state.index,
    )


@lru_cache(maxsize=None)
def reduced_subshell_states(j2: int, q: int) -> tuple[ReducedSubshellState, ...]:
    """Return all reduced states for a relativistic ``j^q`` subshell."""
    return tuple(_to_reduced_state(s) for s in highest_weight_states(j2, q))


def _resolve_state(state: ReducedSubshellState) -> SubshellTermState | None:
    for candidate in highest_weight_states(state.j2, state.q):
        if candidate.J2 == state.J2 and candidate.index == state.index:
            return candidate
    return None


def reduced_creation_coeff(bra: ReducedSubshellState,
                           ket: ReducedSubshellState) -> float:
    """Return ``<bra||a†||ket>`` for one equivalent-electron subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_creation_matrix_element(bra_state, ket_state)


def reduced_annihilation_coeff(bra: ReducedSubshellState,
                               ket: ReducedSubshellState) -> float:
    """Return ``<bra||a||ket>`` via Hermitian conjugation convention.

    This keeps the API explicit for code that removes an electron.  The phase
    convention follows the determinant backend and will be locked down against
    FAC RCFP values as the table benchmarks are added.
    """
    if bra.j2 != ket.j2 or bra.q + 1 != ket.q:
        return 0.0
    return reduced_creation_coeff(ket, bra)


def reduced_w_coeff(bra: ReducedSubshellState,
                    ket: ReducedSubshellState,
                    K2: int) -> float:
    """Return ``<bra||[a† x atilde]^K||ket>`` for one equivalent subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_one_body_tensor_matrix_element(bra_state, ket_state, K2)


def reduced_wxw0_coeff(bra: ReducedSubshellState,
                       ket: ReducedSubshellState,
                       K2: int) -> float:
    """Return scalar ``<bra|[W^K x W^K]^0_0|ket>`` for one subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return coupled_wxw_scalar_matrix_element(bra_state, ket_state, K2)


def reduced_axw_coeff(bra: ReducedSubshellState,
                      ket: ReducedSubshellState,
                      K2: int, R2: int) -> float:
    """Return ``<bra||[A^j x W^K]^R||ket>`` for one subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_axw_matrix_element(bra_state, ket_state, K2, R2)


def reduced_wxa_coeff(bra: ReducedSubshellState,
                      ket: ReducedSubshellState,
                      K2: int, R2: int) -> float:
    """Return ``<bra||[W^K x A^j]^R||ket>`` for one subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_wxa_matrix_element(bra_state, ket_state, K2, R2)


def reduced_pair_creation_coeff(bra: ReducedSubshellState,
                                ket: ReducedSubshellState,
                                K2: int) -> float:
    """Return ``<bra||[A^j x A^j]^K||ket>`` for one subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_pair_creation_matrix_element(bra_state, ket_state, K2)


def reduced_pair_annihilation_coeff(bra: ReducedSubshellState,
                                    ket: ReducedSubshellState,
                                    K2: int) -> float:
    """Return ``<bra||[a x a]^K||ket>`` for one subshell."""
    bra_state = _resolve_state(bra)
    ket_state = _resolve_state(ket)
    if bra_state is None or ket_state is None:
        return 0.0
    return reduced_pair_annihilation_matrix_element(bra_state, ket_state, K2)
