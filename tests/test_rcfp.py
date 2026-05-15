"""Tests for reduced subshell coefficient API."""

import pytest

from atomx.angular.rcfp import (
    ReducedSubshellState,
    reduced_annihilation_coeff,
    reduced_axw_coeff,
    reduced_creation_coeff,
    reduced_pair_annihilation_coeff,
    reduced_pair_creation_coeff,
    reduced_subshell_states,
    reduced_w_coeff,
    reduced_wxa_coeff,
    reduced_wxw0_coeff,
)


def test_reduced_subshell_states_wrap_mscheme_terms():
    states = reduced_subshell_states(3, 2)
    assert [(s.J2, s.index) for s in states] == [(0, 0), (4, 0)]


def test_reduced_creation_coeff_empty_to_one_electron():
    ket = reduced_subshell_states(1, 0)[0]
    bra = reduced_subshell_states(1, 1)[0]
    assert reduced_creation_coeff(bra, ket) == pytest.approx(-(2.0 ** 0.5))


def test_reduced_coeff_returns_zero_for_invalid_transition():
    bra = ReducedSubshellState(j2=1, q=2, J2=0)
    ket = ReducedSubshellState(j2=3, q=1, J2=3)
    assert reduced_creation_coeff(bra, ket) == 0.0
    assert reduced_annihilation_coeff(ket, bra) == 0.0


def test_reduced_annihilation_is_creation_reverse_for_current_convention():
    one = reduced_subshell_states(1, 1)[0]
    closed = reduced_subshell_states(1, 2)[0]
    assert reduced_annihilation_coeff(one, closed) == pytest.approx(
        reduced_creation_coeff(closed, one)
    )


def test_reduced_w_coeff_scalar_one_electron():
    state = reduced_subshell_states(1, 1)[0]
    assert reduced_w_coeff(state, state, K2=0) == pytest.approx(-1.0)


def test_reduced_wxw0_coeff_scalar_one_electron():
    state = reduced_subshell_states(1, 1)[0]
    assert reduced_wxw0_coeff(state, state, K2=0) == pytest.approx(0.5)


def test_reduced_axw_and_wxa_scalar_vacuum_to_one_electron():
    ket = reduced_subshell_states(1, 0)[0]
    bra = reduced_subshell_states(1, 1)[0]
    assert reduced_axw_coeff(bra, ket, K2=0, R2=1) == 0.0
    assert reduced_wxa_coeff(bra, ket, K2=0, R2=1) == pytest.approx(1.0)


def test_reduced_pair_creation_and_annihilation_j_half():
    vac = reduced_subshell_states(1, 0)[0]
    closed = reduced_subshell_states(1, 2)[0]
    assert reduced_pair_creation_coeff(closed, vac, K2=0) == pytest.approx(
        -(2.0 ** 0.5)
    )
    assert reduced_pair_annihilation_coeff(vac, closed, K2=0) == pytest.approx(
        2.0 ** 0.5
    )
