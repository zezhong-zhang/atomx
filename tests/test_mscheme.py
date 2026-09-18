"""Tests for exact m-scheme subshell angular machinery."""

import pytest

from atomx.angular.mscheme import (
    coupled_axw_matrix_element,
    coupled_one_body_tensor_matrix_element,
    coupled_pair_annihilation_matrix_element,
    coupled_pair_creation_matrix_element,
    coupled_wxw_scalar_matrix_element,
    creation_matrix_element,
    highest_weight_states,
    reduced_axw_matrix_element,
    reduced_creation_matrix_element,
    reduced_one_body_tensor_matrix_element,
    reduced_pair_annihilation_matrix_element,
    reduced_pair_creation_matrix_element,
    reduced_wxa_matrix_element,
    subshell_j_counts_mscheme,
)


def test_mscheme_counts_one_electron_shell():
    assert subshell_j_counts_mscheme(1, 1) == {1: 1}
    assert subshell_j_counts_mscheme(3, 1) == {3: 1}


def test_mscheme_counts_closed_shell_j_zero():
    assert subshell_j_counts_mscheme(1, 2) == {0: 1}
    assert subshell_j_counts_mscheme(3, 4) == {0: 1}


def test_mscheme_counts_p_three_half_two_electrons():
    # Two equivalent j=3/2 electrons allow J=0 and J=2 only.
    assert subshell_j_counts_mscheme(3, 2) == {0: 1, 4: 1}


def test_mscheme_counts_particle_hole_symmetry():
    # j^q and j^(g-q) have the same J content.
    assert subshell_j_counts_mscheme(5, 2) == subshell_j_counts_mscheme(5, 4)


def test_highest_weight_states_use_canonical_determinant_phase():
    for state in highest_weight_states(5, 3):
        first = next(c for c in state.coefficients if abs(c) > 1e-12)
        assert first > 0.0


def test_creation_matrix_element_for_empty_to_one_electron():
    ket = highest_weight_states(1, 0)[0]
    bra = highest_weight_states(1, 1)[0]

    assert creation_matrix_element(bra, ket, m2=1) == pytest.approx(1.0)
    assert reduced_creation_matrix_element(bra, ket) == pytest.approx(-(2.0 ** 0.5))


def test_scalar_one_body_tensor_matches_number_operator_phase():
    state = highest_weight_states(1, 1)[0]
    # [a† x atilde]^0_0 = (-1)^(2j) N / sqrt(2j+1) for a single j shell.
    assert coupled_one_body_tensor_matrix_element(state, state, K2=0, Q2=0) == pytest.approx(
        -1.0 / (2.0 ** 0.5)
    )
    assert reduced_one_body_tensor_matrix_element(state, state, K2=0) == pytest.approx(-1.0)


def test_one_body_tensor_blocks_invalid_rank():
    state = highest_weight_states(1, 1)[0]
    assert reduced_one_body_tensor_matrix_element(state, state, K2=4) == 0.0


def test_wxw_scalar_matches_w_squared_for_scalar_rank():
    state = highest_weight_states(1, 1)[0]
    w00 = coupled_one_body_tensor_matrix_element(state, state, K2=0, Q2=0)
    assert coupled_wxw_scalar_matrix_element(state, state, K2=0) == pytest.approx(w00 * w00)


def test_axw_and_wxa_scalar_rank_from_vacuum_are_zero():
    ket = highest_weight_states(1, 0)[0]
    bra = highest_weight_states(1, 1)[0]
    assert coupled_axw_matrix_element(bra, ket, K2=0, R2=1, Q2=1) == 0.0
    assert reduced_axw_matrix_element(bra, ket, K2=0, R2=1) == 0.0
    assert reduced_wxa_matrix_element(bra, ket, K2=0, R2=1) == pytest.approx(1.0)


def test_axw_and_wxa_block_invalid_occupation_change():
    state = highest_weight_states(1, 1)[0]
    assert reduced_axw_matrix_element(state, state, K2=0, R2=1) == 0.0
    assert reduced_wxa_matrix_element(state, state, K2=0, R2=1) == 0.0


def test_pair_creation_and_annihilation_j_half_closed_shell():
    vac = highest_weight_states(1, 0)[0]
    closed = highest_weight_states(1, 2)[0]
    assert coupled_pair_creation_matrix_element(closed, vac, K2=0, Q2=0) == pytest.approx(
        -(2.0 ** 0.5)
    )
    assert reduced_pair_creation_matrix_element(closed, vac, K2=0) == pytest.approx(
        -(2.0 ** 0.5)
    )
    assert reduced_pair_annihilation_matrix_element(vac, closed, K2=0) == pytest.approx(
        2.0 ** 0.5
    )
    assert coupled_pair_annihilation_matrix_element(vac, closed, K2=2, Q2=0) == 0.0
