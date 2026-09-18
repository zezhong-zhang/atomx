"""Tests for multi-subshell CSF recoupling helpers."""

import pytest

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.csf_recoupling import (
    RecoupledCSFState,
    allowed_coupled_J2,
    angular_momenta_couple,
    chain_one_body_excitation_reduced,
    chain_projection_amplitudes,
    chain_state_from_recoupled_csf,
    chain_state_from_simple_csf_auto,
    chain_two_body_scalar_coefficients,
    chain_w_reduced,
    chain_wxw_scalar,
    classify_recoupled_occupation_change,
    discover_chain_two_body_scalar_coefficients,
    discover_double_excitation_coefficients_simple,
    discover_single_excitation_coefficients_simple,
    discover_two_body_coefficients_conservative,
    discover_two_body_scalar_coefficients_simple,
    make_chain_coupled_state,
    make_two_subshell_coupled_state,
    recoupled_state_from_simple_csf,
    scalar_w_subshell_coeff,
    scalar_wxw0_subshell_coeff,
    single_state_for_subshell,
    spectator_subshells_match,
    two_body_scalar_coefficients_simple,
    two_subshell_left_w_reduced,
    two_subshell_right_w_reduced,
    two_subshell_wxw_scalar,
)
from atomx.angular.rcfp import reduced_subshell_states
from atomx.angular.wigner import wigner_3j
from atomx.methods.dirac_fock import Subshell
from atomx.methods.mcdhf.config import CSF


def test_scalar_w_subshell_coeff_matches_single_subshell_primitive():
    active = reduced_subshell_states(1, 1)[0]
    spectator = reduced_subshell_states(3, 2)[0]
    state = RecoupledCSFState((active, spectator), total_J2=0, parity=1)

    assert scalar_w_subshell_coeff(state, state, active=0) == pytest.approx(-1.0)
    assert scalar_wxw0_subshell_coeff(state, state, active=0) == pytest.approx(0.5)


def test_scalar_w_subshell_coeff_requires_matching_spectators():
    active = reduced_subshell_states(1, 1)[0]
    spectator_a, spectator_b = reduced_subshell_states(3, 2)
    bra = RecoupledCSFState((active, spectator_a), total_J2=0, parity=1)
    ket = RecoupledCSFState((active, spectator_b), total_J2=0, parity=1)

    assert not spectator_subshells_match(bra, ket, active=0)
    assert scalar_w_subshell_coeff(bra, ket, active=0) == 0.0


def test_scalar_w_subshell_coeff_blocks_jpi_mismatch():
    active = reduced_subshell_states(1, 1)[0]
    spectator = reduced_subshell_states(3, 2)[0]
    bra = RecoupledCSFState((active, spectator), total_J2=0, parity=1)
    ket_j = RecoupledCSFState((active, spectator), total_J2=2, parity=1)
    ket_p = RecoupledCSFState((active, spectator), total_J2=0, parity=-1)

    assert scalar_w_subshell_coeff(bra, ket_j, active=0) == 0.0
    assert scalar_w_subshell_coeff(bra, ket_p, active=0) == 0.0


def test_single_state_for_subshell_is_conservative():
    assert single_state_for_subshell(1, 0).J2 == 0
    assert single_state_for_subshell(1, 1).J2 == 1
    assert single_state_for_subshell(3, 2) is None


def test_recoupled_state_from_simple_csf_closed_shell():
    csf = CSF(
        subshells=[Subshell(1, -1, 2)],
        total_J2=0,
        parity=1,
        label="1s2 J=0",
    )
    state = recoupled_state_from_simple_csf(csf)

    assert state is not None
    assert state.total_J2 == 0
    assert state.subshells[0].q == 2


def test_chain_state_from_simple_csf_auto_closed_shell():
    csf = CSF(
        subshells=[Subshell(1, -1, 2)],
        total_J2=0,
        parity=1,
        label="1s2 J=0",
    )
    state = chain_state_from_simple_csf_auto(csf)

    assert state is not None
    assert state.cumulative_J2 == (0,)


def test_chain_state_from_simple_csf_auto_two_open_subshells():
    csf = CSF(
        subshells=[Subshell(2, -2, 3), Subshell(3, 2, 1)],
        total_J2=2,
        parity=-1,
        label="2p3/2^3 3d3/2 J=1",
    )
    state = chain_state_from_simple_csf_auto(csf)

    assert state is not None
    assert state.cumulative_J2 == (3, 2)
    assert state.total_J2 == 2
    assert state.parity == -1


def test_chain_state_from_simple_csf_auto_rejects_ambiguous_open_chain():
    csf = CSF(
        subshells=[
            Subshell(2, -1, 1),
            Subshell(2, -2, 1),
            Subshell(3, 2, 1),
        ],
        total_J2=2,
        parity=1,
        label="three open subshells",
    )

    assert chain_state_from_simple_csf_auto(csf) is None


def test_two_subshell_coupled_state_validates_total_j():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]

    assert allowed_coupled_J2(left.J2, right.J2) == (2, 4)
    assert angular_momenta_couple(left.J2, right.J2, 2)
    state = make_two_subshell_coupled_state(left, right, total_J2=4, parity=-1)
    assert state.total_J2 == 4
    assert state.parity == -1


def test_two_subshell_coupled_state_rejects_invalid_total_j():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]

    with pytest.raises(ValueError):
        make_two_subshell_coupled_state(left, right, total_J2=0)


def test_two_subshell_scalar_w_left_and_right_are_diagonal():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = make_two_subshell_coupled_state(left, right, total_J2=2, parity=1)

    assert two_subshell_left_w_reduced(state, state, K2=0) == pytest.approx(
        -(3.0 / 2.0) ** 0.5
    )
    assert two_subshell_right_w_reduced(state, state, K2=0) == pytest.approx(
        -(3.0 ** 0.5) / 2.0
    )


def test_two_subshell_w_blocks_spectator_mismatch():
    left = reduced_subshell_states(1, 1)[0]
    right_a, right_b = reduced_subshell_states(3, 2)
    bra = make_two_subshell_coupled_state(left, right_a, total_J2=1)
    ket = make_two_subshell_coupled_state(left, right_b, total_J2=3)

    assert two_subshell_left_w_reduced(bra, ket, K2=0) == 0.0


def test_two_subshell_wxw_scalar_k0_matches_product_in_state_space():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = make_two_subshell_coupled_state(left, right, total_J2=2, parity=1)

    assert two_subshell_wxw_scalar(state, state, K2=0) == pytest.approx(
        1.0 / (8.0 ** 0.5)
    )


def test_two_subshell_wxw_scalar_blocks_jpi_mismatch():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    bra = make_two_subshell_coupled_state(left, right, total_J2=2, parity=1)
    ket = make_two_subshell_coupled_state(left, right, total_J2=4, parity=1)

    assert two_subshell_wxw_scalar(bra, ket, K2=0) == 0.0


def test_chain_coupled_state_validates_cumulative_j():
    s = reduced_subshell_states(1, 1)[0]
    p = reduced_subshell_states(3, 1)[0]
    d = reduced_subshell_states(5, 1)[0]

    state = make_chain_coupled_state((s, p, d), cumulative_J2=(1, 2, 3), parity=-1)

    assert state.total_J2 == 3
    assert state.parity == -1


def test_chain_coupled_state_rejects_invalid_cumulative_j():
    s = reduced_subshell_states(1, 1)[0]
    p = reduced_subshell_states(3, 1)[0]

    with pytest.raises(ValueError):
        make_chain_coupled_state((s, p), cumulative_J2=(1, 0))


def test_chain_w_reduced_matches_two_subshell_left_and_right():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    two = make_two_subshell_coupled_state(left, right, total_J2=2)
    chain = make_chain_coupled_state((left, right), cumulative_J2=(1, 2))

    assert chain_w_reduced(chain, chain, active=0, K2=0) == pytest.approx(
        two_subshell_left_w_reduced(two, two, K2=0)
    )
    assert chain_w_reduced(chain, chain, active=1, K2=0) == pytest.approx(
        two_subshell_right_w_reduced(two, two, K2=0)
    )


def test_chain_w_reduced_recouples_three_subshell_spectator():
    left = reduced_subshell_states(1, 1)[0]
    middle = reduced_subshell_states(3, 1)[0]
    right = reduced_subshell_states(1, 1)[0]
    state = make_chain_coupled_state(
        (left, middle, right),
        cumulative_J2=(1, 2, 1),
    )

    assert chain_w_reduced(state, state, active=0, K2=0) == pytest.approx(-1.0)
    assert chain_w_reduced(state, state, active=1, K2=0) == pytest.approx(-(0.5 ** 0.5))
    assert chain_w_reduced(state, state, active=2, K2=0) == pytest.approx(-1.0)


def test_chain_projection_amplitudes_are_normalized():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = make_chain_coupled_state((left, right), cumulative_J2=(1, 2))

    norm = sum(coeff * coeff for _, coeff in chain_projection_amplitudes(state))

    assert norm == pytest.approx(1.0)


def test_chain_wxw_scalar_matches_two_subshell_expansion():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    two = make_two_subshell_coupled_state(left, right, total_J2=2)
    chain = make_chain_coupled_state((left, right), cumulative_J2=(1, 2))

    assert chain_wxw_scalar(chain, chain, 0, 1, K2=0) == pytest.approx(
        two_subshell_wxw_scalar(two, two, K2=0)
    )


def test_chain_wxw_scalar_handles_three_subshell_spectator():
    left = reduced_subshell_states(1, 1)[0]
    middle = reduced_subshell_states(3, 1)[0]
    right = reduced_subshell_states(1, 1)[0]
    state = make_chain_coupled_state(
        (left, middle, right),
        cumulative_J2=(1, 2, 1),
    )

    assert chain_wxw_scalar(state, state, 0, 1, K2=0) == pytest.approx(1.0 / (8.0 ** 0.5))
    assert chain_wxw_scalar(state, state, 0, 2, K2=0) == pytest.approx(0.5)
    assert chain_wxw_scalar(state, state, 1, 2, K2=0) == pytest.approx(1.0 / (8.0 ** 0.5))


def test_chain_two_body_scalar_coefficients_wraps_chain_wxw():
    left = reduced_subshell_states(1, 1)[0]
    middle = reduced_subshell_states(3, 1)[0]
    right = reduced_subshell_states(1, 1)[0]
    state = make_chain_coupled_state(
        (left, middle, right),
        cumulative_J2=(1, 2, 1),
    )

    coeffs = chain_two_body_scalar_coefficients(state, state, 0, 1, K2=0)

    assert len(coeffs) == 1
    assert coeffs[0].kind == "chain-WxW0"
    assert coeffs[0].active_left == 0
    assert coeffs[0].active_right == 1
    assert coeffs[0].coefficient == pytest.approx(1.0 / (8.0 ** 0.5))


def test_discover_chain_two_body_scalar_coefficients_scans_pairs():
    left = reduced_subshell_states(1, 1)[0]
    middle = reduced_subshell_states(3, 1)[0]
    right = reduced_subshell_states(1, 1)[0]
    state = make_chain_coupled_state(
        (left, middle, right),
        cumulative_J2=(1, 2, 1),
    )

    coeffs = discover_chain_two_body_scalar_coefficients(state, state)

    assert [(c.active_left, c.active_right) for c in coeffs] == [(0, 1), (0, 2), (1, 2)]
    assert [c.coefficient for c in coeffs] == pytest.approx([
        1.0 / (8.0 ** 0.5),
        0.5,
        1.0 / (8.0 ** 0.5),
    ])


def _one_electron_subshell_vector(state, M2):
    if state.q == 0:
        return {0: 1.0} if M2 == 0 else {}
    if state.q == 1 and state.j2 == 1 and M2 in (-1, 1):
        orbital = 0 if M2 == -1 else 1
        return {1 << orbital: 1.0}
    raise AssertionError(f"Unsupported exact test state: {state}, M2={M2}")


def _one_electron_product_vector(chain):
    offsets = []
    offset = 0
    for subshell in chain.subshells:
        offsets.append(offset)
        offset += subshell.j2 + 1

    out = {}
    for ms, chain_coeff in chain_projection_amplitudes(chain, chain.total_J2):
        partial = {0: chain_coeff}
        for idx, (subshell, M2) in enumerate(zip(chain.subshells, ms, strict=False)):
            next_partial = {}
            for mask, coeff in partial.items():
                for local_mask, local_coeff in _one_electron_subshell_vector(
                    subshell, M2,
                ).items():
                    global_mask = mask | (local_mask << offsets[idx])
                    next_partial[global_mask] = (
                        next_partial.get(global_mask, 0.0)
                        + coeff * local_coeff
                    )
            partial = next_partial
        for mask, coeff in partial.items():
            if abs(coeff) > 1e-14:
                out[mask] = out.get(mask, 0.0) + coeff
    return out, offsets


def _annihilate_global(mask, orbital):
    if not (mask & (1 << orbital)):
        return None
    phase = -1 if (mask & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return phase, mask ^ (1 << orbital)


def _create_global(mask, orbital):
    if mask & (1 << orbital):
        return None
    phase = -1 if (mask & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return phase, mask | (1 << orbital)


def _exact_global_one_body_reduced(bra, ket, source, destination, K2):
    bra_vec, offsets = _one_electron_product_vector(bra)
    ket_vec, _ = _one_electron_product_vector(ket)
    Q2 = bra.total_J2 - ket.total_J2
    actual = 0.0
    for mask, ket_coeff in ket_vec.items():
        for m_create2 in (-1, 1):
            mu2 = Q2 - m_create2
            if mu2 not in (-1, 1):
                continue
            cg = clebsch_gordan(1, m_create2, 1, mu2, K2, Q2)
            if abs(cg) < 1e-14:
                continue
            remove_m2 = -mu2
            remove_orbital = offsets[source] + (0 if remove_m2 == -1 else 1)
            create_orbital = offsets[destination] + (0 if m_create2 == -1 else 1)
            ann = _annihilate_global(mask, remove_orbital)
            if ann is None:
                continue
            phase_a, intermediate = ann
            cre = _create_global(intermediate, create_orbital)
            if cre is None:
                continue
            phase_c, new_mask = cre
            tilde_phase = -1.0 if ((1 - mu2) // 2) % 2 else 1.0
            actual += (
                bra_vec.get(new_mask, 0.0)
                * phase_a * phase_c * tilde_phase * cg * ket_coeff
            )
    if abs(actual) < 1e-14:
        return 0.0
    w3j = wigner_3j(
        bra.total_J2, K2, ket.total_J2,
        -bra.total_J2, Q2, ket.total_J2,
    )
    return actual / w3j


def test_chain_one_body_excitation_reduced_single_electron_scalar():
    source_full = reduced_subshell_states(1, 1)[0]
    source_empty = reduced_subshell_states(1, 0)[0]
    destination_empty = reduced_subshell_states(1, 0)[0]
    destination_full = reduced_subshell_states(1, 1)[0]
    ket = make_chain_coupled_state(
        (source_full, destination_empty),
        cumulative_J2=(1, 1),
    )
    bra = make_chain_coupled_state(
        (source_empty, destination_full),
        cumulative_J2=(0, 1),
    )

    coeff = chain_one_body_excitation_reduced(
        bra,
        ket,
        source=0,
        destination=1,
        K2=0,
    )

    assert coeff == pytest.approx(1.0)


def test_chain_one_body_excitation_reduced_single_electron_vector_normalization():
    source_full = reduced_subshell_states(1, 1)[0]
    source_empty = reduced_subshell_states(1, 0)[0]
    destination_empty = reduced_subshell_states(1, 0)[0]
    destination_full = reduced_subshell_states(1, 1)[0]
    ket = make_chain_coupled_state(
        (source_full, destination_empty),
        cumulative_J2=(1, 1),
    )
    bra = make_chain_coupled_state(
        (source_empty, destination_full),
        cumulative_J2=(0, 1),
    )

    coeff = chain_one_body_excitation_reduced(
        bra,
        ket,
        source=0,
        destination=1,
        K2=2,
    )

    assert coeff == pytest.approx(3.0 ** 0.5)


def test_chain_one_body_excitation_tracks_global_fock_phase_convention():
    source_full = reduced_subshell_states(1, 1)[0]
    source_empty = reduced_subshell_states(1, 0)[0]
    destination_empty = reduced_subshell_states(1, 0)[0]
    destination_full = reduced_subshell_states(1, 1)[0]
    ket = make_chain_coupled_state(
        (source_full, destination_empty),
        cumulative_J2=(1, 1),
    )
    bra = make_chain_coupled_state(
        (source_empty, destination_full),
        cumulative_J2=(0, 1),
    )

    chain_coeff = chain_one_body_excitation_reduced(
        bra,
        ket,
        source=0,
        destination=1,
        K2=0,
    )
    exact_coeff = _exact_global_one_body_reduced(
        bra,
        ket,
        source=0,
        destination=1,
        K2=0,
    )

    assert chain_coeff == pytest.approx(-exact_coeff)


def test_chain_one_body_excitation_global_fock_phase_with_spectator():
    source_full = reduced_subshell_states(1, 1)[0]
    source_empty = reduced_subshell_states(1, 0)[0]
    spectator = reduced_subshell_states(1, 1)[0]
    destination_empty = reduced_subshell_states(1, 0)[0]
    destination_full = reduced_subshell_states(1, 1)[0]
    ket = make_chain_coupled_state(
        (source_full, spectator, destination_empty),
        cumulative_J2=(1, 0, 0),
    )
    bra = make_chain_coupled_state(
        (source_empty, spectator, destination_full),
        cumulative_J2=(0, 1, 0),
    )

    chain_coeff = chain_one_body_excitation_reduced(
        bra,
        ket,
        source=0,
        destination=2,
        K2=0,
    )
    exact_coeff = _exact_global_one_body_reduced(
        bra,
        ket,
        source=0,
        destination=2,
        K2=0,
    )

    assert chain_coeff == pytest.approx(-exact_coeff)


def test_chain_one_body_excitation_reduced_blocks_spectator_mismatch():
    source_full = reduced_subshell_states(1, 1)[0]
    source_empty = reduced_subshell_states(1, 0)[0]
    destination_empty = reduced_subshell_states(1, 0)[0]
    destination_full = reduced_subshell_states(1, 1)[0]
    spectator_a, spectator_b = [
        s for s in reduced_subshell_states(7, 4) if s.J2 == 4
    ][:2]
    ket = make_chain_coupled_state(
        (source_full, destination_empty, spectator_a),
        cumulative_J2=(1, 1, 3),
    )
    bra = make_chain_coupled_state(
        (source_empty, destination_full, spectator_b),
        cumulative_J2=(0, 1, 3),
    )

    coeff = chain_one_body_excitation_reduced(
        bra,
        ket,
        source=0,
        destination=1,
        K2=0,
    )

    assert coeff == 0.0


def test_chain_state_from_recoupled_csf_requires_matching_total_j():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    recoupled = RecoupledCSFState((left, right), total_J2=2)

    chain = chain_state_from_recoupled_csf(recoupled, cumulative_J2=(1, 2))
    assert chain.total_J2 == 2
    with pytest.raises(ValueError):
        chain_state_from_recoupled_csf(recoupled, cumulative_J2=(1, 4))


def test_two_body_scalar_coefficients_simple_returns_wxw_coeff():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = RecoupledCSFState((left, right), total_J2=2, parity=1)

    coeffs = two_body_scalar_coefficients_simple(state, state, 0, 1, K2=0)

    assert len(coeffs) == 1
    assert coeffs[0].active_left == 0
    assert coeffs[0].active_right == 1
    assert coeffs[0].kind == "WxW0"
    assert coeffs[0].coefficient == pytest.approx(1.0 / (8.0 ** 0.5))


def test_two_body_scalar_coefficients_simple_blocks_spectator_mismatch():
    active_l = reduced_subshell_states(1, 1)[0]
    active_r = reduced_subshell_states(3, 1)[0]
    spectator_a, spectator_b = reduced_subshell_states(3, 2)
    bra = RecoupledCSFState((active_l, active_r, spectator_a), total_J2=2, parity=1)
    ket = RecoupledCSFState((active_l, active_r, spectator_b), total_J2=2, parity=1)

    assert two_body_scalar_coefficients_simple(bra, ket, 0, 1, K2=0) == tuple()


def test_two_body_scalar_coefficients_simple_blocks_same_active_index():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = RecoupledCSFState((left, right), total_J2=2, parity=1)

    assert two_body_scalar_coefficients_simple(state, state, 0, 0, K2=0) == tuple()


def test_discover_two_body_scalar_coefficients_simple_scans_pairs():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    state = RecoupledCSFState((left, right), total_J2=2, parity=1)

    coeffs = discover_two_body_scalar_coefficients_simple(state, state)

    assert len(coeffs) == 1
    assert (coeffs[0].active_left, coeffs[0].active_right) == (0, 1)


def test_discover_two_body_scalar_coefficients_simple_blocks_shape_or_jpi_mismatch():
    left = reduced_subshell_states(1, 1)[0]
    right = reduced_subshell_states(3, 1)[0]
    bra = RecoupledCSFState((left, right), total_J2=2, parity=1)
    ket_shape = RecoupledCSFState((left,), total_J2=2, parity=1)
    ket_j = RecoupledCSFState((left, right), total_J2=4, parity=1)
    ket_p = RecoupledCSFState((left, right), total_J2=2, parity=-1)

    assert discover_two_body_scalar_coefficients_simple(bra, ket_shape) == tuple()
    assert discover_two_body_scalar_coefficients_simple(bra, ket_j) == tuple()
    assert discover_two_body_scalar_coefficients_simple(bra, ket_p) == tuple()


def test_classify_recoupled_occupation_change_kinds():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    q2 = reduced_subshell_states(1, 2)[0]
    p1 = reduced_subshell_states(3, 1)[0]

    diagonal = RecoupledCSFState((q1, p1), total_J2=2)
    assert classify_recoupled_occupation_change(diagonal, diagonal).kind == "diagonal"

    bra_single = RecoupledCSFState((q0, p1), total_J2=2)
    ket_single = RecoupledCSFState((q1, q0), total_J2=2)
    single = classify_recoupled_occupation_change(bra_single, ket_single)
    assert single.kind == "single"
    assert single.sources == (0,)
    assert single.destinations == (1,)

    bra_double = RecoupledCSFState((q0, q2), total_J2=0)
    ket_double = RecoupledCSFState((q2, q0), total_J2=0)
    double = classify_recoupled_occupation_change(bra_double, ket_double)
    assert double.kind == "double"
    assert double.sources == (0, 0)
    assert double.destinations == (1, 1)


def test_classify_recoupled_occupation_change_shape_mismatch_is_unsupported():
    q1 = reduced_subshell_states(1, 1)[0]
    bra = RecoupledCSFState((q1,), total_J2=1)
    ket = RecoupledCSFState((q1, q1), total_J2=0)

    assert classify_recoupled_occupation_change(bra, ket).kind == "unsupported"


def test_discover_two_body_coefficients_conservative_dispatches_supported_cases():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    diagonal = RecoupledCSFState((q1, p1), total_J2=2)
    bra_single = RecoupledCSFState((q0, p1), total_J2=2)
    ket_single = RecoupledCSFState((q1, p0), total_J2=2)

    assert len(discover_two_body_coefficients_conservative(diagonal, diagonal)) == 1
    assert len(discover_two_body_coefficients_conservative(bra_single, ket_single)) == 1


def test_discover_single_excitation_coefficients_simple():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    spectator = reduced_subshell_states(1, 2)[0]
    bra = RecoupledCSFState((q0, p1, spectator), total_J2=2, parity=1)
    ket = RecoupledCSFState((q1, p0, spectator), total_J2=2, parity=1)

    coeffs = discover_single_excitation_coefficients_simple(bra, ket)

    assert len(coeffs) == 1
    assert coeffs[0].kind == "single"
    assert coeffs[0].active_left == 0
    assert coeffs[0].active_right == 1
    assert coeffs[0].coefficient != 0.0


def test_discover_single_excitation_coefficients_simple_blocks_spectator_mismatch():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    spectator_a = reduced_subshell_states(3, 2)[0]
    spectator_b = reduced_subshell_states(3, 2)[1]
    bra = RecoupledCSFState((q0, p1, spectator_a), total_J2=2, parity=1)
    ket = RecoupledCSFState((q1, p0, spectator_b), total_J2=2, parity=1)

    assert discover_single_excitation_coefficients_simple(bra, ket) == tuple()


def test_discover_double_excitation_coefficients_simple_paired_case():
    q0 = reduced_subshell_states(1, 0)[0]
    q2 = reduced_subshell_states(1, 2)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p2 = reduced_subshell_states(3, 2)[0]
    spectator = reduced_subshell_states(1, 2)[0]
    bra = RecoupledCSFState((q0, p2, spectator), total_J2=0, parity=1)
    ket = RecoupledCSFState((q2, p0, spectator), total_J2=0, parity=1)

    coeffs = discover_double_excitation_coefficients_simple(bra, ket)

    assert len(coeffs) == 1
    assert coeffs[0].kind == "double"
    assert coeffs[0].active_left == 0
    assert coeffs[0].active_right == 1
    assert coeffs[0].coefficient != 0.0
    assert len(discover_two_body_coefficients_conservative(bra, ket)) == 1


def test_discover_double_excitation_coefficients_simple_blocks_mixed_sources():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    spectator = reduced_subshell_states(1, 2)[0]
    bra = RecoupledCSFState((q0, p0, p1, p1, spectator), total_J2=0, parity=1)
    ket = RecoupledCSFState((q1, p1, p0, p0, spectator), total_J2=0, parity=1)

    coeffs = discover_double_excitation_coefficients_simple(bra, ket)

    assert len(coeffs) == 1
    assert coeffs[0].kind == "double-mixed"
    assert coeffs[0].coefficient != 0.0
    assert len(discover_two_body_coefficients_conservative(bra, ket)) == 1


def test_discover_double_excitation_coefficients_simple_blocks_unsupported_double_shape():
    q0 = reduced_subshell_states(1, 0)[0]
    q1 = reduced_subshell_states(1, 1)[0]
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    p2 = reduced_subshell_states(3, 2)[0]
    bra = RecoupledCSFState((q0, p0, p2), total_J2=0, parity=1)
    ket = RecoupledCSFState((q1, p1, p0), total_J2=0, parity=1)

    assert discover_double_excitation_coefficients_simple(bra, ket) == tuple()
