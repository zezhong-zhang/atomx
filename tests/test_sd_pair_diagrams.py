"""Full signed fermion gates for complete core PP/HH reduced SD diagrams."""

import itertools
from math import sqrt

import numpy as np
import pytest

from atomx.methods.sd_pair_diagrams import CorePairLadders, plan_core_pair_ladders
from atomx.methods.sd_transverse import build_j_coupled_sd_interaction
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, build_spinor_integrals


def fixed_native_model():
    """Four-electron reference on fixed Be2+ orbitals, explicitly noncanonical."""
    from test_spinor_integrals_fscc import native

    from atomx.methods.mbpt import build_virtual_basis

    core, orbitals = native.__wrapped__()
    orbitals = dict(orbitals)
    _, virtuals = build_virtual_basis(
        core, -1, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
    )
    orbitals[3, -1] = virtuals[1][:2]
    return core, orbitals


def fixed_integrals(model, mode="dc", energies=None):
    core, orbitals = model
    return build_spinor_integrals(
        orbitals,
        core._bsp_data,
        4,
        30,
        occupied_keys=[(1, -1), (2, -1)],
        virtual_keys=[(3, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential,
        include_breit=mode != "dc",
        breit_orbital_energies=energies if mode == "dynamic" else None,
    )


@pytest.fixture(scope="module")
def model():
    return fixed_native_model()


def _apply(mask, operators):
    """Independent elementary Fermi action, operators written left to right."""
    sign = 1
    for index, create in reversed(operators):
        bit = 1 << index
        present = bool(mask & bit)
        if present == create:
            return None, 0
        sign *= (-1) ** (mask & (bit - 1)).bit_count()
        mask ^= bit
    return mask, sign


def _action(state, terms):
    out = {}
    for operators, coefficient in terms:
        for mask, amplitude in state.items():
            final, sign = _apply(mask, operators)
            if sign:
                out[final] = out.get(final, 0j) + sign * coefficient * amplitude
    return out


def independent_ladders(a, bridge, ladders):
    """Project explicit normal-ordered second-quantized subgraph operators.

    Does not call the production reduced contraction or its block matrices.
    Uses complete magnetic indices, full creation/annihilation signs and
    unmodified signed spinor ERI. No angular-vector renormalization/fitting.
    """
    reference = {(1 << a.noccupied) - 1: 1.0}
    masks = tuple(
        sum(1 << i for i in row)
        for row in itertools.combinations(range(len(a.labels)), a.noccupied)
    )
    columns, excitation_terms = [], []
    for (twoj, parity), virtual_pair, occupied_pair in ladders.coordinates:
        rows = bridge.channels[twoj, parity]
        v, o = rows.index(virtual_pair), rows.index(occupied_pair)
        terms = []
        for twom in range(-twoj, twoj + 1, 2):
            vectors = bridge.pair_vectors(twoj, parity, twom=twom)
            for p, q, cv in vectors[v]:
                for r, s, co in vectors[o]:
                    terms.append(
                        (
                            ((p, True), (q, True), (s, False), (r, False)),
                            cv * co / sqrt(twoj + 1),
                        )
                    )
        state = _action(reference, terms)
        columns.append([state.get(mask, 0j) for mask in masks])
        excitation_terms.append(terms)
    projection = np.asarray(columns, dtype=np.complex128).T

    def two_body(indices):
        pairs = tuple(itertools.combinations(indices, 2))
        return [
            (((p, True), (q, True), (s, False), (r, False)), a.eri(p, q, r, s))
            for p, q in pairs
            for r, s in pairs
        ]

    all_two = two_body(range(len(a.labels)))
    pp_terms = two_body(range(a.noccupied, len(a.labels)))
    hh_terms = two_body(range(a.noccupied))
    # Core normal ordering of the all-occupied two-body operator, including
    # the one-body contraction and scalar reference term exactly once.
    hh_terms += [
        (((p, True), (q, False)), -sum(a.eri(p, i, q, i) for i in range(a.noccupied)))
        for p, q in itertools.product(range(a.noccupied), repeat=2)
    ]
    hh_constant = 0.5 * sum(
        a.eri(i, j, i, j) for i, j in itertools.product(range(a.noccupied), repeat=2)
    )
    hh_terms.append(((), hh_constant))
    ref_hh = _action(reference, hh_terms)

    def dense(state):
        return np.asarray([state.get(mask, 0j) for mask in masks])

    source = projection.conj().T @ dense(_action(reference, all_two))
    pp_columns, hh_columns = [], []
    for column, terms in enumerate(excitation_terms):
        state = {
            mask: projection[i, column]
            for i, mask in enumerate(masks)
            if projection[i, column] != 0
        }
        pp_columns.append(projection.conj().T @ dense(_action(state, pp_terms)))
        action = dense(_action(state, hh_terms)) - dense(_action(ref_hh, terms))
        hh_columns.append(projection.conj().T @ action)
    return dict(
        projection=projection,
        masks=np.asarray(masks, dtype=np.int64),
        metric=projection.conj().T @ projection,
        source=source,
        pp=np.asarray(pp_columns).T,
        hh=np.asarray(hh_columns).T,
        normal_ordered_hh_reference=dense(ref_hh),
        hh_reference_constant=hh_constant,
    )


@pytest.mark.parametrize("mode", ["dc", "static", "dynamic"])
def test_complete_source_and_normal_ordered_pp_hh_signed_directions(model, mode):
    dc = fixed_integrals(model)
    energies = {x[:2]: dc.fock_Ha[i, i].real for i, x in enumerate(dc.labels)}
    a = fixed_integrals(model, mode, energies)
    bridge = build_j_coupled_sd_interaction(a)
    ladders = CorePairLadders(bridge)
    oracle = independent_ladders(a, bridge, ladders)
    count = len(ladders.source_Ha)
    assert count == 10
    assert plan_core_pair_ladders(bridge)["scalar_doubles"] == count
    np.testing.assert_allclose(oracle["metric"], np.eye(count), atol=1e-12, rtol=0)
    np.testing.assert_allclose(oracle["source"], ladders.source_Ha, atol=1e-11, rtol=0)
    np.testing.assert_allclose(
        oracle["normal_ordered_hh_reference"], 0, atol=1e-12, rtol=0
    )
    for i in range(count):
        amplitude = np.eye(count)[i]
        result = ladders.evaluate(amplitude)
        np.testing.assert_allclose(
            result.particle_particle_Ha, oracle["pp"][:, i], atol=1e-11, rtol=0
        )
        np.testing.assert_allclose(
            result.hole_hole_Ha, oracle["hh"][:, i], atol=1e-11, rtol=0
        )
        assert result.doubles_linked_energy_Ha == pytest.approx(
            oracle["source"][i].conjugate(), abs=1e-11
        )
    assert np.max(abs(oracle["pp"] - np.diag(oracle["pp"].diagonal()))) > 1e-2
    assert np.max(abs(oracle["hh"] - np.diag(oracle["hh"].diagonal()))) > 1e-2
    assert np.max(abs(a.fock_Ha[:4, 4:])) > 1e-3
    assert ladders.interaction is bridge  # complete noncanonical owner retained


def _phase_change(a):
    phases = np.asarray([np.exp(0.37j * (x[0] + x[1])) for x in a.labels])
    packed = [
        phases[p].conjugate()
        * phases[q].conjugate()
        * a.eri(p, q, r, s)
        * phases[r]
        * phases[s]
        for p in range(len(a.labels) - 1)
        for q in range(p + 1, len(a.labels))
        for r in range(p, len(a.labels) - 1)
        for s in range(r + 1, len(a.labels))
    ]
    h = phases.conj()[:, None] * a.one_body_Ha * phases
    f = phases.conj()[:, None] * a.fock_Ha * phases
    h = 0.5 * (h + h.conj().T)
    f = 0.5 * (f + f.conj().T)
    h[np.diag_indices(len(phases))] = h.diagonal().real
    f[np.diag_indices(len(phases))] = f.diagonal().real
    return SpinorIntegrals(
        a.labels,
        a.noccupied,
        h,
        f,
        packed,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        a.identity,
    )


def test_full_complex_phase_signed_action_and_payload_rejection(model):
    a = _phase_change(fixed_integrals(model))
    bridge = build_j_coupled_sd_interaction(a)
    ladders = CorePairLadders(bridge)
    oracle = independent_ladders(a, bridge, ladders)
    t = np.arange(1, 11) * (0.013 + 0.021j)
    result = ladders.evaluate(t)
    for value, expected in (
        (result.source_Ha, oracle["source"]),
        (result.particle_particle_Ha, oracle["pp"] @ t),
        (result.hole_hole_Ha, oracle["hh"] @ t),
    ):
        np.testing.assert_allclose(value, expected, atol=1e-11, rtol=0)
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert result.doubles_linked_energy_Ha == pytest.approx(
        np.vdot(oracle["source"], t), abs=1e-11
    )
    assert np.max(abs(result.source_Ha.imag)) > 1e-3
    with pytest.raises(MemoryError):
        CorePairLadders(bridge, memory_bytes=1)
    for bad in (np.zeros(9), np.full(10, np.nan)):
        with pytest.raises(ValueError):
            ladders.evaluate(bad)
    with pytest.raises(ValueError):
        plan_core_pair_ladders(bridge, batch=0)
    with pytest.raises(TypeError):
        plan_core_pair_ladders(object())


def test_reference_and_frozen_scalar_preserved_without_double_counting(model):
    original = fixed_integrals(model)
    shifted = SpinorIntegrals(
        original.labels,
        original.noccupied,
        original.one_body_Ha,
        original.fock_Ha,
        original.packed_eri_Ha,
        original.frozen_energy_Ha + 0.137,
        original.reference_energy_Ha + 0.137,
        original.identity,
    )
    a = CorePairLadders(build_j_coupled_sd_interaction(original))
    b = CorePairLadders(build_j_coupled_sd_interaction(shifted))
    assert (
        b.interaction.frozen_energy_Ha - a.interaction.frozen_energy_Ha
        == pytest.approx(0.137)
    )
    assert (
        b.interaction.reference_energy_Ha - a.interaction.reference_energy_Ha
        == pytest.approx(0.137)
    )
    np.testing.assert_array_equal(a.source_Ha, b.source_Ha)
    for block in a.pp_Ha:
        np.testing.assert_array_equal(a.pp_Ha[block], b.pp_Ha[block])
        np.testing.assert_array_equal(a.hh_Ha[block], b.hh_Ha[block])
