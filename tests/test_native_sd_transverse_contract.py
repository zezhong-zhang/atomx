"""Independent signed magnetic/CSF gates for the bounded transverse SD bridge."""

import itertools
from unittest.mock import patch

import numpy as np
import pytest
from test_spinor_integrals_fscc import integrals
from test_spinor_integrals_fscc import native as native_fixture
from test_transverse_spinor_integrals import determinant_matrix

from atomx.methods.sd_transverse import (
    build_j_coupled_sd_interaction,
    plan_j_coupled_sd_interaction,
    solve_two_electron_core_sd,
)
from atomx.physics.csf.spinor_integrals import SpinorIntegrals

native = native_fixture


def checked_case(native, mode):
    dc = integrals(native)
    energies = {label[:2]: dc.fock_Ha[i, i].real for i, label in enumerate(dc.labels)}
    options = dict(include_breit=mode != "dc")
    if mode == "dynamic":
        options["breit_orbital_energies"] = energies
    a = integrals(native, **options)
    return a, build_j_coupled_sd_interaction(a)


@pytest.mark.parametrize("mode", ["dc", "static", "dynamic"])
def test_complete_signed_eri_three_electron_fock_and_pair_metric(native, mode):
    a, coupled = checked_case(native, mode)
    assert plan_j_coupled_sd_interaction(a)["magnetic_pair_dimension"] == 45
    for p, q, r, s in itertools.product(range(len(a.labels)), repeat=4):
        assert coupled.eri(p, q, r, s) == pytest.approx(a.eri(p, q, r, s), abs=1e-11)
    masks, full = determinant_matrix(a, 3)
    rebuilt_masks, rebuilt = determinant_matrix(coupled, 3)
    assert masks == rebuilt_masks
    np.testing.assert_allclose(full, rebuilt, atol=1e-11, rtol=0)
    assert np.max(abs(full - np.diag(full.diagonal()))) > 1e-2
    for twoj, parity in coupled.channels:
        for twom in range(-twoj, twoj + 1, 2):
            vectors = coupled.pair_vectors(twoj, parity, twom=twom)
            matrix = np.zeros((45, len(vectors)))
            positions = {
                pair: i for i, pair in enumerate(itertools.combinations(range(10), 2))
            }
            for column, vector in enumerate(vectors):
                for p, q, c in vector:
                    matrix[positions[p, q], column] = c
            np.testing.assert_allclose(
                matrix.T @ matrix, np.eye(len(vectors)), atol=1e-12, rtol=0
            )
    assert (
        max(value for key, value in coupled.gates.items() if key.endswith("Ha"))
        <= 1e-11
    )


def independent_scalar_ci(native, mode, a, coupled):
    """Native jj CSF angular conventions are independent of the CG pair bridge."""
    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    from atomx.config.csf import get_shared_orbital_basis
    from atomx.methods.ci_sd import plan_sd_ci_space
    from atomx.physics.csf.matrix import build_ci_matrix

    core, od = native
    csfs, primary = plan_sd_ci_space(
        ("1s2",), ("2s", "2p"), J2=0, parity=1
    ).materialize()
    csfs = (csfs[primary[0]],) + tuple(c for i, c in enumerate(csfs) if i != primary[0])
    keys = get_shared_orbital_basis(csfs)
    energies = {
        label[:2]: integrals(native).fock_Ha[i, i].real
        for i, label in enumerate(a.labels)
    }
    ci = build_ci_matrix(
        csfs,
        od,
        keys,
        core._bsp_data,
        4,
        30,
        nuclear_potential=core._nuclear_potential,
        include_breit=mode != "dc",
        breit_orbital_energies=energies if mode == "dynamic" else None,
    )
    sorted_labels = tuple(
        (n, k, m) for n, k in sorted(keys) for m in range(-twoj(k), twoj(k) + 1, 2)
    )
    permutation = [a.labels.index(label) for label in sorted_labels]
    masks, _ = determinant_matrix(a, 2)
    columns = []
    for csf in csfs:
        vector = {}
        for mask, c in csf_magnetic_vector(csf_signature(csf), 0, tuple(keys)):
            occupied = [permutation[i] for i in range(10) if mask & (1 << i)]
            inversions = sum(
                left > right
                for i, left in enumerate(occupied)
                for right in occupied[i + 1 :]
            )
            vector[sum(1 << i for i in occupied)] = c * (-1) ** inversions
        columns.append([vector.get(mask, 0) for mask in masks])
    csf_projection = np.asarray(columns).T
    pair_projection = np.zeros((45, len(coupled.channels[0, 1])))
    positions = {mask: i for i, mask in enumerate(masks)}
    for column, vector in enumerate(coupled.pair_vectors(0, 1)):
        for p, q, c in vector:
            pair_projection[positions[(1 << p) | (1 << q)], column] = c
    transfer = pair_projection.T @ csf_projection
    np.testing.assert_allclose(
        transfer.T @ transfer, np.eye(len(csfs)), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        transfer.T @ coupled.hamiltonian_block(0, 1) @ transfer, ci, atol=1e-11, rtol=0
    )
    return ci, transfer


def twoj(kappa):
    return 2 * abs(kappa) - 1


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_independent_csf_offdiagonal_normal_ordering_and_full_scalar_sd(native, mode):
    a, coupled = checked_case(native, mode)
    ci, transfer = independent_scalar_ci(native, mode, a, coupled)
    result = solve_two_electron_core_sd(
        coupled, reference_identity=f"Be2+ {mode} finite anchor"
    )
    assert len(result.amplitudes) == 4  # Independent exact scalar counting.
    assert result.equation_residual_Ha < 1e-12
    assert np.max(abs(result.singles_product_subtraction_Ha)) > 1e-4
    # The actual noncanonical one-body offdiagonals survive; h and F differ.
    assert np.max(abs(a.fock_Ha[:2, 2:])) > 1e-4
    assert np.max(abs(a.one_body_Ha - a.fock_Ha)) > 1
    assert np.max(abs(ci - np.diag(ci.diagonal()))) > 1e-2
    np.testing.assert_allclose(
        transfer @ transfer.T, np.eye(len(ci)), atol=1e-12, rtol=0
    )


def test_phase_covariance_immutable_owner_and_whole_space_rejections(native):
    core, od = native
    original, coupled = checked_case(native, "dynamic")
    changed = dict(od)
    changed[2, -1] = tuple(-x for x in od[2, -1])
    from atomx.physics.csf.spinor_integrals import build_spinor_integrals

    energies = {
        label[:2]: integrals(native).fock_Ha[i, i].real
        for i, label in enumerate(original.labels)
    }
    moved = build_spinor_integrals(
        changed,
        core._bsp_data,
        4,
        30,
        occupied_keys=[(1, -1)],
        virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential,
        include_breit=True,
        breit_orbital_energies=energies,
    )
    result = build_j_coupled_sd_interaction(moved)
    for block, rows in result.channels.items():
        phases = np.array([(-1) ** sum(key == (2, -1) for key in row) for row in rows])
        np.testing.assert_allclose(
            result.blocks_Ha[block],
            phases[:, None] * coupled.blocks_Ha[block] * phases,
            atol=1e-12,
            rtol=0,
        )
        with pytest.raises(ValueError):
            result.blocks_Ha[block].setflags(write=True)
    assert result.identity["input_sha256"] != coupled.identity["input_sha256"]
    with patch.object(
        SpinorIntegrals,
        "eri",
        side_effect=AssertionError("preflight must precede evaluation"),
    ):
        with pytest.raises(MemoryError, match="memory cap"):
            build_j_coupled_sd_interaction(original, memory_bytes=1)
        with pytest.raises(MemoryError, match="work cap"):
            build_j_coupled_sd_interaction(original, work_cap=1)
    with pytest.raises(TypeError):
        plan_j_coupled_sd_interaction(object())
    with pytest.raises(ValueError):
        coupled.pair_vectors(0, 1, twom=2)


def test_nonspherical_onebody_rejects_without_projection_or_fitting(native):
    a = integrals(native)
    h = a.one_body_Ha.copy()
    f = a.fock_Ha.copy()
    h[2, 2] += 0.1
    f[2, 2] += 0.1
    bad = SpinorIntegrals(
        a.labels,
        a.noccupied,
        h,
        f,
        a.packed_eri_Ha,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        a.identity,
    )
    with pytest.raises(ValueError, match="reconstruction failed"):
        build_j_coupled_sd_interaction(bad)


def test_complex_multiplet_phase_keeps_full_interference(native):
    a, original = checked_case(native, "dynamic")
    n = len(a.labels)
    phases = np.array(
        [np.exp(0.37j) if label[:2] == (2, -1) else 1.0 for label in a.labels]
    )
    packed = np.array(
        [
            phases[p].conjugate()
            * phases[q].conjugate()
            * a.eri(p, q, r, s)
            * phases[r]
            * phases[s]
            for p in range(n - 1)
            for q in range(p + 1, n)
            for r in range(p, n - 1)
            for s in range(r + 1, n)
        ]
    )
    one = phases.conj()[:, None] * a.one_body_Ha * phases
    fock = phases.conj()[:, None] * a.fock_Ha * phases
    for values in (one, fock):
        values[np.diag_indices(n)] = values.diagonal().real
    moved = SpinorIntegrals(
        a.labels,
        a.noccupied,
        one,
        fock,
        packed,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        dict(a.identity, phase="common 2s exp(0.37i)"),
    )
    coupled = build_j_coupled_sd_interaction(moved)
    for block, rows in coupled.channels.items():
        pair_phase = np.array(
            [np.exp(0.37j * sum(key == (2, -1) for key in row)) for row in rows]
        )
        np.testing.assert_allclose(
            coupled.blocks_Ha[block],
            pair_phase.conj()[:, None] * original.blocks_Ha[block] * pair_phase,
            atol=1e-12,
            rtol=0,
        )
    assert np.max(abs(coupled.fock_Ha.imag)) > 1e-5
    with pytest.raises(ValueError, match="real symmetric"):
        solve_two_electron_core_sd(
            coupled, reference_identity="complex finite SD is explicitly unsupported"
        )


def test_full_frozen_core_constant_and_onebody_counted_once(native):
    from atomx.physics.csf.spinor_integrals import build_spinor_integrals

    core, od = native
    energies = {
        label[:2]: integrals(native).fock_Ha[i, i].real
        for i, label in enumerate(integrals(native).labels)
    }
    original = integrals(native, include_breit=True, breit_orbital_energies=energies)
    masks, full = determinant_matrix(original, 4)
    retained = [i for i, mask in enumerate(masks) if mask & 3 == 3]
    frozen = build_spinor_integrals(
        od,
        core._bsp_data,
        4,
        30,
        occupied_keys=[(2, -1)],
        virtual_keys=[(2, 1), (2, -2)],
        frozen_keys=[(1, -1)],
        nuclear_potential=core._nuclear_potential,
        include_breit=True,
        breit_orbital_energies=energies,
    )
    coupled = build_j_coupled_sd_interaction(frozen)
    _, reconstructed = determinant_matrix(coupled, 2)
    np.testing.assert_allclose(
        reconstructed, full[np.ix_(retained, retained)], atol=1e-11, rtol=0
    )
    assert coupled.frozen_energy_Ha == frozen.frozen_energy_Ha
    assert abs(coupled.frozen_energy_Ha) > 10


def test_incomplete_magnetic_multiplet_rejects_before_eri_work(native):
    a = integrals(native)
    labels = tuple((3, 1, -1) if i == 4 else label for i, label in enumerate(a.labels))
    partial = SpinorIntegrals(
        labels,
        a.noccupied,
        a.one_body_Ha,
        a.fock_Ha,
        a.packed_eri_Ha,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        a.identity,
    )
    with (
        patch.object(
            SpinorIntegrals,
            "eri",
            side_effect=AssertionError("angular preflight must precede ERIs"),
        ),
        pytest.raises(ValueError, match="complete magnetic multiplets"),
    ):
        build_j_coupled_sd_interaction(partial)


def test_resource_plan_counts_before_radial_pair_channel_materialization(native):
    a = integrals(native)
    with patch(
        "atomx.methods.sd_transverse.combinations_with_replacement",
        side_effect=AssertionError("large pair layout must follow resource acceptance"),
    ):
        plan = plan_j_coupled_sd_interaction(a)
        assert plan["j_pair_coordinates"] == 17
        assert plan["j_block_entries"] == 57
        with pytest.raises(MemoryError, match="memory cap"):
            build_j_coupled_sd_interaction(a, memory_bytes=1)
