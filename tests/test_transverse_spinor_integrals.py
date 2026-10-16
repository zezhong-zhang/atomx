"""Full signed atomic transverse Hamiltonians across integral/CI/FSCC consumers."""

import itertools
import os

import numpy as np
import pytest
from test_spinor_integrals_fscc import integrals
from test_spinor_integrals_fscc import native as native_fixture

from atomx.methods.fscc import (
    atomic_integrals,
    compute,
    export_integrals,
    load_integrals,
)
from atomx.physics.csf.spinor_integrals import build_spinor_integrals

native = native_fixture


def energies(native):
    dc = integrals(native)
    return {label[:2]: dc.fock_Ha[i, i].real for i, label in enumerate(dc.labels)}


def apply(mask, operations):
    """Independent elementary fermion action, including all permutation signs."""
    sign = 1
    for bit, create in reversed(operations):
        if bool(mask & (1 << bit)) == create:
            return None, 0
        sign *= (-1) ** ((mask & ((1 << bit) - 1)).bit_count())
        mask ^= 1 << bit
    return mask, sign


def determinant_matrix(a, electrons):
    n = len(a.labels)
    masks = [
        sum(1 << i for i in c) for c in itertools.combinations(range(n), electrons)
    ]
    index = {mask: i for i, mask in enumerate(masks)}
    matrix = np.eye(len(masks), dtype=complex) * a.frozen_energy_Ha
    for column, mask in enumerate(masks):
        for p, q in itertools.product(range(n), repeat=2):
            target, sign = apply(mask, ((p, True), (q, False)))
            if sign:
                matrix[index[target], column] += sign * a.one_body_Ha[p, q]
        for p, q in itertools.combinations(range(n), 2):
            for r, s in itertools.combinations(range(n), 2):
                target, sign = apply(
                    mask, ((p, True), (q, True), (s, False), (r, False))
                )
                if sign:
                    matrix[index[target], column] += sign * a.eri(p, q, r, s)
    return masks, matrix


@pytest.mark.parametrize("finite", [False, True])
def test_complete_signed_many_electron_ci_and_zero_limit(native, finite):
    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.physics.csf.matrix import build_ci_matrix

    core, od = native
    mapping = energies(native) if finite else {key: 0.0 for key in od}
    exported = integrals(native, include_breit=True, breit_orbital_energies=mapping)
    masks, full = determinant_matrix(exported, 2)
    keys = tuple(od)
    csfs = enumerate_jj_csfs_many(
        ["1s2", "2s2", "2p2", "1s1 2s1", "1s1 2p1", "2s1 2p1"]
    )
    columns = []
    sorted_labels = tuple(
        (n, k, m)
        for n, k in sorted(keys)
        for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
    )
    permutation = [exported.labels.index(label) for label in sorted_labels]
    for csf in csfs:
        vector = {}
        for mask, coefficient in csf_magnetic_vector(
            csf_signature(csf), csf.total_J2, keys
        ):
            occupied = [
                permutation[i] for i in range(len(permutation)) if mask & (1 << i)
            ]
            inversions = sum(
                left > right
                for i, left in enumerate(occupied)
                for right in occupied[i + 1 :]
            )
            vector[sum(1 << i for i in occupied)] = coefficient * (-1) ** inversions
        columns.append([vector.get(mask, 0) for mask in masks])
    projection = np.asarray(columns).T
    np.testing.assert_allclose(projection.T @ projection, np.eye(len(csfs)), atol=1e-13)
    reference = build_ci_matrix(
        csfs,
        od,
        keys,
        core._bsp_data,
        4,
        30,
        nuclear_potential=core._nuclear_potential,
        include_breit=True,
        breit_orbital_energies=mapping if finite else None,
    )
    np.testing.assert_allclose(
        projection.T @ full @ projection, reference, atol=1e-11, rtol=0
    )
    # Actual signed off-diagonals and reference contractions are indispensable.
    assert np.max(abs(reference - np.diag(np.diag(reference)))) > 1e-2
    assert full[0, 0].real == pytest.approx(exported.reference_energy_Ha, abs=1e-12)
    np.testing.assert_allclose(full, full.conj().T, atol=1e-13)


def test_frozen_core_ledger_stream_replay_cache_eviction_and_owned_inputs(
    native, tmp_path
):
    core, original = native
    mapping = energies(native)
    full = integrals(native, include_breit=True, breit_orbital_energies=mapping)
    masks, matrix = determinant_matrix(full, 4)
    selected = [i for i, mask in enumerate(masks) if mask & 3 == 3]
    inputs = {k: tuple(x.copy() for x in pair) for k, pair in original.items()}
    options = dict(
        occupied_keys=[(2, -1)],
        virtual_keys=[(2, 1), (2, -2)],
        frozen_keys=[(1, -1)],
        nuclear_potential=core._nuclear_potential,
        include_breit=True,
        breit_orbital_energies=mapping,
    )
    packed = build_spinor_integrals(inputs, core._bsp_data, 4, 30, **options)
    stream = build_spinor_integrals(
        inputs,
        core._bsp_data,
        4,
        30,
        **options,
        representation="on_demand",
        transverse_cache_size=1,
    )
    inputs[2, 1][0][:] *= -1
    mapping[2, 1] += 100
    _, projected = determinant_matrix(stream, 2)
    np.testing.assert_allclose(
        projected, matrix[np.ix_(selected, selected)], atol=1e-11, rtol=0
    )
    np.testing.assert_array_equal(
        np.concatenate(list(stream.iter_packed_eri())), packed.packed_eri_Ha
    )
    export_integrals(
        stream, tmp_path / "saved", tensor_symmetry="axial", max_chunk_bytes=48
    )
    loaded = load_integrals(tmp_path / "saved")
    np.testing.assert_array_equal(loaded.packed_eri_Ha, packed.packed_eri_Ha)
    np.testing.assert_allclose(loaded.one_body_Ha, packed.one_body_Ha, atol=1e-13)
    assert (
        loaded.identity["frequency_prescription"]
        == "symmetric-two-vertex-orbital-differences"
    )
    for name, cache in stream.diagnostics()["transverse"]["caches"].items():
        assert cache["currsize"] <= cache["maxsize"]
        if name != "values":
            assert cache["maxsize"] == 1


def test_atomic_band_frequencies_and_unsupported_inputs(native):
    core, _ = native
    a = atomic_integrals(
        core,
        virtual_lmax=0,
        virtual_energy_max_Ha=1,
        include_breit=True,
        breit_frequency="orbital_differences",
        representation="on_demand",
        transverse_cache_size=2,
    )
    actual = {
        (row["n"], row["kappa"]): row["energy"]
        for row in a.identity["orbital_energies_Ha"]
    }
    assert actual[1, -1] == core.orbital_energies[core.subshells[0].label]
    assert len(actual) == len(a.identity["occupied_keys"]) + len(
        a.identity["virtual_keys"]
    )
    for key in a.identity["virtual_keys"]:
        assert actual[key] == a.identity["virtual_orbital_energies_Ha"][str(key)]
    with pytest.raises(ValueError, match="requires include_breit"):
        integrals(native, breit_orbital_energies=energies(native))
    with pytest.raises(ValueError, match="every orbital"):
        integrals(native, include_breit=True, breit_orbital_energies={(1, -1): -1.0})
    with pytest.raises(ValueError, match="finite-frequency Breit requires"):
        atomic_integrals(
            core,
            virtual_lmax=0,
            virtual_energy_max_Ha=1,
            breit_frequency="orbital_differences",
        )
    with pytest.raises(MemoryError, match="transverse cache payload"):
        integrals(
            native,
            include_breit=True,
            representation="on_demand",
            max_integral_bytes=20000,
        )


@pytest.mark.parametrize("tensor_symmetry", ["C1", "axial"])
def test_live_finite_frequency_fsccsdt_matches_full_three_electron_hamiltonian(
    native, tmp_path, tensor_symmetry
):
    executable = os.environ.get("ATOMX_EXPT_EXECUTABLE")
    if not executable:
        pytest.skip("set ATOMX_EXPT_EXECUTABLE for the actual EXP-T gate")
    a = integrals(
        native,
        include_breit=True,
        breit_orbital_energies=energies(native),
        representation="on_demand",
        transverse_cache_size=2,
    )
    _, full = determinant_matrix(a, 3)
    result = compute(
        a,
        executable=executable,
        directory=tmp_path / "fsccsdt",
        model="CCSDT",
        sector="0h1p",
        active_indices=range(2, 10),
        tile_size=4,
        tolerance=1e-12,
        residual_bound_tolerance_Ha=1e-9,
        tensor_symmetry=tensor_symmetry,
    )
    exact = np.linalg.eigvalsh(full)
    # Complete low model multiplets, including degenerate M roots; no nearest
    # energy relabeling, reference shift, root deletion or tolerance change.
    np.testing.assert_allclose(result.sector_energies_Ha, exact[:8], atol=1e-9, rtol=0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Retained EXP-T 0h1p nonzero-Fov defect: transverse_precision007/fov008",
)
def test_live_transverse_noncanonical_fsccsdt_stricter_fci_target(native, tmp_path):
    executable = os.environ.get("ATOMX_EXPT_EXECUTABLE")
    if not executable:
        pytest.skip("set ATOMX_EXPT_EXECUTABLE for the actual EXP-T gate")
    a = integrals(
        native,
        include_breit=True,
        breit_orbital_energies=energies(native),
        representation="on_demand",
        transverse_cache_size=2,
    )
    assert np.max(abs(a.fock_Ha[:2, 2:])) > 1e-4
    _, full = determinant_matrix(a, 3)
    result = compute(
        a,
        executable=executable,
        directory=tmp_path / "stricter_fsccsdt",
        model="CCSDT",
        sector="0h1p",
        active_indices=range(2, 10),
        tile_size=4,
        tolerance=1e-14,
        residual_bound_tolerance_Ha=1e-11,
    )
    defect = np.max(abs(result.sector_energies_Ha - np.linalg.eigvalsh(full)[:8]))
    assert defect < 1e-9  # Preserve the original finite Hamiltonian gate.
    assert defect < 1e-10  # Measured unfinished target; no tolerance relaxation.
