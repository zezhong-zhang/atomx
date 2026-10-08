"""Shared radial integrals preserve Hamiltonians and reject stale snapshots."""

import copy

import numpy as np
import pytest

from atomx import Atom, Basis, CIChargeStateAverage, CIStateAverage, Correlation
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci import ci_solve
from atomx.physics.csf.matrix import GridCIWorkspace, build_ci_matrix
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot


@pytest.fixture(scope="module")
def seed():
    return (
        Atom(
            10,
            config=["1s2 2s1", "1s1 2s1 2p1"],
            basis=Basis(ns=40, ks=9),
            corrections="none",
            correlation=Correlation(
                state_average=CIChargeStateAverage(
                    tuple((n, 1, CIStateAverage()) for n in (3, 2, 1))
                )
            ),
        )
        .solve(tol=1e-6)
        .result
    )


def inputs(r):
    return (
        r.orbitals_dict,
        tuple(sorted(r.orbitals_dict)),
        r.bsp_data,
        r.Z,
        r.bsp_data["ns"],
    )


def test_cross_charge_reuse_is_bitwise_exact_and_keeps_signed_values(seed):
    args = inputs(seed)
    snap = CIRadialSnapshot(
        args[0], args[2], args[3], args[4], nuclear_potential=seed._nuclear_potential
    )
    for n in (3, 2, 1):
        csfs = seed.csfs if n == 3 else enumerate_jj_orbital_space(args[1], n)
        ordinary = build_ci_matrix(
            csfs, *args, nuclear_potential=seed._nuclear_potential
        )
        cached = build_ci_matrix(
            csfs, *args, nuclear_potential=seed._nuclear_potential, radial_snapshot=snap
        )
        np.testing.assert_array_equal(ordinary, cached)
        a = ci_solve(csfs, *args, nuclear_potential=seed._nuclear_potential)
        b = ci_solve(
            csfs, *args, nuclear_potential=seed._nuclear_potential, radial_snapshot=snap
        )
        np.testing.assert_array_equal(a.state_energies, b.state_energies)
        np.testing.assert_array_equal(a.mixing_coefficients, b.mixing_coefficients)
    before = snap.diagnostics()
    build_ci_matrix(
        seed.csfs,
        *args,
        nuclear_potential=seed._nuclear_potential,
        radial_snapshot=snap,
    )
    after = snap.diagnostics()
    assert after["radial_cache_misses"] == before["radial_cache_misses"]
    assert after["radial_cache_hits"] > before["radial_cache_hits"]
    evaluate = snap.evaluator_for(
        args[0], args[2], args[3], args[4], nuclear_potential=seed._nuclear_potential
    )
    assert evaluate(("h", 1, -1, 1)) < 0
    assert after["radial_cache_entries"] <= after["radial_cache_capacity"]


def test_snapshot_is_immutable_and_rejects_orbital_geometry_Z_and_nucleus_mutation(
    seed,
):
    orbitals, _, ws, Z, ns = inputs(seed)
    local = {key: tuple(v.copy() for v in wave) for key, wave in orbitals.items()}
    geometry = copy.deepcopy(ws)
    scale = [1.0]

    def nuclear(r):
        return scale[0] * seed._nuclear_potential(r)

    snap = CIRadialSnapshot(local, geometry, Z, ns, nuclear_potential=nuclear)
    evaluate = snap.evaluator_for(local, geometry, Z, ns, nuclear_potential=nuclear)
    original = evaluate(("h", 1, -1, 1))
    from atomx.physics.csf.matrix import _radial_evaluator

    reference = _radial_evaluator(orbitals, ws, Z, ns, nuclear_potential=nuclear)
    off_diagonal = reference(("h", 1, -1, 2))
    # Force an uncached one-body evaluation after caller mutation: frozen copy.
    local[1, -1][0][:] *= 0.9
    assert evaluate(("h", 1, -1, 2)) == off_diagonal
    with pytest.raises(ValueError, match="snapshot does not match"):
        snap.evaluator_for(local, geometry, Z, ns, nuclear_potential=nuclear)
    local = {key: tuple(v.copy() for v in wave) for key, wave in orbitals.items()}
    geometry["bsp_p"][0, 0, 0] += 1e-9
    with pytest.raises(ValueError, match="snapshot does not match"):
        snap.evaluator_for(local, geometry, Z, ns, nuclear_potential=nuclear)
    geometry = copy.deepcopy(ws)
    for z, size in ((Z + 1, ns), (Z, ns + 1)):
        with pytest.raises(ValueError, match="snapshot does not match"):
            snap.evaluator_for(local, geometry, z, size, nuclear_potential=nuclear)
    scale[0] = 1.1
    with pytest.raises(ValueError, match="snapshot does not match"):
        snap.evaluator_for(local, geometry, Z, ns, nuclear_potential=nuclear)
    assert evaluate(("h", 1, -1, 1)) == original
    with pytest.raises(ValueError):
        snap._orbitals[1, -1][0][0] = 0


def test_per_build_effective_shifts_do_not_pollute_shared_cache(seed):
    args = inputs(seed)
    snap = CIRadialSnapshot(
        args[0], args[2], args[3], args[4], nuclear_potential=seed._nuclear_potential
    )
    base = build_ci_matrix(
        seed.csfs,
        *args,
        nuclear_potential=seed._nuclear_potential,
        radial_snapshot=snap,
    )
    shifts = {(1, -1): 0.2, (2, -1): -0.1}
    expected = build_ci_matrix(
        seed.csfs,
        *args,
        nuclear_potential=seed._nuclear_potential,
        one_body_shifts=shifts,
    )
    actual = build_ci_matrix(
        seed.csfs,
        *args,
        nuclear_potential=seed._nuclear_potential,
        one_body_shifts=shifts,
        radial_snapshot=snap,
    )
    np.testing.assert_array_equal(actual, expected)
    replay = build_ci_matrix(
        seed.csfs,
        *args,
        nuclear_potential=seed._nuclear_potential,
        radial_snapshot=snap,
    )
    np.testing.assert_array_equal(replay, base)
    assert abs(actual - base).max() > 0.1


def test_owned_solve_survives_external_mutation_with_exact_nuclear_operator(seed):
    orbitals, keys, ws, Z, ns = inputs(seed)
    local = {key: tuple(v.copy() for v in wave) for key, wave in orbitals.items()}
    geometry = copy.deepcopy(ws)
    scale = [1.0]

    def nuclear(radius):
        return scale[0] * seed._nuclear_potential(radius)

    baseline = ci_solve(
        seed.csfs, local, keys, geometry, Z, ns, nuclear_potential=nuclear
    )
    snap = CIRadialSnapshot(local, geometry, Z, ns, nuclear_potential=nuclear)
    local[1, -1][0][:] *= 0.5
    geometry["gr"][:] *= 0.9
    scale[0] = 2.0
    # The first owned solve also exercises cold one-body/nuclear assembly.
    replay = snap.solve(seed.csfs, keys)
    np.testing.assert_array_equal(replay.state_energies, baseline.state_energies)
    np.testing.assert_array_equal(
        replay.mixing_coefficients, baseline.mixing_coefficients
    )
    with pytest.raises(ValueError, match="snapshot does not match"):
        snap.evaluator_for(local, geometry, Z, ns, nuclear_potential=nuclear)
    with pytest.raises(TypeError):
        snap.solve(seed.csfs, keys, nuclear_potential=nuclear)


def test_breit_and_matrix_free_builds_preserve_their_own_operator(seed):
    args = inputs(seed)
    snap = CIRadialSnapshot(
        args[0], args[2], args[3], args[4], nuclear_potential=seed._nuclear_potential
    )
    options = dict(
        nuclear_potential=seed._nuclear_potential,
        include_breit=True,
        one_body_shifts={(1, -1): 0.1},
    )
    ordinary = build_ci_matrix(seed.csfs, *args, **options)
    cached = build_ci_matrix(seed.csfs, *args, **options, radial_snapshot=snap)
    np.testing.assert_array_equal(cached, ordinary)
    a = build_ci_matrix(seed.csfs, *args, **options, as_operator=True)
    b = build_ci_matrix(
        seed.csfs, *args, **options, as_operator=True, radial_snapshot=snap
    )
    vector = np.linspace(-1, 1, len(seed.csfs))
    np.testing.assert_array_equal(a @ vector, b @ vector)
    np.testing.assert_allclose(b @ vector, ordinary @ vector, rtol=0, atol=1e-12)


def test_grid_snapshot_rejected():
    grid = GridCIWorkspace(np.linspace(0, 10, 10), np.ones(10), -np.ones(10))
    with pytest.raises(ValueError, match="B-spline workspace"):
        CIRadialSnapshot({}, grid, 1, 10)


def test_sampled_nuclear_operator_is_exact_and_rejects_other_grids(seed):
    from atomx.numerics.bspline.integrals import SampledNuclearPotential, nuclear_matrix

    ws = seed.bsp_data
    potential = seed._nuclear_potential
    values = np.array([potential(r) for r in ws["gr"].ravel()]).reshape(ws["gr"].shape)
    sampled = SampledNuclearPotential(ws["gr"], values)
    args = (ws["nv"], ws["ksp"], ws["ns"], ws["bsp_p"], ws["gr"], ws["grw"], seed.Z)
    np.testing.assert_array_equal(
        nuclear_matrix(*args, sampled), nuclear_matrix(*args, potential)
    )
    values[:] = 0
    assert sampled(ws["gr"][0, 0]) == potential(ws["gr"][0, 0])
    with pytest.raises(ValueError, match="grid mismatch"):
        sampled.on_grid(ws["gr"] * 0.9)
    with pytest.raises(KeyError):
        sampled(-1.0)
