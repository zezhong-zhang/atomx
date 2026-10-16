"""Standalone reduced SD against independent finite primary/Fock-space anchors."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from test_monovalent_sd import dense_linked, full_operator

from atomx.angular.sd_rho import rho_coefficient
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.monovalent_sd import FiniteMonovalentSD
from atomx.methods.reduced_monovalent_sd import (
    ReducedMonovalentSD,
    plan_reduced_monovalent_sd,
)
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


@pytest.fixture(
    scope="module", params=("original", "streamed", "rotated", "occupied_rotated")
)
def reduced_core(tmp_path_factory, request):
    source, compiler = (
        os.environ.get("ATOMX_PCI_SOURCE"),
        os.environ.get("ATOMX_PCI_COMPILER"),
    )
    if not source or not compiler:
        pytest.skip("pinned pCI source/compiler required for reduced standalone SD")
    options = {}
    if request.param == "streamed":
        blas = os.environ.get("ATOMX_PCI_BLAS")
        if not blas:
            pytest.skip("pinned OpenMP BLAS required for complete streamed gate")
        options = dict(
            blas_library=blas,
            term21_threads=2,
            term21_mode="streamed",
            core_singles_blas=True,
            core_single_to_doubles_blas=True,
        )
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    try:
        spec = importlib.util.spec_from_file_location(
            "standalone_builder", tools / "build_pci_monovalent_backend.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
    finally:
        sys.path.remove(str(tools))
    radial = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    occupied, virtual = ((1, -1),), ((2, -1), (3, -1), (2, 1), (2, -2))
    orbitals = {(1, -1): radial.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            radial, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[: 2 if k == -1 else 1]):
            orbitals[i + 2, k] = state[:2]
    if request.param == "rotated":
        a, b = orbitals[2, -1], orbitals[3, -1]
        c, s = np.cos(0.37), np.sin(0.37)
        orbitals[2, -1] = tuple(c * x + s * y for x, y in zip(a, b, strict=True))
        orbitals[3, -1] = tuple(-s * x + c * y for x, y in zip(a, b, strict=True))
    if request.param == "occupied_rotated":
        # Changes the core projector deliberately; compare only the same new H.
        a, b = orbitals[1, -1], orbitals[2, -1]
        c, s = np.cos(0.08), np.sin(0.08)
        orbitals[1, -1] = tuple(c * x + s * y for x, y in zip(a, b, strict=True))
        orbitals[2, -1] = tuple(-s * x + c * y for x, y in zip(a, b, strict=True))
    core_options = {k: v for k, v in options.items() if k != "blas_library"}
    features = PolynomialCellCoulombFeatures(radial._bsp_data)
    plan = plan_core_sd(occupied, virtual, features.identity, **core_options)
    val = plan_valence_sd(
        plan,
        virtual,
        valence_term2_blas=request.param == "streamed",
        valence_term2_threads=2 if request.param == "streamed" else 1,
    )
    standalone = plan_reduced_monovalent_sd(
        plan,
        virtual,
        valence_term2_blas=request.param == "streamed",
        valence_term2_threads=2 if request.param == "streamed" else 1,
    )
    assert standalone["complete_payload_bound_bytes"] < 256 * 1024**2
    directory = (
        tmp_path_factory.mktemp("reduced-monovalent-" + request.param) / "backend"
    )
    builder.build_monovalent(
        source,
        directory,
        plan["dimensions"],
        compiler,
        nxv=val["nxv"],
        nvh=val["nvh"],
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
        valence_term2_blas=request.param == "streamed",
        valence_term2_threads=2 if request.param == "streamed" else 1,
        **options,
    )
    core = CoreSD(
        orbitals,
        radial._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=directory,
        nuclear_potential=radial._nuclear_potential,
        radial_representation="polynomial",
        memory_bytes=256 * 1024**2,
        **core_options,
    )
    result = core.solve(residual_tolerance_Ha=1e-12)
    return core, result, ReducedMonovalentSD(core, result, valence_keys=virtual)


def expanded(problem, key, vector, finite):
    """Decode every direct rho coordinate, then particle antisymmetrize.

    This representation bridge does not consume a native map, its source,
    denominators, nonlinear solution or energy estimate.
    """
    core = problem.core
    labels, groups = core.integrals.labels, core.plan["groups"]
    virtual = core.plan["virtual_keys"]
    singles = problem.plan["single_keys"][key]
    positions = {
        row: (p, group.index(row)) for p, group in enumerate(groups) for row in group
    }
    tables, cursor = {}, len(singles)
    for b, im, jn, rank in problem.plan["channels"][key]:
        ms = [row for row in groups[im] if row in virtual]
        ns = [row for row in groups[jn] if row in virtual]
        size = len(ms) * len(ns)
        table = np.asarray(vector[cursor : cursor + size]).reshape(
            (len(ms), len(ns)), order="F"
        )
        tables[b, im, jn, rank] = {
            (m, n): table[i, j] for i, m in enumerate(ms) for j, n in enumerate(ns)
        }
        cursor += size
    assert cursor == len(vector)

    def rho(m, n, b):
        im, _ = positions[labels[m][:2]]
        jn, _ = positions[labels[n][:2]]
        core_index = core.plan["occupied_keys"].index(labels[b][:2])
        total = 0.0
        for rank in range(core.plan["dimensions"]["KK"] + 1):
            table = tables.get((core_index, im, jn, rank))
            if table is not None:
                total += (
                    rho_coefficient(
                        rank,
                        labels[m],
                        labels[n],
                        labels[finite.valence_index],
                        labels[b],
                    )
                    * table[labels[m][:2], labels[n][:2]]
                )
        return total

    result = []
    for particles, holes, _ in finite.excitations:
        if len(particles) == 1:
            row = labels[particles[0]][:2]
            # A scalar standalone wave operator has no different-J singles.
            result.append(vector[singles.index(row)] if row in singles else 0.0)
        else:
            m, n = particles
            result.append(rho(m, n, holes[1]) - rho(n, m, holes[1]))
    return np.array(result)


MAGNETIC_COLUMNS = [
    (key, m)
    for key in ((2, -1), (3, -1), (2, 1), (2, -2))
    for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2)
]


@pytest.mark.parametrize("key,m", MAGNETIC_COLUMNS)
def test_complete_reduced_signed_map_against_fock_and_primary_anchor(
    reduced_core, key, m
):
    core, result, reduced = reduced_core
    v = core.integrals.labels.index((*key, m))
    finite = FiniteMonovalentSD(core, result, v, work_cap=200_000_000)
    vector = (
        np.random.default_rng(962341).normal(size=reduced.plan["amplitudes"][key])
        * 0.013
    )
    physical = expanded(reduced, key, vector, finite)
    energy, rhs = reduced.evaluate(key, vector)
    expected, _ = dense_linked(finite, physical)
    actual = expanded(reduced, key, rhs, finite) + finite.diagonal_gaps_Ha * physical
    np.testing.assert_allclose(
        energy + finite.orbital_energy_Ha, expected[0], rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(actual, expected[1:], rtol=0, atol=1e-11)
    zero = np.zeros_like(vector)
    constant, source = reduced.evaluate(key, zero)
    linear_e, linear_rhs = reduced.evaluate(key, vector, linear=True)
    np.testing.assert_allclose(energy, constant + linear_e, rtol=0, atol=1e-11)
    np.testing.assert_allclose(rhs, source + linear_rhs, rtol=0, atol=1e-11)
    # Core actions mutate COMMON; independent calls restore the actual core.
    core.action(result.amplitudes)
    again, again_rhs = reduced.evaluate(key, vector)
    assert again == energy
    np.testing.assert_array_equal(again_rhs, rhs)


@pytest.mark.parametrize("key,m", MAGNETIC_COLUMNS)
def test_reduced_feedback_root_matches_complete_finite_reference(reduced_core, key, m):
    core, result, reduced = reduced_core
    v = core.integrals.labels.index((*key, m))
    finite = FiniteMonovalentSD(core, result, v, work_cap=200_000_000).solve()
    actual = reduced.solve(key, residual_tolerance_Ha=1e-12)
    np.testing.assert_allclose(
        actual.correlation_energy_Ha, finite.correlation_energy_Ha, rtol=0, atol=1e-11
    )
    problem = finite._owner
    physical = expanded(reduced, key, actual.amplitudes, problem)
    np.testing.assert_allclose(physical, finite.amplitudes, rtol=0, atol=1e-10)
    independent, (masks, _, tc, sv) = dense_linked(problem, physical)
    raw = independent[1:] - actual.correlation_energy_Ha * physical
    assert np.max(abs(raw)) < 1e-10
    assert abs(independent[0] - actual.attachment_energy_Ha) < 1e-10
    assert actual.residual_max_Ha < 1e-12 and actual.energy_residual_Ha < 1e-12
    assert len(actual.continuation_path) == 8
    assert not actual.amplitudes.flags.writeable
    # Complete reduced scalar waves must retain pure J in every M, including
    # noncanonical occupied/virtual reference controls.
    labels = core.integrals.labels
    raising = []
    for p, (nr, k, projection) in enumerate(labels):
        target = (nr, k, projection + 2)
        if target in labels:
            j, mj = abs(k) - 0.5, projection / 2
            raising.append(
                (
                    np.sqrt(j * (j + 1) - mj * (mj + 1)),
                    ((labels.index(target), True), (p, False)),
                )
            )
    _, jp = full_operator(len(labels), core.integrals.noccupied + 1, raising)
    psi = (np.eye(len(masks)) + tc + sv)[:, masks.index(problem.reference_mask)]
    psi /= np.linalg.norm(psi)
    j, mj = abs(key[1]) - 0.5, m / 2
    assert np.max(abs(jp.T @ jp @ psi + (mj * (mj + 1) - j * (j + 1)) * psi)) < 1e-9


def test_preflight_identity_and_original_fixed_energy_consumer(reduced_core):
    core, result, reduced = reduced_core
    key = (2, -1)
    old_cap = core.memory_bytes
    core.memory_bytes = 1
    try:
        with (
            patch.object(
                reduced.valence,
                "_initialize",
                side_effect=AssertionError("preflight too late"),
            ),
            pytest.raises(MemoryError, match="complete reduced"),
        ):
            reduced.solve(key)
    finally:
        core.memory_bytes = old_cap
    core.memory_bytes = reduced.plan["complete_payload_bound_bytes"] - 1
    try:
        with (
            patch.object(
                core,
                "action",
                side_effect=AssertionError("parent action before preflight"),
            ),
            pytest.raises(MemoryError, match="complete reduced"),
        ):
            ReducedMonovalentSD(
                core,
                replace(result, total_energy_Ha=result.total_energy_Ha + 0.01),
                valence_keys=reduced.plan["valence_keys"],
            )
    finally:
        core.memory_bytes = old_cap
    with pytest.raises(MemoryError, match="Arnoldi"):
        reduced.solve(key, krylov_vectors=1)
    with pytest.raises(ValueError, match="continuation"):
        reduced.solve(key, continuation=(0.5, 0.25, 1))
    with pytest.raises(ValueError, match="raw tolerance"):
        reduced.solve(key, residual_tolerance_Ha=1e-5)
    wrong = replace(result, correlation_energy_Ha=result.correlation_energy_Ha + 0.01)
    # A fake energy must not be accepted as a physical core attachment zero.
    with pytest.raises(ValueError, match="core|energy"):
        ReducedMonovalentSD(core, wrong, valence_keys=(key,))
    original = ValenceSD(core, result, valence_keys=(key,))
    i = core.integrals.labels.index((2, -1, -1))
    parameter = core.integrals.fock_Ha[i, i].real + 0.075
    before = original.solve(key, parameter, residual_tolerance_Ha=1e-11)
    reduced.solve(key, residual_tolerance_Ha=1e-12)
    after = original.solve(key, parameter, residual_tolerance_Ha=1e-11)
    np.testing.assert_array_equal(after.numerators, before.numerators)
    np.testing.assert_array_equal(after.sigma1_Ha, before.sigma1_Ha)
