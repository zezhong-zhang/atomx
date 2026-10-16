"""Live complete native fixed-energy valence API and physical-input guards."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


def test_complete_valence_plan_rejects_unbound_columns_and_counts_full_pool():
    # Resource planning needs only the bound feature dimensions, no orbitals.
    geometry = dict(
        auxiliary_nodes=32,
        quadrature_nodes=(3, 4),
        same_cell_edges=30,
        coefficient_sizes=(12, 12),
        component_orders=(4, 4),
        payload_estimate_bytes=2048,
    )
    core = plan_core_sd([(1, -1)], [(2, -1), (2, 1), (2, -2)], geometry)
    valence = plan_valence_sd(core, core["virtual_keys"])
    assert (
        valence["complete_payload_bound_bytes"] > core["complete_payload_bound_bytes"]
    )
    assert valence["nxv"] > 0 and valence["nvh"] > 0
    assert all(
        valence["raw_amplitudes"][k] == len(valence["channels"][k])
        for k in valence["valence_keys"]
    )
    streamed = plan_valence_sd(core, core["virtual_keys"], valence_term2_blas=True)
    assert streamed["valence_term2_scratch_bytes"] > 0
    assert streamed["complete_payload_bound_bytes"] == (
        valence["complete_payload_bound_bytes"]
        + streamed["valence_term2_scratch_bytes"]
    )
    parallel = plan_valence_sd(
        core, core["virtual_keys"], valence_term2_blas=True, valence_term2_threads=8
    )
    d = core["dimensions"]
    assert parallel["valence_term2_scratch_bytes"] == (
        streamed["valence_term2_scratch_bytes"]
        + 7 * 8 * (d["NHF"] * (4 * d["NXX"] + 2) + 4 * d["NXX"] ** 2)
    )
    for workers in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="worker count"):
            plan_valence_sd(
                core,
                core["virtual_keys"],
                valence_term2_blas=True,
                valence_term2_threads=workers,
            )
    with pytest.raises(ValueError, match="original path"):
        plan_valence_sd(core, core["virtual_keys"], valence_term2_threads=8)
    with pytest.raises(ValueError, match="distinct valence"):
        plan_valence_sd(core, [(1, -1)])
    with pytest.raises(ValueError, match="distinct valence"):
        plan_valence_sd(core, [(2, -1), (2, -1)])


@pytest.fixture(
    params=(
        "original",
        "streamed_blas",
        "streamed_valence",
        "streamed_valence_parallel",
    )
)
def live_valence(tmp_path, request):
    source, compiler = (
        os.environ.get("ATOMX_PCI_SOURCE"),
        os.environ.get("ATOMX_PCI_COMPILER"),
    )
    if not source or not compiler:
        pytest.skip(
            "pinned pCI source and compiler required for live valence equations"
        )
    options = {}
    if request.param != "original":
        blas = os.environ.get("ATOMX_PCI_BLAS")
        if not blas:
            pytest.skip("bound OpenMP BLAS required for optimized valence ABI check")
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
            "valence_builder", tools / "build_pci_valence_backend.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
    finally:
        sys.path.remove(str(tools))
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    occupied = ((1, -1),)
    virtual = ((2, -1), (2, 1), (2, -2))
    orbitals = {key: core.orbitals_by_subshell()[key][:2] for key in occupied}
    for key in virtual:
        _, states = build_virtual_basis(
            core, key[1], e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        orbitals[key] = states[0][:2]
    feature = PolynomialCellCoulombFeatures(core._bsp_data)
    plan = plan_core_sd(
        occupied,
        virtual,
        feature.identity,
        **{k: v for k, v in options.items() if k != "blas_library"},
    )
    assert plan["dimensions"]["NX"] > plan["dimensions"]["NXX"]
    term2_blas = request.param.startswith("streamed_valence")
    term2_threads = 8 if request.param == "streamed_valence_parallel" else 1
    val_plan = plan_valence_sd(
        plan,
        virtual,
        valence_term2_blas=term2_blas,
        valence_term2_threads=term2_threads,
    )
    directory = tmp_path / "native"
    builder.build_valence(
        source,
        directory,
        plan["dimensions"],
        compiler,
        nxv=val_plan["nxv"],
        nvh=val_plan["nvh"],
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
        valence_term2_blas=term2_blas,
        valence_term2_threads=term2_threads,
        **options,
    )
    problem = CoreSD(
        orbitals,
        core._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=directory,
        nuclear_potential=core._nuclear_potential,
        radial_representation="polynomial",
        **{k: v for k, v in options.items() if k != "blas_library"},
    )
    result = problem.solve(residual_tolerance_Ha=1e-11)
    return problem, result, ValenceSD(problem, result, valence_keys=virtual)


def test_live_complete_valence_initial_guesses_recertify_full_root_and_response(
    live_valence,
):
    core, _, problem = live_valence
    key = problem.plan["valence_keys"][0]
    i = next(i for i, label in enumerate(core.integrals.labels) if label[:2] == key)
    energy = float(core.integrals.fock_Ha[i, i].real + 0.075)
    reference = problem.solve(key, energy, residual_tolerance_Ha=1e-11)
    rng = np.random.default_rng(741041)
    initial = reference.numerators + rng.normal(size=reference.numerators.size) * 0.01
    prime = reference.numerator_derivative + rng.normal(size=initial.size) * 0.01
    snapshots = []
    restored = problem.solve(
        key,
        energy,
        residual_tolerance_Ha=1e-11,
        initial_numerators=initial,
        initial_numerator_derivative=prime,
        iteration_callback=snapshots.append,
    )
    assert restored.residual_max_Ha < 1e-11
    assert restored.derivative_residual_max_Ha < 1e-11
    for name in (
        "numerators",
        "numerator_derivative",
        "sigma1_Ha",
        "sigma1_derivative",
    ):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(reference, name), rtol=0, atol=1e-11
        )
    assert all(restored.provenance["initial_guess_sha256"])
    assert restored.provenance["iteration_snapshots"] == len(snapshots)
    assert {row["stage"] for row in snapshots} == {"root", "response"}
    for row in snapshots:
        assert row["key"] == key and row["energy_Ha"] == energy
        assert row["certified"] is False
        assert row["numerators"].shape == initial.shape
        assert not row["numerators"].flags.writeable
        assert row["packed_coordinates_sha256"] and row["denominators_sha256"]
        assert row["valence_source_sha256"]
        with pytest.raises(ValueError):
            row["numerators"][0] = 0
    recovered = problem.solve(
        key,
        energy,
        residual_tolerance_Ha=1e-11,
        initial_numerators=next(
            row["numerators"] for row in snapshots if row["stage"] == "root"
        ),
        initial_numerator_derivative=next(
            row["numerators"] for row in snapshots if row["stage"] == "response"
        ),
    )
    np.testing.assert_allclose(
        recovered.numerators, reference.numerators, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        recovered.numerator_derivative,
        reference.numerator_derivative,
        rtol=0,
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="callable iteration"):
        problem.solve(key, energy, iteration_callback=False)

    def interrupted(row):
        raise RuntimeError("checkpoint consumer interruption")

    with pytest.raises(RuntimeError, match="consumer interruption"):
        problem.solve(
            key, energy, initial_numerators=initial, iteration_callback=interrupted
        )
    for bad in (initial[:-1], initial.astype(complex), initial * np.nan):
        with pytest.raises(ValueError, match="finite complete real numerator"):
            problem.solve(key, energy, initial_numerators=bad)
        with pytest.raises(ValueError, match="finite complete real numerator"):
            problem.solve(key, energy, initial_numerator_derivative=bad)
    with pytest.raises(ValueError, match="requires requested response"):
        problem.solve(
            key, energy, solve_response=False, initial_numerator_derivative=prime
        )
    np.testing.assert_array_equal(
        initial,
        reference.numerators
        + np.random.default_rng(741041).normal(size=initial.size) * 0.01,
    )


def test_live_complete_valence_api_core_common_restore_and_energy_response(
    live_valence,
):
    core, core_result, problem = live_valence
    for key in problem.plan["valence_keys"]:
        labels = core.integrals.labels
        i = next(i for i, label in enumerate(labels) if label[:2] == key)
        energy = float(core.integrals.fock_Ha[i, i].real + 0.075)
        result = problem.solve(key, energy, residual_tolerance_Ha=1e-11)
        assert result.residual_max_Ha <= 1e-11
        assert result.derivative_residual_max_Ha <= 1e-11
        assert not result.numerators.flags.writeable
        assert not result.sigma1_derivative.flags.writeable
        assert result.provenance["forward_accuracy_certified"] is False
        assert result.provenance["equation_coverage"]["core_common_patches"] == 8
        before = problem.evaluate(
            key,
            energy,
            result.numerators,
            numerator_derivative=result.numerator_derivative,
        )
        group = core.plan["groups"][
            next(i for i, g in enumerate(core.plan["groups"]) if key in g)
        ]
        rows = [i for i, k in enumerate(group) if k in core.plan["virtual_keys"]]
        # The complete original affine derivative remains the independent
        # oracle for the solver's source-free response decomposition.
        np.testing.assert_allclose(
            result.sigma1_derivative, before[2][rows], atol=1e-11, rtol=0
        )
        core.action(core_result.amplitudes * 0.51)  # mutates all core COMMON arrays
        after = problem.evaluate(
            key,
            energy,
            result.numerators,
            numerator_derivative=result.numerator_derivative,
        )
        assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
        step = 1e-4
        plus = problem.solve(key, energy + step, solve_response=False)
        minus = problem.solve(key, energy - step, solve_response=False)
        derivative = (plus.sigma1_Ha - minus.sigma1_Ha) / (2 * step)
        np.testing.assert_allclose(
            result.sigma1_derivative, derivative, atol=1e-9, rtol=1e-6
        )
        with pytest.raises(ValueError, match="finite complete real"):
            problem.evaluate(key, energy, np.full_like(result.numerators, np.nan))
        with pytest.raises(ValueError, match="finite complete real"):
            problem.evaluate(key, energy, None)
        with pytest.raises(ValueError, match="finite energy"):
            problem.evaluate(key, energy + 1j, result.numerators)


def test_live_current_core_certificate_and_reconstructed_result_recheck(live_valence):
    core, result, problem = live_valence
    keys = problem.plan["valence_keys"]
    before = core._matvecs
    ValenceSD(core, result, valence_keys=keys)
    assert core._matvecs == before
    # A copied/archive-style result has no transferable producer certificate.
    ValenceSD(core, replace(result), valence_keys=keys)
    assert core._matvecs == before + 1
    assert not core._has_current_solution_certificate(
        result, result.residual_max_Ha * 0.5
    )
    for field in (
        "reference_energy_Ha",
        "correlation_energy_Ha",
        "total_energy_Ha",
        "residual_max_Ha",
        "residual_l2_Ha",
    ):
        saved = getattr(result, field)
        object.__setattr__(result, field, saved + 0.001)
        try:
            assert not core._has_current_solution_certificate(result, 1e-10)
        finally:
            object.__setattr__(result, field, saved)
    source = core.source
    core.source = source + 0.001
    try:
        with pytest.raises(ValueError, match="actual full raw residual"):
            ValenceSD(core, result, valence_keys=keys)
    finally:
        core.source = source
    denominator = core.denominators
    core.denominators = denominator + 0.001
    try:
        assert not core._has_current_solution_certificate(result, 1e-10)
    finally:
        core.denominators = denominator
    # Even bypassing frozen dataclass protection cannot preserve the proof.
    amplitudes = result.amplitudes
    object.__setattr__(result, "amplitudes", amplitudes + 1)
    try:
        with pytest.raises(ValueError, match="actual full raw residual"):
            ValenceSD(core, result, valence_keys=keys)
    finally:
        object.__setattr__(result, "amplitudes", amplitudes)
    original = core.action
    core.action = lambda value: original(value)
    try:
        before = core._matvecs
        ValenceSD(core, result, valence_keys=keys)
        assert core._matvecs == before + 1
    finally:
        del core.action
    resources = core.provenance["resources"]
    cached = resources["radial_pair_cache_size"]
    resources["radial_pair_cache_size"] += 1
    try:
        assert not core._has_current_solution_certificate(result, 1e-10)
    finally:
        resources["radial_pair_cache_size"] = cached


def test_live_valence_result_identity_and_complete_resource_rejection(live_valence):
    core, result, problem = live_valence
    wrong = replace(
        result, provenance=dict(result.provenance, hamiltonian="Dirac-Coulomb-Breit")
    )
    with pytest.raises(ValueError, match="identity differs"):
        ValenceSD(core, wrong, valence_keys=problem.plan["valence_keys"])
    bad = replace(result, amplitudes=result.amplitudes + 1)
    with pytest.raises(ValueError, match="actual full raw residual"):
        ValenceSD(core, bad, valence_keys=problem.plan["valence_keys"])
    if problem.plan["valence_term2_blas"]:
        manifest = core._manifest["valence_sd"]["term2"]
        scratch = manifest["scratch_bytes"]
        manifest["scratch_bytes"] += 8
        try:
            with pytest.raises(ValueError, match="source/scratch identity"):
                ValenceSD(core, result, valence_keys=problem.plan["valence_keys"])
        finally:
            manifest["scratch_bytes"] = scratch
        if problem.plan["valence_term2_threads"] > 1:
            runtime = core._blas_runtime
            core._blas_runtime = dict(runtime, threads=2)
            try:
                with pytest.raises(ValueError, match="inner-one OpenMP BLAS"):
                    ValenceSD(core, result, valence_keys=problem.plan["valence_keys"])
            finally:
                core._blas_runtime = runtime
    saved = core.memory_bytes
    core.memory_bytes = problem.plan["complete_payload_bound_bytes"] - 1
    try:
        with pytest.raises(MemoryError, match="Krylov response"):
            problem.solve(problem.plan["valence_keys"][0], -0.1)
    finally:
        core.memory_bytes = saved


def test_live_homogeneous_value_action_preserves_signed_values_and_response(
    live_valence,
):
    core, _, problem = live_valence
    rng = np.random.default_rng(814120)
    for key in problem.plan["valence_keys"]:
        i = next(i for i, label in enumerate(core.integrals.labels) if label[:2] == key)
        energy = float(core.integrals.fock_Ha[i, i].real + 0.075)
        count = problem.plan["raw_amplitudes"][key]
        numerator, prime = rng.normal(size=(2, count)) * 0.02
        with core._lock:
            context = problem._initialize(key, energy)
            full = problem._evaluate(energy, numerator, prime, context, linear=True)
            values = problem._evaluate_value(energy, numerator, context)
            assert len(values) == 2
            assert all(
                np.array_equal(a, b) for a, b in zip(values, full[:2], strict=True)
            )
            restored = problem._evaluate(energy, numerator, prime, context, linear=True)
            assert all(
                np.array_equal(a, b) for a, b in zip(full, restored, strict=True)
            )
            with pytest.raises(ValueError, match="finite complete real"):
                problem._evaluate_value(energy, numerator.astype(complex), context)
            with pytest.raises(ValueError, match="finite complete real"):
                problem._evaluate_value(energy, numerator[:-1], context)
