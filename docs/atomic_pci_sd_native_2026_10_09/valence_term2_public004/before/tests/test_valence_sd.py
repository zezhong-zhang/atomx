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
    with pytest.raises(ValueError, match="distinct valence"):
        plan_valence_sd(core, [(1, -1)])
    with pytest.raises(ValueError, match="distinct valence"):
        plan_valence_sd(core, [(2, -1), (2, -1)])


@pytest.fixture(params=("original", "streamed_blas"))
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
    if request.param == "streamed_blas":
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
    val_plan = plan_valence_sd(plan, virtual)
    directory = tmp_path / "native"
    builder.build_valence(
        source,
        directory,
        plan["dimensions"],
        compiler,
        nxv=val_plan["nxv"],
        nvh=val_plan["nvh"],
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
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
    saved = core.memory_bytes
    core.memory_bytes = problem.plan["complete_payload_bound_bytes"] - 1
    try:
        with pytest.raises(MemoryError, match="Krylov response"):
            problem.solve(problem.plan["valence_keys"][0], -0.1)
    finally:
        core.memory_bytes = saved
