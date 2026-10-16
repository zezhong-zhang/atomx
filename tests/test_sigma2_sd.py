"""Live public directed screening, state response and native identity guards."""

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
from atomx.methods.sigma2_sd import Sigma2SD, plan_sigma2_sd
from atomx.methods.valence_sd import ValenceSD
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


def test_complete_screening_plan_and_no_undeclared_valence():
    geometry = dict(
        auxiliary_nodes=32,
        quadrature_nodes=(3, 4),
        same_cell_edges=30,
        coefficient_sizes=(12, 12),
        component_orders=(4, 4),
        payload_estimate_bytes=2048,
    )
    core = plan_core_sd([(1, -1)], [(2, -1), (2, 1), (2, -2)], geometry)
    plan = plan_sigma2_sd(core, core["virtual_keys"])
    assert plan["sizes"]["NZ"] == 3
    assert plan["sizes"]["NVH2"] >= plan["sizes"]["NVH"]
    assert plan["sizes"]["NVWH"] > 1
    assert plan["complete_payload_bound_bytes"] > core["complete_payload_bound_bytes"]
    with pytest.raises(ValueError, match="distinct valence"):
        plan_sigma2_sd(core, [(1, -1)])
    with pytest.raises(ValueError, match="distinct valence"):
        plan_sigma2_sd(core, [(2, -1), (2, -1)])


@pytest.fixture(
    params=(
        "original",
        "streamed_blas",
        "streamed_valence",
        "streamed_valence_parallel",
    )
)
def native_sigma2(tmp_path, request):
    source, compiler = (
        os.environ.get("ATOMX_PCI_SOURCE"),
        os.environ.get("ATOMX_PCI_COMPILER"),
    )
    if not source or not compiler:
        pytest.skip(
            "pinned pCI source and compiler required for live screening equations"
        )
    options = {}
    if request.param != "original":
        blas = os.environ.get("ATOMX_PCI_BLAS")
        if not blas:
            pytest.skip("bound OpenMP BLAS required for optimized screening ABI check")
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
            "screening_builder", tools / "build_pci_sigma2_backend.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(tools))
    atom = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    occupied = ((1, -1),)
    virtual = ((2, -1), (2, 1), (2, -2))
    orbitals = {key: atom.orbitals_by_subshell()[key][:2] for key in occupied}
    for key in virtual:
        _, states = build_virtual_basis(
            atom, key[1], e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        orbitals[key] = states[0][:2]
    feature = PolynomialCellCoulombFeatures(atom._bsp_data)
    plan = plan_core_sd(
        occupied,
        virtual,
        feature.identity,
        **{k: v for k, v in options.items() if k != "blas_library"},
    )
    assert plan["dimensions"]["NX"] > plan["dimensions"]["NXX"]
    term2_blas = request.param.startswith("streamed_valence")
    term2_threads = 8 if request.param == "streamed_valence_parallel" else 1
    screening = plan_sigma2_sd(
        plan,
        virtual,
        valence_term2_blas=term2_blas,
        valence_term2_threads=term2_threads,
    )
    module.build_sigma2(
        source,
        tmp_path / "backend",
        plan["dimensions"],
        compiler,
        sizes=dict(screening["sizes"]),
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
        valence_term2_blas=term2_blas,
        valence_term2_threads=term2_threads,
        **options,
    )
    core = CoreSD(
        orbitals,
        atom._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=tmp_path / "backend",
        nuclear_potential=atom._nuclear_potential,
        radial_representation="polynomial",
        **{k: v for k, v in options.items() if k != "blas_library"},
    )
    result = core.solve(residual_tolerance_Ha=1e-11)
    valence = ValenceSD(core, result, valence_keys=virtual)
    labels = core.integrals.labels
    eps = core.integrals.fock_Ha.diagonal().real
    energies = {
        key: float(eps[next(i for i, label in enumerate(labels) if label[:2] == key)])
        + 0.075
        for key in virtual
    }
    columns = tuple(
        valence.solve(key, energies[key], residual_tolerance_Ha=1e-11)
        for key in virtual
    )
    return core, result, valence, Sigma2SD(valence), columns


def test_live_screening_common_restore_signed_query_and_full_joint_response(
    native_sigma2,
):
    core, core_result, valence, problem, columns = native_sigma2
    result = problem.evaluate(columns)
    assert not result.radial_Ha.flags.writeable
    assert not result.joint_derivative.flags.writeable
    assert not result.provenance["ci_metric_constructed"]
    assert not result.provenance["experiment_validated"]
    assert result.provenance["equation_coverage"]["core_common_patches"] == 3
    core.action(0.37 * core_result.amplitudes)
    replay = problem.evaluate(tuple(reversed(columns)))
    assert np.array_equal(result.radial_Ha, replay.radial_Ha)
    assert np.array_equal(result.joint_derivative, replay.joint_derivative)
    for key in problem.plan["valence_keys"]:
        label = (*key, -2 * abs(key[1]) + 1)
        assert np.isfinite(result.direct_element(label, label, label, label))
        assert np.isfinite(
            result.direct_element(label, label, label, label, derivative=True)
        )
    step = 1e-5
    shifted = []
    for sign in (-1, 1):
        trial = tuple(
            valence.solve(
                c.column_key,
                c.energy_parameter_Ha + sign * step,
                solve_response=False,
                residual_tolerance_Ha=1e-11,
            )
            for c in columns
        )
        out = problem.evaluate(trial, joint_response=False)
        assert out.joint_derivative is None
        shifted.append(out.radial_Ha)
    np.testing.assert_allclose(
        (shifted[1] - shifted[0]) / (2 * step),
        result.joint_derivative,
        atol=1e-9,
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="every declared"):
        problem.evaluate(columns[:-1])
    with pytest.raises(ValueError, match="complete bound"):
        result.direct_element((1, -1, 1), (2, -1, 1), (2, -1, 1), (2, -1, 1))
    with pytest.raises(ValueError, match="joint response was not"):
        out.direct_element(
            (2, -1, 1), (2, -1, 1), (2, -1, 1), (2, -1, 1), derivative=True
        )


def test_live_screening_rejects_wrong_core_response_and_resource(native_sigma2):
    core, _, _, problem, columns = native_sigma2
    bad = replace(
        columns[0],
        provenance=dict(columns[0].provenance, core_amplitudes_sha256="wrong"),
    )
    with pytest.raises(ValueError, match="identity differs"):
        problem.evaluate((bad,) + columns[1:])
    bad = replace(columns[0], numerator_derivative=columns[0].numerator_derivative + 1)
    with pytest.raises(ValueError, match="implicit response residual"):
        problem.evaluate((bad,) + columns[1:])
    bad = replace(columns[0], numerators=columns[0].numerators + 1)
    with pytest.raises(ValueError, match="valence root residual"):
        problem.evaluate((bad,) + columns[1:])
    bad = replace(columns[0], numerator_derivative=None)
    with pytest.raises(ValueError, match="implicit numerator response"):
        problem.evaluate((bad,) + columns[1:])
    previous = core.memory_bytes
    core.memory_bytes = problem.plan["complete_payload_bound_bytes"] - 1
    try:
        with pytest.raises(MemoryError, match="current cap"):
            problem.evaluate(columns)
    finally:
        core.memory_bytes = previous
