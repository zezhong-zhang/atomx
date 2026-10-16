"""Signed angular, complete planning and live pCI/native core SD gates."""

import hashlib
import importlib.util
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from test_spinor_integrals_fscc import native as native_fixture

from atomx.angular.coefficients import reduced_C_jj
from atomx.angular.csf_hamiltonian import two_body_angular
from atomx.angular.sd_rho import kappa_from_partial, partial_index, rho_coefficient
from atomx.methods.atomic_space import build_atomic_space
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.linearized_sd import compute as dense_sd
from atomx.numerics.bspline.coulomb_features import (
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)

native = native_fixture


def test_signed_rho_angular_anchor_including_null_electric_channels():
    # Original S relation was separately executed against pinned pCI.
    for kappas in itertools.product((-1, 1, -2), repeat=4):
        states = [
            [(1, k, m) for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)] for k in kappas
        ]
        norms = np.zeros(4)
        for labels in itertools.product(*states):
            a, b, i, j = labels
            direct = dict(
                two_body_angular(a[1], a[2], b[1], b[2], i[1], i[2], j[1], j[2])
            )
            for rank in range(4):
                coefficient = rho_coefficient(rank, a, b, i, j)
                sa = (-1) ** (abs(a[1]) + abs(i[1]) + rank - 1) * reduced_C_jj(
                    a[1], i[1], rank
                )
                sb = (-1) ** (abs(b[1]) + abs(j[1]) + rank - 1) * reduced_C_jj(
                    b[1], j[1], rank
                )
                assert coefficient * ((-1) ** rank) * sa * sb == pytest.approx(
                    direct.get(rank, 0), abs=1e-13
                )
                norms[rank] += coefficient**2
        for rank, norm in enumerate(norms):
            allowed = (
                max(
                    abs(abs(kappas[0]) - abs(kappas[2])),
                    abs(abs(kappas[1]) - abs(kappas[3])),
                )
                <= rank
                <= min(
                    abs(kappas[0]) + abs(kappas[2]) - 1,
                    abs(kappas[1]) + abs(kappas[3]) - 1,
                )
            )
            assert norm == pytest.approx(
                1 / (2 * rank + 1) if allowed else 0, abs=1e-13
            )
    # Odd-rank s-s rho has nonzero angular amplitude despite C^1(s,s)=0.
    assert reduced_C_jj(-1, -1, 1) == 0
    assert rho_coefficient(1, (1, -1, 1), (1, -1, -1), (1, -1, 1), (1, -1, -1)) != 0


def test_complete_plan_counts_and_preparation_guards(native):
    core, orbitals = native
    features = CellCoulombFeatures(core._bsp_data)
    plan = plan_core_sd([(1, -1)], [(2, -1), (2, 1), (2, -2)], features.identity)
    assert plan["independent_scalar_amplitudes"] == 4
    assert plan["raw_amplitudes"] == 9
    assert plan["dimensions"]["NCH"] == 8
    assert plan["complete_payload_bound_bytes"] > plan["raw_amplitudes"] * 8
    assert plan["radial_pair_cache_size"] == 16384
    smaller = plan_core_sd(
        [(1, -1)],
        [(2, -1), (2, 1), (2, -2)],
        features.identity,
        radial_pair_cache_size=1,
    )
    assert smaller["dimensions"] == plan["dimensions"]
    assert smaller["raw_amplitudes"] == plan["raw_amplitudes"]
    assert plan["complete_payload_bound_bytes"] - smaller[
        "complete_payload_bound_bytes"
    ] == (
        plan["radial_pair_cache_array_bound_bytes"]
        - smaller["radial_pair_cache_array_bound_bytes"]
    )
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="cache"):
            plan_core_sd(
                [(1, -1)], [(2, -1)], features.identity, radial_pair_cache_size=value
            )
    threaded = plan_core_sd(
        [(1, -1)],
        [(2, -1), (2, 1), (2, -2)],
        features.identity,
        term21_threads=4,
    )
    assert threaded["term21_threads"] == 1  # one complete core pair
    assert threaded["raw_amplitudes"] == plan["raw_amplitudes"]
    assert (
        threaded["complete_payload_bound_bytes"] - plan["complete_payload_bound_bytes"]
        == threaded["term21_parallel_scratch_bytes"]
    )
    singles = plan_core_sd(
        [(1, -1)],
        [(2, -1), (2, 1), (2, -2)],
        features.identity,
        term21_threads=4,
        core_singles_blas=True,
    )
    assert singles["raw_amplitudes"] == plan["raw_amplitudes"]
    assert (
        singles["complete_payload_bound_bytes"]
        - threaded["complete_payload_bound_bytes"]
        == singles["core_singles_parallel_scratch_bytes"]
    )
    d = singles["dimensions"]
    assert singles["core_singles_parallel_scratch_bytes"] == 8 * d["NHF"] * (
        d["NX"] * d["NK"] + d["NXX"]
    ) + singles["term21_threads"] * 8 * (
        d["NHF"] * d["NXX"] + d["NXX"] ** 2 + 2 * d["NHF"]
    )
    for kwargs in (
        dict(core_singles_blas=True),
        dict(core_singles_blas=1, term21_threads=1),
    ):
        with pytest.raises(ValueError, match="core singles"):
            plan_core_sd([(1, -1)], [(2, -1)], features.identity, **kwargs)
    single_to_doubles = plan_core_sd(
        [(1, -1)],
        [(2, -1), (2, 1), (2, -2)],
        features.identity,
        term21_threads=4,
        core_single_to_doubles_blas=True,
    )
    assert single_to_doubles["raw_amplitudes"] == plan["raw_amplitudes"]
    assert (
        single_to_doubles["complete_payload_bound_bytes"]
        - threaded["complete_payload_bound_bytes"]
        == single_to_doubles["core_single_to_doubles_scratch_bytes"]
    )
    for kwargs in (
        dict(core_single_to_doubles_blas=True),
        dict(core_single_to_doubles_blas=1, term21_threads=1),
    ):
        with pytest.raises(ValueError, match="single-to-doubles"):
            plan_core_sd([(1, -1)], [(2, -1)], features.identity, **kwargs)
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError, match="worker"):
            plan_core_sd([(1, -1)], [(2, -1)], features.identity, term21_threads=value)
    more = plan_core_sd(
        [(1, -1)],
        [(n, k) for k in (-1, 1, -2) for n in range(2, 15)],
        features.identity,
    )
    assert more["raw_amplitudes"] == 8 * 13**2 + 13
    with pytest.raises(MemoryError, match="complete"):
        CoreSD(
            orbitals,
            core._bsp_data,
            4,
            30,
            occupied_keys=[(1, -1)],
            virtual_keys=[(2, -1), (2, 1), (2, -2)],
            backend_directory="/missing",
            nuclear_potential=core._nuclear_potential,
            memory_bytes=1,
        )
    for index in range(1, 20):
        assert partial_index(kappa_from_partial(index)) == index
    with pytest.raises(ValueError):
        kappa_from_partial(0)
    with pytest.raises(ValueError):
        partial_index(0)


def test_shared_complete_virtual_space_and_owned_components(native):
    core, _ = native
    from atomx.methods.mbpt import build_virtual_basis

    space = build_atomic_space(core, virtual_lmax=1, virtual_energy_max_Ha=50)
    expected = []
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        n = (
            max(
                [0 if k == -1 else 1] + [a[0] for a in space.occupied_keys if a[1] == k]
            )
            + 1
        )
        expected.extend((n + i, k) for i in range(len(states)))
    assert space.virtual_keys == tuple(expected)
    for pair in space.orbitals.values():
        for component in pair:
            with pytest.raises(ValueError):
                component.setflags(write=True)
    for kw in (
        dict(virtual_lmax=True, virtual_energy_max_Ha=50),
        dict(virtual_lmax=1, virtual_energy_max_Ha=1j),
    ):
        with pytest.raises(ValueError):
            build_atomic_space(core, **kw)


@pytest.mark.parametrize(
    "mode,singles,single_to_doubles",
    [("streamed", False, False), ("compact", True, False), ("compact", False, True)],
)
def test_parallel_streamed_blas_preflight(
    native, tmp_path, mode, singles, single_to_doubles
):
    source, compiler = (
        os.environ.get("ATOMX_PCI_SOURCE"),
        os.environ.get("ATOMX_PCI_COMPILER"),
    )
    serial = os.environ.get("ATOMX_PCI_SERIAL_BLAS")
    if not source or not compiler or not serial:
        pytest.skip("set live pCI and serial BLAS paths for concurrency rejection gate")
    script = Path(__file__).resolve().parents[1] / "tools/build_pci_sd_backend.py"
    spec = importlib.util.spec_from_file_location("build_pci_sd_backend", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    core, _ = native
    plan = plan_core_sd(
        [(1, -1), (2, -1)],
        [(2, 1), (2, -2)],
        CellCoulombFeatures(core._bsp_data).identity,
        term21_threads=3,
        term21_mode=mode,
        core_singles_blas=singles,
        core_single_to_doubles_blas=single_to_doubles,
    )
    directory = tmp_path / "rejected"
    with pytest.raises(ValueError, match="OpenMP"):
        builder.build(
            source,
            directory,
            plan["dimensions"],
            compiler,
            runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
            blas_library=serial,
            term21_threads=3,
            term21_mode=mode,
            core_singles_blas=singles,
            core_single_to_doubles_blas=single_to_doubles,
        )
    assert not directory.exists()


@pytest.mark.parametrize(
    "initialization,term21_mode,core_singles_blas,term21_threads,core_single_to_doubles_blas",
    [
        ("original", "compact", False, 0, False),
        ("original", "compact", False, 1, False),
        ("blas", "compact", False, 0, False),
        ("blas", "compact", False, 1, False),
        ("blas", "streamed", False, 0, False),
        ("blas", "streamed", False, 1, False),
        ("blas", "compact", True, 1, False),
        ("blas", "streamed", True, 1, False),
        ("blas", "compact", False, 1, True),
        ("blas", "streamed", True, 1, True),
    ],
)
@pytest.mark.parametrize("radial_representation", ["native", "polynomial"])
def test_live_direct_core_sd_vs_independent_native_csf(
    native,
    tmp_path,
    initialization,
    term21_threads,
    term21_mode,
    core_singles_blas,
    core_single_to_doubles_blas,
    capfd,
    radial_representation,
):
    source = os.environ.get("ATOMX_PCI_SOURCE")
    compiler = os.environ.get("ATOMX_PCI_COMPILER")
    if not source or not compiler:
        pytest.skip(
            "set pinned ATOMX_PCI_SOURCE and ATOMX_PCI_COMPILER for the live equation backend gate"
        )
    blas = os.environ.get("ATOMX_PCI_BLAS") if initialization == "blas" else None
    if initialization == "blas" and not blas:
        pytest.skip("set ATOMX_PCI_BLAS for live LP64 BLAS initialization")
    script = Path(__file__).resolve().parents[1] / "tools/build_pci_sd_backend.py"
    spec = importlib.util.spec_from_file_location("build_pci_sd_backend", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    core, orbitals = native
    occupied, virtual = [(1, -1)], [(2, -1), (2, 1), (2, -2)]
    features_type = (
        CellCoulombFeatures
        if radial_representation == "native"
        else PolynomialCellCoulombFeatures
    )
    features = features_type(core._bsp_data)
    plan = plan_core_sd(
        occupied,
        virtual,
        features.identity,
        term21_threads=term21_threads,
        term21_mode=term21_mode,
        core_singles_blas=core_singles_blas,
        core_single_to_doubles_blas=core_single_to_doubles_blas,
    )
    directory = tmp_path / "backend"
    builder.build(
        source,
        directory,
        plan["dimensions"],
        compiler,
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
        blas_library=blas,
        term21_threads=term21_threads,
        term21_mode=term21_mode,
        core_singles_blas=core_singles_blas,
        core_single_to_doubles_blas=core_single_to_doubles_blas,
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
        term21_threads=term21_threads,
        term21_mode=term21_mode,
        core_singles_blas=core_singles_blas,
        core_single_to_doubles_blas=core_single_to_doubles_blas,
        radial_representation=radial_representation,
    )
    result = problem.solve(residual_tolerance_Ha=1e-12)
    independent = dense_sd(
        "1s2",
        ["2s", "2p"],
        orbitals,
        core._bsp_data,
        4,
        30,
        reference_identity="same-space native CSF DC gate",
        nuclear_potential=core._nuclear_potential,
    )
    assert result.correlation_energy_Ha == pytest.approx(
        independent.correlation_energy_Ha, abs=1e-12
    )
    assert result.reference_energy_Ha == pytest.approx(
        independent.reference_energy_Ha, abs=1e-12
    )
    assert result.residual_max_Ha < 1e-12
    assert result.matrix_vector_products > 1
    assert not result.provenance["space_converged"]
    assert result.provenance["direct_sd_source_sha256"]
    replay = problem.solve(
        residual_tolerance_Ha=1e-12, initial_amplitudes=result.amplitudes
    )
    assert replay.residual_max_Ha < 1e-12
    assert replay.matrix_vector_products >= 2
    assert replay.correlation_energy_Ha == pytest.approx(
        result.correlation_energy_Ha, abs=1e-12
    )
    assert (
        replay.provenance["initial_amplitudes_sha256"]
        == hashlib.sha256(result.amplitudes.tobytes()).hexdigest()
    )
    for invalid in (
        result.amplitudes[:-1],
        result.amplitudes.astype(complex),
        np.full_like(result.amplitudes, np.nan),
    ):
        with pytest.raises(ValueError, match="initial amplitude"):
            problem.solve(initial_amplitudes=invalid)
    capfd.readouterr()
    problem.set_diagram_progress(True)
    assert np.max(abs(problem.action(result.amplitudes) - problem.source)) < 1e-12
    problem.set_diagram_progress(False)
    assert "native SD stage term21" in capfd.readouterr().err
    with pytest.raises(ValueError, match="boolean"):
        problem.set_diagram_progress(1)
    with pytest.raises(TypeError):
        problem.plan["raw_amplitudes"] = 0
    with pytest.raises(TypeError):
        problem.plan["dimensions"]["NHF"] = 0
    with pytest.raises(ValueError):
        result.amplitudes.setflags(write=True)
    with pytest.raises(MemoryError):
        problem.memory_bytes = plan["complete_payload_bound_bytes"]
        problem.solve(restart=30)
    with pytest.raises(ValueError, match="real"):
        problem.energy(result.amplitudes.astype(complex))
    with pytest.raises(ValueError):
        problem.action(np.zeros(len(result.amplitudes) - 1))
    # Valid finite inputs can overflow the native contractions. A NaN/Inf
    # result must fail before an absolute residual comparison can accept it.
    with pytest.raises(ValueError, match="nonfinite"):
        problem.action(np.full_like(result.amplitudes, np.finfo(float).max))
    with pytest.raises(ValueError, match="COMMON"):
        CoreSD(
            orbitals,
            core._bsp_data,
            4,
            30,
            occupied_keys=occupied,
            virtual_keys=virtual,
            backend_directory=directory,
            nuclear_potential=core._nuclear_potential,
            term21_threads=term21_threads,
            term21_mode=term21_mode,
            core_singles_blas=core_singles_blas,
            core_single_to_doubles_blas=core_single_to_doubles_blas,
            radial_representation=radial_representation,
        )
