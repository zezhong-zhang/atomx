#!/usr/bin/env python3
"""Replay the complete bounded transverse-to-J-pair and finite SD anchor."""

import argparse
import hashlib
import json
import os
import resource
import sys
import tarfile
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value):
    if hasattr(value, "items"):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise SystemExit("refuse to overwrite scientific result")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))
    owners = [Path(__file__).resolve(), Path(__file__).with_name("contract.json")]
    sources = (
        sorted((root / "src/atomx").rglob("*.py"))
        + [
            root / "tests/test_spinor_integrals_fscc.py",
            root / "tests/test_transverse_spinor_integrals.py",
            root / "tests/test_native_sd_transverse_contract.py",
        ]
        + owners
    )
    before = {str(path.relative_to(root)): sha(path) for path in sources}
    archive = output / "sources.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sources:
            tar.add(path, arcname=str(path.relative_to(root)))
    result = dict(
        status="running",
        highest_level_reached=False,
        action="native_sd_transverse_bridge",
        scope="complete J-pair bridge and finite two-electron linked scalar SD",
        native_scalable_sd_transverse_complete=False,
        caps=dict(address_bytes=4 * 1024**3, threads=1, cpu_seconds=120),
        source_sha256=before,
        source_archive_sha256=sha(archive),
        contract_sha256=sha(Path(__file__).with_name("contract.json")),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    try:
        import numpy as np
        from test_native_sd_transverse_contract import independent_scalar_ci
        from test_spinor_integrals_fscc import integrals, native
        from test_transverse_spinor_integrals import determinant_matrix

        from atomx.methods.sd_transverse import (
            build_j_coupled_sd_interaction,
            solve_two_electron_core_sd,
        )

        model = native.__wrapped__()
        dc = integrals(model)
        energies = {
            label[:2]: dc.fock_Ha[i, i].real for i, label in enumerate(dc.labels)
        }
        arrays, cases = {}, {}
        for mode in ("dc", "static", "dynamic"):
            options = dict(include_breit=mode != "dc")
            if mode == "dynamic":
                options["breit_orbital_energies"] = energies
            a = integrals(model, **options)
            bridge = build_j_coupled_sd_interaction(a)
            masks, original = determinant_matrix(a, 3)
            rebuilt_masks, rebuilt = determinant_matrix(bridge, 3)
            if masks != rebuilt_masks:
                raise AssertionError("full three-electron determinant order changed")
            matrix_error = float(np.max(abs(original - rebuilt)))
            if matrix_error > 1e-11:
                raise AssertionError(
                    "full signed three-electron Hamiltonian gate failed"
                )
            ci, transfer = independent_scalar_ci(model, mode, a, bridge)
            pair_h = bridge.hamiltonian_block(0, 1)
            csf_error = float(np.max(abs(transfer.T @ pair_h @ transfer - ci)))
            sd = solve_two_electron_core_sd(
                bridge,
                reference_identity=f"Be2+ {mode} 10-spinor complete finite scalar SD",
            )
            if sd.equation_residual_Ha > 1e-12:
                raise AssertionError(
                    "original finite linked SD raw residual gate failed"
                )
            arrays[f"{mode}_labels"] = np.asarray(a.labels, dtype=np.int64)
            for name, value in (
                ("one_body", a.one_body_Ha),
                ("fock", a.fock_Ha),
                ("packed_eri", a.packed_eri_Ha),
                ("full_three_electron", original),
                ("rebuilt_three_electron", rebuilt),
                ("scalar_ci", ci),
                ("scalar_transfer", transfer),
                ("scalar_pair_h", pair_h),
                ("sd_amplitudes", sd.amplitudes),
                ("sd_equations", sd.equation_matrix_Ha),
                ("sd_singles_subtraction", sd.singles_product_subtraction_Ha),
            ):
                arrays[f"{mode}_{name}"] = value
            for block, value in bridge.blocks_Ha.items():
                arrays[f"{mode}_j{block[0]}_parity{block[1]}"] = value
            cases[mode] = dict(
                gates=dict(bridge.gates),
                full_three_electron_matrix_max_Ha=matrix_error,
                independent_csf_matrix_max_Ha=csf_error,
                finite_core_sd_raw_residual_Ha=sd.equation_residual_Ha,
                finite_core_sd_total_energy_Ha=sd.total_energy_Ha,
                finite_core_sd_correlation_energy_Ha=sd.correlation_energy_Ha,
                noncanonical_fov_max_Ha=float(np.max(abs(a.fock_Ha[:2, 2:]))),
                singles_product_subtraction_max_Ha=float(
                    np.max(abs(sd.singles_product_subtraction_Ha))
                ),
                resources=dict(bridge.resources),
                hamiltonian_identity=dict(a.identity),
                bridge_input_sha256=bridge.identity["input_sha256"],
            )
        np.savez_compressed(output / "arrays.npz", **arrays)
        after = {str(path.relative_to(root)): sha(path) for path in sources}
        if before != after:
            raise RuntimeError("loaded source changed during scientific calculation")
        result.update(
            status="passed",
            gates_passed=True,
            gates={
                mode: dict(
                    case["gates"],
                    full_three_electron_matrix_max_Ha=case[
                        "full_three_electron_matrix_max_Ha"
                    ],
                    independent_csf_matrix_max_Ha=case["independent_csf_matrix_max_Ha"],
                    finite_core_sd_raw_residual_Ha=case[
                        "finite_core_sd_raw_residual_Ha"
                    ],
                )
                for mode, case in cases.items()
            },
            card_summary=(
                "完整 signed J/宇称横向积分桥与有限两电子 SD 已验证；"
                "DC/static/dynamic 的完整 120×120 三电子矩阵及独立 jj-CSF 对照均通过 1e-11 Ha 门，"
                "有限 linked SD 原始残差通过 1e-12 Ha 门。"
                "native_scalable_sd_transverse_complete=false：大规模 pCI 核/价/Sigma2 横向图项仍未完成。"
            ),
            source_unchanged=True,
            arrays_sha256=sha(output / "arrays.npz"),
            cases=cases,
            source_after_sha256=after,
            input_identity=dict(
                Z=4,
                ns=30,
                ks=7,
                spinors=10,
                occupied_keys=[(1, -1)],
                virtual_keys=[(2, -1), (2, 1), (2, -2)],
                fixed_dynamic_energies_Ha={
                    str(k): float(v) for k, v in energies.items()
                },
            ),
            next_direction="Full tensor-invariant transverse substitutions in every native core, valence and Sigma2 diagram, retaining noncanonical Fock/singles terms; no scalar RINT injection.",
        )
    except Exception as error:
        result.update(
            status="failed",
            gates_passed=False,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        result.update(
            observed_wall_seconds=time.monotonic() - started,
            timing_scope="imports, native DC fixture, three Hamiltonians, complete bridge, full magnetic/CSF controls and finite SD; excludes source archival",
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            performance_claim="uncontrolled observation; no speedup or HPC certification",
        )
        (output / "result.json").write_text(
            json.dumps(jsonable(result), indent=2, default=repr) + "\n"
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "gates_passed",
                    "observed_wall_seconds",
                    "process_peak_rss_bytes",
                    "native_scalable_sd_transverse_complete",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
