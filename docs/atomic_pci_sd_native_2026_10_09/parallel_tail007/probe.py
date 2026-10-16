"""Bounded complete Yb tail counting/replay; never runs correlation roots."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OWNER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]


def main(output):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Read/copy contract before any scientific import or radial work.
    contract_file = Path(__file__).with_name("contract.json")
    contract = json.loads(contract_file.read_text())
    (output / "contract.json").write_bytes(contract_file.read_bytes())
    cap = contract["caps"]["probe_address_space_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"

    def timeout(signum, frame):
        raise TimeoutError("complete tail preflight exhausted original120s probe cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["caps"]["probe_wall_seconds"])

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    try:
        from atomic_tail_preflight import (
            YB_PANEL,
            YB_STATES,
            admit,
            digest,
            estimate_task,
            identity_hash,
            machine_snapshot,
            retained_band,
        )

        baseline = REPO / "docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default"
        inputs = OWNER / "yb_full_run008/yb_full_inputs.npz"
        run = OWNER / "yb_full_monovalent006"
        payloads = [
            baseline / name
            for name in (
                "result.pkl",
                "core.json",
                "status.json",
                "orbitals_workspace.npz",
                "structure.json",
            )
        ]
        payloads += [
            inputs,
            OWNER / "yb_full_run008/nuclear_model.json",
            run / "resource.json",
            run / "numeric64_resource.json",
            run / "result.json",
            OWNER / "yb_monovalent_norm001/resource.json",
            OWNER / "yb_monovalent_norm001/result.json",
            OWNER / "yb_monovalent_experiment003/reference.json",
            OWNER / "yb_monovalent_experiment003/comparison.json",
        ]
        comparison = json.loads(
            (OWNER / "yb_monovalent_experiment003/comparison.json").read_text()
        )
        for name, expected_sha in comparison["computed_payloads_sha256"].items():
            if Path(name).name != name or digest(run / name) != expected_sha:
                raise ValueError("actual NIST comparison root payload identity changed")
            payloads.append(run / name)
        source_postcheck = run / "source_postcheck.json"
        if digest(source_postcheck) != comparison["source_postcheck_sha256"]:
            raise ValueError("actual source audit for NIST comparison changed")
        payloads.append(source_postcheck)
        sources = sorted((REPO / "src/atomx").rglob("*.py")) + sorted(
            (REPO / "src/atomx").rglob("*.f90")
        )
        sources += [
            Path(__file__),
            contract_file,
            REPO / "tools/atomic_tail_preflight.py",
            REPO / "docs/yb_diagnosis_repair_2026_10_08/common.py",
            OWNER / "probe_yb_monovalent_subspace.py",
            OWNER / "monovalent_term31_blas_control.py",
            OWNER / "valence_terma4_blas_control.py",
            OWNER / "cuda_full_term2_control.py",
            OWNER / "cuda_density_control.py",
            OWNER / "cuda_valence_bridge.py",
            OWNER / "monovalent_norm_control.py",
        ]
        hashes = {str(path.relative_to(REPO)): digest(path) for path in sources}
        bindings = {str(path.relative_to(REPO)): digest(path) for path in payloads}
        save("source_sha256.json", hashes)
        save("input_sha256.json", bindings)
        # Freeze the actual dirty sources used by replay/planning, without
        # copying scientific roots or compiler caches into a new directory.
        for path in sources:
            target = output / "source" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        import numpy as np

        from atomx.methods.atomic_space import build_atomic_space
        from atomx.numerics.bspline.coulomb_features import (
            PolynomialCellCoulombFeatures,
        )

        sys.path.insert(0, str(REPO / "docs/yb_diagnosis_repair_2026_10_08"))
        from common import load_reference

        reference = json.loads(
            (OWNER / "yb_monovalent_experiment003/reference.json").read_text()
        )
        if comparison["reference_sha256"] != digest(
            OWNER / "yb_monovalent_experiment003/reference.json"
        ):
            raise ValueError("actual frozen NIST reference/comparison identity changed")
        if tuple(map(tuple, reference["matching_contract"]["keys"])) != YB_STATES:
            raise ValueError("same five joint state identities required")
        core, _, _, reference_identity = load_reference()
        nuclear = json.loads((OWNER / "yb_full_run008/nuclear_model.json").read_text())
        if (
            core.Z != 70
            or not core.converged
            or core._bsp_data["ns"] != 123
            or sum(s.occupation for s in core.subshells) != 68
        ):
            raise ValueError("unchanged converged V68/ns123 reference required")
        if nuclear["model"] != "fermi" or nuclear["radius_mass_number"] != 175:
            raise ValueError("original Fermi A175 radius identity required")
        maximal = build_atomic_space(core, virtual_lmax=4, virtual_energy_max_Ha=10.0)
        features = PolynomialCellCoulombFeatures(core._bsp_data)
        machine = machine_snapshot(output)
        save("machine.json", machine)
        old_resource = json.loads((run / "resource.json").read_text())
        old_virtual = tuple(map(tuple, old_resource["core_plan"]["virtual_keys"]))
        original = retained_band(maximal, lmax=3, energy_max_Ha=5.0)
        key_equal = (
            original.virtual_keys == old_virtual
            and original.occupied_keys
            == tuple(map(tuple, old_resource["core_plan"]["occupied_keys"]))
        )
        coefficient_error = 0.0
        workspace_equal = True
        with np.load(inputs, allow_pickle=False) as data:
            for name in data:
                if name.startswith("workspace_"):
                    workspace_equal = workspace_equal and np.array_equal(
                        core._bsp_data[name.removeprefix("workspace_")], data[name]
                    )
            for key in original.occupied_keys + original.virtual_keys:
                for component, wave in zip(
                    ("P", "Q"), original.orbitals[key], strict=True
                ):
                    coefficient_error = max(
                        coefficient_error,
                        float(
                            np.max(abs(data[f"{component}_{key[0]}_{key[1]}"] - wave))
                        ),
                    )
        if (
            not key_equal
            or not workspace_equal
            or coefficient_error
            > contract["acceptance"]["baseline_coefficient_absolute_gate"]
        ):
            save(
                "radial_replay.json",
                {
                    "keys_equal": key_equal,
                    "coefficient_absolute_max": coefficient_error,
                    "status": "fail",
                },
            )
            raise ValueError(
                "complete original radial replay identity failed; no tail solve admitted"
            )
        tasks, spaces = [], {}
        for lmax, emax in YB_PANEL:
            space = retained_band(maximal, lmax=lmax, energy_max_Ha=emax)
            spaces[lmax, emax] = set(space.virtual_keys)
            plan1 = estimate_task(space, features.identity, worker_threads=1)
            plan8 = estimate_task(space, features.identity, worker_threads=8)
            band = {
                "lmax": lmax,
                "energy_max_Ha": emax,
                "occupied_keys": space.occupied_keys,
                "virtual_keys": space.virtual_keys,
                "energies_Ha": dict(space.identity["virtual_orbital_energies_Ha"]),
            }
            orbital_sha = hashlib.sha256()
            for key in space.occupied_keys + space.virtual_keys:
                orbital_sha.update(str(key).encode())
                for wave in space.orbitals[key]:
                    orbital_sha.update(np.asarray(wave).tobytes())
            identity = {
                "Z": 70,
                "core_electrons": 68,
                "hamiltonian": "Dirac-Coulomb",
                "nucleus": nuclear,
                "workspace_ns": 123,
                "band_sha256": identity_hash(band),
                "orbital_coefficients_sha256": orbital_sha.hexdigest(),
                "reference_inputs_sha256": bindings,
                "loaded_source_sha256": hashes,
                "states": YB_STATES,
                "common_energy_zero": "same-task6s attachment energy; no fitted shift",
                "metric": "common original native B-spline overlap",
            }
            task_id = (
                "yb_tail_l"
                + str(lmax)
                + "_e"
                + str(int(emax))
                + "_"
                + identity_hash(identity)[:20]
            )
            save(task_id + "_band.json", band)
            tasks.append(
                {
                    "task_id": task_id,
                    "physical_identity": identity,
                    "lmax": lmax,
                    "energy_max_Ha": emax,
                    "serial_cpu_resource": plan1,
                    "existing_gpu8_resource": plan8,
                    "admission": admit(plan8, machine),
                    "all_joint_states_required": True,
                    "correlation_execution_started": False,
                    "benchmark_expected_errors": None,
                    "independent_process_boundary": "one complete physical space owns core/COMMON/backend and joint states",
                    "shared_owner_state_parallelism": "forbidden: native COMMON/core lock serializes this owner's states",
                }
            )
        baseline_task = tasks[0]
        p = baseline_task["existing_gpu8_resource"]
        count_equal = (
            p["state_amplitudes"] == old_resource["standalone_plan"]["amplitudes"]
        )
        raw_equal = (
            p["core_raw_amplitudes"] == old_resource["core_plan"]["raw_amplitudes"]
        )
        bound_equal = (
            p["maximum_legacy_numeric64_host_payload_bytes"]
            == json.loads((run / "numeric64_resource.json").read_text())[
                "maximum_bound_bytes"
            ]
        )
        nesting = (
            spaces[3, 5.0] <= spaces[3, 10.0] <= spaces[4, 10.0]
            and spaces[3, 5.0] <= spaces[4, 5.0] <= spaces[4, 10.0]
        )
        gates = {
            "original_complete_radial_keys": key_equal,
            "original_complete_workspace_bitwise_equal": workspace_equal,
            "original_complete_radial_coefficients": coefficient_error <= 1e-11,
            "original_all_five_state_coordinate_counts": count_equal,
            "original_core_coordinate_count": raw_equal,
            "original_numeric64_bound_formula": bound_equal,
            "complete_declared_band_nesting": nesting,
            "same_five_states_and_original_NIST_reference": True,
            "sources_unchanged": all(
                digest(REPO / path) == sha for path, sha in hashes.items()
            ),
            "inputs_unchanged": all(
                digest(REPO / path) == sha for path, sha in bindings.items()
            ),
        }
        if not all(gates.values()):
            save("gate_failure.json", gates)
            raise ValueError("complete counting/source preflight gate failed")
        save(
            "radial_replay.json",
            {
                "keys_equal": True,
                "coefficient_absolute_max": coefficient_error,
                "status": "pass",
                "reference_identity": reference_identity,
            },
        )
        save(
            "result.json",
            {
                "status": "pass",
                "card_summary": "完整tail计数/径向回放通过：lmax3/4 × Emax5/10Ha 四档、全部5态。发现l4完整norm Gram scratch超过原64MiB，需要分块复用；GPU staging已计入，长算executor/GPU实际准入仍deferred。尚无新收敛谱或实验精度改善证书。",
                "scope": contract["scope"],
                "gates": gates,
                "tasks": tasks,
                "reference_comparison": {
                    "path": str(
                        (
                            OWNER / "yb_monovalent_experiment003/comparison.json"
                        ).relative_to(REPO)
                    ),
                    "sha256": digest(
                        OWNER / "yb_monovalent_experiment003/comparison.json"
                    ),
                    "experimental_gate_relative": reference["matching_contract"][
                        "relative_error_gate"
                    ],
                    "new_tail_spectra_computed": False,
                    "maximum_excitation_relative_error": max(
                        abs(row["relative_error"])
                        for row in comparison["rows"]
                        if row["relative_error"] is not None
                    ),
                    "maximum_splitting_relative_error": max(
                        abs(row["relative_error"]) for row in comparison["splittings"]
                    ),
                    "rows": comparison["rows"],
                    "splittings": comparison["splittings"],
                    "excitation_gate_pass": comparison["excitation_gate_pass"],
                    "splitting_gate_pass": comparison["splitting_gate_pass"],
                    "state_matching_scope": reference["matching_contract"]["scope"],
                },
                "resource_correction": {
                    "legacy_baseline_payload_bytes": p[
                        "maximum_legacy_numeric64_host_payload_bytes"
                    ],
                    "previous_complete_norm_payload_bytes": 16629201972,
                    "new_explicit_cuda_host_staging_bytes": p[
                        "cuda_host_staging_bytes"
                    ],
                    "baseline_corrected_payload_bytes": p[
                        "maximum_complete_host_payload_bytes"
                    ],
                    "runtime_reservation_separate": True,
                    "historical_certificates_restamped": False,
                },
                "parallel_policy": {
                    "independent_space_tasks": 4,
                    "probe_cpu_threads": 1,
                    "all_tasks_resource_sum_required": True,
                    "simultaneous_gpu_admission": "exclusive GPU task by default; overlap requires complete summed host/device/disk reservations and fresh live load",
                    "atomic_space_cache_scope": "one maximal immutable VN solve; complete exact band subsets with their own identity",
                    "full_tail_executor_ready": False,
                },
                "seconds": time.perf_counter() - start,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "highest_level_reached": False,
            },
        )
    except BaseException as error:
        save(
            "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - start,
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
