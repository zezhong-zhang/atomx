"""Independent review of preserved complete finite benchmark evidence."""

import argparse
import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_arrays(reader, directory, spinors):
    with np.load(directory / "model_space_matrix.npz") as archive:
        result = {"HEFF": archive["matrix_Ha"].copy()}
    for name in ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c"):
        result[name] = reader.decode_diagram(
            directory / "scratch" / (name + ".dg"), spinors
        )[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    owner = run.parent
    repo = owner.parents[2]
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "output_sha256.json").read_text())
    assert all(sha(run / p) == value for p, value in manifest.items())
    assert all(sha(repo / p) == value for p, value in result["source_sha256"].items())
    assert all(result["gates"].values())
    assert result["performance_ranking"] is None
    assert not result["highest_level_reached"]
    accuracy_rows = {}
    for case in ("two_core", "four_core"):
        with (
            np.load(run / f"accuracy_{case}_default/full_signed_audit.npz") as d,
            np.load(run / f"accuracy_{case}_corrected/full_signed_audit.npz") as c,
        ):
            assert np.array_equal(
                d["full_core_hamiltonian_Ha"], c["full_core_hamiltonian_Ha"]
            )
            assert np.array_equal(
                d["full_attachment_hamiltonian_Ha"], c["full_attachment_hamiltonian_Ha"]
            )
            pi = c["model_indices"]
            assert c["omega"].shape[1] == len(pi) == 8
            normalization = float(np.max(abs(c["omega"][pi] - np.eye(8))))
            projected = c["attachment_ranks"] <= 2
            equation = float(np.max(abs(c["attachment_residual_Ha"][projected])))
            full = float(np.max(abs(c["attachment_residual_Ha"])))
            heff_error = float(
                np.max(
                    abs((c["attachment_hbar_Ha"] @ c["omega"])[pi] - c["total_heff_Ha"])
                )
            )
            all_roots_error = float(
                np.max(
                    abs(
                        np.sort(np.linalg.eigvals(c["total_heff_Ha"]).real)
                        - c["exact_attachment_roots_Ha"][:8]
                    )
                )
            )
            assert normalization <= 1e-12 and equation <= 1e-11 and heff_error <= 1e-10
            if case == "two_core":
                assert full <= 1e-11 and all_roots_error <= 1e-11
            else:
                assert full > 1e-11 and all_roots_error > 1e-11
            assert float(np.max(abs(d["attachment_residual_Ha"][projected]))) > 1e-11
            accuracy_rows[case] = dict(
                normalization=normalization,
                retained_equation_Ha=equation,
                full_Bloch_residual_Ha=full,
                signed_heff_projection_Ha=heff_error,
                FCI_method_error_Ha=all_roots_error,
                complete_rows=c["omega"].shape[0],
                complete_columns=8,
            )
    raw = json.loads((run / "raw_samples.json").read_text())
    reader_path = (
        repo / "docs/atomic_pci_sd_native_2026_10_09/parallel_fscc012/probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "preserved_tensor_reader", reader_path
    )
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    baselines = {
        case: complete_arrays(reader, run / f"accuracy_{case}_corrected", spinors)
        for case, spinors in (("two_core", 10), ("four_core", 12))
    }
    repeat_max_error = 0.0
    for row in raw["backend"]:
        assert (
            row["correctness_pass"]
            and row["actual_runtime"]["observed_backend_max_threads"] <= 8
        )
        case = (
            "two_core"
            if Path(row["directory"]).name.startswith("wrapper_pair")
            else "four_core"
        )
        observed = complete_arrays(
            reader, owner / row["directory"], 10 if case == "two_core" else 12
        )
        error = max(
            float(np.max(abs(value - baselines[case][key])))
            for key, value in observed.items()
        )
        assert error <= 1e-11
        repeat_max_error = max(repeat_max_error, error)
        for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json"):
            assert row["inputs"][name] == result["inputs"][case][name]
        assert all(
            sha(p) == value
            for p, value in row["actual_runtime"][
                "mapped_runtime_library_sha256"
            ].items()
        )
    for field in (
        "speedup",
        "logical_requested_team_parallel_efficiency",
        "observed_process_thread_resource_efficiency",
    ):
        assert result["backend_scaling"]["4"][field] is None
        assert result["backend_scaling"]["8"][field] is None
    for archive in sorted(run.glob("finite_ci_*.npz")):
        with np.load(archive) as a:
            for part in ("core", "attachment"):
                h, v, e = (
                    a[part + "_H_Ha"],
                    a[part + "_all_vectors"],
                    a[part + "_all_roots_Ha"],
                )
                assert np.max(abs(h @ v - v * e)) <= 1e-11
                assert np.max(abs(v.conj().T @ v - np.eye(len(e)))) <= 1e-12
    # Historical source bytes are checked in their actual preserved archives;
    # they are not re-labelled as current benchmark.py.
    archived_epochs = {}
    for epoch in ("run001", "run002", run.name):
        contract = json.loads((owner / epoch / "contract.json").read_text())
        hashes = contract["source_sha256"]
        with tarfile.open(owner / epoch / "source_inputs.tar.gz") as archive:
            for key, value in hashes.items():
                original = archive.extractfile(key).read()
                assert hashlib.sha256(original).hexdigest() == value
        archived_epochs[epoch] = {
            "verified_source_files": len(hashes),
            "source_archive_sha256": sha(owner / epoch / "source_inputs.tar.gz"),
        }
    for name in ("thread_preflight001", "thread_preflight002"):
        folder = owner / name
        contract = json.loads((folder / "contract.json").read_text())
        for original, value in contract["source_sha256"].items():
            assert sha(folder / ("source_" + Path(original).name)) == value
    prior = json.loads((owner / "run002/timing_review_withdrawal.json").read_text())
    assert prior["original_result_sha256"] == sha(owner / "run002/result.json")
    assert prior["performance_speedup"] is None
    proof = {
        "status": "complete_evidence_review_pass",
        "run_result_sha256": sha(run / "result.json"),
        "output_manifest_sha256": sha(run / "output_sha256.json"),
        "verified_output_files": len(manifest),
        "verified_current_source_files": len(result["source_sha256"]),
        "archived_source_epochs": archived_epochs,
        "backend_samples": len(raw["backend"]),
        "independently_redecoded_complete_repeat_arrays_max_error_Ha": repeat_max_error,
        "finite_CI_samples": len(raw["finite_ci"]),
        "accuracy": accuracy_rows,
        "old_requested_only_resource_failure_retained": True,
        "polling_timer_review_retained": True,
        "actual_runtime_dependency_hashes_pass": True,
        "highest_level_reached": False,
    }
    (owner / "validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
