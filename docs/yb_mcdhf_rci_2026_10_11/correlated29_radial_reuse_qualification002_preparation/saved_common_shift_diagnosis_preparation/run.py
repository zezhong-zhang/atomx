"""Fixed common-shift complete saved-H discriminator; no production source import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def admit(execution):
    contract = json.loads((HERE / "contract.json").read_text())
    seal = json.loads(execution.read_text())
    assert seal["execution_authorized_by_root"] is True
    assert seal["driver_sha256"] == sha(Path(__file__))
    assert seal["contract_sha256"] == sha(HERE / "contract.json")
    assert seal["inputs"] == contract["inputs"]
    assert seal["resources"] == contract["resources"]
    for row in contract["inputs"].values():
        assert sha(REPO / row["path"]) == row["sha256"], row["path"]
    assert not (HERE / "actual").exists(), "Once-only result path already exists"
    assert contract["resources"]["CPU"] in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {contract["resources"]["CPU"]})
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    ceiling = int(contract["resources"]["memory_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))

    def timeout(_signum, _frame):
        raise TimeoutError("Fixed30-second common-shift budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["inner_seconds"])
    return contract, seal


def calculate(contract, output, counts, checks):
    import numpy as np
    import scipy
    from scipy import linalg

    assert np.__version__ == contract["resources"]["NumPy_version"]
    assert scipy.__version__ == contract["resources"]["SciPy_version"]

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Undeclared eigen/SVD/QR call")

    for name in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "qr"):
        setattr(np.linalg, name, forbidden)
    for name in ("eig", "eigvals", "eigvalsh", "svd", "qr"):
        setattr(linalg, name, forbidden)
    arrays = []
    for tag in ("ordinary", "candidate"):
        counts["NPZ_loads"] += 1
        assert counts["NPZ_loads"] <= contract["resources"]["NPZ_loads"]
        with np.load(REPO / contract["inputs"][tag + "_arrays"]["path"],
                     allow_pickle=False) as stored:
            arrays.append({key: stored[key].copy() for key in ("H", "energies", "mixing", "keys", "P", "Q")})
    n = contract["fixed"]["n_csf"]
    for values in arrays:
        for key, shape in (("H", (n, n)), ("mixing", (n, n)), ("energies", (n,))):
            assert values[key].dtype == np.dtype("float64")
            assert values[key].shape == shape and np.isfinite(values[key]).all()
    for key in ("keys", "P", "Q"):
        assert arrays[0][key].dtype == arrays[1][key].dtype
        assert arrays[0][key].shape == arrays[1][key].shape
        assert arrays[0][key].tobytes() == arrays[1][key].tobytes()
    sectors = [json.loads((REPO / contract["inputs"][tag + "_sectors"]["path"]).read_text())
               for tag in ("ordinary", "candidate")]
    assert sectors[0] == sectors[1] and len(sectors[0]) == 5
    assert sorted(i for row in sectors[0] for i in row["columns"]) == list(range(n))
    assert sorted(i for row in sectors[0] for i in row["CSF_rows"]) == list(range(n))
    targets = [json.loads((REPO / contract["inputs"][tag + "_targets"]["path"]).read_text())["targets"]
               for tag in ("ordinary", "candidate")]
    assert [(r["label"], r["UID"], r["column"]) for r in targets[0]] == [
        (r["label"], r["UID"], r["column"]) for r in targets[1]]
    assert len(targets[0]) == 5
    for row, declaration in zip(sectors[0], contract["fixed"]["complete_blocks"], strict=True):
        assert row["sector"] == declaration["sector"]
        assert len(row["columns"]) == len(row["CSF_rows"]) == declaration["roots"]
        assert row["columns"][0] == declaration["alpha_source_column"]
        assert arrays[0]["energies"][row["columns"][0]] == declaration["alpha_Ha"]
        target = next(t for t in targets[0] if t["UID"][:2] == row["sector"])
        assert target["UID"][2] == 0 and target["column"] == row["columns"][0]

    def gate(name, error, limit):
        checks.append({"name": name, "error": float(error), "limit": limit,
                       "passed": bool(np.isfinite(error) and error <= limit)})

    shifted = []
    block_records = []
    for endpoint, values in zip(("ordinary", "candidate"), arrays, strict=True):
        energies = np.zeros(n, dtype=np.float64)
        mixing = np.zeros((n, n), dtype=np.float64)
        shifted.append({"energies": energies, "mixing": mixing})
        for row, declaration in zip(sectors[0], contract["fixed"]["complete_blocks"], strict=True):
            rows, columns = row["CSF_rows"], row["columns"]
            block = values["H"][np.ix_(rows, rows)].copy()
            alpha = declaration["alpha_Ha"]
            block[np.diag_indices(len(rows))] -= alpha
            counts["scipy_eigh_started"] += 1
            assert counts["scipy_eigh_started"] <= contract["resources"]["scipy_eigh_calls"]
            write(output / "progress.json", {"counts": counts, "active_endpoint": endpoint,
                                              "active_sector": row["sector"], "alpha_Ha": alpha})
            start = time.monotonic()
            lambdas, vectors = linalg.eigh(block, subset_by_index=None, driver="evd")
            elapsed = time.monotonic() - start
            counts["scipy_eigh_completed"] += 1
            assert lambdas.shape == (len(columns),) and vectors.shape == block.shape
            assert np.isfinite(lambdas).all() and np.isfinite(vectors).all()
            energies[columns] = lambdas + alpha
            mixing[np.ix_(rows, columns)] = vectors
            block_records.append({"endpoint": endpoint, "sector": row["sector"],
                                  "complete_roots": len(columns), "alpha_Ha": alpha,
                                  "source_column": declaration["alpha_source_column"],
                                  "eigh_seconds_uncontrolled": elapsed})
            write(output / "progress.json", {"counts": counts, "completed_blocks": block_records})
    assert counts["scipy_eigh_started"] == counts["scipy_eigh_completed"] == 10
    original_correspondence = []
    for endpoint, old, new in zip(("ordinary", "candidate"), arrays, shifted, strict=True):
        residual = np.linalg.norm(old["H"] @ new["mixing"] - new["mixing"] * new["energies"], axis=0)
        gram = new["mixing"].T @ new["mixing"] - np.eye(n)
        gate(endpoint + "_original_H_complete_column_residual", np.max(residual),
             contract["fixed"]["column_residual_2norm_Ha"])
        gate(endpoint + "_complete_Gram", np.max(abs(gram)), contract["fixed"]["Gram_max_abs"])
        phases = np.where(np.sum(old["mixing"] * new["mixing"], axis=0) < 0., -1., 1.)
        differences = np.linalg.norm(new["mixing"] * phases - old["mixing"], axis=0)
        new["original_H_residual_column2norm_Ha"] = residual
        new["unshifted_alignment_phase"] = phases
        new["unshifted_phase_aligned_vector2norm"] = differences
        original_correspondence.append({"endpoint": endpoint,
                                        "max_phase_aligned_vector2norm": float(np.max(differences)),
                                        "column296_phase_aligned_vector2norm": float(differences[296]),
                                        "column296_original_H_residual_Ha": float(residual[296]),
                                        "max_abs_restored_energy_minus_unshifted_Ha": float(np.max(abs(new["energies"] - old["energies"]))),
                                        "mapped_global_energies_nondecreasing": bool((np.diff(new["energies"]) >= 0.).all())})
        for row in sectors[0]:
            outside = sorted(set(range(n)) - set(row["CSF_rows"]))
            gate(endpoint + "_exact_sector_support_" + str(row["sector"]),
                 np.max(abs(new["mixing"][np.ix_(outside, row["columns"])]), initial=0.), 0.)
    gate("complete_shifted_endpoint_energy_difference", np.max(abs(shifted[1]["energies"] - shifted[0]["energies"])),
         contract["fixed"]["complete_endpoint_energy_max_Ha"])

    def clusters(columns, first, second):
        start = 0
        for stop in range(1, len(columns) + 1):
            if stop == len(columns) or min(abs(first[columns[stop]] - first[columns[stop - 1]]),
                                          abs(second[columns[stop]] - second[columns[stop - 1]])) > contract["fixed"]["cluster_gap_Ha"]:
                yield columns[start:stop]
                start = stop

    span_records = []
    target_spans = []
    for row in sectors[0]:
        columns, rows = row["columns"], row["CSF_rows"]
        original_clusters = list(clusters(columns, arrays[0]["energies"], arrays[1]["energies"]))
        shifted_clusters = list(clusters(columns, shifted[0]["energies"], shifted[1]["energies"]))
        gate("same_original_cluster_membership_" + str(row["sector"]),
             int(original_clusters != shifted_clusters), 0.)
        for cluster in original_clusters:
            a = shifted[0]["mixing"][np.ix_(rows, cluster)]
            b = shifted[1]["mixing"][np.ix_(rows, cluster)]
            error = float(np.max(abs(a @ a.T - b @ b.T)))
            gate("complete_shifted_signed_cluster_" + str(cluster), error, contract["fixed"]["projector_gate"])
            record = {"sector": row["sector"], "columns": cluster, "projector_max_abs": error}
            span_records.append(record)
            for target in targets[0]:
                if target["column"] in cluster:
                    target_spans.append({"label": target["label"], "UID": target["UID"], **record})
    assert sorted(i for r in span_records for i in r["columns"]) == list(range(n))
    assert len(target_spans) == 5
    correspondence_spans = []
    for endpoint, old, new in zip(("ordinary", "candidate"), arrays, shifted, strict=True):
        for row in sectors[0]:
            for cluster in clusters(row["columns"], old["energies"], new["energies"]):
                a, b = old["mixing"][np.ix_(row["CSF_rows"], cluster)], new["mixing"][np.ix_(row["CSF_rows"], cluster)]
                correspondence_spans.append({"endpoint": endpoint, "sector": row["sector"], "columns": cluster,
                                             "projector_max_abs": float(np.max(abs(a @ a.T - b @ b.T))),
                                             "observation_only": True})
    values = {endpoint + "_" + key: value for endpoint, data in zip(("ordinary", "candidate"), shifted, strict=True)
              for key, value in data.items()}
    with (output / "shifted_arrays.npz").open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    failures = [row for row in checks if not row["passed"]]
    result = {"status": "diagnostic_passed" if not failures else "diagnostic_failed",
              "all_diagnostic_gates_passed": not failures, "route_qualification_passed": False,
              "original_route_failure_preserved": True, "counts": counts, "checks": checks,
              "block_calls": block_records, "shifted_complete_cluster_spans": span_records,
              "five_target_spans_no_exemption": target_spans,
              "original_unshifted_correspondence": original_correspondence,
              "original_unshifted_cluster_correspondence_observation_only": correspondence_spans,
              "column296_shifted_cluster": next(r for r in span_records if 296 in r["columns"]),
              "shifted_arrays_sha256": sha(output / "shifted_arrays.npz"),
              "source_inputs": contract["inputs"],
              "scope": "Only fixed-alpha saved-H complete eigensolves. No new physical operator/gradient, production source change or full route qualification."}
    write(output / "result.json", result)
    if failures:
        raise RuntimeError("Fixed-alpha diagnostic gates failed: " + str([r["name"] for r in failures]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True, type=Path)
    args = parser.parse_args()
    contract, seal = admit(args.execution)
    output = HERE / "actual"
    output.mkdir()
    start = time.monotonic()
    counts = {"NPZ_loads": 0, "scipy_eigh_started": 0, "scipy_eigh_completed": 0,
              "H": 0, "CI_construction": 0, "gradient": 0, "other_eigen": 0,
              "SVD": 0, "QR": 0, "Dirac": 0, "optimizer": 0, "GPU": 0}
    checks = []
    write(output / "once_marker.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                       "execution_sha256": sha(args.execution), "root_seal": seal})
    try:
        calculate(contract, output, counts, checks)
        for row in contract["inputs"].values():
            assert sha(REPO / row["path"]) == row["sha256"]
    except BaseException:
        write(output / "failure.json", {"traceback": traceback.format_exc(), "counts": counts,
                                        "checks": checks, "route_qualification_passed": False})
        raise
    finally:
        signal.alarm(0)
        write(output / "receipt.json", {"elapsed_seconds": time.monotonic() - start,
                                        "maximum_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                        "affinity": sorted(os.sched_getaffinity(0)), "counts": counts,
                                        "original_route_failure_preserved": True})


if __name__ == "__main__":
    main()
