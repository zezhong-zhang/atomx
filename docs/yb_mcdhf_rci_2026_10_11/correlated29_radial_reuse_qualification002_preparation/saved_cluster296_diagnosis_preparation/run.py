"""Bounded stored-array diagnosis; no Atomx, spectral solver, or route certificate."""
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


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
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
    assert not (HERE / "actual").exists(), "Once-only output already exists"
    assert 30 in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {contract["resources"]["CPU"]})
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    ceiling = int(contract["resources"]["memory_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))

    def timeout(_signum, _frame):
        raise TimeoutError("Original30-second saved-array budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["inner_seconds"])
    return contract, seal


def calculate(contract, output, counts):
    import numpy as np

    assert np.__version__ == contract["resources"]["NumPy_version"]

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Forbidden new eigen/SVD/QR call")

    for name in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "qr"):
        setattr(np.linalg, name, forbidden)
    fixed = contract["fixed"]
    n = fixed["n_csf"]
    arrays = []
    for tag in ("ordinary", "candidate"):
        counts["NPZ_loads"] += 1
        assert counts["NPZ_loads"] <= contract["resources"]["NPZ_loads"]
        with np.load(REPO / contract["inputs"][tag + "_arrays"]["path"],
                     allow_pickle=False) as stored:
            arrays.append({name: stored[name].copy() for name in
                           ("H", "energies", "mixing", "keys", "P", "Q")})
    ordinary, candidate = arrays
    for values in arrays:
        for name, shape in (("H", (n, n)), ("mixing", (n, n)), ("energies", (n,))):
            assert values[name].shape == shape
            assert values[name].dtype == np.dtype("float64")
            assert np.isfinite(values[name]).all()
    for name in ("keys", "P", "Q"):
        assert ordinary[name].dtype == candidate[name].dtype
        assert ordinary[name].shape == candidate[name].shape
        assert ordinary[name].tobytes() == candidate[name].tobytes()
    sectors0, sectors1 = [json.loads((REPO / contract["inputs"][tag + "_sectors"]["path"]).read_text())
                          for tag in ("ordinary", "candidate")]
    assert sectors0 == sectors1
    target0, target1 = [json.loads((REPO / contract["inputs"][tag + "_targets"]["path"]).read_text())
                        for tag in ("ordinary", "candidate")]
    targets = target0["targets"]
    assert len(targets) == 5
    assert [t["column"] for t in targets] == fixed["five_target_global_columns"]
    assert [(t["label"], t["UID"], t["column"]) for t in targets] == [
        (t["label"], t["UID"], t["column"]) for t in target1["targets"]]
    assert target0["residual_metric_sha256"] == target1["residual_metric_sha256"]
    failure = json.loads((REPO / contract["inputs"]["failed_comparison"]["path"]).read_text())
    failed_checks = [row for row in failure["checks"] if not row["passed"]]
    assert len(failed_checks) == 1
    assert failed_checks[0]["name"] == "signed_cluster_[296]"
    assert failed_checks[0]["error"] == fixed["expected_failure_error"]
    sector = next(row for row in sectors0 if row["sector"] == fixed["failed_sector"])
    local = fixed["failed_sector_local_root"]
    root = fixed["failed_global_column"]
    columns, csf_rows = sector["columns"], sector["CSF_rows"]
    assert columns[local] == root and 0 < local < len(columns) - 1
    neighbors = [columns[local - 1], columns[local + 1]]
    requested = sorted(set([root, *neighbors, *(t["column"] for t in targets)]))
    assert len(requested) <= contract["resources"]["max_unique_root_columns"]
    counts["unique_root_columns"] = len(requested)
    h0, h1 = ordinary["H"], candidate["H"]
    c0, c1 = ordinary["mixing"], candidate["mixing"]
    e0, e1 = ordinary["energies"], candidate["energies"]
    dh = h1 - h0
    location = tuple(int(i) for i in np.unravel_index(np.argmax(abs(dh)), dh.shape))
    offdiag = dh.copy()
    np.fill_diagonal(offdiag, 0.)
    matrix_difference = {
        "max_abs_Ha": float(np.max(abs(dh))),
        "Frobenius_Ha": float(np.linalg.norm(dh)),
        "max_diagonal_abs_Ha": float(np.max(abs(np.diag(dh)))),
        "max_offdiagonal_abs_Ha": float(np.max(abs(offdiag))),
        "largest_entry_indices": list(location),
        "largest_entry_signed_Ha": float(dh[location]),
        "ordinary_max_symmetry_error_Ha": float(np.max(abs(h0 - h0.T))),
        "candidate_max_symmetry_error_Ha": float(np.max(abs(h1 - h1.T))),
    }
    root_rows = []
    for column in requested:
        a, b = c0[:, column], c1[:, column]
        phase = 1 if float(a @ b) >= 0 else -1
        root_rows.append({
            "column": column, "ordinary_energy_Ha": float(e0[column]),
            "candidate_energy_Ha": float(e1[column]),
            "energy_difference_Ha": float(e1[column] - e0[column]),
            "ordinary_norm": float(np.linalg.norm(a)), "candidate_norm": float(np.linalg.norm(b)),
            "ordinary_own_residual_2norm_Ha": float(np.linalg.norm(h0 @ a - e0[column] * a)),
            "candidate_own_residual_2norm_Ha": float(np.linalg.norm(h1 @ b - e1[column] * b)),
            "ordinary_vector_in_candidate_residual_2norm_Ha": float(np.linalg.norm(h1 @ a - e0[column] * a)),
            "candidate_vector_in_ordinary_residual_2norm_Ha": float(np.linalg.norm(h0 @ b - e1[column] * b)),
            "chosen_whole_column_phase": phase,
            "phase_aligned_vector_2norm": float(np.linalg.norm(phase * b - a)),
            "projector_max_abs": float(np.max(abs(np.outer(a, a) - np.outer(b, b)))),
        })
    failed_c0 = c0[np.ix_(csf_rows, [root])]
    failed_c1 = c1[np.ix_(csf_rows, [root])]
    replay_error = float(np.max(abs(failed_c0 @ failed_c0.T - failed_c1 @ failed_c1.T)))
    neighbor_rows = []
    for column in neighbors:
        gap0, gap1 = float(e0[column] - e0[root]), float(e1[column] - e1[root])
        assert gap0 != 0. and gap1 != 0.
        coupling0 = float(c0[:, column] @ dh @ c0[:, root])
        coupling1 = float(c1[:, column] @ dh @ c1[:, root])
        neighbor_rows.append({
            "column": column, "ordinary_signed_gap_Ha": gap0, "candidate_signed_gap_Ha": gap1,
            "ordinary_basis_signed_dH_coupling_Ha": coupling0,
            "candidate_basis_signed_dH_coupling_Ha": coupling1,
            "ordinary_coupling_over_signed_gap": coupling0 / gap0,
            "candidate_coupling_over_signed_gap": coupling1 / gap1,
            "cross_endpoint_stored_vector_overlap": float(c0[:, column] @ c1[:, root]),
            "scope": "Neighbor-only first-order heuristic; no uniform eigenvector perturbation bound",
        })
    gap = fixed["original_cluster_gap_Ha"]
    target_rows = []
    for target in targets:
        block = next(row for row in sectors0 if row["sector"] == target["UID"][:2])
        all_columns, rows = block["columns"], block["CSF_rows"]
        index = all_columns.index(target["column"])
        assert index == target["UID"][2]
        lo = hi = index
        while lo > 0 and min(abs(e0[all_columns[lo]] - e0[all_columns[lo - 1]]),
                             abs(e1[all_columns[lo]] - e1[all_columns[lo - 1]])) <= gap:
            lo -= 1
        while hi + 1 < len(all_columns) and min(abs(e0[all_columns[hi + 1]] - e0[all_columns[hi]]),
                                               abs(e1[all_columns[hi + 1]] - e1[all_columns[hi]])) <= gap:
            hi += 1
        cluster = all_columns[lo:hi + 1]
        assert len(cluster) <= contract["resources"]["max_cluster_rank"]
        a, b = c0[np.ix_(rows, cluster)], c1[np.ix_(rows, cluster)]
        delta = a @ a.T - b @ b.T
        outer_gaps = []
        if lo:
            outer_gaps.append(float(abs(e0[all_columns[lo]] - e0[all_columns[lo - 1]])))
        if hi + 1 < len(all_columns):
            outer_gaps.append(float(abs(e0[all_columns[hi + 1]] - e0[all_columns[hi]])))
        target_rows.append({
            "label": target["label"], "UID": target["UID"], "column": target["column"],
            "cluster_columns": cluster, "projector_max_abs": float(np.max(abs(delta))),
            "projector_Frobenius": float(np.linalg.norm(delta)),
            "original_limit": fixed["original_projector_gate"],
            "under_original_limit_as_target_only_observation": bool(np.max(abs(delta)) <= fixed["original_projector_gate"]),
            "nearest_outer_gap_ordinary_Ha": min(outer_gaps) if outer_gaps else None,
        })
    counts["target_clusters"] = len(target_rows)
    assert counts["target_clusters"] <= contract["resources"]["max_target_clusters"]
    result = {
        "status": "stored_array_diagnosis_completed",
        "route_qualification_passed": False, "original_failure_preserved": True,
        "scope": fixed["report_domain"], "new_physical_calls": 0,
        "matrix_difference": matrix_difference, "root_rows": root_rows,
        "failed_root": {"sector": sector["sector"], "sector_root_count": len(columns),
                        "local_root": local, "column": root,
                        "stored_first_failure": failed_checks[0],
                        "saved_projector_error_replay": replay_error,
                        "replay_minus_stored_error": replay_error - failed_checks[0]["error"],
                        "still_exceeds_original_limit": bool(replay_error > fixed["original_projector_gate"]),
                        "ordinary_nearest_same_sector_gap_Ha": float(min(abs(e0[root] - e0[k]) for k in columns if k != root)),
                        "candidate_nearest_same_sector_gap_Ha": float(min(abs(e1[root] - e1[k]) for k in columns if k != root))},
        "neighbor_rows": neighbor_rows, "five_target_cluster_observations": target_rows,
        "original_metric_and_indicator_failures": [
            {"label": tag, "residual": data["residual"], "indicator_Ha": data["indicator_Ha"],
             "stationarity_passed": data["stationarity_passed"], "indicator_passed": data["indicator_passed"]}
            for tag, data in (("ordinary", target0), ("candidate", target1))],
        "counts": counts,
        "limits": "No spectral norm, eigenvectors, global perturbation bound, performance certificate or corrected physical spectrum was produced.",
    }
    write(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True, type=Path)
    args = parser.parse_args()
    contract, seal = admit(args.execution)
    output = HERE / "actual"
    output.mkdir()
    start = time.monotonic()
    counts = {"NPZ_loads": 0, "unique_root_columns": 0, "target_clusters": 0,
              "H": 0, "CI": 0, "gradient": 0, "eigen": 0, "SVD": 0,
              "QR": 0, "Dirac": 0, "optimizer": 0, "GPU": 0}
    write(output / "once_marker.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                       "execution_sha256": sha(args.execution),
                                       "root_seal": seal, "contract_sha256": sha(HERE / "contract.json")})
    try:
        calculate(contract, output, counts)
        for row in contract["inputs"].values():
            assert sha(REPO / row["path"]) == row["sha256"]
    except BaseException:
        write(output / "failure.json", {"route_qualification_passed": False,
                                        "traceback": traceback.format_exc(), "counts": counts})
        raise
    finally:
        signal.alarm(0)
        write(output / "receipt.json", {"elapsed_seconds": time.monotonic() - start,
                                        "maximum_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                        "affinity": sorted(os.sched_getaffinity(0)), "counts": counts,
                                        "original_failure_preserved": True})


if __name__ == "__main__":
    main()
