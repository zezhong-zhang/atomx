"""Independent saved H/CI/gradient algebra; zero Atomx/science calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-contract", required=True, type=Path)
    args = parser.parse_args()
    contract = load(args.verification_contract)
    if (contract.get("saved_array_review_authorized") is not True
            or contract.get("verifier_sha256") != sha(Path(__file__))
            or contract.get("new_H_CI_gradient_optimizer_calls") != 0
            or contract.get("mode") not in ("failed_parent_arrays_only", "complete_successor")):
        raise ValueError("Missing exact prospective saved-array review authorization")
    for name, expected in contract["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Immutable saved input changed: " + name)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3 or contract["chosen_CPU"] != affinity[-3]:
        raise ValueError("Prospective original accounting CPU affinity changed")
    os.sched_setaffinity(0, {contract["chosen_CPU"]})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    cap = 2**30
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Saved-array30s cap")))
    signal.alarm(30)
    started = time.monotonic()
    import numpy as np

    directory = ROOT / contract["data_directory"]
    science_contract = load(directory / "contract.json")
    gates, fixed = science_contract["acceptance"], science_contract["fixed_conditions"]
    archive = np.load(directory / "arrays.npz", allow_pickle=False)
    original = np.load(ROOT / contract["original_point_arrays"], allow_pickle=False)
    layout = load(ROOT / contract["original_layout"])
    records = load(ROOT / contract["basis_identity"])["CSFs"]
    checks = []

    def check(name, error, limit=0.):
        error = float(error)
        checks.append({"name": name, "error": error, "limit": limit,
                       "passed": bool(np.isfinite(error) and error <= limit)})

    for name, expected in science_contract["inputs_sha256"].items():
        check("science_input:" + name, sha(ROOT / name) != expected)
    source = load(ROOT / science_contract["candidate_source_identity_path"])
    source_dir = ROOT / science_contract["candidate_frozen_source_path"]
    actual_source = {"src/" + str(p.relative_to(source_dir)): sha(p)
                     for p in sorted((source_dir / "atomx").rglob("*.py"))}
    check("actual_complete354_source", len(actual_source) != 354
          or actual_source != source["all_atomx_python_sha256"]
          or canonical_sha(actual_source) != science_contract["candidate_source_map_required_sha256"])
    for name in original.files:
        check("original_point_all_saved_fields:" + name,
              not np.array_equal(original[name], archive[name]))
    check("complete_point_dimension_and_finiteness", archive["x"].shape != (2856,)
          or archive["P"].shape != (24, 123) or archive["Q"].shape != (24, 123)
          or archive["orbital_keys"].shape != (24, 2)
          or any(not np.isfinite(archive[n]).all() for n in archive.files))
    keys = [tuple(map(int, row)) for row in archive["orbital_keys"]]
    positions = {key: i for i, key in enumerate(keys)}
    check("complete24_distinct_keys", len(keys) != len(set(keys)) or len(keys) != 24)
    check("raw_gradient_order_matches_point_keys",
          not np.array_equal(archive["radial_gradient_keys"], archive["orbital_keys"]))
    for model in ("ordinary", "retained"):
        check(model + "/raw_gradient_finite246x24", archive[model + "_radial_gradient"].shape != (24, 246)
              or archive[model + "_radial_gradient"].dtype != np.float64)
        check(model + "/g_pg_finite2856", any(archive[model + suffix].shape != (2856,)
              or archive[model + suffix].dtype != np.float64
              for suffix in ("_projected_g", "_preconditioned_pg")))
        check(model + "/all24_raw_gradient_equality", np.max(abs(archive[model + "_radial_gradient"]
              - archive["ordinary_radial_gradient"])))
        check(model + "/complete_g_originalpoint_replay", np.max(abs(archive[model + "_projected_g"]
              - original["original_projected_gradient"])), gates["complete_projected_coordinate_g_replay_max"])
        check(model + "/complete_pg_originalpoint_replay", np.max(abs(archive[model + "_preconditioned_pg"]
              - original["original_preconditioned_gradient"])), gates["complete_preconditioned_pg_replay_max"])
    digest = hashlib.sha256()
    independent_g = {m: np.zeros(2856) for m in ("ordinary", "retained")}
    independent_pg = {m: np.zeros(2856) for m in ("ordinary", "retained")}
    for i, row in enumerate(layout):
        group_keys = [tuple(key) for key in row["keys"]]
        shape, bounds = tuple(row["shape"]), tuple(row["slice"])
        p = archive[f"projector_{i}"]
        digest.update(repr((group_keys, shape)).encode())
        digest.update(p.tobytes())
        y = archive["x"][slice(*bounds)].reshape(shape)
        s = archive[f"actual_S_{i}"]
        radial_columns = np.column_stack([np.r_[archive["P"][positions[key]], archive["Q"][positions[key]]]
                                          for key in group_keys])
        check(f"projector_Px_exact_actual_waves:{i}", not np.array_equal(p @ y, radial_columns))
        check(f"original_P_S_P:{i}", np.max(abs(p.T @ s @ p - np.eye(shape[0]))), gates["radial_Gram_error"])
        check(f"actual_radial_Gram:{i}", np.max(abs(radial_columns.T @ s @ radial_columns
              - np.eye(shape[1]))), gates["radial_Gram_error"])
        check(f"coordinate_Gram:{i}", np.max(abs(y.T @ y - np.eye(shape[1]))), gates["radial_Gram_error"])
        outside = radial_columns - p @ (p.T @ s @ radial_columns)
        check(f"no_pair_original_space:{i}",
              np.sqrt(max(0., float(np.max(np.sum(outside * (s @ outside), axis=0))))),
              gates["projector_leakage_norm"])
        for model in independent_g:
            # Independent per-column projection; do not call producer helper.
            projected = np.column_stack([p.T @ archive[model + "_radial_gradient"][positions[key]]
                                         for key in group_keys])
            multipliers = .5 * (y.T @ projected + projected.T @ y)
            g = projected - y @ multipliers
            z = np.column_stack([archive[f"inverse_blocks_{i}"][j] @ g[:, j]
                                 for j in range(shape[1])])
            pg = z - y @ (.5 * (y.T @ z + z.T @ y))
            independent_g[model][slice(*bounds)] = g.ravel()
            independent_pg[model][slice(*bounds)] = pg.ravel()
    digest.update(archive["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for inverse in archive[f"inverse_blocks_{i}"]:
            digest.update(inverse.tobytes())
    check("original_legacy_metric_byte_SHA", digest.hexdigest() != fixed["original_metric_sha256"])
    residuals = {}
    for model in independent_g:
        g, pg = independent_g[model], independent_pg[model]
        check(model + "/independent_raw_radial_to2856g", np.max(abs(g - archive[model + "_projected_g"])),
              gates["complete_projected_coordinate_g_replay_max"])
        check(model + "/independent_original_inverse_tangent_pg",
              np.max(abs(pg - archive[model + "_preconditioned_pg"])),
              gates["complete_preconditioned_pg_replay_max"])
        residual, indicator = float(np.linalg.norm(pg)), .5 * abs(float(g @ pg))
        residuals[model] = {"original_orbital_residual": residual, "original_energy_indicator_Ha": indicator,
                            "original_stationarity_passed": residual <= 1e-7 and indicator <= 1e-8}
        check(model + "/original_stationarity_FAILURE_retained", residual <= 1e-7 or indicator >= 1e-8)
    symmetry = [(int(row["twoJ"]), int(row["parity"])) for row in records]
    groups = {}
    for row, key in zip(records, symmetry, strict=True):
        groups.setdefault(key, []).append(row["index"])
    check("complete531_16_signed_symmetries", len(records) != 531 or len(groups) != 16)
    columns_by_model = {}
    for model in ("ordinary", "retained"):
        h, e, v = (archive[model + suffix] for suffix in ("_Ha", "_energies_Ha", "_mixing"))
        check(model + "/full_H_CI_finite_dimensions", h.shape != (531, 531) or e.shape != (531,)
              or v.shape != (531, 531) or any(a.dtype != np.float64 for a in (h, e, v)))
        check(model + "/H_bitwise_original_and_ordinary", not np.array_equal(h, archive["matrix_Ha"])
              or not np.array_equal(h, archive["ordinary_Ha"]))
        check(model + "/all531_energy_vs_original", np.max(abs(e - archive["energies_Ha"])),
              gates["all_root_energy_vs_frozen320_max_Ha"])
        check(model + "/all531_energy_vs_ordinary", np.max(abs(e - archive["ordinary_energies_Ha"])))
        check(model + "/complete_CI_Gram", np.max(abs(v.T @ v - np.eye(531))), gates["CI_orthogonality"])
        check(model + "/Hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
        for col in range(531):
            vector = v[:, col]
            check(f"{model}/independent_Rayleigh:{col}", abs(float(vector @ h @ vector) - e[col]),
                  gates["CI_residual_Ha"])
            check(f"{model}/independent_eigenpair:{col}", np.linalg.norm(h @ vector - e[col] * vector),
                  gates["CI_residual_Ha"])
        columns_by_model[model] = {}
        for key, indices in groups.items():
            outside = [j for j in range(531) if j not in indices]
            check(model + "/cross_symmetry:" + str(key), np.max(abs(h[np.ix_(indices, outside)])),
                  gates["cross_symmetry_zero_Ha"])
            support = np.sum(v[indices]**2, axis=0)
            cols = np.flatnonzero(support > 1 - 1e-10).tolist()
            check(model + "/complete_signed_symmetry_columns:" + str(key), len(cols) != len(indices))
            columns_by_model[model][key] = cols
        weights = archive[model + "_state_weights"]
        check(model + "/five_equal_actual_weights", weights.shape != (531,) or np.count_nonzero(weights) != 5
              or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
    check("ordinary_retained_all_actual_weights_exact",
          not np.array_equal(archive["ordinary_state_weights"], archive["retained_state_weights"]))
    maximum_span_error = 0.
    for key, columns in columns_by_model["ordinary"].items():
        first = 0
        while first < len(columns):
            end = first + 1
            while end < len(columns) and (archive["ordinary_energies_Ha"][columns[end]]
                    - archive["ordinary_energies_Ha"][columns[end - 1]] <= gates["numerical_cluster_span_gap_Ha"]):
                end += 1
            candidate_cols = columns_by_model["retained"][key][first:end]
            reference = archive["ordinary_mixing"][:, columns[first:end]]
            candidate = archive["retained_mixing"][:, candidate_cols]
            saved_support = np.sum(archive["mixing"][groups[key]]**2, axis=0)
            saved_cols = np.flatnonzero(saved_support > 1 - 1e-10).tolist()[first:end]
            saved = archive["mixing"][:, saved_cols]
            error = max(float(np.max(abs(reference @ reference.T - candidate @ candidate.T))),
                        float(np.max(abs(reference @ reference.T - saved @ saved.T))))
            maximum_span_error = max(maximum_span_error, error)
            check(f"independent_signed_symmetry_span:{key}:{first}:{end}", error,
                  gates["same_signed_CI_span_projector_max"])
            first = end
    header = load(directory / "owner_header.json")
    check("owner_header_archive_exact_binding", header["saved_array_sha256"] != sha(directory / "arrays.npz"))
    check("owner_header_H_matrix_byte_identity", header["matrix_sha256"]
          != hashlib.sha256(repr((archive["retained_Ha"].shape, archive["retained_Ha"].dtype.str)).encode()
          + archive["retained_Ha"].tobytes()).hexdigest())
    check("owner_canonical_payload_input_SHA", header["input_sha256"] != canonical_sha(header["identity_payload"]))
    check("owner_complete_signed_CSF_N_and_operator", len(header["identity_payload"]["CSFs"]) != 531
          or any(row[0] != 69 for row in header["identity_payload"]["CSFs"])
          or header["identity_payload"]["hamiltonian_terms"] != ["dirac-coulomb"])
    check("owner_ordered24_radials_and_DC_cap", header["identity_payload"]["orbital_keys"]
          != archive["orbital_keys"].tolist() or header["identity_payload"]["max_determinants"] != 2000
          or header["identity_payload"]["radial_binding"]["Z"] != 70
          or header["identity_payload"]["radial_binding"]["ns"] != 123)
    if contract["mode"] == "complete_successor":
        result = load(directory / "result.json")
        rdm = load(directory / "signed_public_RDM.json")
        check("complete_actual_numeric_qualification_ledger", result["qualification_numeric_passed"] is not True
              or result["checks_count"] != len(result["checks"])
              or not all(row["passed"] for row in result["checks"]))
        check("actual_call_budget_completed", result["actual_calls"] != {"started": {
            "H_build": 3, "CI_solve": 2, "gradient": 2}, "completed": {"H_build": 3, "CI_solve": 2, "gradient": 2}}
              or result["optimizer_calls"] != 0)
        check("result_full_source_point_metric_arrays_binding", result["arrays_sha256"] != sha(directory / "arrays.npz")
              or result["candidate_source_map_sha256"] != science_contract["candidate_source_map_required_sha256"]
              or result["point_arrays_sha256"] != fixed["point_arrays_sha256"]
              or result["original_metric_sha256"] != digest.hexdigest())
        check("RDM_complete_signed_archive_count", rdm["count"] != len(rdm["terms"])
              or rdm["negative_terms"] != sum(row["value"] < 0 for row in rdm["terms"])
              or not all(np.isfinite(row["value"]) for row in rdm["terms"])
              or rdm["ordinary_retained_values_and_order_exact"] is not True)
        check("RDM_coverage_key_unique",
              len({json.dumps(row["key"]) for row in rdm["terms"]}) != len(rdm["terms"]))
        check("RDM_actual_full_public_signed_reuse_gate", not any(row["name"] == "all_public_signed_RDM_values_exact"
              and row["passed"] and row["error"] == 0 for row in result["checks"]))
        check("stale_owner_rejects_without_physical_fallback", len(result["stale_owner_controls"]) != 14
              or any(row["rejected"] is not True or row["extra_physical_calls"] != 0
                     for row in result["stale_owner_controls"]))
        check("no_unearned_stationarity_or_performance_claim", any(result[k] is not False for k in (
            "original_stationarity_passed", "production_certified", "experimental_gate_passed",
            "highest_level_reached", "controlled_performance_claim")))
        check("result_original_residual_replay", abs(result["actual_original_orbital_residual"]
              - residuals["retained"]["original_orbital_residual"]), 1e-10)
        check("result_original_indicator_replay", abs(result["actual_original_energy_indicator_Ha"]
              - residuals["retained"]["original_energy_indicator_Ha"]), 1e-12)
    else:
        check("failed_parent_not_certified", (directory / "result.json").exists()
              or not (directory / "failure.json").exists())
    check("zero_Atomx_import", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    check("saved_review_resource_caps", elapsed >= 30 or rss >= 1024
          or sorted(os.sched_getaffinity(0)) != [contract["chosen_CPU"]])
    result = {"all_passed": all(row["passed"] for row in checks), "checks_count": len(checks), "checks": checks,
              "mode": contract["mode"], "data_directory": contract["data_directory"],
              "arrays_sha256": sha(directory / "arrays.npz"), "owner_header_sha256": sha(directory / "owner_header.json"),
              "verifier_sha256": sha(Path(__file__)), "verification_contract_sha256": sha(args.verification_contract),
              "inputs_sha256": contract["inputs_sha256"], "independent_original_metrics": residuals,
              "max_signed_symmetry_span_error": maximum_span_error, "seconds": elapsed, "peak_RSS_MiB": rss,
              "Atomx_imports": 0, "new_H_builds": 0, "new_CI_solves": 0, "new_gradients": 0, "optimizer_calls": 0,
              "original_stationarity_passed": False, "production_certified": False,
              "RDM_recomputed_from_saved_angular_structure": False,
              "scope": "Independent saved H/CI rootwise Rayleigh, Gram, signed numerical-cluster spans, raw radial to original-metric g/pg and metadata integrity; no physical RDM contraction or new science evaluation."}
    (ROOT / contract["output_path"]).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    signal.alarm(0)
    print(json.dumps({"all_passed": result["all_passed"], "checks": len(checks), "seconds": elapsed,
                      "peak_RSS_MiB": rss, "failed": [row["name"] for row in checks if not row["passed"]]}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
