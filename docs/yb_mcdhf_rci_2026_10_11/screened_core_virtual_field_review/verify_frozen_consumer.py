"""Read only the frozen screened consumer and sources; no physical imports."""

import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FINAL = {
    "run.py": "91146b180b1c60a9daed52b373f34d1f5d8d5c8ac44a8cf74232df336c69b996",
    "contract.json": "7d8ac218370b77d55cee19ddd40597fdc3960216c281a8607dabc5519edda201",
    "screened_operator.py": "4b50850d1f78d2ec8793a6bd02a25fcac36613cd7f3d61981bd7b42fdd976742",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    prospective = json.loads((HERE / "frozen_consumer_contract.json").read_text())
    cap = prospective["resources"]["address_space_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {original_affinity[0]})
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(name)

    consumer = ROOT / prospective["consumer_root"]
    consumed = ROOT / prospective["actual_consumed_source_root"]
    before = {}
    for name, expected in FINAL.items():
        path = consumer / name
        check("frozen_preparation/" + name, sha(path) == expected)
        before[str(path.relative_to(ROOT))] = sha(path)
    contract = json.loads((consumer / "contract.json").read_text())
    fixed = contract["fixed_conditions"]
    check("source_exact_consumed_root", fixed["frozen_source_root"] == prospective["actual_consumed_source_root"])
    check("source_exact_epoch", fixed["source_map_sha256"] == prospective["actual_consumed_source_map_sha256"])
    identity_path = ROOT / prospective["actual_consumed_source_identity"]
    identity = json.loads(identity_path.read_text())
    before[str(identity_path.relative_to(ROOT))] = sha(identity_path)
    source_map = identity["all_atomx_python_sha256"]
    check("source_complete353", len(source_map) == prospective["actual_consumed_source_count"])
    check("source_identity_epoch", identity["source_map_sha256"] == fixed["source_map_sha256"])
    for path, expected in source_map.items():
        archived = consumed / path.removeprefix("src/")
        check("frozen_source/" + path, sha(archived) == expected)
        before[str(archived.relative_to(ROOT))] = expected
    check("matrix_exact_old_consumed_epoch", source_map["src/atomx/physics/csf/matrix.py"]
          == prospective["expected_consumed_matrix_sha256"])
    prior_review = json.loads((HERE / "review.json").read_text())
    for row in prior_review["reviewed_source"]:
        check("prior71_review_consumed_source/" + row["path"], source_map[row["path"]] == row["sha256"])
    for path, expected in contract["inputs_sha256"].items():
        check("consumer_scientific_input/" + path, sha(ROOT / path) == expected)
        before[path] = expected
    check("point_160_exact_identity", fixed["checkpoint"]["manifest_sha256"]
          == "1ce49cd2bcc49bb22ed7bf21ec8f1e21fe3f37f945a2ef5ca3a3669e1a15c93c"
          and fixed["checkpoint"]["arrays_sha256"]
          == "b4ddfbd8cdb791d16bf01fb942dae1f9e5f387f60d080e18507aacea34a6b3a1")
    check("original_metric_exact", fixed["original_metric_sha256"]
          == "aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe")
    check("original_stationarity_failure_retained", fixed["checkpoint"]["stationarity_gates_passed"] is False
          and fixed["checkpoint"]["original_residual"] == 4.717047646600387e-5)
    core = fixed["new_virtuals"]["closed_core"]
    check("full19_closed68_sources", len(core) == 19 and sum(row["occupation"] for row in core) == 68
          and all(row["occupation"] == 2 * abs(row["key"][1]) for row in core))
    callbacks = []
    for k in fixed["new_virtuals"]["kappas"]:
        l = k if k > 0 else -k - 1
        ranks = sum(sum((rank + l + (row["key"][1] if row["key"][1] > 0 else -row["key"][1] - 1)) % 2 == 0
                        for rank in range(abs(abs(k) - abs(row["key"][1])), abs(k) + abs(row["key"][1])))
                    for row in core)
        callbacks.append({"kappa": k, "ordered_16pairs_callback_upper": 16 * (19 + ranks)})
    check("integer_scalar_callback_upper_within_budget", sum(row["ordered_16pairs_callback_upper"] for row in callbacks)
          == 3584 < contract["resources"]["scalar_radial_callback_calls_budget"])
    helper = (consumer / "screened_operator.py").read_text()
    driver = (consumer / "run.py").read_text()
    for name, text in (("helper", helper), ("driver", driver)):
        tree = ast.parse(text)
        physical_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                            and ((isinstance(node, ast.ImportFrom) and (node.module or "").startswith("atomx"))
                                 or (isinstance(node, ast.Import) and any(alias.name.startswith("atomx") for alias in node.names)))]
        check("no_toplevel_atomx_import/" + name, not physical_imports)
    structural = {
        "zero_target_explicit_DFS_adapter": "DFSolver.__new__(DFSolver)" in helper,
        "core_source_only_coefficient_row": "core.get(key, 0) for key in keys" in helper,
        "normalization_primary1_scaled3": "for scale in (1., 3.):" in helper and "normalizations = np.ones(len(keys))" in helper,
        "workspace_cache_exact_match": "geometry.cache.same_grid(data)" in helper and "field._convol_cache = geometry.cache" in helper,
        "all_source_samekappa_complete_shape_guards": "auxiliary_complete_same_kappa_source_order" in helper and "auxiliary_source_shape/" in helper,
        "full_quad_no_skip": "skip_pq_exchange=False" in helper and "np.block([[pp, pq], [qp, qq]])" in helper,
        "h_nuclear_added_once": "full = h_nuclear + jk" in helper,
        "public_scalar_core_and_cell_oracle": "analytic = closed_core_one_body(probes, core, radial, bare_one_body=bare)" in helper and "compute_slater_rk_cell(*probe_waves[a], *probe_waves[b]," in helper,
        "all80_signed_quadrants_guard": "auxiliary_scalar_exact80_entries" in helper and "auxiliary_scalar_cross_relative/" in helper and "auxiliary_measured_cross_anchor/" in helper,
        "frozen_source_insert_import_boundary": "sys.path.insert(0, str(source_root))" in driver and "Auxiliary Atomx import escaped frozen10d481" in driver,
        "original_P_metric_and_inverse_byte_guards": "complete_original_metric_SHA" in driver and "original_P_inverse_bytes/" in driver and "coordinate_oldwave/" in driver,
        "all_old_samekappa_complement": "frozen.T @ metric @ projector" in driver and "virtual_whole_SGram/" in driver,
        "aux_field_not_retained_CI": "auxiliary_operator_not_retained_H=True" in driver and "include_breit=False, max_determinants=2000" in driver,
        "original_tracker_equalweights": "tracker.resolve(space, ci, current, geometry.metric)" in driver and "np.repeat(.2, 5)" in driver,
        "failed_stationarity_no_expanded_claim": "original_stationarity_passed=False, expanded_stationarity_evaluated=False" in driver,
        "new_exact_root_authorization_required": "Missing/mismatched NEW root binding; no science" in driver,
        "counter_initial_plus_CI_onebody_separate": "initial_one_body_matrix_attempts" in driver and "internal_CI_onebody_attempts" in driver,
    }
    for name, passed in structural.items():
        check("consumer_structure/" + name, passed)
    caps = contract["resources"]
    check("cap_exact10JK_1scalar_2CI_4H_zeroopt", caps["screened_JK_builder_calls_budget"] == 10
          and caps["closed_core_one_body_calls_budget"] == 1 and caps["complete_CI_calls_budget"] == 2
          and caps["total_many_electron_H_builds_budget"] == 4 and caps["optimizer_calls_budget"] == 0
          and caps["orbital_gradients_budget"] == 0 and caps["nuclear_Dirac_spectral_calls"] == 0)
    check("cap_distinct_initial7_CIinternal28", caps["initial_one_body_matrix_kappa_count"] == 7
          and caps["internal_CI_onebody_builds_budget"] == 28)
    check("cross_anchor_prospective_nonzero_gates", contract["acceptance"]["auxiliary_cross_bilinear_magnitude_strict_min_Ha"] == 1e-7
          and contract["acceptance"]["auxiliary_scalar_PQ_QP_relative_error"] == 1e-7
          and contract["acceptance"]["auxiliary_scalar_bilinear_error_Ha"] == 5e-9)
    prior_proof = json.loads((HERE / "verification.json").read_text())
    check("prior71_proof_preserved", prior_proof["all_passed"] and prior_proof["checks_passed"] == 71)
    check("prior71_exact_driver_copy", sha(HERE / "verify_readonly071_source_epoch.py")
          == prior_proof["owned_inputs_sha256"]["verify_readonly.py"])
    check("prior71_exact_readme_copy", sha(HERE / "README071_source_epoch.md")
          == prior_proof["owned_inputs_sha256"]["README.md"])
    refusal = json.loads((HERE / "readonly_source_refusal002.json").read_text())
    check("mutable_source_drift_refusal_preserved", refusal["no_physical_calls"]
          and refusal["reviewed_old_source_sha256"] == prospective["expected_consumed_matrix_sha256"]
          and refusal["current_source_sha256"] != prospective["expected_consumed_matrix_sha256"])
    check("before_after_bound_inputs_identical", all(sha(ROOT / path) == expected for path, expected in before.items()))
    check("no_atomx_imported", not any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    result = {
        "all_passed": True, "checks_passed": len(checks), "checks": checks,
        "read_only_consumer_review_passed": True, "actual_physical_qualification": False,
        "source_root": prospective["actual_consumed_source_root"], "source_map_sha256": fixed["source_map_sha256"],
        "consumer_inputs_sha256": before, "freeze_receipt_sha256": FINAL,
        "own_inputs_sha256": {name: sha(HERE / name) for name in ("frozen_consumer_contract.json", "verify_frozen_consumer.py", "review.json", "verification.json", "readonly_source_refusal002.json")},
        "integer_selection_callback_upper": callbacks, "total_callback_upper": 3584,
        "science_calls": 0, "tests_executed": 0, "atomx_imports": 0,
        "original_stationarity_passed": False, "source_mutable_successor_excluded": True,
        "resources": {"original_affinity": original_affinity, "chosen_CPU": original_affinity[0],
                      "address_space_cap_bytes": cap, "outer_timeout_seconds": 30,
                      "elapsed_seconds": time.monotonic() - start, "max_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "verdict": "No blocking algebra/source-identity/consumer issue found. Actual all-quadrant scalar and original-space numerical gates remain prospective for root's separately authorized execution.",
    }
    path = HERE / "frozen_consumer_verification.json"
    if path.exists():
        raise FileExistsError("Preserve existing independent proof")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_passed": True, "checks_passed": len(checks), "sha256": sha(path)}))


if __name__ == "__main__":
    main()
