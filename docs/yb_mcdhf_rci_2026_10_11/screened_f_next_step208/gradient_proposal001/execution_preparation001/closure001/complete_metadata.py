"""Once-only SOURCE/JSON closure; never open numerical arrays or import Atomx."""
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = next(p for p in BASE.parents if (p / "AGENTS.md").is_file())
CHECKS = []
INPUTS = {}


def sha(path):
    if path.suffix == ".npz":
        raise ValueError("NPZ reads and rehashes forbidden in this closure")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    CHECKS.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise ValueError(name)


def read(name):
    path = BASE / name
    INPUTS[str(path.relative_to(REPO))] = sha(path)
    return json.loads(path.read_text())


def key(value):
    return json.dumps(value, separators=(",", ":"))


def main():
    if (BASE / "completion.json").exists():
        raise FileExistsError("completion already exists; no retry or overwrite")
    contract = read("contract.json")
    prepared = read("handoff.json")
    read("preparation.json")
    receipt = read("root_execution.json")
    binding = read("execution_binding.json")
    result = read("actual/result.json")
    check("closed_exact_once", receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0 and receipt["no_retry"])
    check("root_receipt_identity", INPUTS[str((BASE / "root_execution.json").relative_to(REPO))] == "e6dca7629a7341e51e710cd6c17f5cea81f7c4aee2497df83e7f0c01790cba6f")
    check("actual_result_identity", INPUTS[str((BASE / "actual/result.json").relative_to(REPO))] == "cc085154a25df3a0dde5c3ba7f961662d57ecb2efacf75c070cc1378113ee0e0")
    for name, expected in prepared["files_sha256"].items():
        path = REPO / name
        check("preserved_preparation/" + name, sha(path) == expected)
        INPUTS[str(path.relative_to(REPO))] = expected
    check("authorized_source", result["source_map_sha256"] == prepared["source_map_sha256"] == "f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe")
    check("root_bound_contract_driver", result["contract_sha256"] == receipt["contract_sha256"] == prepared["contract_sha256"] and result["driver_sha256"] == receipt["driver_sha256"] == prepared["driver_sha256"])
    check("root_bound_authorization", result["exact_execution_binding_sha256"] == receipt["binding_sha256"] == sha(BASE / "execution_binding.json"))
    check("binding_is_root_authorization", binding["execution_authorized_by_root"] and binding["resources"] == contract["resources"])
    check("438_actual_live_controls", len(result["checks"]) == 438 and all(c["passed"] and c["error"] <= c["limit"] for c in result["checks"]) and result["all_declared_controls_passed"])
    check("attempted_completed_match", result["attempted"] == result["completed"])
    check("all_calls_bounded", all(result["attempted"][k] <= limit for k, limit in contract["resources"]["calls"].items()))
    check("actual_2H2CI3g_and_zero_optimizer", [result["completed"][k] for k in ["H", "CI", "gradient_from_weights", "optimizer"]] == [2, 2, 3, 0])
    check("actual_5_unfiltered_tables", result["completed"]["unfiltered_target_RDM"] == 5 and result["completed"]["retained_RDM"] == 1)
    hs = result["hS_requests"]
    check("hS_request_cache_identity", hs["attempted"] == hs["completed"] == hs["hits"] + hs["misses"] == 76 and hs["misses"] == result["completed"]["CI_internal_hS"] == 7)
    check("cache_carried_original_tuples", len(hs["carry_events"]) == 2 and [v["carried"] for v in hs["carry_events"]] == [0, 7] and all(v["identical_tuples"] for v in hs["carry_events"]))
    identities = {"6s_1/2": [1, 1, 0], "5d_3/2": [3, 1, 0], "5d_5/2": [5, 1, 0], "6p_1/2": [1, -1, 0], "6p_3/2": [3, -1, 0]}
    for point, evidence in result["points"].items():
        rows = {v["label"]: v for v in evidence["rows"]}
        check(point + "/complete_original_UIDs", set(rows) == set(identities) and all(v["UID"] == identities[label] and v["selected_level"] == v["UID"] and v["weight"] == .2 for label, v in rows.items()))
        check(point + "/reported_DC_not_rebased", evidence["operator"] == "dirac-coulomb" and evidence["original_tracker_not_rebased"] and evidence["physical_point"] == point)
        tracking = read("actual/" + point + "_tracking.json")
        owner = read("actual/" + point + "_owner.json")
        check(point + "/real_full_owner", owner["units"] == "Hartree" and len(owner["CSFs"]) == 1595 and len(owner["orbital_keys"]) == 31)
        check(point + "/signed_tracker_five_bindings", len(tracking["tracked"]) == 5 and all(t["per_level_weight"] == .2 for t in tracking["tracked"]))
    recovery = read("actual/point_recovery.json")
    check("separate_canonical_recovered_points", recovery["canonical_and_recovered_samepoint_claim"] is False and len(result["recovery"]) == 31)
    check("original29_coordinates_and_recovery_gate", sum(v["original29_coordinate_preserved"] for v in result["recovery"]) == 29 and max(v["canonical_vs_public_max"] for v in result["recovery"]) <= contract["gates"]["recovery_max"])
    layout = read("actual/new_layout.json")
    check("complete_seven_kappa3689_layout", len(layout) == 7 and sum(len(v["keys"]) for v in layout) == 31 and layout[-1]["slice"][1] == 3689)
    check("new_metric_not_old3451", result["new_metric_sha256"] == recovery["new3689_metric_SHA"] != result["original3451_metric_sha256"] and result["original3451_metric_sha256"] == recovery["original3451_metric_SHA"])
    objective = read("actual/new_objective_identity.json")
    check("new_objective_identity", objective["identity"] == result["objective_identity"] == recovery["new_objective_identity"])
    tables = read("actual/signed_tables.json")
    recovered = {v["label"]: v for v in result["points"]["recovered"]["rows"]}
    check("full_unfiltered_target_norm_binding", len(tables["targets"]) == 5 and all(t["UID"] == recovered[t["label"]]["UID"] and t["column"] == recovered[t["label"]]["column"] and int(t["exact_CI_norm"][0]) > 0 and int(t["exact_CI_norm"][1]) > 0 for t in tables["targets"]))
    check("normalization_and_zero_only_scope", tables["exact_zero_only"] and tables["coefficient_counts_are_not_bounds"] and tables["delta_order"] == "full order then retained-only original order")
    full = {key(k): float(v) for k, v in tables["full_normalized"]}
    retained = {key(k): float(v) for k, v in tables["retained"]}
    delta = {key(k): float(v) for k, v in tables["full_minus_retained"]}
    check("ordered_weight_tables_no_duplicate_keys", len(full) == len(tables["full_normalized"]) and len(retained) == len(tables["retained"]) and len(delta) == len(tables["full_minus_retained"]))
    check("complete_signed_delta_JSON_algebra", all(delta.get(k, 0.) == full.get(k, 0.) - retained.get(k, 0.) for k in dict.fromkeys([*full, *retained])))
    check("finite_signed_values_preserve_negative", all(math.isfinite(v) for d in [full, retained, delta] for v in d.values()) and any(v < 0 for v in full.values()))
    tangent = read("actual/complete_tangent_constraints.json")
    check("scaled_defects_keep_absolute_and_input_norm", len(tangent) == 44 and all(v["scaled_defect"] == v["absolute_defect"] / max(1., v["input_G_norm"]) and v["scaled_defect"] <= contract["gates"]["tangent_identity_scaled"] and v["no_stationarity_or_error_bound"] for v in tangent))
    metrics = read("actual/action_metrics.json")
    accessible = read("actual/newf_accessible_actions.json")
    check("whole_metrics_and_actions_not_relabelled", metrics["metrics"] == result["action_metrics"] and accessible == result["accessible_actions"] and metrics["declared_diagnostic_metric"] == result["new_metric_sha256"])
    check("HF_partial_not_CI_response", result["full_gradient_is_HF_partial"] and not result["finite_CI_response_qualified"] and metrics["HF_partial_not_finiteCIresponse"])
    check("no_optimizer_stationarity_experiment_claim", not result["expanded_MCDHF_optimized"] and not result["original_stationarity_passed"] and not result["experimental_gate_passed"] and result["no_old3451_residual_improvement_claim"] and result["no_energy_callback"])
    check("new3689_both_original_thresholds_fail", all(v["metric_residual"] > 1e-7 and v["indicator_Ha"] > 1e-8 and v["strict_thresholds"] == [1e-7, 1e-8] and not v["energy_or_spectrum_error_bound"] for v in result["action_metrics"].values()))
    check("actual_resource_and_uncontrolled_timing", result["actual_affinity"] == [31] and result["peak_RSS_MiB"] < 3.95 * 1024 and not result["timing_claim_valid"] and not receipt["timing_claim_valid"])
    completion = {"status": "closed_SOURCE_JSON_metadata_only", "all_completion_metadata_checks_passed": True, "metadata_checks": CHECKS, "metadata_checks_count": len(CHECKS), "all_actual_live_controls_passed": True, "actual_live_controls_count": 438, "root_foreground_closed": True, "root_exit_code": 0, "inputs_sha256": INPUTS, "actual_calls": result["completed"], "canonical_recovered_points": result["points"], "metric_identity": {k: result[k] for k in ["new_metric_sha256", "new_geometry_sha256", "new_state_sha256", "objective_identity", "original3451_metric_sha256", "source_map_sha256"]}, "action_metrics": result["action_metrics"], "accessible_actions": accessible, "recovery_max": max(v["canonical_vs_public_max"] for v in result["recovery"]), "peak_RSS_MiB": result["peak_RSS_MiB"], "root_elapsed_uncontrolled_s": receipt["elapsed_uncontrolled_s"], "independent_numerical_array_replay_performed": False, "NPZ_reads_or_rehashes": 0, "NumPy_Atomx_imports": 0, "new_physical_calls": 0, "gradient_is_HF_partial": True, "finite_CI_response_qualified": False, "old_FAIL_and_no31_optimization_preserved": True, "deferred_array_identities_from_actual_runtime": result["archives"]}
    (BASE / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps({"all_completion_metadata_checks_passed": True, "metadata_checks_count": len(CHECKS), "actual_live_controls_count": 438, "completion_sha256": sha(BASE / "completion.json")}))


if __name__ == "__main__":
    main()
