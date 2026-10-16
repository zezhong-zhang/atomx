"""Once-only saved source/JSON closure; no NumPy, Atomx or NPZ access."""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]


def sha(path):
    if path.suffix == ".npz":
        raise ValueError("NPZ reads/rehashing are outside this metadata closure")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    started = time.monotonic()
    if (HERE / "completion.json").exists():
        raise FileExistsError("Once-only immutable completion already exists")
    contract = load(HERE / "closure/contract.json")
    checks, deferred = [], []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError("Source/JSON closure gate: " + name)

    for name, expected in contract["inputs_sha256"].items():
        path = ROOT / name
        if path.suffix == ".npz":
            deferred.append(dict(path=name, expected_SHA_from_frozen_runtime_contract=expected,
                                 independently_read_or_rehashed=False))
            continue
        check("source_JSON_input_SHA/" + name, sha(path) == expected)
    r, receipt = load(HERE / "actual/result.json"), load(HERE / "root_execution.json")
    c, binding = load(HERE / "contract.json"), load(HERE / "execution_binding.json")
    handoff = load(HERE / "handoff.json")
    check("all12_frozen_handoff_files_unchanged", len(handoff["paths_sha256"]) == 12
          and all(sha(ROOT / name) == expected for name, expected in handoff["paths_sha256"].items()))
    check("exact_parent_closed_receipt_result_SHA", sha(HERE / "root_execution.json") == contract["exact_actual_root_receipt_sha256"]
          and sha(HERE / "actual/result.json") == contract["exact_actual_result_sha256"])
    check("root_foreground_closed_exit0_no_retry", receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0
          and receipt["no_retry"] is True and receipt["timing_claim_valid"] is False)
    check("root_runtime_driver_contract_binding_exact", receipt["driver_sha256"] == sha(HERE / "run.py")
          and receipt["contract_sha256"] == sha(HERE / "contract.json")
          and receipt["binding_sha256"] == sha(HERE / "execution_binding.json")
          and receipt["launcher_sha256"] == sha(HERE / "root_launch.py")
          and receipt["log_sha256"] == sha(HERE / "execution.log"))
    check("actual_snapshot_run_source_exact", sha(HERE / "actual/execution_source.py") == sha(HERE / "run.py"))
    check("explicit_exact_root_authorization", binding["execution_authorized_by_root"] is True
          and binding["pending_exact_root_execution_binding"] is False
          and binding["scientific_inputs_sha256"] == c["inputs_sha256"])
    check("actual_source_point_metric_identity", r["source_map_sha256"] == c["fixed"]["source_map_sha256"]
          and r["checkpoint_arrays_sha256"] == c["input_roles"]["literal208_arrays"]["sha256"]
          and r["original_metric_sha256"] == c["fixed"]["checkpoint_metric_sha256"]
          and r["driver_sha256"] == sha(HERE / "run.py")
          and r["contract_sha256"] == sha(HERE / "contract.json")
          and r["execution_binding_sha256"] == sha(HERE / "execution_binding.json"))
    live = r["checks"]
    check("all941_saved_live_gates_true_within_original_limits", len(live) == contract["reported_actual_gates"] == 941
          and all(row["passed"] is True and math.isfinite(row["error"]) and math.isfinite(row["limit"])
                  and row["error"] <= row["limit"] for row in live)
          and r["all_original_numeric_identity_gates_passed"] is True)
    check("producer_attempts_complete_and_capped", r["attempted"] == r["completed"]
          and all(0 <= value <= c["resources"]["call_limits"][name] for name, value in r["attempted"].items()))
    counts = r["completed"]
    check("actual2H2CI4JK_no_extra_physics", counts["H"] == counts["CI"] == 2 and counts["auxiliary_JK_builder"] == 4
          and counts["auxiliary_angular_precompute"] == 2 and counts["closed_core_scalar"] == 1
          and all(counts[name] == 0 for name in ("Dirac", "external_hS", "gradient", "optimizer", "new_RDM")))
    check("actual12EVD_2complementSVD_2full118refinements", counts["full_scipy_eigh"] == 12 and counts["complement_SVD"] == 2
          and r["refinement"]["completed_complete_spectrum_refinements"] == 2
          and [row["whole_spectrum_rows"] for row in r["virtuals"]] == [118, 118])
    scalar = r["signed_scalar_controls"]
    check("scalar32_signed_full_quadrants_saved", scalar["ordered_bilinears"] == 32
          and len(scalar["signed_bilinears"]) == 32 and len(scalar["rows"]) == 8
          and scalar["zero_bare_one_body"] is True)
    scalar_max = max(abs(row["signed_difference_Ha"]) for row in scalar["signed_bilinears"])
    check("saved_scalar32_errors_within_original_gate", scalar_max <= c["acceptance"]["auxiliary_scalar_bilinear_error_Ha"])
    hS = r["hS_request_ledger"]
    check("hS_requests_hits_misses_actual7producer", hS["attempted"] == hS["completed"]
          and hS["completed"] <= c["resources"]["maximum_hS_consumer_requests"]
          and hS["cache_hits"] + hS["cache_misses"] == hS["completed"]
          and hS["cache_misses"] == counts["CI_internal_hS"] == 7)
    check("bothCI_coverall7kappa", [row["covered_kappas"] for row in hS["by_CI"]]
          == [[-4, -3, -2, -1, 1, 2, 3], [-4, -3, -2, -1, 1, 2, 3]])
    carry = hS["carry_events"]
    check("cache_carry0_7_identity_nucleus_geometry", [row["carried_h0_blocks"] for row in carry] == [0, 7]
          and all(carry[1][name] is True for name in ("same_workspace_before", "same_nucleus_before",
                  "same_geometry_before_after", "carried_tuple_identity")))
    check("hS_ledger_exact_JSON_copy", load(HERE / "actual/hS_requests_producers_cache_carry.json") == hS)
    models = r["models"]
    expected_UIDs = {label: uid for label, uid in c["fixed"]["target_UIDs"]}
    for name, csfs, orbitals in (("baseline", 1046, 29), ("complete5f", 1595, 31)):
        model, rows = models[name], models[name]["rows"]
        check(name + "/exact_target_scope", model["complete_target_CSFs"] == csfs and model["radial_orbitals"] == orbitals
              and model["fixed_literal208"] is True and model["original_stationarity_passed"] is False
              and model["expanded_stationarity_evaluated"] is False and model["full_spectrum_certified"] is False)
        check(name + "/unique_physical5UIDs", len(rows) == 5
              and {row["label"]: row["reference_UID"] for row in rows} == expected_UIDs
              and all(row["selected_level"][:2] == row["reference_UID"][:2] for row in rows))
        check(name + "/saved_summary_exact", load(HERE / ("actual/" + name + "_same_point_evidence.json")) == model)
        check(name + "/nominal1percent_goal_FAIL_preserved", model["nominal_four_1percent_goal_passed"] is False
              and any(abs(row["nominal_relative_gap"]) > .01 for row in rows if row["nominal_relative_gap"] is not None))
        tracking = load(HERE / ("actual/" + name + "_signed_tracking.json"))
        check(name + "/five_equal_point2weights_original_reference", tracking["original_reference_unchanged"] is True
              and len([weight for weight in tracking["weights"] if weight != 0]) == 5
              and all(weight == .2 for weight in tracking["weights"] if weight != 0))
    mapping = load(HERE / "actual/nested_old1046_mapping.json")["mapping"]
    basis = load(HERE / "actual/basis_rows.json")
    check("whole1046_signed_nested_mapping", len(mapping) == 1046 and len(set(mapping)) == 1046
          and all(basis["complete5f"]["retained_signatures"][j] == basis["baseline"]["retained_signatures"][i]
                  for i, j in enumerate(mapping)))
    check("baseline_bitwise_H_roots_signedC_owner_UID_live_gates", all(any(row["name"] == name and row["passed"] is True
          and row["error"] == 0 and row["limit"] == 0 for row in live) for name in (
              "literal208_baseline_H_bitwise", "literal208_baseline_reported_roots_bitwise", "literal208_baseline_signed_C_bitwise",
              "literal208_baseline_owner_identity", "literal208_baseline_tracking_UIDs")))
    check("nonstationary_nonexperimental_boundaries", all(r[name] is False for name in (
              "original_stationarity_passed", "expanded_MCDHF_executed", "physical_completeness_certified", "literal208_experimental_gate_passed"))
          and r["no_best_gap_selection"] is True)
    check("no_actual_failure_complete_progress", not (HERE / "actual/failure.json").exists()
          and load(HERE / "actual/progress.json")["phase"] == "complete")
    support = load(HERE / "actual/radial_support.json")
    check("nine_support_rows_no_bound", len(support["rows"]) == 9 and support["same_saved_geometry"] is True
          and all(row["adequacy_bound"] is False for row in support["rows"]))
    closure = dict(status="closed_SOURCE_JSON_only_metadata_verified", all_completion_metadata_checks_passed=True,
        all_actual_live_controls_passed=True, total_actual_live_controls=941, metadata_checks=checks,
        metadata_checks_count=len(checks), root_foreground_exit_code=0,
        inputs_sha256=contract["inputs_sha256"], deferred_NPZ_identities_not_read_or_rehashed=deferred,
        completed_runtime_array_SHA_declarations=r["completed_archives"], runtime_array_hashes_not_independently_replayed=True,
        actual_counts=counts, actual_refinement=r["refinement"], actual_hS_requests=hS,
        scalar32_max_error_Ha=scalar_max, actual_models=models, virtual_Fock_Rayleighs=r["virtuals"], radial_support=support,
        source_map_sha256=r["source_map_sha256"], original3451_metric_sha256=r["original_metric_sha256"],
        literal208_arrays_sha256=r["checkpoint_arrays_sha256"],
        root_elapsed_uncontrolled_s=receipt["elapsed_uncontrolled_s"], runtime_elapsed_uncontrolled_s=r["elapsed_uncontrolled_s"],
        runtime_peak_RSS_MiB=r["peak_RSS_MiB"], metadata_elapsed_s=time.monotonic() - started,
        metadata_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metadata_affinity=sorted(os.sched_getaffinity(0)),
        completion_contract_sha256=sha(HERE / "closure/contract.json"), completion_script_sha256=sha(Path(__file__)),
        actual_result_sha256=sha(HERE / "actual/result.json"), root_receipt_sha256=sha(HERE / "root_execution.json"),
        NumPy_Atomx_imports=0, NPZ_reads=0, H_CI_gradient_Dirac_optimizer_cells_eigensolver_calls=0,
        independent_numerical_array_replay_performed=False, original_stationarity_passed=False,
        expanded_MCDHF_executed=False, expanded_stationarity_evaluated=False, experimental_gate_passed=False,
        limitations="941livecontrols inspectedfromsavedJSON,notsecondnumericalreplay. Closed68Fockseed/fixed31RCI notcorrelatedEuler/f-CV-CCconvergence. Oldnuclear14counterFAIL andoriginal208stationarityFAIL remain.")
    (HERE / "completion.json").write_text(json.dumps(closure, indent=2) + "\n")
    print(json.dumps(dict(all_completion_metadata_checks_passed=True, metadata_checks=len(checks),
        inspected_runtime_controls=941, actual_counts=counts, science_replays=0, NPZ_reads=0)))


if __name__ == "__main__":
    main()
