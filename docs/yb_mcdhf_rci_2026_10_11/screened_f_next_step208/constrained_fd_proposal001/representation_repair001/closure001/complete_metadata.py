"""Once-only stdlib FD closure; refuse actual data until exact root closure."""
import hashlib
import json
import math
import os
import resource
import signal
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = next(p for p in BASE.parents if (p / "AGENTS.md").is_file())
CHECKS = []
INPUTS = {}


def sha(path):
    if path.suffix == ".npz":
        raise ValueError("No numerical array reads or rehashes in this closure")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    CHECKS.append(dict(name=name, passed=bool(condition)))
    if not condition:
        raise ValueError(name)


def read(path):
    INPUTS[str(path.relative_to(REPO))] = sha(path)
    return json.loads(path.read_text())


def ratio(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def main():
    contract = read(HERE / "contract.json")
    if contract["metadata_execution_authorized_after_exact_root_closure"] is not True:
        raise ValueError("Prepared only: exact root closure and input hashes required")
    if sha(Path(__file__)) != contract["closure_driver_sha256"]:
        raise ValueError("Changed closure consumer")
    if (BASE / "completion.json").exists():
        raise FileExistsError("Preserve completion; no retry/overwrite")
    caps = contract["resources"]
    if caps["CPU"] not in os.sched_getaffinity(0):
        raise ValueError("Declared closure CPU unavailable")
    os.sched_setaffinity(0, {caps["CPU"]})
    memory = int(caps["AS_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("metadata cap")))
    signal.alarm(caps["inner_seconds"])
    start = time.monotonic()
    try:
        # The receipt is the only runtime artifact permitted before confirming
        # closure. No arrays are inspected even after closure.
        receipt = read(BASE / "root_execution.json")
        check("exact_closed_receipt", INPUTS[str((BASE / "root_execution.json").relative_to(REPO))]
              == contract["closed_input_sha256"]["root_execution.json"]
              and receipt["status"] == "foreground_closed" and receipt["no_retry"])
        actual_path = BASE / contract["actual_record"]
        actual = read(actual_path)
        check("exact_closed_actual_record", sha(actual_path) == contract["closed_input_sha256"][contract["actual_record"]])
        frozen = read(BASE / "handoff.json")
        science_contract = read(BASE / "contract.json")
        binding = read(BASE / "execution_binding.json")
        for name, expected in contract["frozen_preparation_sha256"].items():
            check("preserved/" + name, sha(BASE / name) == expected)
            INPUTS[str((BASE / name).relative_to(REPO))] = expected
        check("root_execution_binding", receipt["binding_sha256"] == sha(BASE / "execution_binding.json")
              == contract["scientific_execution_binding_sha256"])
        check("root_bound_driver_contract", receipt["driver_sha256"] == sha(BASE / "run.py")
              and receipt["contract_sha256"] == sha(BASE / "contract.json")
              and binding["resources"] == science_contract["resources"]
              and binding["source_map_sha256"] == frozen["source_map_sha256"])
        check("declared_gate_formula_preserved", science_contract["gates"]["FD_absolute"] == 1e-8
              and science_contract["gates"]["FD_relative"] == 1e-5
              and science_contract["fixed"]["h"] == [1e-5, 5e-6])
        checks = actual["checks"]
        calls = actual["completed"]
        attempted = actual["attempted"]
        check("actual_call_counters_schema", set(calls) == set(attempted)
              == set(science_contract["resources"]["calls"]) and all(
              isinstance(v, int) and 0 <= v <= attempted[k] for k, v in calls.items()))
        check("no_new_optimizer_gradient_GPU", calls["optimizer"] == calls["gradient_from_weights"]
              == calls["gradient_pair_potential"] == calls["GPU"] == 0)
        check("same_counter_names_controls", all(isinstance(r["passed"], bool)
              and math.isfinite(r["error"]) and math.isfinite(r["limit"]) for r in checks))
        check("all_array_archives_identity_only", all(r["path"].endswith(".npz") and len(r["sha256"]) == 64
              for r in actual["archives"]))
        check("no_actual_success_inferred_from_exit", receipt["exit_code"] in range(256)
              or receipt["exit_code"] < 0)
        completed_run = contract["actual_record"] == "actual/result.json"
        domains = actual.get("derivative_domains", {})
        points = actual["points"]
        ids = {label: uid for label, uid in science_contract["fixed"]["original_UIDs"]}
        summaries = {}
        for name, evidence in points.items():
            rows = {r["label"]: r for r in evidence["rows"]}
            check(name + "/physical_UIDs_and_weights", set(rows) == set(ids)
                  and all(r["UID"] == ids[label] and r["selected_level"][:2] == r["UID"][:2]
                          and r["weight"] == .2 for label, r in rows.items()))
            check(name + "/reported_point_scope", evidence["physical_point"] == name
                  and evidence["operator"] == "dirac-coulomb" and evidence["original_tracker_not_rebased"])
            check(name + "/lower_reference_identity", rows["6s_1/2"]["UID"] == [1, 1, 0])
            check(name + "/same_point_four_lines", all(r["excitation_cm1"] == 0. for label, r in rows.items()
                  if label == "6s_1/2") and len([label for label in rows if label != "6s_1/2"]) == 4)
            summaries[name] = dict(reported_EO_Ha=evidence["EOL_reported_Ha"], five_states=evidence["rows"],
                                  multiplets=evidence["multiplets"], ULP_Ha=evidence["actual_EO_ULP_Ha"])
        if completed_run:
            check("full_five_point_record", set(points) == {"baseline", "h0_minus", "h0_plus", "h1_minus", "h1_plus"})
            check("own_source_state_metric", all(actual[k] == science_contract["fixed"][k] for k in (
                  "source_map_sha256", "new_metric_sha256", "new_geometry_sha256", "new_state_sha256", "objective_identity")))
            check("actual_seal_driver_contract_chain", actual["exact_execution_binding_sha256"] == sha(BASE / "execution_binding.json")
                  and actual["driver_sha256"] == sha(BASE / "run.py") and actual["contract_sha256"] == sha(BASE / "contract.json"))
            check("full_completed_calls", all(calls[k] == v for k, v in dict(H=5, CI=5,
                  public_reported_energy_evaluate=5, mixed_ci_finite_increment=4, unfiltered_target_RDM=20,
                  initial_hS=7, mixed_radial_hS=28, validation_Dirac=7, geometry_SPD_eigvalsh=38,
                  polar_retraction=28, tracker_resolve=18, tracking_SVD=90, tracker_overlap=19).items()))
            check("completed_budget_and_attempts", calls == attempted
                  and all(attempted[k] <= v for k, v in science_contract["resources"]["calls"].items()))
            check("two_separate_h_domains", set(domains) == {str(h) for h in science_contract["fixed"]["h"]})
            for index, h in enumerate(science_contract["fixed"]["h"]):
                domain = domains[str(h)]
                ordinary = domain["ordinary_reported"]
                expected = (points[f"h{index}_plus"]["EOL_reported_Ha"] - points[f"h{index}_minus"]["EOL_reported_Ha"]) / (2 * h)
                threshold = max(1e-8, 1e-5 * abs(ordinary["analytic"]))
                check(f"h{index}/ordinary_algebra_and_original_gate", ordinary["derivative"] == expected
                      and ordinary["threshold"] == threshold and ordinary["error"] == expected - ordinary["analytic"]
                      and ordinary["passed"] == (abs(ordinary["error"]) <= threshold))
                check(f"h{index}/centered_separate_two_references", len(domain["independent_centered"]) == 2
                      and {r["reference"] for r in domain["independent_centered"]} == {"fixedLambda_ideal", "public_retained"}
                      and domain["production_callback"] is False and domain["no_uniform_error_bound"] is True)
                for row in domain["independent_centered"]:
                    error = ratio(row["derivative"]) - ratio(row["analytic"])
                    threshold = max(1e-8, 1e-5 * abs(float(ratio(row["analytic"]))))
                    check(f"h{index}/centered_JSON_ratio_" + row["reference"], error == ratio(row["error"])
                          and row["threshold"] == threshold and row["passed"] == (abs(error) <= Fraction.from_float(threshold)))
            check("actual_allgate_flag_truth", actual["all_declared_controls_passed"] == all(r["passed"] for r in checks))
            check("ordinary_FAIL_not_upgraded", actual["single_direction_finite_measurement_only"]
                  and not actual["full31_space_certificate"] and not actual["original_stationarity_passed"]
                  and not actual["expanded_MCDHF_optimized"] and actual["ordinary_reported_objective_unchanged"]
                  and actual["old_failed_scientific_gates_unchanged"])
            check("actual_resource_domain", actual["actual_affinity"] == [31] and not actual["timing_claim_valid"]
                  and actual["peak_RSS_MiB"] <= 3.95 * 1024)
        # Every closed output JSON is identity-bound without touching NPZs.
        # Scalar/domain cross-checks reuse saved metadata, not H/CI/g producers.
        output_JSON = {path.name: read(path) for path in sorted((BASE / "actual").glob("*.json"))}
        if completed_run:
            check("output_reported_points_same_as_result", output_JSON["five_point_reported_states.json"] == points)
            check("output_domain_rows_same_as_result", output_JSON["derivative_domains.json"] == domains)
            state = output_JSON["state_identity.json"]
            check("output_original_state_metric_identity", all(state[k] == actual[v] for k, v in (
                ("metric_sha256", "new_metric_sha256"), ("geometry_sha256", "new_geometry_sha256"),
                ("state_sha256", "new_state_sha256"), ("objective_identity", "objective_identity")))
                  and not state["initial_polar_repair"] and not state["metric_refit"])
            check("output_complete_capacity_unpruned", output_JSON["complete_angular_capacity.json"]["terms"]
                  == actual["angular_capacity"] and output_JSON["complete_angular_capacity.json"]["exact_zero_only"])
            check("output_predetermined_direction_not_gap_selected", output_JSON["direction_identity.json"]["no_gap_selection"]
                  and output_JSON["direction_identity.json"]["no_per_kappa_normalization"])
            check("output_exact_analytic_identity", output_JSON["baseline_full_signed_Lambda.json"]["analytic"]
                  == actual["diagnostics"]["analytic"] and output_JSON["baseline_full_signed_Lambda.json"]["exact_all_signed_offdiagonal"])
            for name in points:
                owner = output_JSON[name + "_owner.json"]
                tracking = output_JSON[name + "_tracking.json"]
                check(name + "/output_complete_signed_owner", owner["units"] == "Hartree"
                      and len(owner["CSFs"]) == 1595 and len(owner["orbital_keys"]) == 31)
                check(name + "/output_five_tracker_UIDs", len(tracking["tracked"]) == 5
                      and all(len(t["reference_levels"]) == 1 and len(t["selected_levels"]) == 1
                              and t["per_level_weight"] == .2 for t in tracking["tracked"]))
                if name != "baseline":
                    diagnostic = actual["diagnostics"][name]
                    check(name + "/output_mixed_complete_change", output_JSON[name + "_mixed_finite_change.json"]
                          == diagnostic["mixed_change"] and diagnostic["mixed_change"]["diagnostic_only"]
                          and not diagnostic["mixed_change"]["certified_error_bound"])
                    check(name + "/output_exact_centering_identity", output_JSON[name + "_centered_diagnostic.json"]
                          == diagnostic and all(v["all_Gram_identities_exact"]
                            and v["no_offdiagonal_or_quadratic_omission"]
                            for v in diagnostic["samepoint_fixedLambda"].values()))
        ordinary_flags = [d["ordinary_reported"]["passed"] for d in domains.values()]
        centered_flags = [r["passed"] for d in domains.values() for r in d["independent_centered"]]
        completion = dict(status="closed_stdlib_metadata_only", all_completion_metadata_checks_passed=True,
            metadata_checks=CHECKS, metadata_checks_count=len(CHECKS), inputs_sha256=INPUTS,
            root_foreground_closed=True, root_exit_code=receipt["exit_code"],
            actual_record=contract["actual_record"], actual_live_controls_count=len(checks),
            actual_live_controls_passed=sum(r["passed"] for r in checks),
            all_actual_live_controls_passed=completed_run and all(r["passed"] for r in checks),
            first_failed_actual_control=next((r for r in checks if not r["passed"]), None),
            completed_all_four_trial_domains=completed_run,
            ordinary_both_h_passed=len(ordinary_flags) == 2 and all(ordinary_flags),
            independent_centered_both_h_both_references_passed=len(centered_flags) == 4 and all(centered_flags),
            derivative_domains=domains, actual_calls=calls, attempted_calls=attempted,
            actual_cache_observer=actual["hS_requests"], all_reported_points=summaries,
            deferred_array_identities_from_actual_runtime=actual["archives"],
            all_closed_output_JSON_sha256={name: INPUTS[str((BASE / "actual" / name).relative_to(REPO))] for name in output_JSON},
            original_stationarity_passed=False, no31_optimizer=True,
            no_full_space_gradient_certificate=True, old_FAIL_not_upgraded=True,
            independent_numerical_array_replay_performed=False, NPZ_reads_or_rehashes=0,
            NumPy_Atomx_imports=0, new_physical_calls=0, no_API=True,
            elapsed_metadata_s=time.monotonic() - start)
        with (BASE / "completion.json").open("x") as stream:
            json.dump(completion, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(metadata_checks=len(CHECKS), ordinary_both_h_passed=completion["ordinary_both_h_passed"],
            centered_both_h_both_references_passed=completion["independent_centered_both_h_both_references_passed"],
            root_exit_code=receipt["exit_code"], completion_sha256=sha(BASE / "completion.json"))))
    except BaseException as error:
        with (HERE / "closure_failure.json").open("x") as stream:
            json.dump(dict(exception=type(error).__name__, message=str(error), checks=CHECKS,
                           inputs_sha256=INPUTS, new_physical_calls=0), stream, indent=2)
            stream.write("\n")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
