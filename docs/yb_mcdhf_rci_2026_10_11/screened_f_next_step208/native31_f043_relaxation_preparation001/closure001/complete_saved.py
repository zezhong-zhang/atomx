"""Closed-only scalar/JSON ledger review, with no NPZ access or physics calls."""
import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    path = Path(path)
    if path.suffix == ".npz":
        raise ValueError("NPZ reads or rehashes forbidden in scalar closure")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def fraction(row):
    return Fraction(int(row["numerator"]), int(row["denominator"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-receipt-sha256", required=True)
    parser.add_argument("--closed-result-sha256", required=True)
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    checks, inputs = [], {}

    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("Frozen metadata changed: " + str(path))
        inputs[str(Path(path).relative_to(REPO))] = actual
        return actual

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    bind(RUN / "root_execution.json", args.closed_receipt_sha256)
    receipt = load(RUN / "root_execution.json")
    check("explicit_foreground_closed_before_actual_read", receipt["status"] == "foreground_closed"
          and type(receipt["exit_code"]) is int)
    for name, expected in contract["frozen_metadata_sha256"].items():
        bind(RUN / name, expected)
    binding, run_contract = load(RUN / "execution_binding.json"), load(RUN / "contract.json")
    bind(RUN / "actual/result.json", args.closed_result_sha256)
    result = load(RUN / "actual/result.json")
    bind(RUN / "foreground.log", receipt["log_sha256"])
    check("root_binding_source_and_driver_match", receipt["binding_sha256"] == sha(RUN / "execution_binding.json")
          == contract["frozen_metadata_sha256"]["execution_binding.json"]
          and result["driver_sha256"] == binding["driver_sha256"] == sha(RUN / "run.py")
          and result["contract_sha256"] == binding["contract_sha256"] == sha(RUN / "contract.json")
          and result["execution_binding_sha256"] == receipt["binding_sha256"]
          and result["source_map_sha256"] == binding["source_map_sha256"] == run_contract["source"]["map_sha256"])
    check("root_postclosure_source_input_preservation", receipt["frozen_source_unchanged"] is True
          and receipt["complete_shared_source_unchanged"] is True and receipt["input_unchanged"] is True)
    successful = result["status"] == "bounded_f043_public_optimizer_returned" and result["error"] is None
    check("actual_exit_status_consistent", (successful and receipt["exit_code"] == 0 and not receipt["timed_out"])
          or (not successful and receipt["exit_code"] != 0))
    actual_json = {str(p.relative_to(REPO)): bind(p) for p in sorted((RUN / "actual").rglob("*.json"))}
    expected = {label: uid for label, uid in run_contract["fixed"]["original_UIDs"]}
    records, previous = [], None
    for entry in result["trajectory"]:
        path = REPO / entry["manifest"]
        bind(path, entry["manifest_sha256"])
        point = load(path)
        rows = {r["label"]: r for r in point["rows"]}
        check("point_" + str(entry["iteration"]) + "_UID_lineage", len(rows) == len(point["rows"]) == 5
              and set(rows) == set(expected) and all(rows[k]["UID"] == uid and rows[k]["weight"] == .2
              and rows[k]["selected_level"][:2] == uid[:2] for k, uid in expected.items()))
        check("point_" + str(entry["iteration"]) + "_qualified_metric_source", point["qualified"] is True
              and point["metric_sha256"] == run_contract["fixed"]["new_metric_sha256"]
              and point["geometry_sha256"] == run_contract["fixed"]["new_geometry_sha256"]
              and point["objective_identity"] == run_contract["fixed"]["objective_identity"]
              and point["source_map_sha256"] == result["source_map_sha256"]
              and point["diagnostics"] == entry["diagnostics"]
              and point["array_sha256"] == entry["array_sha256"])
        decision = point["exact_acceptance"]
        if previous is None:
            check("point0_has_no_line_search_acceptance", entry["iteration"] == 0 and decision is None
                  and point["diagnostics"]["accepted_line_search"] is None)
        else:
            ls = point["diagnostics"]["accepted_line_search"]
            e0, e1 = previous["diagnostics"]["energy_ha"], point["diagnostics"]["energy_ha"]
            change, rhs = Fraction(ls["energy_change_ha"]), Fraction(ls["armijo_rhs_ha"])
            margin = (abs(fraction(previous["exact_point"]["quotient_minus_actual_EO"]))
                      + abs(fraction(point["exact_point"]["quotient_minus_actual_EO"]))
                      + abs(Fraction(e1) - Fraction(e0) - change)
                      + abs(Fraction(1e-4) * Fraction(ls["step"]) * Fraction(ls["directional_derivative"]) - rhs))
            check("point_" + str(entry["iteration"]) + "_original_exact_acceptance_ledger",
                  decision["qualified"] is True and decision["actual_optimizer_comparison_passed"] is True
                  and fraction(decision["margin_exact"]) == margin
                  and fraction(decision["guard_lhs"]) == change + margin
                  and fraction(decision["recorded_rhs"]) == rhs and change + margin <= rhs
                  and fraction(decision["stored_quotient_change"]) == fraction(point["exact_point"]["weighted_quotient"])
                      - fraction(previous["exact_point"]["weighted_quotient"])
                  and fraction(decision["stored_quotient_change"]) < 0
                  and ls["parent_energy_ha"] == e0 and ls["trial_energy_ha"] == e1 and e1 < e0
                  and ls["stable_callback_used"] is False)
        rms = math.sqrt(math.fsum(r["nominal_gap_cm1"]**2 for k, r in rows.items() if k != "6s_1/2") / 4)
        multiplets = [dict(label=label, centroid_cm1=(wa * rows[a]["excitation_cm1"]
                     + wb * rows[b]["excitation_cm1"]) / (wa + wb),
                     splitting_cm1=rows[b]["excitation_cm1"] - rows[a]["excitation_cm1"])
                     for label, a, b, wa, wb in (("5d", "5d_3/2", "5d_5/2", 4, 6),
                                               ("6p", "6p_1/2", "6p_3/2", 2, 4))]
        stationary = all(m["residual"] < 1e-7 and m["indicator_Ha"] < 1e-8
                         for m in point["fullsigned_metrics"].values())
        check("point_" + str(entry["iteration"]) + "_saved_stationarity_and_spectra", stationary
              == point["original_stationarity_passed"] == entry["original_stationarity_passed"]
              and rms == point["nominal_four_gap_RMS_cm1"] == entry["nominal_four_gap_RMS_cm1"]
              and point["multiplets"] == multiplets and rows["6s_1/2"]["UID"] == [1, 1, 0])
        records.append(dict(iteration=entry["iteration"], manifest=entry["manifest"],
            manifest_sha256=entry["manifest_sha256"], array_sha256=entry["array_sha256"],
            reported_EO_Ha=point["diagnostics"]["energy_ha"], metrics=point["fullsigned_metrics"],
            rows=point["rows"], multiplets=multiplets, RMS_cm1=rms,
            original_stationarity_passed=stationary,
            finite_stored_H_acceptance_qualified=None if decision is None else decision["qualified"]))
        previous = point
    check("literal_actual_horizon_and_terminal_not_gap_selected", [r["iteration"] for r in records]
          == list(range(len(records))) and len(records) <= 17
          and result["last_qualified_checkpoint"] == (records[-1]["manifest"] if records else None)
          and result["original_stationarity_passed"] == (records[-1]["original_stationarity_passed"] if records else False))
    check("attempted_completed_work_within_prospective_caps", all(0 <= result["completed"][k]
          <= result["attempted"][k] <= limit for k, limit in run_contract["resources"]["calls"].items()))
    check("scope_objective_history_no_adoption", result["ordinary_reported_objective_unchanged"] is True
          and result["no_energy_callback"] is True and result["current_source_adopted"] is False
          and result["current_regression_failures_unchanged"] is True
          and run_contract["public_optimizer"]["history_size"] == 10)
    failed_live = [row for row in result["checks"] if row["passed"] is not True]
    check("actual_live_control_boolean_matches_ledger", result["all_completed_controls_passed"] == (not failed_live))
    unqualified = []
    qualified_paths = {r["manifest"] for r in records}
    for path in sorted((RUN / "actual/checkpoints").glob("accepted_*.json")):
        if path.name.endswith("_signed_tables.json"):
            continue
        m = load(path)
        if str(path.relative_to(REPO)) not in qualified_paths:
            unqualified.append(dict(path=str(path.relative_to(REPO)), manifest_sha256=sha(path),
                qualified=m["qualified"], iteration=m["iteration"], diagnostics=m["diagnostics"],
                exact_acceptance=m.get("exact_acceptance"), array_sha256=m.get("array_sha256"),
                forensic_array_sha256=m.get("forensic_array_sha256"), previous_qualified=m["previous_qualified"]))
            check("failed_candidate_never_promoted", m["qualified"] is False)
    changes = []
    if records:
        initial_rows = {r["label"]: r for r in records[0]["rows"]}
        changes = [dict(label=r["label"], UID=r["UID"], delta_energy_Ha=r["energy_Ha"]
                       - initial_rows[r["label"]]["energy_Ha"], transition_shift_cm1=r["excitation_cm1"]
                       - initial_rows[r["label"]]["excitation_cm1"], terminal_excitation_cm1=r["excitation_cm1"],
                       terminal_nominal_gap_cm1=r["nominal_gap_cm1"]) for r in records[-1]["rows"]]
    check("zero_numerical_imports_or_physical_replay", not any(n == "numpy" or n.startswith("numpy.")
          or n == "atomx" or n.startswith("atomx.") for n in sys.modules))
    output = dict(all_scalar_completion_controls_passed=True, checks=checks, inputs_sha256=inputs,
        output_JSON_sha256=actual_json, scientific_calls=0, NPZ_reads_or_rehashes=0,
        independent_H_C_gradient_replay=False, scope="saved metadata and exact ledger arithmetic only",
        actual_status=result["status"], actual_error=result["error"], actual_exit_code=receipt["exit_code"],
        actual_live_control_count=len(result["checks"]), failed_live_controls=failed_live,
        actual_new_qualified_steps=max(0, len(records) - 1), trajectory=records,
        last_qualified_checkpoint=result["last_qualified_checkpoint"], unqualified_candidates=unqualified,
        final_optimizer_diagnostics=result["optimizer_diagnostics"], attempted=result["attempted"], completed=result["completed"],
        exact_guard_budget=result["exact_budget"], finite_stored_H_acceptance_count=max(0, len(records) - 1),
        terminal_target_changes=changes, original_stationarity_passed=result["original_stationarity_passed"],
        EO_delta_Ha=records[-1]["reported_EO_Ha"] - records[0]["reported_EO_Ha"] if records else None,
        RMS_delta_cm1=records[-1]["RMS_cm1"] - records[0]["RMS_cm1"] if records else None,
        source_map_sha256=result["source_map_sha256"], current_source_adopted=False,
        current_regression_failures_unchanged=True, no_automatic_continuation=True,
        experiment_or_physical_completeness_certified=False, elapsed_uncontrolled_s=result["elapsed_uncontrolled_s"],
        peak_RSS_MiB=result["peak_RSS_MiB"], closure_driver_sha256=sha(__file__))
    path = HERE / "completion.json"
    with path.open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(all_scalar_completion_controls_passed=True, checks=len(checks),
                         science_calls=0, completion_sha256=sha(path))))


if __name__ == "__main__":
    main()
