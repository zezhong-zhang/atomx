"""Post-run source/JSON-only closure; never import physics or decode arrays."""
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []

    def verify(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    handoff, contract = load(HERE / "handoff.json"), load(HERE / "contract.json")
    result, receipt = load(HERE / "actual/result.json"), load(HERE / "execution_receipt.json")
    execution = load(HERE / "execution_contract.json")
    files = {}
    for name, expected in handoff["files_sha256"].items():
        verify("frozen_preparation_" + name, sha(REPO / name) == expected)
    for key, expected in handoff["root_execution_expected"].items():
        verify("root_seal_" + key, execution[key] == expected)
    seal_sha = sha(HERE / "execution_contract.json")
    verify("actual_and_receipt_exact_seal", result["execution_contract_sha256"] ==
           receipt["execution_contract_sha256"] == seal_sha)
    verify("sole_launch_marker", load(HERE / "launch_once_marker.json")["sole_launch"] is True)
    verify("outer_foreground_closed_exit0", receipt["status"] == "foreground_closed" and
           receipt["exit_code"] == 0 and receipt["no_retry"] is True)
    verify("outer_log_identity", sha(HERE / "execution.log") == receipt["log_sha256"])
    verify("caps_wall_and_RSS", result["wall_seconds"] < 300 and
           result["memory"]["peak_combined_worker_RSS_bytes"] <= int(3.95 * 2**30))
    verify("recorded_scientific_PASS", result["all_science_gates_passed"] is True)
    cumulative_check_count = 0
    recorded_maxima = {}
    for variant in ("ordinary", "candidate"):
        identity = load(REPO / contract[variant]["identity"])
        verify(variant + "_frozen356_source", len(identity["all_atomx_python_sha256"]) == 356 and all(
            sha(REPO / contract[variant]["archive"] / name.removeprefix("src/")) == expected
            for name, expected in identity["all_atomx_python_sha256"].items()))
        ledger = result["final_worker_ledgers"][variant]
        verify(variant + "_closed7", ledger["status"] == "closed" and ledger["completed_chains"] == 7)
        expected = {**contract["worker_exact_calls"], **contract["variant_exact_calls"][variant]}
        verify(variant + "_every_forward_started_completed_count", all(
            ledger[side].get(name, 0) == count for name, count in expected.items()
            for side in ("started", "completed")))
        verify(variant + "_no_unaccounted_counter", set(ledger["started"]) <= set(expected) and
               ledger["started"] == ledger["completed"])
        verify(variant + "_all_recorded_gate_rows_PASS", all(row["passed"] is True and
               ("error" not in row or math.isfinite(row["error"]) and row["error"] <= row["limit"])
               for row in ledger["checks"]))
        verify(variant + "_final_progress_identity", load(HERE / "actual" / variant / "progress.json")[
               "checks"] == ledger["checks"])
        cumulative_check_count += len(ledger["checks"])
        for row in ledger["checks"]:
            if "error" in row:
                recorded_maxima[row["name"]] = max(row["error"], recorded_maxima.get(row["name"], 0.))
    total_calls = {name: sum(result["final_worker_ledgers"][variant]["completed"].get(name, 0)
                            for variant in ("ordinary", "candidate"))
                   for name in contract["worker_exact_calls"]}
    verify("derived14H14CI14g70eigh28hS", total_calls["H"] == total_calls["CI"] ==
           total_calls["gradient"] == result["total_H_CI_gradient_calls"] == 14 and
           total_calls["scipy_eigh"] == result["total_scipy_eigh"] == 70 and
           total_calls["initial_hS"] + total_calls["CI_internal_hS"] == result["total_hS"] == 28)
    expected_labels = ["cold", "warmup", "pair1", "pair2", "pair3", "pair4", "pair5"]
    verify("complete_predeclared7pair_order", [p["label"] for p in result["pairs"]] == expected_labels)
    host_failures = []
    ratios = []
    for pair in result["pairs"]:
        label = pair["label"]
        verify(label + "_expected_alternating_order", pair["order"] ==
               (["candidate", "ordinary"] if label in ("pair2", "pair4") else ["ordinary", "candidate"]))
        verify(label + "_complete1046_comparison", pair["complete_scientific_comparison"]["status"] ==
               "compared" and pair["complete_scientific_comparison"]["cluster_count"] == 1046)
        for variant in ("ordinary", "candidate"):
            folder = HERE / "actual" / variant / label
            chain = load(folder / "result.json")
            verify(variant + label + "_recorded_trial_identity", chain == pair["outcomes"][variant] and
                   chain["status"] == "chain_passed")
            verify(variant + label + "_recorded_gate_PASS", all(row["passed"] for row in chain["checks"]))
            verify(variant + label + "_12exact_archive_readback_controls", sum(
                row["name"].startswith("archive_exact_") and row["error"] == row["limit"] == 0.
                for row in chain["checks"]) == 12)
            header = load(folder / "source_header.json")
            verify(variant + label + "_source_input_header", header["source_map_sha256"] ==
                   contract[variant]["source_map_sha256"] and header["input_files_sha256"] == contract["inputs"]
                   and header["driver_sha256"] == sha(HERE / "run.py") and
                   header["helper_sha256"] == sha(HERE / "helper.py") and header["fresh_owner"] is True)
            times = chain["timings"]
            verify(variant + label + "_primary_timer_sum", times["primary_seconds"] ==
                   times["energy_trial_seconds"] + times["accepted_gradient_seconds"])
        recomputed_host = pair["host_idle_guard"]["passed"] and pair["host_running_guard"]["passed"]
        verify(label + "_host_failure_retained", recomputed_host == pair["host_passed"])
        if not recomputed_host:
            h = pair["host_idle_guard"]
            host_failures.append({"label": label, "idle_busy_fraction": h["busy_fraction"],
                "idle_limit": .05, "idle_tick_deltas": [b-a for a, b in zip(
                    h["before"]["cpu30_ticks"], h["after"]["cpu30_ticks"], strict=True)],
                "cause": "Unknown. Aggregate CPU ticks do not identify the process or establish external interference."})
        if label.startswith("pair"):
            ratios.append(pair["outcomes"]["ordinary"]["timings"]["primary_seconds"] /
                          pair["outcomes"]["candidate"]["timings"]["primary_seconds"])
    median = statistics.median(ratios)
    verify("exact_saved5ratios_and_median", ratios == result["warm_pair_ratios"] and
           median == result["median_ordinary_over_candidate_primary_ratio"])
    verify("host_FAIL_no_scoped_certificate", host_failures and result["all_host_gates_passed"] is False and
           result["scoped_warm_improvement_certified"] is False)
    metadata_files = [p for p in HERE.rglob("*.json") if "numba_cache" not in str(p) and
                      p.name not in ("completion.json", "document_impact.json", "actual_handoff.json")]
    for path in metadata_files:
        files[str(path.relative_to(REPO))] = sha(path)
    files[str((HERE / "close.py").relative_to(REPO))] = sha(__file__)
    completion = {"status": "closed_science_PASS_host_FAIL_observational_only", "closure_all_passed": True,
        "closure_checks_count": len(checks), "closure_checks": checks,
        "closure_NumPy_Atomx_imports_NPZ_decodes_H_CI_gradient_eigen_NumBa_calls": 0,
        "closure_scope": "Source hashes and JSON recorded-control/counter/statistic audit only; no independent array or physical replay.",
        "actual_cumulative_recorded_gate_rows_PASS": cumulative_check_count,
        "actual_counter_totals": total_calls, "actual_worker_ledgers": {
            v: result["final_worker_ledgers"][v]["completed"] for v in ("ordinary", "candidate")},
        "actual_wall_seconds": result["wall_seconds"], "peak_combined_worker_RSS_bytes":
            result["memory"]["peak_combined_worker_RSS_bytes"],
        "all_science_gates_passed": True, "all_host_gates_passed": False,
        "scoped_warm_improvement_certified": False, "warm_pair_ratios_observational": ratios,
        "median_primary_ratio_observational": median, "host_failures": host_failures,
        "recorded_gate_maxima": recorded_maxima, "input_source_and_actual_metadata_sha256": files,
        "counter_provenance": "Forward-only native wrappers; started/completed match every prospective count. Getters include cache hits; no cache-miss or per-kernel production count inferred.",
        "prelaunch_metadata_refusal_retained": load(HERE / "root_prelaunch_attempt001_failure.json"),
        "original_route001_002_FAIL_native_physical_stationarity_experiment_status_unchanged": True,
        "optimizer_GPU_parallel_whole_run_speed_certificate": False}
    (HERE / "completion.json").write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"closure_checks_PASS": len(checks), "actual_gate_rows_PASS": cumulative_check_count,
                      "completion_sha256": sha(HERE / "completion.json")}))


if __name__ == "__main__":
    main()
