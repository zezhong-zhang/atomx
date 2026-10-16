"""Report paired statistics without rewriting any frozen run summary."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run, owner = args.run.resolve(), args.run.resolve().parent
    result = json.loads((run / "result.json").read_text())
    raw = json.loads((run / "raw_samples.json").read_text())
    scaling, wrapper = {}, {}
    for sample in raw["backend"]:
        if Path(sample["directory"]).name.startswith("scaling_"):
            scaling.setdefault(sample["repeat"], {})[sample["workers"]] = sample
        else:
            wrapper.setdefault(sample["repeat"], {})[sample["scope"]] = sample
    ratios = [
        scaling[i][1]["wall_seconds"] / scaling[i][2]["wall_seconds"]
        for i in sorted(scaling)
    ]
    extra = [
        wrapper[i]["wrapper"]["wall_seconds"] - wrapper[i]["bare"]["wall_seconds"]
        for i in sorted(wrapper)
    ]
    peak = result["backend_scaling"]["2"]["observed_total_threads_peak"]
    paired = {
        "run_result_sha256": sha(run / "result.json"),
        "raw_samples_sha256": sha(run / "raw_samples.json"),
        "scaling_statistic": "Median of five within-round T1/T2 ratios; time rows retain per-team median/min/max. No confidence interval or large-scale extrapolation.",
        "all_round_speedups_T1_over_T2": ratios,
        "median_paired_speedup": statistics.median(ratios),
        "logical_requested_team_efficiency": statistics.median(ratios) / 2,
        "process_thread_resource_efficiency": statistics.median(ratios) / peak,
        "observed_process_thread_peak": peak,
        "resource_efficiency_statistic": "median paired speedup / aggregate observed p2 total-thread peak; distinct from CPU utilization or instantaneous active-thread efficiency.",
        "private_wrapper_label": "Private low-overhead wait wrapper: unchanged Atomx export/validation/readback plus temporary blocking-wait launcher; not untouched production wrapper latency.",
        "all_round_private_wrapper_extra_seconds": extra,
        "median_paired_private_wrapper_extra_seconds": statistics.median(extra),
        "difference_of_time_medians_private_wrapper_extra_seconds": result[
            "wrapper_same_backend_overhead"
        ]["median_wrapper_extra_seconds"],
    }
    dump(owner / "paired_statistics.json", paired)
    audit = json.loads((owner / "validation.json").read_text())
    tests = json.loads((owner / "test_results.json").read_text())
    rejected = json.loads(
        (owner / "thread_preflight002/p4/runtime_thread_cap_rejection.json").read_text()
    )
    summary = {
        "status": "finite_backend_accuracy_cost_scaling_evidence_complete",
        "authoritative_run": run.name,
        "raw_result": str((run / "result.json").relative_to(owner)),
        "raw_result_sha256": sha(run / "result.json"),
        "private_launcher_source_sha256": sha(owner / "benchmark.py"),
        "parent_contract_sha256": json.loads((run / "contract.json").read_text())[
            "parent_contract_sha256"
        ],
        "accuracy": result["accuracy"],
        "backend_scaling_raw_ratio_of_time_medians": result["backend_scaling"],
        "resource_rejection_review": {
            "4": {
                "status": "resource_rejected",
                "first_rejection_threshold_total_threads": 9,
                "final_observed_total_threads_peak": rejected[
                    "observed_backend_max_threads"
                ],
                "scope": "Frozen run003 row records first observed over-cap value 9; final rejection observer reached 10 before owned child exit. Frozen run row remains unchanged.",
                "speedup": None,
                "efficiency": None,
                "evidence": "thread_preflight002/p4/runtime_thread_cap_rejection.json",
            },
            "8": {
                "status": "resource_rejected",
                "final_observed_total_threads_peak": 25,
                "scope": "Requested-only run001 observation; not repeated under the new limiting configuration.",
                "speedup": None,
                "efficiency": None,
                "evidence": "run001/stopped_thread_budget.json",
            },
        },
        "authoritative_paired_statistics": paired,
        "finite_CI_oracle_cost": result["finite_CI_oracle_cost"],
        "timing_scope_and_limits": result["limits"],
        "resource_accounting": {
            "sampled_backend_peak_RSS_bytes": max(
                sample["actual_runtime"]["sampled_backend_peak_RSS_bytes"]
                for sample in raw["backend"]
            ),
            "benchmark_parent_peak_RSS_bytes": result["parent_process_peak_RSS_bytes"],
            "RSS_scope": "Backend /proc VmHWM sampled lower bound; parent resource.ru_maxrss separate. These are not a whole-process-tree exact peak.",
        },
        "validation": audit,
        "tests": tests,
        "failures_and_review_overrides": {
            "default_noncanonical_operator_failure": "run003/accuracy_two_core_default/accuracy.json",
            "larger_core_CCSDT_truncation": "run003/accuracy_four_core_corrected/accuracy.json",
            "requested_only_thread_budget_failure": "run001/stopped_thread_budget.json",
            "stricter_requested_team_preflight_rejection": "thread_preflight001/p2/runtime_thread_cap_rejection.json",
            "parent_thread_cap_preflight_rejection": "thread_preflight002/p4/runtime_thread_cap_rejection.json",
            "timeout_polling_inference_withdrawal": "run002/timing_review_withdrawal.json",
        },
        "performance_ranking": None,
        "precision_equivalent_software_ranking": False,
        "highest_level_reached": False,
        "production_changes": False,
    }
    dump(owner / "result.json", summary)
    print(json.dumps(paired))


if __name__ == "__main__":
    main()
