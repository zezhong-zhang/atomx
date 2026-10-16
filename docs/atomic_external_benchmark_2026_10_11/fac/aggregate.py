"""Extend the delivered campaign with complete, separately scoped FAC evidence."""

import copy
import csv
import hashlib
import json
import statistics
from pathlib import Path

OWNER = Path(__file__).resolve().parent
PARENT = OWNER.parent
INPUTS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path.relative_to(PARENT))] = digest(path)
    return json.loads(path.read_text())


def save(name, value):
    (OWNER / name).write_text(json.dumps(value, indent=2) + "\n")


def scaling_rows(performance):
    groups = performance["groups"]
    rows = []
    epoch_pass = (
        performance["source_guard_passed"]
        and performance["runtime_guard_passed"]
    )

    def admitted(group):
        return all(
            group.get(name, False)
            for name in (
                "science_ok", "resource_ok", "load_ok", "full_array_correctness_ok"
            )
        )

    for measured in performance["cpu_rows"]:
        backend = measured["workload"].split()[0]
        selected = sorted(
            [g for g in groups if g["backend"] == backend
             and g["workers"] == measured["workers"]],
            key=lambda g: g["round"],
        )
        if not selected:
            # A preflight rejection is a missing measurement, never a zero time.
            row = copy.deepcopy(measured)
            for key in (
                "speedup", "speedup_min", "speedup_max", "efficiency",
                "efficiency_min", "efficiency_max", "process_thread_efficiency",
            ):
                row[key] = None
            row["controlled"] = False
            rows.append(row)
            continue
        pairs = [
            next(g for g in groups if g["backend"] == backend
                 and g["round"] == sample["round"] and g["workers"] == 1)
            for sample in selected
        ]
        # Sum all four complete, freshly launched child lifetimes independently.
        times = []
        ratios = []
        for sample, baseline in zip(selected, pairs, strict=True):
            for group in (sample, baseline):
                if [c["z"] for c in group["cases"]] != [18, 26, 54, 82]:
                    raise ValueError("The entire fixed four-ion group is required")
                if group["seconds"] != sum(c["seconds"] for c in group["cases"]):
                    raise ValueError("Child sums do not reproduce the group time")
            times.append(sample["seconds"])
            ratios.append(baseline["seconds"] / sample["seconds"])
        controlled = (
            len(selected) == 3 and {g["round"] for g in selected} == {0, 1, 2}
            and epoch_pass and all(admitted(g) for g in selected + pairs)
        )
        threads = max(c["actual_threads_peak"] or 0
                      for g in selected for c in g["cases"])
        controlled = controlled and 0 < threads <= 8
        speedup = statistics.median(ratios) if controlled else None
        row = {
            "workload": measured["workload"], "workers": measured["workers"],
            "rounds": len(times), "seconds_median": statistics.median(times),
            "seconds_min": min(times), "seconds_max": max(times),
            "speedup": speedup,
            "speedup_min": min(ratios) if controlled else None,
            "speedup_max": max(ratios) if controlled else None,
            "efficiency": speedup / measured["workers"] if controlled else None,
            "efficiency_min": min(ratios) / measured["workers"] if controlled else None,
            "efficiency_max": max(ratios) / measured["workers"] if controlled else None,
            "actual_threads_peak": threads,
            "process_thread_efficiency": speedup / threads if controlled else None,
            "controlled": controlled, "scope": measured["scope"],
            "statistic": "median of within-round paired T1/Tp; times separate",
            "observed_times_seconds": times, "observed_paired_speedups": ratios,
        }
        for key in (
            "seconds_median", "seconds_min", "seconds_max", "speedup",
            "speedup_min", "speedup_max", "efficiency", "controlled",
        ):
            if row[key] != measured[key]:
                raise ValueError(f"Owner/root statistic or admission disagrees: {key}")
        rows.append(row)
    return rows


def main():
    contract = read(OWNER / "contract.json")
    previous = read(PARENT / "aggregation_provenance.json")
    for name, expected in previous["inputs_sha256"].items():
        path = PARENT / name
        if digest(path) != expected:
            raise ValueError(f"Prior campaign input changed: {name}")
        INPUTS[name] = expected
    base = read(PARENT / "metrics.json")
    snapshot = OWNER / "prior_campaign_snapshot/metrics.json"
    if digest(snapshot) != digest(PARENT / "metrics.json"):
        raise ValueError("The previously delivered metrics must remain immutable")
    precision = read(OWNER / "precision/result.json")
    performance = read(OWNER / "performance/result.json")
    model_spread = read(OWNER / "performance/model_spread.json")
    stages = read(OWNER / "performance/stages.json")
    gaps = read(OWNER / "gaps.json")
    review = read(OWNER / "review/final_audit.json")
    if not review["passed"]:
        raise ValueError("Independent provenance/complete-array/statistic audit failed")
    for case in precision["cases"]:
        # Gate outcomes include failures. Preserve both absolute and relative tests.
        for key in ("common_integrals", "2000", "4000"):
            if key not in case["gates"]:
                raise ValueError("A complete precision grid is required")
    if precision["total_levels"] != 192 or precision["total_integrals"] != 2003:
        raise ValueError("Truncated precision panel")
    if not precision["source_and_runtime_unchanged"]:
        raise ValueError("Precision epoch source/runtime changed")
    combined = copy.deepcopy(base)
    combined["cpu"].extend(scaling_rows(performance))
    combined["FAC"] = {
        "contract": contract,
        "precision": precision,
        "model_spread": model_spread,
        "stages": stages,
        "next_directions": gaps,
        "workflow": {k: v for k, v in performance.items()
                     if k not in ("groups", "warmups", "cpu_rows")},
        "independent_audit": "docs/atomic_external_benchmark_2026_10_11/fac/review/final_audit.json",
        "same_integral_agreement_is_experimental_accuracy": False,
        "equal_accuracy_cross_software_speedup_claim": False,
    }
    combined["highest_level_reached"] = False
    combined["whole_production_software_rank"] = False
    save("metrics.json", combined)
    fields = [key for key in combined["cpu"][0]
              if not key.startswith("observed_")]
    with (OWNER / "performance.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined["cpu"])
    save("aggregation_provenance.json", {
        "inputs_sha256": INPUTS, "script_sha256": digest(Path(__file__).resolve()),
        "original_metrics_retained": True,
        "complete_groups_required": True,
        "paired_statistic": "median of same-round complete child T1/Tp",
        "experimental_spectra_rerun": False,
        "failed_precision_and_resource_gates_retained": True,
    })


if __name__ == "__main__":
    main()
