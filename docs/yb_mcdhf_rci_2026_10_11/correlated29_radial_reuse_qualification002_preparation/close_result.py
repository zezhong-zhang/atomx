"""Close the sole root-authorized run using stored JSON/files only.

No Atomx/numerical imports, array loading, qualification replay or timing.
The producer and root receipt must already be closed before this is invoked.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    root = load(HERE / "root_execution.json")
    if root.get("status") != "foreground_closed":
        raise ValueError("Root foreground has not closed; no actual closure is allowed")
    if sha(HERE / "execution_contract.json") != "971fa0ffeee20122a1fee234a66c94e9b3cb6ad5d1b4f1d406e2239b7d2813e2":
        raise ValueError("Sole root execution seal changed")
    contract, handoff = load(HERE / "contract.json"), load(HERE / "handoff.json")
    for name, expected in handoff["files_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Frozen preparation changed: " + name)
    for row in contract["inputs"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Frozen input changed: " + row["path"])
    for variant in ("ordinary", "candidate"):
        source = contract[variant]
        identity = load(REPO / source["identity"])
        for name, expected in identity["all_atomx_python_sha256"].items():
            if sha(REPO / source["archive"] / name.removeprefix("src/")) != expected:
                raise ValueError("Frozen source changed: " + name)

    actual = HERE / "actual"
    phases = {}
    for phase in ("anchors", "ordinary", "candidate"):
        path = actual / phase / "result.json"
        if path.exists():
            phases[phase] = load(path)
        else:
            failure = actual / phase / "failure.json"
            if failure.exists():
                phases[phase] = {"all_passed": False, "failure": load(failure)}
    outcome = load(actual / "result.json") if (actual / "result.json").exists() else None
    comparison_path = actual / "comparison.json"
    if not comparison_path.exists():
        comparison_path = actual / "comparison_failed.json"
    comparison = load(comparison_path) if comparison_path.exists() else None
    passed = (root.get("exit_code") == 0 and outcome is not None
              and outcome.get("all_equivalence_gates_passed") is True
              and comparison is not None and comparison.get("all_passed") is True
              and set(phases) == {"anchors", "ordinary", "candidate"}
              and all(row.get("all_passed") is True for row in phases.values()))
    if passed:
        for phase, row in phases.items():
            if not all(check["passed"] is True for check in row["checks"]):
                raise ValueError("Stored failed gate in successful phase " + phase)
            expected = (contract["anchor_calls"] if phase == "anchors"
                        else contract["chain_exact_calls"])
            for name, count in expected.items():
                if not (row["calls_started"].get(name, 0) == count
                        and row["calls_completed"].get(name, 0) == count):
                    raise ValueError("Stored exact-call ledger mismatch " + phase + ": " + name)
        if not all(check["passed"] is True for check in comparison["checks"]):
            raise ValueError("Stored comparison is not all passed")

    evidence = {}
    for path in sorted(actual.rglob("*")):
        if path.is_file() and "_numba_cache" not in str(path):
            evidence[str(path.relative_to(HERE))] = sha(path)
    gate_summary = {}
    interesting = ("full_H", "full_roots", "full_CI", "radial_gradient", "raw_g", "tangent_g",
                   "metric_pg", "owned_H", "owner", "UID", "target", "coordinate_Gram",
                   "radial_Gram", "no_pair", "same_reported", "signed_RDM")
    for phase, row in phases.items():
        gate_summary[phase] = [check for check in row.get("checks", [])
                               if any(token in check["name"] for token in interesting)]
    if comparison:
        gate_summary["comparison"] = [check for check in comparison["checks"]
            if not check["name"].startswith("signed_cluster_")]

    completion = {"status": "closed_equivalence_pass" if passed else "closed_first_failure_retained",
        "all_recorded_equivalence_gates_passed": passed,
        "root_exit_code": root.get("exit_code"),
        "root_execution_sha256": sha(HERE / "root_execution.json"),
        "execution_contract_sha256": sha(HERE / "execution_contract.json"),
        "source_maps": handoff["source_maps"], "frozen_inputs_unchanged": True,
        "actual_files_sha256": evidence, "actual_gate_summary": gate_summary,
        "first_saved_comparison_failure": (next((check for check in comparison["checks"]
                                                   if check["passed"] is not True), None)
                                            if comparison else None),
        "actual_calls": {phase: {"started": row.get("calls_started", row.get("failure", {}).get("started", {})),
                                 "completed": row.get("calls_completed", row.get("failure", {}).get("completed", {}))}
                         for phase, row in phases.items()},
        "actual_stages_uncontrolled_seconds": {phase: row.get("stages_uncontrolled_seconds", {})
                                               for phase, row in phases.items()},
        "actual_call_seconds_uncontrolled": {phase: row.get("call_wall_seconds_uncontrolled", {})
                                              for phase, row in phases.items()},
        "closure_array_loads_or_scientific_calls": 0,
        "original_failed_route_qualification_unchanged": True,
        "shared_source_adoption_authorized": False,
        "physical_stationarity_experiment_speedup_parallel_GPU_certified": False}
    write(HERE / "completion.json", completion)
    lines = ["# Semiseparable H route successor — actual closure", "",
        ("The sole root foreground run passed its stored equivalence gates."
         if passed else "The sole root foreground run failed; the first failure and partial calls remain unchanged."),
        "This closure reads JSON and hashes stored files; it performs no new physical calculation, array replay or timing.", "",
        "The ordinary 62a109 epoch and private 608f8f9d candidate share the corrected inner-cell operator; the private candidate changes only the CI pair-potential route. Original route/derivative/stationarity/experimental failures remain historical failures.", ""]
    for phase, row in phases.items():
        lines.extend([f"{phase}: stored pass={row.get('all_passed', False)}; calls completed "
                      + json.dumps(row.get("calls_completed", row.get("failure", {}).get("completed", {}))),
                      "Stages (uncontrolled seconds): " + json.dumps(row.get("stages_uncontrolled_seconds", {})), ""])
    lines.extend(["See completion.json for exact H/CI/gradient/UID/owner gates and stored file hashes. Full signed cluster gates remain in actual/comparison.json.", "",
        "Fresh-process imports/JIT, cProfile, validation, source/input hashing and archive overhead make this an uncontrolled correctness/profile observation. It certifies no speedup, full-optimizer, parallel or GPU improvement; kernel getter counts include cache hits. Shared adoption needs a separate reviewed source change.", ""])
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    write(HERE / "documentation_impact_actual.json", {"new_actual_closure": True,
        "completion_sha256": sha(HERE / "completion.json"), "RESULTS_sha256": sha(HERE / "RESULTS.md"),
        "frozen_preparation_or_shared_source_or_closed_parent_evidence_changed": False,
        "original_FAIL_not_upgraded": True, "performance_certificate": False})
    write(HERE / "actual_handoff.json", {"status": completion["status"],
        "files_sha256": {name: sha(HERE / name) for name in
                         ("completion.json", "RESULTS.md", "documentation_impact_actual.json", "close_result.py")},
        "root_execution_sha256": completion["root_execution_sha256"],
        "actual_result_sha256": sha(actual / "result.json") if outcome else None,
        "scientific_calls": 0, "shared_adoption_authorized": False})
    print(json.dumps({"status": completion["status"],
                      "completion_sha256": sha(HERE / "completion.json"),
                      "actual_handoff_sha256": sha(HERE / "actual_handoff.json"), "science_calls": 0}))


if __name__ == "__main__":
    main()
