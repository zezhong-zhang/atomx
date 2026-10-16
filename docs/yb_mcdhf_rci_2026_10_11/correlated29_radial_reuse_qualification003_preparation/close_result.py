"""Close exact003 JSON/hash evidence without array loads or scientific imports."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    root = json.loads((HERE / "root_execution.json").read_text())
    result = json.loads((HERE / "actual/result.json").read_text())
    comparison = json.loads((HERE / "actual/comparison.json").read_text())
    assert root["status"] == "foreground_closed" and root["exit_code"] == 0
    assert sha(HERE / "root_execution.json") == "27fbab86c8be671bb0b390b96cf621e68444f70c5073e45fa61039e6d9649946"
    assert sha(HERE / "actual/result.json") == "04592757c7ef05a22b181453d31d708c6d41d3fe6d433b372447c42369c73836"
    assert sha(HERE / "actual/comparison.json") == "a4bda374b3246398e480ac1d21ac50a17c7e1e4c7824989908d71deb74b79f76"
    assert result["all_equivalence_gates_passed"] is True and comparison["all_passed"] is True
    assert result["driver_sha256"] == sha(HERE / "run.py")
    assert result["contract_sha256"] == sha(HERE / "contract.json")
    assert result["comparison_sha256"] == sha(HERE / "actual/comparison.json")
    for row in contract["inputs"].values():
        assert sha(REPO / row["path"]) == row["sha256"], row["path"]
    for variant in ("ordinary", "candidate"):
        row = contract[variant]
        identity = json.loads((REPO / row["identity"]).read_text())
        assert identity["source_map_sha256"] == row["source_map_sha256"]
        for name, expected in identity["all_atomx_python_sha256"].items():
            assert sha(REPO / row["archive"] / name.removeprefix("src/")) == expected
    phases = {phase: json.loads((HERE / "actual" / phase / "result.json").read_text())
              for phase in ("anchors", "ordinary", "candidate")}
    for phase, data in phases.items():
        assert data["all_passed"] is True and all(row["passed"] for row in data["checks"])
        assert data["calls_started"] == data["calls_completed"]
        expected = contract["anchor_calls"] if phase == "anchors" else contract["chain_exact_calls"][phase]
        for name, count in expected.items():
            assert data["calls_completed"].get(name, 0) == count, (phase, name)
        if phase != "anchors":
            assert data["arrays_sha256"] == sha(HERE / "actual" / phase / "arrays.npz")
            assert data["owner_sha256"] == sha(HERE / "actual" / phase / "owner.json")
    assert len(comparison["checks"]) == 1126 and all(row["passed"] for row in comparison["checks"])
    prior = json.loads((REPO / contract["inputs"]["prior002_closed_failure"]["path"]).read_text())
    assert prior["all_recorded_equivalence_gates_passed"] is False
    aggregate = {}
    for data in phases.values():
        for name, count in data["calls_completed"].items():
            aggregate[name] = aggregate.get(name, 0) + count
    assert [aggregate[k] for k in ("H", "CI", "gradient", "scipy_eigh", "Gauss_setup_eigvalsh")] == [2, 2, 2, 17, 7]
    assert aggregate["initial_hS"] + aggregate["CI_internal_hS"] == 28
    physical = {phase: json.loads((HERE / "actual" / phase / "targets_metric.json").read_text())
                for phase in ("ordinary", "candidate")}
    assert all(not r["stationarity_passed"] and not r["indicator_passed"] for r in physical.values())
    observed = {row["name"]: row for row in comparison["checks"] if row["name"].startswith("whole_")}
    spans = [row for row in comparison["checks"] if row["name"].startswith("signed_cluster_")]
    assert len(spans) == 1046
    replay = [row for row in phases["ordinary"]["checks"] if row["name"] == "new_centered_epoch_all_roots_bitwise_replay"]
    assert len(replay) == 1 and replay[0]["passed"]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "actual").rglob("*")) if p.is_file()}
    closure = {
        "status": "closed_new_centered_epoch_equivalence_passed",
        "new_epoch_equivalence_passed": True, "original_route002_qualification": "FAIL preserved",
        "root_execution_sha256": sha(HERE / "root_execution.json"),
        "actual_files_sha256": inputs,
        "source_maps": {v: contract[v]["source_map_sha256"] for v in ("ordinary", "candidate")},
        "gate_counts": {**{phase: len(data["checks"]) for phase, data in phases.items()}, "saved_comparison": 1126},
        "actual_completed_calls": {phase: data["calls_completed"] for phase, data in phases.items()},
        "aggregate_completed_calls": aggregate,
        "original_H_bitwise_new_ordinary_centered_energy_bitwise_replay": True,
        "old_unshifted_energy_comparison": "Historical source evolution under existing allroot1e-8, never bitwise replay",
        "full_comparison_whole_array_gates": observed,
        "full1046span_maximum": max(spans, key=lambda row: row["error"]),
        "column296": next(row for row in spans if row["name"] == "signed_cluster_[296]"),
        "samepoint_physical_failures": {phase: {k: r[k] for k in ("residual", "indicator_Ha", "stationarity_passed", "indicator_passed")}
                                       for phase, r in physical.items()},
        "actual_stages_uncontrolled_seconds": {phase: data["stages_uncontrolled_seconds"] for phase, data in phases.items()},
        "closure_array_loads_and_new_science_calls": 0,
        "shared_adoption_or_optimizer_or_speedup_or_GPU_certified": False,
    }
    with (HERE / "completion.json").open("x") as handle:
        json.dump(closure, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"completion_sha256": sha(HERE / "completion.json"),
                      "gate_counts": closure["gate_counts"], "aggregate_calls": aggregate}, indent=2))


if __name__ == "__main__":
    main()
