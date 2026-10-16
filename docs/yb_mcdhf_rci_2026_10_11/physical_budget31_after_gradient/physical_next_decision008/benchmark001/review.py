"""Standard-library-only receipt review; never invokes a scientific worker."""

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


c = read(HERE / "contract.json")
result = read(HERE / "actual/result.json")
checks = {}
checks["driver_hash"] = sha(HERE / "run.py") == c["driver_sha256"]
checks["exact_root_seal"] = read(HERE / "root_execution_seal.json") == {
    "execution_authorized_by_root": True, "once_only": True,
    "driver_sha256": sha(HERE / "run.py"), "contract_sha256": sha(HERE / "contract.json"),
    "resources": c["resources"]}
checks["source_manifest"] = all(sha(ROOT / p) == h for p, h in c["sources"].items())
checks["input_manifest"] = all(sha(ROOT / v["path"]) == v["sha256"] for v in c["inputs"].values())
checks["once_only_ABBA_success"] = result["success"] and result["order"] == ["old", "new", "new", "old"]
workers = []
for row in result["workers"]:
    path = ROOT / row["path"]
    checks["worker_hash/" + path.stem] = sha(path) == row["sha256"] and row["returncode"] == 0
    workers.append(read(path))
checks["four_workers_ten_blocks_each"] = len(workers) == 4 and all(w["success"] and len(w["blocks"]) == 10 for w in workers)
checks["all_source_imports_bound"] = all(
    all(c["sources"].get(path) == expected for path, expected in w["imported_sources"].items())
    for w in workers)
baseline = [b["identity"] for b in workers[0]["blocks"]]
checks["all40_bitwise_complete_block_identities"] = all(
    b["identity"] == baseline[i] for w in workers for i, b in enumerate(w["blocks"]))
checks["all40_signed_direct_action_checks"] = all(
    b["signed_bilinear_action_error"] <= c["gates"]["signed_bilinear_action_max_abs"]
    for w in workers for b in w["blocks"])
checks["all40_signed_negative_controls"] = all(
    b["signed_absolute_control_difference"] >= c["gates"]["signed_absolute_control_min_difference"]
    for w in workers for b in w["blocks"])
checks["all40_cache_repeat_identities"] = all(b["cache_identity"] for w in workers for b in w["blocks"])
checks["all_host_guards"] = all(w["controlled_timing_admissible"] for w in workers)
checks["worker_caps"] = all(w["elapsed_seconds"] < c["resources"]["worker_alarm_seconds"]
                            and w["RSS_highwater_KiB"] * 1024 < c["resources"]["worker_AS_bytes"]
                            for w in workers)
checks["same_numerical_versions"] = len({(w["numpy"], w["scipy"]) for w in workers}) == 1
checks["complete_expected_CSF_counts"] = all(
    sum(b["CSFs"] for b in w["blocks"] if b["arm"] == "A") == 1595
    and sum(b["CSFs"] for b in w["blocks"] if b["arm"] == "B") == 2574 for w in workers)
timings = []
for w in workers:
    timings.append({"variant": w["variant"], "worker_seconds": w["elapsed_seconds"],
                    "table_seconds": sum(b["table_seconds"] for b in w["blocks"]),
                    "projection_seconds": sum(b["projection_seconds"] for b in w["blocks"]),
                    "connection_pull_seconds": sum(b["connected_seconds"] for b in w["blocks"]),
                    "RSS_MiB": w["RSS_highwater_KiB"] / 1024})
means = {variant: {key: statistics.mean(t[key] for t in timings if t["variant"] == variant)
                   for key in ("worker_seconds", "table_seconds", "projection_seconds", "connection_pull_seconds", "RSS_MiB")}
         for variant in ("old", "new")}
summary = {
    "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
    "pass_count": sum(checks.values()), "check_count": len(checks), "samples_per_variant": 2,
    "timings": timings, "means": means,
    "mean_table_speed_ratio_old_over_new": means["old"]["table_seconds"] / means["new"]["table_seconds"],
    "mean_table_time_reduction_fraction": 1 - means["new"]["table_seconds"] / means["old"]["table_seconds"],
    "mean_worker_RSS_reduction_fraction": 1 - means["new"]["RSS_MiB"] / means["old"]["RSS_MiB"],
    "pairwise_table_speed_ratios_old_over_new": [timings[0]["table_seconds"] / timings[1]["table_seconds"],
                                               timings[3]["table_seconds"] / timings[2]["table_seconds"]],
    "retained_payload_identical": {"ten_table_CSR_bytes": sum(b["table_bytes"] for b in workers[0]["blocks"]),
                                   "ten_table_nnz": sum(b["table_nnz"] for b in workers[0]["blocks"]),
                                   "ten_connected_candidates": sum(b["connected_candidates"] for b in workers[0]["blocks"]),
                                   "blocks": [{key: b[key] for key in ("arm", "sector", "CSFs", "raw_determinants", "radial_keys", "connected_candidates", "table_nnz", "table_bytes", "projection_nnz")}
                                              for b in workers[0]["blocks"]]},
    "max_signed_action_bilinear_error": max(b["signed_bilinear_action_error"] for w in workers for b in w["blocks"]),
    "min_signed_abs_control_difference": min(b["signed_absolute_control_difference"] for w in workers for b in w["blocks"]),
    "evidence": {str(p.relative_to(HERE)): sha(p) for p in
                 [HERE / "run.py", HERE / "contract.json", HERE / "actual/result.json",
                  HERE / "root_execution_seal.json",
                  *sorted((HERE / "actual").glob("worker_*.json"))]},
    "scope": "Angular construction, cold in-process caches, one CPU, n=2 per variant. No radial/H/CI solve, end-to-end or parallel scaling, 4f12 energy, or spectroscopy precision claim.",
}
(HERE / "review.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: summary[key] for key in ("status", "pass_count", "check_count", "means", "mean_table_speed_ratio_old_over_new", "mean_worker_RSS_reduction_fraction", "retained_payload_identical")}))
if not all(checks.values()):
    raise SystemExit(1)
