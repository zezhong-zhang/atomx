"""Closed-result JSON/source review only: no NumPy, NPZ decoding or solve."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


contract = read(HERE / "contract.json")
result = read(HERE / "actual/result.json")
seal = read(HERE / "root_execution_seal.json")
checks = []


def check(name, condition):
    checks.append({"name": name, "PASS": bool(condition)})
    if not condition:
        raise RuntimeError(name)


check("status", result["status"] == "PASS_native_sparse_full_basis_Davidson_fixed_R0")
check(
    "273_original_controls",
    len(result["checks"]) == 273 and all(r["PASS"] for r in result["checks"]),
)
check(
    "driver_contract_seal",
    seal["driver_sha256"] == sha(HERE / "run.py")
    and seal["contract_sha256"] == sha(HERE / "contract.json"),
)
check("root_once_only", seal["execution_authorized_by_root"] and seal["once_only"])
check(
    "resources_source",
    seal["resources"] == contract["resources"]
    and result["source_map_sha256"]
    == seal["source_map_sha256"]
    == contract["source_map_sha256"],
)
for key, spec in contract["inputs"].items():
    check("input_SHA256/" + key, sha(ROOT / spec["path"]) == spec["sha256"])
source = ROOT / contract["source_archive"]
for key, spec in result["actual_imported_sources"].items():
    check(
        "imported_SHA256/" + key,
        sha(source / spec["path"].removeprefix("src/")) == spec["sha256"],
    )
check("81_imported_native_modules", len(result["actual_imported_sources"]) == 81)
check(
    "attempted_calls_complete",
    all(result["counts"].get(k, 0) == v for k, v in result["attempted"].items()),
)
check(
    "call_caps",
    all(
        result["attempted"].get(k, 0) <= v
        for k, v in contract["resources"]["calls"].items()
    ),
)
check(
    "scope_limits",
    result["old_failures_preserved"]
    and not any(
        result[k]
        for k in [
            "experimental_accuracy_qualified",
            "larger_4f12_calculated",
            "public_iterative_owner_or_tracker_qualified",
            "controlled_timing",
        ]
    ),
)
check("whole_angular_census", result["counts"]["angular_nullspace"] == 43)
check(
    "two_operators_ten_blocks_ten_Davidson",
    all(
        result["counts"][k] == v
        for k, v in [
            ("H", 2),
            ("angular_table", 10),
            ("H_projection", 10),
            ("Davidson", 10),
            ("NPZ_load", 5),
            ("NPZ_save", 2),
        ]
    ),
)
for arm, n in [("A", 1595), ("B", 2574)]:
    point = result["points"][arm]
    check(
        arm + "/complete_CSF_rows_columns",
        sum(r["CSFs"] for r in point["blocks"]) == n
        and all(r["CSFs"] == r["full_columns"] for r in point["blocks"]),
    )
    check(
        arm + "/complete_sector_set",
        {tuple(r["sector"]) for r in point["blocks"]}
        == {(1, 1), (1, -1), (3, 1), (3, -1), (5, 1)},
    )
    check(
        arm + "/five_owned_UIDs",
        [r["UID"] for r in point["UIDs"]] == [uid for _, uid in contract["UIDs"]],
    )
    check(
        arm + "/same_radial_binding",
        point["input_binding"] == result["work"]["nucleus"]["binding"],
    )
    for row in point["blocks"]:
        sector = str(row["sector"])
        check(
            arm + "/full_action/" + sector,
            row["whole_reference_action_max_abs_Ha"]
            <= contract["gates"]["whole_action_max_abs_Ha"],
        )
        check(
            arm + "/full_residual/" + sector,
            row["full_reference_residual_max_abs_Ha"]
            <= contract["gates"]["full_reference_eigenpair_max_abs_Ha"],
        )
        check(
            arm + "/selected_direct_L2/" + sector,
            len(row["direct_residual_L2_Ha"]) == 2
            and max(row["direct_residual_L2_Ha"])
            <= contract["gates"]["Davidson_selected_eigenpair_L2_Ha"],
        )
        check(
            arm + "/iterations/" + sector,
            row["iterations"] <= contract["gates"]["Davidson_max_iter"],
        )
check(
    "table_projection_payloads",
    len(result["work"]["tables"]) == len(result["work"]["projections"]) == 10
    and sum(r["bytes"] for r in result["work"]["tables"]) <= 600_000_000,
)
launch = read(HERE / "launcher_failure001.json")
check(
    "pre_interpreter_launcher_failure_preserved",
    launch["stage"] == "launcher_before_interpreter"
    and launch["science_calls"] == 0
    and not launch["actual_created"]
    and not launch["driver_changed"]
    and not launch["contract_changed"],
)
review = {
    "status": "PASS_JSON_source_review",
    "checks": checks,
    "checks_count": len(checks),
    "NPZ_loads": 0,
    "science_calls": 0,
    "scope": "JSON and source/hash consistency; arrays are not independently decoded or recomputed",
    "bindings": {
        key: sha(HERE / key)
        for key in [
            "run.py",
            "contract.json",
            "root_execution_seal.json",
            "launcher_failure001.json",
            "actual/result.json",
            "actual/A.json",
            "actual/B.json",
            "actual/A_selected.npz",
            "actual/B_selected.npz",
        ]
    },
}
(HERE / "result_review.json").write_text(json.dumps(review, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": review["status"],
            "checks": len(checks),
            "NPZ_loads": 0,
            "science_calls": 0,
        }
    )
)
