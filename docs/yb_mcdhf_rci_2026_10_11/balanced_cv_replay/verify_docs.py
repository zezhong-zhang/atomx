"""Factual presentation, evidence hashes and local documentation link checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    readme = (HERE / "README.md").read_text()
    result, verification, presentation = [json.loads((HERE / name).read_text())
        for name in ("result.json", "verification.json", "presentation.json")]
    checks = []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})

    for target in re.findall(r"\]\(([^)]+)\)", readme):
        check("local_link:" + target, (HERE / target).resolve().exists())
    check("scientific_replay_passed", verification["all_passed"])
    check("scientific_gate_count_in_README", str(verification["checks_count"]) in readme)
    check("execution_gate_count_in_README", str(len(result["checks"])) in readme)
    check("scientific_result_unchanged_by_presentation", verification["artifacts_sha256"]["result.json"]
          == presentation["scientific_result_sha256"] == sha(HERE / "result.json"))
    check("presentation_verified_input", presentation["verification_sha256"] == sha(HERE / "verification.json"))
    check("presentation_script", presentation["script_sha256"] == sha(HERE / "make_report.py"))
    check("presentation_csv_hashes", all(sha(HERE / filename) == expected
          for filename, expected in presentation["artifacts_sha256"].items()))
    check("presentation_final_state_rows", presentation["final_static_rows"] ==
          result["stages"][-1]["tracked_levels"]["static_Breit"][1:])
    for stage in result["stages"]:
        for model, rows in stage["tracked_levels"].items():
            for row in rows[1:]:
                check(stage["name"] + "/" + model + "/" + row["label"] + "/table_energy",
                      f"{row['excitation_from_tracked_6s_cm1']:.3f}" in readme)
        for row in stage["tracked_levels"]["static_Breit"]:
            check(stage["name"] + "/" + row["label"] + "/absolute_shift_Ha",
                  f"{row['energy_delta_from_previous_Ha']:.12f}".replace("-", "−") in readme)
        for row in stage["tracked_levels"]["static_Breit"][1:]:
            check(stage["name"] + "/" + row["label"] + "/same_state_transition_increment",
                  f"{row['transition_delta_from_previous_cm1']:+.3f}".replace("-", "−") in readme)
        for label, moments in stage["multiplets"]["static_Breit"].items():
            check(stage["name"] + "/" + label + "/centroid", f"{moments['centroid_cm1']:.3f}" in readme)
            check(stage["name"] + "/" + label + "/splitting", f"{moments['splitting_cm1']:.3f}" in readme)
    for row in result["stages"][-1]["tracked_levels"]["static_Breit"][1:]:
        check(row["label"] + "/final_gap", f"{row['gap_cm1']:+.3f}".replace("-", "−") in readme)
        check(row["label"] + "/final_relative_error", f"{100*row['relative_error']:+.3f}%".replace("-", "−") in readme)
    check("bounded_observation_time", f"{result['resource']['total_seconds_uncontrolled']:.3f} s" in readme)
    check("bounded_observation_memory", f"{result['resource']['peak_RSS_MiB']:.3f} MiB" in readme)
    check("owner_scientific_scope", "not a production spectrum" in readme and "**not** undergone" in readme
          and "complete\ncore-valence" in readme)
    output = {"all_passed": all(c["passed"] for c in checks), "checks_count": len(checks), "checks": checks,
              "README_sha256": sha(HERE / "README.md"), "verification_sha256": sha(HERE / "verification.json"),
              "presentation_sha256": sha(HERE / "presentation.json"), "script_sha256": sha(Path(__file__)),
              "scope": "Local links, source-bound presentation hashes and rounded scientific table facts; does not rewrite scientific evidence."}
    (HERE / "documentation_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"all_passed": output["all_passed"], "checks_count": len(checks),
                      "failed": [c for c in checks if not c["passed"]]}))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
