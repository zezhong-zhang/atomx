"""Check stage009 evidence, current guide links and claims without physics work."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    result = read(TOP / "results_stage009.json")
    for name, binding in result["inputs"].items():
        check("input_SHA/" + name, sha(ROOT / binding["path"]) == binding["sha256"])
    card = read(TOP / "delivery_stage009/delivery_proof.json")
    check("actual_card_readback", card["successful_send_and_readback"])
    check("card_current_result", card["stage009_sha256"] == sha(TOP / "results_stage009.json"))
    check("actual_dispatch", card["iteration_dispatch_ok"])
    for name, digest in card["files_sha256"].items():
        check("card_SHA/" + name, sha(TOP / "delivery_stage009" / name) == digest)
    decision = read(TOP / "virtual_only_discriminator/decision.json")
    for name, digest in decision["bound_outputs_sha256"].items():
        check("virtual_decision_SHA/" + name,
              sha(TOP / "virtual_only_discriminator" / name) == digest)
    check("original_stationarity_fail", not decision["stationarity_passed"])
    check("original_experiment_fail", not decision["experimental_gate_passed"])
    check("virtual_independent3112", decision["independent_saved_array_checks_passed"] == 3112)
    check("performance_independent10041",
          result["retained_H_chain_independent_verification"]["checks"] == 10041)
    check("v4_published", result["website_publication"]["deployment_status"] == "succeeded")
    check("audience_preserved", result["website_publication"]["audience_unchanged"])
    check("v4_models_rows", (result["website_publication"]["models"],
                             result["website_publication"]["rows"]) == (14, 56))
    guides = [TOP / "README.md", ROOT / "docs/NIST_SPECTRUM_ATLAS.md",
              ROOT / "docs/atomic_native_roadmap_2026_10_11/README.md"]
    guide = guides[0].read_text()
    for term in ["1.94348e−6", "19.52%", "10,041", "3,112", "0.287–0.348",
                 "完整1046目标块", "尚未执行数值资格", "results_stage009.json",
                 "delivery_stage009/delivery_proof.json"]:
        check("current_guide_claim/" + term, term in guide)
    for path in guides:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.strip().split("#")[0]
            if not target or re.match(r"(?:https?://|mailto:)", target):
                continue
            check("local_link/" + str(path.relative_to(ROOT)) + "/" + target,
                  (path.parent / target).exists())
        check("guide_whitespace/" + str(path.relative_to(ROOT)),
              all(line.rstrip() == line for line in path.read_text().splitlines()))
    output = {"status": "stage009_current_documentation_and_delivery_verified",
              "all_passed": all(row["passed"] for row in checks),
              "checks_count": len(checks), "checks": checks,
              "driver_sha256": sha(Path(__file__)),
              "guide_sha256": {str(path.relative_to(ROOT)): sha(path) for path in guides},
              "new_scientific_calls": 0,
              "documentation_impact": {
                  "updated": [str(path.relative_to(ROOT)) for path in guides],
                  "retained": "All frozen scientific results, failures, source epochs and prior cards",
                  "proposal": "Mixed-CI source edit authorized only; no finite numerical pass or optimizer adoption",
                  "distillation": "Radial-quality inference remains topic-scoped; no new cross-task rule"}}
    with (HERE / "verification.json").open("x") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ["status", "all_passed", "checks_count", "new_scientific_calls"]}))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
