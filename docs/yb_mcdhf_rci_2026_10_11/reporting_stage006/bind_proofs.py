"""Append direct sector and publication identities to the completed stage006."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
ROOT = TOPIC.parents[1]


def binding(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    original = TOPIC / "results_stage006.json"
    result = json.loads(original.read_text())
    for item in result["inputs"].values():
        assert binding(ROOT / item["path"])["sha256"] == item["sha256"]
    sector = TOPIC / "target_sector_equivalence002"
    proof_path = sector / "saved_array_audit/verification.json"
    assert binding(proof_path)["sha256"] == "56363aee43e70800df3144a3690630eeda40790e56ebe48cfee8d3a7a5999b92"
    proof = json.loads(proof_path.read_text())
    assert all(c["passed"] for c in proof["checks"]) and len(proof["checks"]) == 1795
    result["inputs"]["previous_stage006"] = binding(original)
    for relative in ("result.json", "completion.json", "saved_array_audit/contract.json", "saved_array_audit/verification.json"):
        result["inputs"]["exact_target_sector/" + relative] = binding(sector / relative)
    publication = TOPIC / "atlas_update002"
    deployed = json.loads((publication / "deployment.json").read_text())
    access = json.loads((publication / "access_after_deploy.json").read_text())
    assert deployed["status"] == "succeeded" and access["audience_unchanged"]
    for relative in ("handoff_manifest.json", "root_publication_preflight.json", "root_push_and_archive.json", "deployment.json", "access_after_deploy.json"):
        result["inputs"]["website_version3/" + relative] = binding(publication / relative)
    result["exact_target_sector_equivalence"] = {"full_CSFS": 531, "target_CSFS": 261,
        "target_count": 5, "gradient_radial_orbitals": 24, "independent_checks": 1795,
        "all_spectrum_or_global_ground_certified": False, "speedup_measured": False,
        "stationarity_passed": False}
    result["website_publication"] = {"version": 3, "url": deployed["url"],
        "models": 12, "rows": 48, "current_model_point": 160,
        "current320_static_RCI_claimed": False, "audience_unchanged": True}
    output = TOPIC / "results_stage006b.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(binding(output)))


if __name__ == "__main__":
    main()
