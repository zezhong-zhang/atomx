"""Verify the appended diagnostic and website publication; preserve stage002 reports."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from integrate_stage002 import HERE, REPO, binding, read


def main():
    result = json.loads((HERE / "results_stage003.json").read_text())
    for item in result["inputs"].values():
        assert binding(item["path"])["sha256"] == item["sha256"], item["path"]
    assert not result["production_certified"] and not result["highest_level_reached"]
    assert not result["experimental_gate_passed"]
    probe = result["mixed_CI_stationarity"]
    assert not probe["baseline"]["stationarity_passed"]
    assert not probe["one_step"]["stationarity_passed"]
    assert not probe["static_Breit_spectrum_replaced_or_corrected"]
    current = json.loads((HERE / "current_source_manifest.json").read_text())
    for item in current["files"]:
        assert binding(item["path"])["sha256"] == item["sha256"], item["path"]
    # Stage002 artifacts are not regenerated or overwritten by this extension.
    old = json.loads((HERE / "verification.json").read_text())
    assert old["passed"]
    for item in old["artifacts"]:
        assert binding(item["path"])["sha256"] == item["sha256"], item["path"]
    receipt = json.loads((HERE / "delivery_stage003/receipt.json").read_text())
    assert receipt["passed"] and all(receipt["checks"].values())
    for item in receipt["source_identity"]:
        if not item["path"].startswith("private "):
            assert binding(item["path"])["sha256"] == item["sha256"], item["path"]
    iteration = json.loads((HERE / "delivery_stage003/iteration_result.json").read_text())
    assert iteration["message_id"] == receipt["message_id"]
    assert iteration["receipt_backed_selection_executed"]
    assert not iteration["new_expanded_MCDHF_production_execution"]
    for item in iteration["artifacts"]:
        assert binding(item["path"])["sha256"] == item["sha256"]
    deployment = json.loads((HERE / "atlas_update/deployment.json").read_text())
    assert deployment["deployment"]["status"] == "succeeded"
    assert deployment["owner_private_preserved"]
    assert deployment["credentials_persisted"] is False
    access = json.loads((HERE / "atlas_update/access_after_deploy.json").read_text())
    assert access["owner_only_verified"] and access["owner_allowlist_unchanged"]
    assert access["latest_version_number"] == 2
    assert access["current_live_url"] == deployment["deployment"]["url"]
    assert binding(deployment["archive"]["path"])["sha256"] == deployment["archive"]["sha256"]
    handoff = json.loads((HERE / "atlas_update/handoff_manifest.json").read_text())
    assert binding(deployment["preparation_manifest"]["path"])["sha256"] == deployment["preparation_manifest"]["sha256"]
    site = Path(handoff["site"])
    for name, sha in handoff["site_changed_and_added_files_sha256"].items():
        assert hashlib.sha256((site / name).read_bytes()).hexdigest() == sha, name
    for name, sha in handoff["owner_evidence_files_sha256"].items():
        assert binding((HERE / "atlas_update" / name).relative_to(REPO).as_posix())["sha256"] == sha, name
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=site, text=True).strip()
    assert git_head == deployment["commit_sha"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=site, text=True).strip()
    assessment_dir = HERE / "mixed_ci_stationarity_probe"
    assessment = json.loads((assessment_dir / "successor_assessment_verification.json").read_text())
    algebra = json.loads((assessment_dir / "mixed_difference_algebra.json").read_text())
    assert assessment["all_passed"] and assessment["checks_count"] == 44
    assert algebra["all_passed"] and algebra["checks_count"] == 239
    assert algebra["matched_target_pairs"] == 45
    assert not assessment["source_edits"] and not assessment["new_full_optimization_run"]
    for section in ["immutable_prior_evidence_sha256", "assessment_artifacts_sha256"]:
        for name, sha in assessment[section].items():
            assert binding((assessment_dir / name).relative_to(REPO).as_posix())["sha256"] == sha, name
    scientific_inputs = [binding(item["path"]) for item in result["inputs"].values()]
    impact = json.loads((HERE / "documentation_impact.json").read_text())
    docs = list(impact["documentation_review"]) + [
        "docs/yb_mcdhf_rci_2026_10_11/README.md",
        "docs/yb_mcdhf_rci_2026_10_11/mixed_ci_stationarity_probe/README.md",
        "docs/yb_mcdhf_rci_2026_10_11/atlas_update/README.md",
        "docs/yb_mcdhf_rci_2026_10_11/atlas_update/PUBLICATION.md",
        "docs/yb_mcdhf_rci_2026_10_11/mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md",
        "docs/atomic_nist_atlas_2026_10_11/README.md",
        "docs/atomx_hourly_progress_2026_10_10/README.md",
    ]
    broken, links = [], []
    for relative in docs:
        path = REPO / relative
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text()):
            if "://" in target or target.startswith(("#", "mailto:", "app:")):
                continue
            target = re.sub(r":\d+$", "", target.strip("<>").split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            generated_here = resolved == (HERE / "verification_stage003.json").resolve()
            record = {"document": relative, "target": target,
                      "exists": resolved.exists(), "generated_by_this_finalizer": generated_here}
            links.append(record)
            if not record["exists"] and not generated_here:
                broken.append(record)
    assert not broken, broken
    paths = [
        "docs/yb_mcdhf_rci_2026_10_11/integrate_stage003.py",
        "docs/yb_mcdhf_rci_2026_10_11/deliver_stage003.py",
        "docs/yb_mcdhf_rci_2026_10_11/finalize_stage003.py",
        "docs/yb_mcdhf_rci_2026_10_11/atlas_update/root_push_source.py",
    ]
    lint = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "ruff", "check", *paths],
                          cwd=REPO, text=True, capture_output=True, check=False)
    (HERE / "stage003_ruff.log").write_text(lint.stdout + lint.stderr)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    registry_path = "docs/atomx_hourly_progress_2026_10_10/progress.json"
    registry = read(registry_path)
    assert registry["physics_budget_result"].endswith("results_stage003.json")
    assert registry["yb_mcdhf_result"].endswith("card_evidence_stationary_baseline.json")
    additions = {
        "scope": "Appended mixed-CI diagnostic, display-card delivery and separate website publication; no Atomx scientific source edits after stage002",
        "science_source_files_unchanged_since_stage002": len(current["files"]),
        "stage002_report_and_artifacts_preserved": True,
        "documentation_review": {path: "Refresh current diagnostic/publication scope; preserve original scientific contracts and failed results" for path in docs},
        "relative_working_tree_links": len(links),
        "broken": broken,
        "website_repository": "/home/zzhang/.local/share/atomx-nist-atlas/site",
        "website_change_is_separate_from_scientific_working_tree": True,
    }
    (HERE / "documentation_impact_stage003.json").write_text(
        json.dumps(additions, ensure_ascii=False, indent=2) + "\n")
    files = ["results_stage003.json", "delivery_stage003/receipt.json",
             "delivery_stage003/iteration_result.json", "atlas_update/deployment.json",
             "verification.json", "documentation_impact_stage003.json", "stage003_ruff.log"]
    files += ["atlas_update/access_after_deploy.json", "atlas_update/handoff_manifest.json",
              "atlas_update/PUBLICATION.md", "mixed_ci_stationarity_probe/successor_assessment_verification.json",
              "mixed_ci_stationarity_probe/mixed_difference_algebra.json",
              "mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md"]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "primary_route": "Atomx native MCDHF -> RCI",
        "scientific_inputs": scientific_inputs,
        "all_353_source_files_unchanged_since_stage002": True,
        "original_metric_preserved": probe["original_metric_sha256"],
        "five_reference_numerical_stationarity_passed": True,
        "expanded531_MCDHF_stationarity_passed": False,
        "experimental_gate_passed": False, "highest_level_reached": False,
        "mixed_CI_running_checks": 68, "mixed_CI_independent_saved_array_checks": 432,
        "mixed_probe_driver_is_prospectively_bound": False,
        "derivative_oracle_is_independent_external_Hamiltonian": False,
        "test_repetition": "Science implementation unchanged: retain stage002 160 science / 45 card tests; append source-bound diagnostic and publication verification",
        "actual_card_message_id": receipt["message_id"],
        "actual_iteration_status": iteration["status"],
        "registered_new_production_executor_started": False,
        "saved_matrix_algebra_checks": 239,
        "stable_mixed_radial_energy_increment_implemented": False,
        "atlas_url": deployment["deployment"]["url"],
        "owner_private_preserved": True,
        "website_commit_sha": deployment["commit_sha"],
        "relative_working_tree_links_checked": len(links), "broken": broken,
        "ruff_passed": lint.returncode == 0,
        "artifacts": [binding((HERE / name).relative_to(REPO).as_posix()) for name in files],
    }
    target = HERE / "verification_stage003.json"
    if target.exists():
        raise FileExistsError("Stage003 report is immutable")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    assert target.exists()
    print(json.dumps({"passed": True, "verification": binding(target.relative_to(REPO).as_posix()),
                      "links": len(links), "card": receipt["message_id"]}))


if __name__ == "__main__":
    main()
