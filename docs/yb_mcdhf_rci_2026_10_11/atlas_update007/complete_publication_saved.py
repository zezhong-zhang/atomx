"""Once saved-metadata closure of root's v8 publication; no API or browser."""

import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    contract = json.loads((HERE / "publication_completion_contract.json").read_text())
    assert contract["script_sha256"] == sha(Path(__file__))
    for name, expected in contract["inputs_sha256"].items():
        assert sha(HERE / name) == expected, name
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        assert passed, name

    def load(name):
        return json.loads((HERE / name).read_text())
    data = load("data_verification.json")
    browser = load("browser_result.json")
    receipt = load("browser_execution.json")
    preflight = load("root_publication_preflight.json")
    commit = load("root_source_commit.json")
    push = load("root_push_and_archive.json")
    version = load("saved_site_version.json")
    deployment = load("deployment_result.json")
    audience = load("root_owner_audience.json")
    check("local_data_preservation", data["all_saved_data_and_preservation_checks_passed"] is True)
    check("desktop_mobile57_saved_gates", browser["all_passed"] is True and browser["checks_count"] == 57
          and len(browser["checks"]) == 57 and all(row["passed"] for row in browser["checks"]))
    check("browser_once_closed_and_owned_cleanup", receipt["status"] == "foreground_closed"
          and receipt["exit_code"] == receipt["browser_exit_code"] == 0
          and receipt["browser_attempts"] == receipt["execution_attempts"] == 1
          and receipt["cleanup"]["all_owned_descendants_stopped"] is True
          and receipt["cleanup"]["remaining_pids"] == [])
    check("actual_observed_browser_resources", receipt["actual_affinity"] == [30]
          and receipt["elapsed_uncontrolled_seconds"] < 90
          and receipt["peak_observed_aggregate_rss_bytes"] < 2147483648
          and receipt["maximum_observed_sample_gap_seconds"] <= 0.25
          and receipt["memory_kernel_hard_cap"] is False)
    check("saved_browser_result_source_identity", receipt["browser_result_sha256"] == sha(HERE / "browser_result.json")
          == preflight["browser_result_sha256"]
          and preflight["browser_execution_sha256"] == sha(HERE / "browser_execution.json"))
    check("counts_and_preserved_fork_identity", browser["models"] == 21 and browser["rows"] == 84
          and browser["trajectory_rows"] == 277 and browser["distinct_saved_point_identities"] == 262
          and browser["fork_shared_local_indices"] == list(range(12)))
    check("website_only17_source_files", commit["reviewed_files"] == len(commit["staged_files"]) == 17
          and set(commit["staged_files"]) == set(preflight["site_changed_and_added_files_sha256"])
          and commit["Atomx_source_committed"] is False and push["Atomx_source_committed"] is False)
    check("pushed_source_to_saved_version", push["pushed"] is True and push["remote_HEAD_verified"] is True
          and push["commit_sha"] == commit["website_only_commit"] == version["source"]["commit_sha"]
          == "381643cf53c0cf258b60a419b41896456b5bec3f")
    check("local_tar_size_saved_version_size", (HERE / "site.tar").stat().st_size
          == push["archive_bytes"] == version["archive_storage"]["size_bytes"] == 71608320
          and push["archive_built_from_pushed_commit"] is True)
    check("deployment_success_exact_version", version["version_number"] == 8
          and deployment["status"] == "succeeded" and deployment["failure_message"] is None
          and deployment["version_id"] == version["id"]
          and deployment["project_id"] == version["project_id"])
    policy = audience["site"]["access_policy"]
    check("root_owner_private_audience_evidence", audience["site"]["id"] == deployment["project_id"]
          and audience["site"]["current_user_role"] == "owner"
          and policy == {"access_mode": "custom", "revision": 1, "users": 1, "editors": 0,
                         "groups": 0, "external_visitors": 0}
          and audience["private_deployment_enforced_owner_only"] is True
          and audience["deployment"] == deployment)
    check("zero_new_science_api_browser_calls", True)

    archive = HERE / "prepublication_docs"
    archive.mkdir(exist_ok=False)
    for name in ("README.md", "docimpact.json"):
        (archive / name).write_bytes((HERE / name).read_bytes())
    manifest = {
        "status": "root_private_v8_deployment_closed_saved_metadata_verified",
        "project_id": deployment["project_id"], "version_id": version["id"], "version_number": 8,
        "deployment": deployment, "source_commit": commit["website_only_commit"],
        "source_parent_commit": commit["parent_commit"], "website_only_files": 17,
        "changed_website_files_sha256": preflight["site_changed_and_added_files_sha256"],
        "archive_identity": {
            "local_tar_sha256": push["archive_sha256"], "local_tar_hash_basis": "raw site.tar byte stream, root archived receipt",
            "local_tar_bytes": push["archive_bytes"], "Sites_archive_storage": version["archive_storage"],
            "Sites_content_hash_basis": "Sites-reported normalized archive/file-tree identity; distinct from raw local tar SHA",
            "local_tar_rehashed_by_this_closure": False,
        },
        "audience": {"current_user_role": "owner", "access_policy": policy,
                     "safe_get_site_receipt_latest_version": audience["site"]["latest_version_number"],
                     "query_phase": "root read before v8 deployment, latest_version7",
                     "private_v8_deployment_owner_only_enforced": True, "child_API_calls": 0},
        "browser": {"checks": 57, "result_sha256": sha(HERE / "browser_result.json"),
                    "execution_sha256": sha(HERE / "browser_execution.json"), "actual_CPU": 30,
                    "uncontrolled_seconds": receipt["elapsed_uncontrolled_seconds"],
                    "peak_observed_aggregate_RSS_bytes": receipt["peak_observed_aggregate_rss_bytes"],
                    "memory_hard_kernel_cap": False, "owned_descendants_stopped": True},
        "data": {"models": 21, "rows": 84, "old_models_preserved": 19, "old_rows_preserved": 76,
                 "NIST_gzip_payloads_preserved": 6019, "run_objects": 5, "trajectory_records": 277,
                 "five_state_records": 1385, "common_lineage_point_identities": 209,
                 "new_fork_records": 66, "distinct_saved_point_identities": 262,
                 "shared_fork_local_indices": list(range(12)), "global240_is_branch_specific": True},
        "scientific_scope": "Display saved f043 native measurements; original stationarity and experimental FAIL remain. No numerical recalculation or new physics claim.",
        "inputs_sha256": contract["inputs_sha256"], "closure_script_sha256": sha(Path(__file__)),
        "child_science_calls": 0, "child_array_calls": 0, "child_browser_calls": 0,
        "child_API_calls": 0, "child_commits": 0, "main_campaign_README_changed": False,
    }
    write(HERE / "publication_manifest.json", manifest)
    (HERE / "RESULTS.md").write_text(
        "Root deployed atlas v8 privately at 2026-10-11 17:21:19.628589 UTC. The website-only commit is "
        "`381643cf53c0cf258b60a419b41896456b5bec3f`; all 17 reviewed website files were pushed. "
        "[The sanitized successful deployment](deployment_result.json) binds the saved v8 version.\n\n"
        "The atlas shows 21 models / 84 lines. The previous 19 models/76 lines, 6019 compressed NIST payloads, "
        "old trajectories, renderer and hosting bytes remain preserved. Two literal 208 forks contribute "
        "33 records each; they share identities at local0–11. Five run objects contain 277 records/1385 state "
        "records and 262 distinct saved point identities. Their global240 terminals remain separate, "
        "with original stationarity and experiment FAIL.\n\n"
        "[Desktop/mobile browser verification](browser_result.json) passed all 57 checks. "
        "[Its sole foreground receipt](browser_execution.json) records 2.630101675 seconds, CPU30, "
        "peak observed aggregate RSS 1,455,394,816bytes and complete owned-process cleanup. "
        "The 2 GiB RSS rule was sampled stopping, not a kernel hard memory cap; timing is uncontrolled.\n\n"
        "The root's raw local tar SHA is `c776d43f51699a359f7dbb4ebfc7fea1e31248aef02bb3959207839e7af9b01a` "
        "(71,608,320bytes). Sites reports normalized content hash "
        "`6abd2be089f6e93bc29a5dcda4bbc2549a084ca79613a94518d6b11cd0a2ff22` and 6114 files. "
        "These identify different representations; equality is not claimed.\n\n"
        "[Owner-audience evidence](root_owner_audience.json) records the deployment-before query at v7 "
        "with custom revision 1, one user, no editors/groups/external visitors, followed by owner-only "
        "private v8 deployment. [The publication manifest](publication_manifest.json) binds the "
        "source, version, audience and receipt hashes. This closure reads saved JSON/stat metadata only: "
        "no new science, arrays, browser, API, publication or commit. Historical failed and preparation "
        "artifacts remain unchanged.\n"
    )
    (HERE / "README.md").write_text(
        "Atlas v8 is privately deployed: 21 models / 84 rows, including the separate history10/history40 "
        "fork terminals at 32 local updates/global240 from literal 208. Both original stationarity gates "
        "and experimental gates remain FAIL. The old 19 models/76 rows, three old trajectory objects, "
        "20 old evidence assets,6019 NIST gzip payloads, renderer and hosting bytes are preserved.\n\n"
        "The common lineage has 209 saved point identities. The forks add 66 records, sharing local0–11; "
        "the five run objects contain 277 records and 262 distinct identities. Global240 is branch-specific, "
        "and the fork measurements retain their f043 source epoch.\n\n"
        "[The saved-data proof](data_verification.json) passed. [The desktop/mobile browser proof](browser_result.json) "
        "passed 57 checks; its [single execution receipt](browser_execution.json) records owned server/browser "
        "cleanup. Root pushed the 17-file website-only commit and [successfully deployed v8](deployment_result.json) "
        "to the existing owner-private site. The [publication manifest](publication_manifest.json) separates "
        "raw local tar SHA from Sites normalized content identity and binds the exact audience receipt.\n\n"
        "[RESULTS](RESULTS.md) gives the closed evidence and limits. The old pending-browser/publication "
        "README and docimpact are preserved in[prepublication_docs](prepublication_docs/README.md); "
        "original prospective contracts/handoffs and all failures remain historical phase evidence. "
        "This final metadata closure performs no new scientific/array/browser/API calls or commits, "
        "and does not edit the main campaign README or registry.\n"
    )
    impact = {
        "scope": "Saved-metadata closure after root browser and private v8 deployment",
        "dispositions": {"README.md": "Current closed publication status; exact prepublication copy archived",
                         "RESULTS.md": "New concise result/limits/archive identity domains",
                         "publication_manifest.json": "New receipt/source/audience/phase hash binding",
                         "prepublication_docs/": "Retain exact old README/docimpact, no failure rewriting",
                         "prospective contracts and handoffs": "Retain historical immutable preparation phase",
                         "BROWSER_LAUNCHER.md": "Retain source-preparation narrative; actual execution in current README/receipt",
                         "old website/assets/NIST and native proofs": "Retain; no edits by this closure",
                         "main campaign README/registry/source": "Owned by parent/physics; untouched"},
        "validation": {"saved_metadata_checks": len(checks), "browser_actual": "root once57PASS",
                       "deployment_actual": "root once private v8 succeeded", "closure_new_science_arrays_API_browser_calls": 0},
        "lasting_insight": "Fork record counts differ from distinct saved identities; normalized Sites content hashes differ in domain from raw local tar SHA.",
        "inputs_sha256": contract["inputs_sha256"],
    }
    write(HERE / "docimpact.json", impact)
    with (HERE / "trace.md").open("a") as trace:
        trace.write("\nPublication closure EARS: preserve append-only saved native evidence and owner-private audience; "
                    "root once browser 57 PASS and website-only v8 private deployment succeeded. "
                    "Child saved-metadata closure checked receipts/source/version/audience, archived old pending docs, "
                    "and wrote current RESULTS/README/manifest. 21 models / 84 rows / 277 records / 262 identities; "
                    "stationarity/experiment FAIL remains. No new science, arrays, browser, API or commit. "
                    "Raw tar and Sites normalized content identity remain distinct domains.\n")
    outputs = ("publication_manifest.json", "RESULTS.md", "README.md", "docimpact.json")
    completion = {"status": "saved_metadata_publication_closed", "all_passed": True,
                  "checks_count": len(checks), "checks": checks,
                  "contract_sha256": sha(HERE / "publication_completion_contract.json"),
                  "script_sha256": sha(Path(__file__)), "inputs_sha256": contract["inputs_sha256"],
                  "outputs_sha256": {name: sha(HERE / name) for name in outputs},
                  "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "new_science_arrays_API_browser_calls": 0, "commits": 0}
    write(HERE / "publication_completion.json", completion)
    write(HERE / "publication_handoff.json", {
        "status": "root_private_v8_deployed_metadata_closed", "project_id": deployment["project_id"],
        "version_id": version["id"], "deployment_id": deployment["id"], "url": deployment["url"],
        "source_commit": commit["website_only_commit"], "models": 21, "rows": 84,
        "browser_checks": 57, "saved_closure_checks": len(checks),
        "completion_sha256": sha(HERE / "publication_completion.json"),
        "outputs_sha256": completion["outputs_sha256"], "child_new_calls": 0, "child_commits": 0,
    })
    print(json.dumps({"all_passed": True, "checks": len(checks),
                      "completion_sha256": sha(HERE / "publication_completion.json"),
                      "handoff_sha256": sha(HERE / "publication_handoff.json")}))


if __name__ == "__main__":
    main()
