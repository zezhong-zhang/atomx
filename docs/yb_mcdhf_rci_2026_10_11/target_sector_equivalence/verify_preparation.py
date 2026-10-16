"""JSON/hash/AST controls only; never imports NumPy or an Atomx consumer."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Preparation check failed: " + name)

    contract = load(HERE / "contract.json")
    proposal = load(HERE / "input_binding_proposal.json")
    audit = load(HERE / "mathematical_source_audit.json")
    check("original_pending_contract_preserved",
          sha(HERE / "contract_planning001.json") == contract["planning001_contract_sha256"])
    check("original_pending_audit_preserved",
          sha(HERE / "mathematical_source_audit_planning001.json") == audit["planning001_audit_sha256"])
    check("proposal_does_not_authorize_execution", proposal["execution_authorized_by_root"] is False)
    check("proposal_has_no_root_tracker_attestation",
          proposal["tracker_inputs_attested_original_objective_by_root"] is False)
    check("proposal_exact_contract", proposal["preparation_contract_sha256"] == sha(HERE / "contract.json"))
    check("proposal_exact_driver", proposal["driver_sha256"] == sha(HERE / "run.py"))
    for name, expected in contract["inputs_sha256"].items():
        check("immutable_input:" + name, sha(REPO / name) == expected)
    files = {role: REPO / entry["path"] for role, entry in proposal["files"].items()}
    for role, entry in proposal["files"].items():
        check("proposal_input:" + role, sha(files[role]) == entry["sha256"])
    checkpoint = load(files["checkpoint_manifest"])
    proof = load(files["checkpoint_verification"])
    source = load(files["source_identity"])
    objective = load(files["objective_identity"])
    check("checkpoint_accepted_complete", checkpoint["accepted_complete_point"] is True)
    check("endpoint_independent_identity_pass_988",
          proof["all_identity_replay_checks_passed"] is True and proof["checks_count"] == 988)
    check("failed_stationarity_not_relabelled",
          checkpoint["stationarity_passed"] is False and proof["original_stationarity_gates_passed"] is False
          and checkpoint["diagnostics"]["orbital_residual"] > 1e-7)
    basis = load(files["basis_identity"])
    sectors = {(row["twoJ"], row["parity"]) for row in contract["fixed_conditions"]["sector_selection"]["sector_counts"]}
    kept = [i for i, row in enumerate(basis["CSFs"]) if (row["twoJ"], row["parity"]) in sectors]
    check("independent_complete_selected_sector_indices",
          kept == contract["fixed_conditions"]["sector_selection"]["indices_full531"] and len(kept) == 261)
    for row in contract["fixed_conditions"]["sector_selection"]["sector_counts"]:
        check("independent_sector_count:" + str((row["twoJ"], row["parity"])),
              sum((item["twoJ"], item["parity"]) == (row["twoJ"], row["parity"])
                  for item in basis["CSFs"]) == row["CSFs"])
    full_keys = sorted({tuple(sh[:2]) for row in basis["CSFs"] for sh in row["subshells"]})
    sector_keys = sorted({tuple(sh[:2]) for i in kept for sh in basis["CSFs"][i]["subshells"]})
    check("all24_radial_keys_survive_complete_sector_selection",
          full_keys == sector_keys == [tuple(key) for key in contract["fixed_conditions"]["radial_keys"]])
    check("complete_objective_CSF_rows", objective["new"]["CSFs"] == basis["CSFs"])
    source_root = REPO / proposal["frozen_source_root"]
    for name, expected in source["all_atomx_python_sha256"].items():
        check("frozen_source:" + name, sha(source_root / Path(name).relative_to("src")) == expected)
    for name, expected in contract["audited_critical_source_sha256"].items():
        check("audited_consumer:" + name, source["all_atomx_python_sha256"][name] == expected)

    tree = ast.parse((HERE / "run.py").read_text())
    compile(tree, str(HERE / "run.py"), "exec")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    check("no_optimizer_callsite", not any(isinstance(node.func, ast.Name)
          and node.func.id == "optimize_orbitals" for node in calls))
    check("one_gradient_evaluation_callsite_in_exact_two_model_loop",
          sum(isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate"
              for node in calls) == 1)
    check("one_explicit_H_callsite_in_exact_two_model_loop",
          sum(isinstance(node.func, ast.Name) and node.func.id == "build_ci_matrix"
              for node in calls) == 1)
    check("prospective_two_gradient_and_two_CI_budget",
          contract["resources"]["gradient_budget"] == contract["resources"]["CI_solves_budget"] == 2)
    check("prospective_single_CPU_4GiB_120s",
          contract["resources"]["CPU_threads"] == 1
          and contract["resources"]["address_space_cap_GiB"] == 4
          and contract["resources"]["wall_cap_s"] == 120)
    check("declared_affinity_minus2_no_silent_fallback",
          contract["resources"]["minimum_available_affinity_CPUs"] == 3
          and "affinity[-2]" in (HERE / "run.py").read_text())

    # Test only the metadata authorization function with an in-memory binding.
    # No persistent root-authorized contract is produced. All scientific imports
    # are blocked even if an accidental future edit tries to enter a consumer.
    original_import = builtins.__import__

    def no_science_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "atomx"}:
            raise RuntimeError("Scientific imports forbidden during preparation validation")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = no_science_import
    try:
        spec = importlib.util.spec_from_file_location("prepared_sector_driver", HERE / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check("default_preparation_has_zero_science_calls",
              module.prepare()["gradient_calls"] == module.prepare()["CI_calls"] == 0)
        try:
            module.authorize(HERE / "input_binding_proposal.json")
        except ValueError as error:
            check("actual_pending_proposal_rejected_before_science", "Pending" in str(error))
        else:
            check("actual_pending_proposal_rejected_before_science", False)
        binding = copy.deepcopy(proposal)
        binding["execution_authorized_by_root"] = True
        binding["tracker_inputs_attested_original_objective_by_root"] = True
        sentinel = HERE / "IN_MEMORY_METADATA_CONTROL_ONLY"
        original_load = module.load
        module.load = lambda path: binding if Path(path) == sentinel else original_load(path)
        answer = module.authorize(sentinel)
        check("positive_metadata_identity_control_only_no_execution",
              answer[3]["iteration"] == 160 and answer[4]["all_identity_replay_checks_passed"] is True)
        mutations = [
            ("root_absent", lambda x: x.update(execution_authorized_by_root=False)),
            ("pending_checkpoint", lambda x: x.update(input_status="pending")),
            ("wrong_preparation_SHA", lambda x: x.update(preparation_contract_sha256="0" * 64)),
            ("wrong_driver_SHA", lambda x: x.update(driver_sha256="0" * 64)),
            ("missing_tracker_attestation", lambda x: x.update(tracker_inputs_attested_original_objective_by_root=False)),
            ("wrong_objective_SHA", lambda x: x.update(expected_checkpoint_objective_sha256="0" * 64)),
            ("wrong_source_map_SHA", lambda x: x.update(expected_source_map_sha256="0" * 64)),
            ("wrong_checkpoint_arrays_SHA", lambda x: x["files"]["checkpoint_arrays"].update(sha256="0" * 64)),
            ("incomplete_scientific_roles", lambda x: x["files"].pop("original_geometry_arrays")),
        ]
        for name, mutate in mutations:
            binding = copy.deepcopy(proposal)
            binding["execution_authorized_by_root"] = True
            binding["tracker_inputs_attested_original_objective_by_root"] = True
            mutate(binding)
            try:
                module.authorize(sentinel)
            except ValueError:
                check("metadata_rejection:" + name, True)
            else:
                check("metadata_rejection:" + name, False)
    finally:
        builtins.__import__ = original_import
    check("no_scientific_module_imported", not any(name == "numpy" or name.startswith("numpy.")
          or name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    check("no_scientific_execution_record_created",
          not any((HERE / name).exists() for name in ("result.json", "failure.json", "progress.json",
                                                       "resource_admission.json", "equivalence_arrays.npz")))
    record = {"all_preparation_checks_passed": all(row["passed"] for row in checks),
              "checks_count": len(checks), "checks": checks,
              "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(HERE / "run.py"),
              "audit_sha256": sha(HERE / "mathematical_source_audit.json"),
              "input_binding_proposal_sha256": sha(HERE / "input_binding_proposal.json"),
              "verifier_sha256": sha(__file__), "source_map_sha256": source["source_map_sha256"],
              "known_checkpoint_sha256": sha(files["checkpoint_manifest"]),
              "known_endpoint_independent_verification_sha256": sha(files["checkpoint_verification"]),
              "physics_imports": 0, "H_builds": 0, "CI_solves": 0, "gradient_calls": 0,
              "optimizer_calls": 0, "execution_authorized": False, "equivalence_passed": False,
              "scope": "Prospective preparation/source/identity/guard verification only. The in-memory positive control cannot authorize execution. No numerical equivalence or speedup claim."}
    (HERE / "preparation_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("all_preparation_checks_passed", "checks_count",
          "H_builds", "CI_solves", "gradient_calls", "execution_authorized", "equivalence_passed")}))


if __name__ == "__main__":
    main()
