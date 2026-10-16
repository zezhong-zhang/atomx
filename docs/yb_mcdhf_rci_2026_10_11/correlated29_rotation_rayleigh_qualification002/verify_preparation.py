"""Standard-library source/metadata qualification; zero numeric imports/calls."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    source = (HERE / "run.py").read_text()
    tree = ast.parse(source)
    checks = []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})

    check("source_ast_parses", isinstance(tree, ast.Module))
    predecessor = contract["predecessor_failure"]["inputs"]
    previous_source = (REPO / predecessor["driver"]["path"]).read_text()
    before = 'base[1].hamiltonian_snapshot.identity_payload() != load("saved_point_owner")'
    after = 'plain(base[1].hamiltonian_snapshot.identity_payload()) != load("saved_point_owner")'
    check("exact_one_representation_fix_only", previous_source.count(before) == 1 and
          source == previous_source.replace(before, after))
    archived = json.loads((REPO / predecessor["actual_owner"]["path"]).read_text())
    original = json.loads((REPO / contract["input_roles"]["saved_point_owner"]["path"]).read_text())
    check("predecessor_owner_whole_JSON_equal", archived == original)
    check("predecessor_owner_whole_bytes_equal", sha(REPO / predecessor["actual_owner"]["path"]) ==
          sha(REPO / contract["input_roles"]["saved_point_owner"]["path"]))
    check("canonical_whole_JSON_exact_equality", json.dumps(archived, sort_keys=True, separators=(",", ":")) ==
          json.dumps(original, sort_keys=True, separators=(",", ":")))
    previous_contract = json.loads((REPO / predecessor["contract"]["path"]).read_text())
    check("all_original_science_and_budget_fields_unchanged", all(contract[key] == previous_contract[key]
          for key in ("source", "fixed", "point_policy", "metric", "new_scientific_target", "direction",
                      "finite_step", "gates", "resources", "kinematic_resources", "procedure", "scope")))
    check("predecessor_root_failure_preserved", json.loads((REPO / predecessor["root_execution"]["path"]).read_text())["exit_code"] == 1)
    check("preparer_executes_no_runtime_driver", not (HERE / "actual").exists())
    for name, role in contract["input_roles"].items():
        check("bound_input_" + name, sha(REPO / role["path"]) == role["sha256"])
    for name, role in contract["pending_prerequisites"].items():
        check("bound_prerequisite_" + name, sha(REPO / role["path"]) == role["sha256"])
    identity = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    frozen = REPO / contract["source"]["candidate_archive"]
    source_map = identity["all_atomx_python_sha256"]
    check("full356_source_identity", len(source_map) == 356 and
          identity["source_map_sha256"] == contract["source"]["candidate356_map_sha256"])
    check("all356_frozen_source_bytes", all(sha(frozen / key.removeprefix("src/")) == value
          for key, value in source_map.items()))
    audit = json.loads((REPO / contract["input_roles"]["saved_gradient_audit"]["path"]).read_text())
    check("saved341_audit_pass_scope", audit["all_saved_array_checks_passed"] is True and
          audit["original_probe_qualified"] is False and
          audit["whole_public_samepoint_gradient_replay_executed"] is False)
    check("same_saved3451_metric", audit["new_metric_sha256"] == contract["metric"]["residual_metric_sha256"])
    check("old_failure_still_failure", audit["original_reported_derivative_failed"] is True and
          audit["original_Rayleigh_derivative_gate_executed"] is False)
    check("prior_stationarity_failure_preserved", audit["saved_newmetric_stationarity_gates_passed"] is False)
    names = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    allowed = {"argparse", "hashlib", "json", "os", "resource", "signal", "sys", "time", "pathlib"}
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("no_top_level_numeric_import", all(
          all(alias.name.split(".")[0] in allowed for alias in node.names) if isinstance(node, ast.Import)
          else node.module.split(".")[0] in allowed for node in top_imports))
    check("authorize_metadata_only", "bind_inputs(contract, execution)" in ast.unparse(names["main"]) and
          "return" in ast.unparse(names["main"]))
    check("admission_before_numeric_imports", source.index("root, accounting = admit(contract, execution)") <
          source.index("import numpy as np"))
    check("frozen_source_escape_guard", "is_relative_to(root.resolve())" in source)
    calls = [node for node in ast.walk(names["probe"]) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    check("no_optimizer_invocation", "optimize_orbitals" not in attributes and "minimize" not in attributes)
    check("no_gradient_true_invocation", all(not (key.arg == "gradient" and isinstance(key.value, ast.Constant)
          and key.value.value is True) for node in calls for key in node.keywords))
    check("no_manufactured_retained_owner", source.count("CIResult(") == 1 and
          "CIResult(reference[" in source and "hamiltonian_snapshot=" not in source)
    check("same_saved_x_not_recovered_again", 'coordinates=saved_gradient["x"]' in source and
          'projector.T @ functional.metric' not in source)
    check("same_saved_inverse_and_P", 'inverse_diagonal=saved_gradient["inverse_diagonal"]' in source and
          'saved_gradient[f"inverse_blocks_{i}"]' in source and 'old7_P_bytes_' in source)
    check("baseline_no_polar_seed_repair", source.index('base = complete_point("fresh_recovered_coordinate_point", waves)') <
          source.index('candidate[where] = _polar_retraction'))
    check("fresh_complete_baseline_bitwise_and_real_owner", 'fresh_baseline_bitwise_' in source and
          'fresh_baseline_real_owner_identity' in source)
    check("only_original_rotation_seed", source.count("np.random.PCG64(") == 1 and
          'np.random.PCG64(20261012)' in source and 'omega[j, i] = -omega[i, j]' in source)
    check("no_gradient_dependent_direction", source.index("direction /= norm") < source.index("analytical = float(g @ direction)"))
    check("same_h_and_signs", 'for sign in (1., -1.):' in source and 'candidate = x + sign * 1e-6 * direction' in source)
    check("explicit_P_mapped_increment", 'increment = unpack(candidate - x)' in source and 'increments=increment' in source)
    check("normalized_actual_Rayleigh_norm", 'norm = c @ c' in source and 'correction = c @ residual_vector / norm' in source)
    check("NEW_Rayleigh_acceptance_only", 'check("direction1_NEW_Rayleigh_derivative", abs(rayleigh - analytical)' in source and
          'check("direction1_reported_derivative"' not in source)
    check("original_derivative_thresholds", '"gate": max(1e-8, 1e-5 * abs(analytical))' in source)
    check("finite_response_not_claimed_callback", 'diagnostic["diagnostic_only"] is True' in source and
          'diagnostic["certified_error_bound"] is False' in source and 'diagnostic["optimizer_adapter"] is False' in source)
    check("whole1046_original_tracker_cap", 'selected_complete1046' in source and 'same1046signed_order' in source and
          'max_determinant_pairs=1_000_000' in source and 'require_full=True' in source)
    check("same_whole_original_gates", '1e-8)' in source and '_all_CI_Gram' in source and
          '_P_leakage_' in source and '1e-12)' in source)
    caps = contract["resources"]
    check("threepoint_zero_g_budget", tuple(caps[k] for k in ("max_H", "max_CI", "max_full_gradient", "max_optimizer")) == (3, 3, 0, 0))
    check("raw_and_spectral_caps", (caps["max_total_hS"], caps["max_scipy_eigh"], caps["max_polar_retraction"]) == (42, 36, 14))
    check("CPU31_120s_3point95GiB", (caps["CPU"], caps["wall_seconds"], caps["address_space_GiB"]) == (31, 120, 3.95))
    check("partial_evidence_before_derivative_gate", source.index('write(output / "direction1_central.json"') <
          source.index('check("direction1_NEW_Rayleigh_derivative"'))
    check("failure_keeps_oldfailure", '"original_failure_unchanged": True' in source)
    check("new_objective_not_old_pass", contract["new_scientific_target"]["name"] == "EOL_Rayleigh" and
          contract["new_scientific_target"]["old_reported_root_contract_not_passed"] is True)
    result = {"all_passed": all(row["passed"] for row in checks), "checks_count": len(checks),
              "checks": checks, "runner_sha256": sha(HERE / "run.py"),
              "contract_sha256": sha(HERE / "contract.json"), "verifier_sha256": sha(__file__),
              "source_map_sha256": identity["source_map_sha256"], "source_files": len(source_map),
              "saved_gradient_audit_sha256": contract["input_roles"]["saved_gradient_audit"]["sha256"],
              "source_only": True, "numeric_imports": 0, "physical_calls": 0,
              "derivative_qualification_executed": False, "optimizer_authorized": False}
    (HERE / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
