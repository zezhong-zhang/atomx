"""Static metadata/refusal preparation proof; do not import Atomx or evaluate."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    specification = importlib.util.spec_from_file_location("qualification_preparation_driver", HERE / "run.py")
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    contract = driver.load(HERE / "contract.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    metadata = {
        "execution_authorized_by_root": True, "qualification_authorized": True,
        "driver_sha256": driver.sha(HERE / "run.py"),
        "preparation_contract_sha256": driver.sha(HERE / "contract.json"),
        "candidate_source_identity_sha256": driver.sha(driver.ROOT / contract["candidate_source_identity_path"]),
        "candidate_source_map_sha256": contract["candidate_source_map_required_sha256"],
        "point_arrays_sha256": contract["fixed_conditions"]["point_arrays_sha256"],
        "max_full_gradients": 2, "max_CI_solves": 2, "max_public_H_builds": 3, "max_optimizer_calls": 0,
    }
    # This only calls the metadata predicate. It writes no fake root authority
    # and never enters qualify(), imports the physical package or allocates H.
    driver.validate_execution(metadata, contract)
    check("synthetic_exact_metadata_predicate_only", True)
    for name in metadata:
        invalid = dict(metadata)
        invalid[name] = None
        try:
            driver.validate_execution(invalid, contract)
        except ValueError:
            check("reject_missing_or_wrong:" + name, True)
        else:
            check("reject_missing_or_wrong:" + name, False)
    try:
        driver.validate_execution({}, contract)
    except ValueError:
        check("reject_empty_authority", True)
    else:
        check("reject_empty_authority", False)
    tree = ast.parse((HERE / "run.py").read_text())
    qualify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    calls = [node for node in ast.walk(qualify) if isinstance(node, ast.Call)]
    check("authorization_precedes_physical_imports",
          isinstance(qualify.body[2], ast.Expr) and isinstance(qualify.body[2].value, ast.Call)
          and isinstance(qualify.body[2].value.func, ast.Name)
          and qualify.body[2].value.func.id == "validate_execution")
    check("no_optimizer_call_in_qualification_consumer",
          not any(isinstance(call.func, ast.Name) and call.func.id == "optimize_orbitals" for call in calls))
    check("declared_forward_counted_original_functions",
          all(name in (HERE / "run.py").read_text() for name in (
              'counted("H_build", raw_build)', 'counted("CI_solve", raw_ci)',
              'counted("gradient", raw_gradient)')))
    check("full_g_pg_original_gates_kept", contract["acceptance"]["complete_projected_coordinate_g_replay_max"] == 1e-9
          and contract["acceptance"]["complete_preconditioned_pg_replay_max"] == 1e-10
          and contract["acceptance"]["original_orbital_tolerance"] == 1e-7
          and contract["acceptance"]["original_energy_tolerance_Ha"] == 1e-8)
    check("full_point_stays531_no_model_shrink", contract["fixed_conditions"]["CSFs"] == 531
          and contract["fixed_conditions"]["radial_orbitals"] == 24
          and contract["fixed_conditions"]["electrons"] == 69)
    check("preflight_current_driver_contract_bound",
          driver.load(HERE / "preflight.json")["all_passed"]
          and driver.load(HERE / "preflight.json")["driver_sha256"] == driver.sha(HERE / "run.py")
          and driver.load(HERE / "preflight.json")["contract_sha256"] == driver.sha(HERE / "contract.json"))
    check("no_Atomx_import_in_metadata_proof", not any(name == "atomx" or name.startswith("atomx.")
          for name in sys.modules))
    import numpy as np

    payload = {"negative_terms": np.int64(17), "terms": [{"key": (np.int64(2), np.int32(-3)),
               "value": np.float64(-.125)}], "passed": np.bool_(True), "signed": np.float32(-.5)}
    encoded = json.dumps(payload, allow_nan=False, default=driver.json_numeric_scalar)
    check("nested_NumPy_numeric_metadata_serializes_exactly", json.loads(encoded)
          == {"negative_terms": 17, "terms": [{"key": [2, -3], "value": -.125}],
              "passed": True, "signed": -.5})
    for name, unsupported in (("array", np.array([1., 2.])), ("nonfinite", np.float64(np.nan)),
                              ("arbitrary_object", object())):
        try:
            json.dumps(unsupported, allow_nan=False, default=driver.json_numeric_scalar)
        except (TypeError, ValueError):
            check("serialization_rejects:" + name, True)
        else:
            check("serialization_rejects:" + name, False)
    parent = HERE.parent / "hamiltonian_snapshot_qualification"
    old_tree = ast.parse((parent / "run.py").read_text())
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("prepare_point", "source_and_input_checks", "validate_execution", "host_resources"):
        check("unchanged_context_and_science_boundary:" + name,
              ast.dump(old_functions[name], include_attributes=False)
              == ast.dump(new_functions[name], include_attributes=False))
    old_contract = driver.load(parent / "contract.json")
    for field in ("acceptance", "fixed_conditions", "resources", "candidate_source_map_required_sha256"):
        check("unchanged_model_gates_source_caps:" + field, old_contract[field] == contract[field])
    physical_calls = {"evaluate", "build_ci_matrix", "hamiltonian_for", "original_gradients",
                      "ci_level_blocks", "make_nuclear_potential", "CIResult", "CIHamiltonianSnapshot"}
    def calls_in(tree):
        return sorted(ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and (
                      isinstance(node.func, ast.Name) and node.func.id in physical_calls or
                      isinstance(node.func, ast.Attribute) and node.func.attr in physical_calls))
    check("every_physical_call_and_arguments_AST_unchanged", calls_in(old_tree) == calls_in(tree))
    check("persistent_controls_before_after_gates_and_actual_calls",
          all(text in (HERE / "run.py").read_text() for text in (
              'persist_controls(name, "before_gate")', 'persist_controls(name, "after_gate")',
              'persist_controls(kind, "call_started")', 'persist_controls(kind, "call_completed")',
              'save_arrays("complete_gradients_before_remaining_gates")')))
    check("failed_parent_has_arrays_but_no_final_numeric_certificate",
          (parent / "failure.json").exists() and (parent / "arrays.npz").exists()
          and not (parent / "result.json").exists())
    result = {"all_passed": True, "checks_count": len(checks), "checks": checks,
              "preflight_checks_count": driver.load(HERE / "preflight.json")["checks_count"],
              "driver_sha256": driver.sha(HERE / "run.py"), "contract_sha256": driver.sha(HERE / "contract.json"),
              "preflight_sha256": driver.sha(HERE / "preflight.json"),
              "scope": "Metadata predicate, static AST and source/input admission only; not root authority or numerical qualification",
              "new_H_builds": 0, "new_CI_solves": 0, "new_gradients": 0, "optimizer_calls": 0,
              "science_execution_pending_exact_root_contract": True}
    driver.atomic_json(HERE / "preparation_verification.json", result)
    print(json.dumps({"all_passed": True, "metadata_refusal_checks": len(checks),
                      "readonly_preflight_checks": result["preflight_checks_count"]}, indent=2))


if __name__ == "__main__":
    main()
