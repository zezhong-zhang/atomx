"""Source/JSON-only admission: never import Atomx/NumPy or open NPZ bytes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError("Source-only preparation failed: " + name)

    contract = json.loads((HERE / "contract.json").read_text())
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    decoded, deferred = {}, []
    for name, expected in contract["inputs_sha256"].items():
        path = ROOT / name
        if path.suffix == ".npz":
            deferred.append(dict(path=name, sha256=expected, verified_at_runtime_only=True))
            continue
        check("input_sha/" + name, sha(path) == expected)
        if path.suffix == ".json":
            decoded[name] = json.loads(path.read_text())
    check("six_unique_frozen_NPZ_inputs_not_opened", len(deferred) == 6)
    check("input_roles_exact_manifest",
          all(contract["inputs_sha256"][row["path"]] == row["sha256"] for row in contract["input_roles"].values()))
    check("models_exact_original_bytes", sha(HERE / "models.json") == contract["models_sha256"])
    models = json.loads((HERE / "models.json").read_text())["models"]
    check("complete_target1046_1595", [models[n]["complete_target_CSFs"] for n in ("baseline", "complete5f")] == [1046, 1595])
    check("complete_parent2163_3789", [models[n]["full_CSFs"] for n in ("baseline", "complete5f")] == [2163, 3789])
    check("NR69_91_families", [len(models[n]["configs"]) for n in ("baseline", "complete5f")] == [69, 91])
    source = json.loads(roles["candidate_source_identity"].read_text())
    source_root = ROOT / contract["source"]["candidate_archive"]
    source_trees = {}
    for name, expected in source["all_atomx_python_sha256"].items():
        path = source_root / Path(name).relative_to("src")
        if sha(path) != expected:
            raise ValueError("Immutable source changed: " + name)
        source_trees[str(Path(name).relative_to("src")).removesuffix(".py").replace("/", ".")] = ast.parse(path.read_text())
    check("whole356_source_SHA_and_AST", len(source_trees) == 356
          and source["source_map_sha256"] == contract["fixed"]["source_map_sha256"])
    trees = {name: ast.parse((HERE / name).read_text()) for name in ("run.py", "signed_scalar_controls.py")}
    for role in ("closed68_adapter", "refinement_helper"):
        trees[role] = ast.parse(roles[role].read_text())
    for name, expected in contract["consumer_helpers_sha256"].items():
        check("owned_helper_SHA/" + name, sha(HERE / name) == expected)
    direct_imports = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("atomx"):
                continue
            module = node.module
            module_tree = source_trees.get(module, source_trees.get(module + ".__init__"))
            if module_tree is None:
                raise ValueError("Declared native import module absent: " + module)
            symbols = set()
            for part in ast.walk(module_tree):
                if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.add(part.name)
                elif isinstance(part, (ast.Import, ast.ImportFrom)):
                    symbols.update(alias.asname or alias.name.split(".")[-1] for alias in part.names)
                elif isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store):
                    symbols.add(part.id)
            for alias in node.names:
                child = module + "." + alias.name
                if alias.name not in symbols and child not in source_trees and child + ".__init__" not in source_trees:
                    raise ValueError("Declared native import symbol absent: " + module + "/" + alias.name)
                direct_imports.append(dict(consumer=name, module=module, symbol=alias.name))
    check("all_direct_native_import_modules_and_symbols_exist", bool(direct_imports))
    for tree in trees.values():
        compile(tree, "source_only_AST", "exec")
    check("all_consumer_helpers_compile_without_execution", True)
    top_imports = [node for node in trees["run.py"].body if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("top_level_imports_stdlib_only", all(not (getattr(n, "module", "") or "").startswith("atomx")
          and all(not a.name.startswith(("atomx", "numpy", "scipy")) for a in n.names) for n in top_imports))
    functions = {node.name: node for node in trees["run.py"].body if isinstance(node, ast.FunctionDef)}
    execute = functions["execute"]
    check("authorization_first_before_numerical_imports", isinstance(execute.body[0], ast.Assign)
          and isinstance(execute.body[0].value, ast.Call) and execute.body[0].value.func.id == "authorize")
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    scalar_line = next(n.lineno for n in calls if isinstance(n.func, ast.Subscript)
                       and isinstance(n.func.value, ast.Name) and n.func.value.id == "scalar_helper")
    spectral_line = next(n.lineno for n in calls if isinstance(n.func, ast.Attribute)
                         and isinstance(n.func.value, ast.Name) and n.func.value.id == "scipy_linalg" and n.func.attr == "eigh")
    CI_line = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "ci_solve")
    check("all32_scalar_controls_before_Fock_spectra_and_CI", scalar_line < spectral_line < CI_line)
    old = ast.parse(roles["nuclear_failed_driver"].read_text())

    def named_call(tree, name):
        return next(ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name)

    check("public_native_CI_arguments_unchanged", named_call(old, "ci_solve") == named_call(trees["run.py"], "ci_solve"))
    check("public_CI_state_average_constructor_unchanged", named_call(old, "CIStateAverage") == named_call(trees["run.py"], "CIStateAverage"))
    fixed, caps = contract["fixed"], contract["resources"]
    core = {tuple(row["key"]): row["occupation"] for row in fixed["closed68_core_rows"]}
    check("exact19_closed68_core", len(core) == 19 and sum(core.values()) == 68 and all(q == 2 * abs(k[1]) for k, q in core.items()))
    check("two_zero_occupation_f_targets", fixed["new_radial_keys"] == [[5, 3], [5, -4]] and not set(map(tuple, fixed["new_radial_keys"])) & set(core))
    triangle_upper = sum(16 * sum(1 + 2 * min(abs(kappa), abs(k[1])) for k in core) for kappa in (3, -4))
    check("integer_scalar_work_upper2752", triangle_upper == 2752 == caps["scalar_triangle_only_callback_upper"])
    check("complete118_pair_rotation_bound", caps["numerical_method_caps"]["maximum_attempted_pair_rotations"] == 2 * 12 * 118 * 117 // 2)
    check("H_CI2_and_no_external_producers", caps["call_limits"]["H"] == caps["call_limits"]["CI"] == 2
          and all(caps["call_limits"][name] == 0 for name in ("Dirac", "external_hS", "gradient", "optimizer", "new_RDM")))
    check("hS_producer7_forecast74_ceiling14_128", caps["source_expected_CI_internal_hS"] == 7
          and caps["source_expected_hS_consumer_requests"] == 29 + 7 + 31 + 7
          and caps["call_limits"]["CI_internal_hS"] == 14 and caps["maximum_hS_consumer_requests"] == 128)
    manifest = json.loads(roles["literal208_checkpoint"].read_text())
    check("literal208_stationarity_failed_qualified_checkpoint", manifest["global_accepted_step"] == 208
          and manifest["checkpoint_qualified"] is True and manifest["stationarity_passed"] is False
          and manifest["array_sha256"] == contract["input_roles"]["literal208_arrays"]["sha256"]
          and manifest["metric_sha256"] == fixed["checkpoint_metric_sha256"])
    prior = json.loads(roles["nuclear_failed_evidence"].read_text())
    check("nuclear_counter_failure_preserved", prior["checks"][-1]["name"] == "expected_completed_CI_internal_hS"
          and prior["checks"][-1]["passed"] is False and prior["scientific_success_claim"] is False
          and prior["completed"]["CI_internal_hS"] == 7 and all(row["passed"] is True for row in prior["checks"][:-1]))
    for i in range(2):
        expected = contract["input_roles"]["nuclear_probe_seed_" + str(i)]
        check("nuclear_seed_saved_stage_binding_" + str(i), any(row["sha256"] == expected["sha256"]
              and row["name"] == f"virtual_{i}_qualified_complete.npz" for row in prior["completed_archives"]))
    check("fixed_point_not_expanded_stationary_MCDHF", fixed["original_stationarity_passed"] is False
          and fixed["expanded_MCDHF_executed"] is False and fixed["experimental_gate_passed"] is False)
    proof = dict(status="source_JSON_only_prepared_not_executed", all_source_JSON_preparation_checks_passed=True,
        checks=checks, total_checks=len(checks), driver_sha256=sha(HERE / "run.py"),
        contract_sha256=sha(HERE / "contract.json"), helper_sha256=sha(HERE / "signed_scalar_controls.py"),
        models_sha256=sha(HERE / "models.json"), preparation_script_sha256=sha(Path(__file__)),
        source_map_sha256=source["source_map_sha256"], parsed_source_files=len(source_trees),
        source_AST_direct_imports=direct_imports, deferred_NPZ_identity_checks=deferred,
        NumPy_Atomx_imports=0, NPZ_reads=0, scientific_calls=0, actual_source_execution_qualified=False,
        preserved_prior_counter_FAIL=True, new_CI_expanded_stationarity_claim=False)
    (HERE / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(all_source_JSON_preparation_checks_passed=True, checks=len(checks),
        science=0, imports=0, NPZ_reads=0, source_files=len(source_trees), deferred_NPZ_checks=len(deferred))))


if __name__ == "__main__":
    main()
