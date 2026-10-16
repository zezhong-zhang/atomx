"""Static import/JSON schema and exact predecessor-diff checks; no imports of driver."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]
OLD = TOP / "complete5f_fixed208_preparation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def bindings(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = type(node).__name__
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]] = "imported_name"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        result[child.id] = "assigned_name"
    return result


def main():
    destination = HERE / "preparation.json"
    if destination.exists():
        raise FileExistsError("Keep preparation immutable")
    contract = read(HERE / "contract.json")
    previous = read(OLD / "contract.json")
    checks, imports, schemas = [], [], []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    before, after = (OLD / "run.py").read_text(), (HERE / "run.py").read_text()
    check("driver_exact_one_import_line_diff", before.count("from atomx.numerics.bspline import dirac\n") == 1
          and before.replace("from atomx.numerics.bspline import dirac\n",
          "from atomx.numerics.bspline import dirac_assembly as dirac\n") == after)
    for field in ("fixed", "source", "hypothesis", "models_sha256", "acceptance", "resources", "numerical_method", "stopping", "execution_binding_required"):
        check("unchanged_physics_gates_caps/" + field, contract[field] == previous[field])
    check("models_exact_original_bytes", (HERE / "models.json").read_bytes() == (OLD / "models.json").read_bytes())
    for role, row in previous["input_roles"].items():
        check("same_original_role/" + role, contract["input_roles"][role] == row)
    for path, expected in contract["inputs_sha256"].items():
        check("input_bytes/" + path, sha(ROOT / path) == expected)
    failure, receipt = read(OLD / "actual/failure.json"), read(OLD / "root_execution.json")
    check("original_import_failure_retained", failure["error_type"] == "ImportError"
          and "cannot import name 'dirac'" in failure["message"] and receipt["exit_code"] == 1)
    check("original_all_producer_counters_zero", all(value == 0 for table in (failure["attempted"], failure["completed"]) for value in table.values()))
    check("original_no_array_archives", failure["completed_archives"] == [])
    check("predecessor_source_retained", (OLD / "actual/execution_source.py").read_bytes() == (OLD / "run.py").read_bytes())
    check("original540_stays_original_SCOPE", read(OLD / "preparation.json")["checks_count"] == 540)
    source = read(ROOT / contract["input_roles"]["candidate_source_identity"]["path"])
    source_root = ROOT / contract["source"]["candidate_archive"]
    check("original_frozen356", len(source["all_atomx_python_sha256"]) == 356
          and source["source_map_sha256"] == contract["fixed"]["source_map_sha256"])
    mismatches = [name for name, value in source["all_atomx_python_sha256"].items()
                  if sha(source_root / Path(name).relative_to("src")) != value]
    check("all356_source_bytes_exact", not mismatches)
    tree = ast.parse(after)
    check("driver_AST_not_evaluated_or_compiled", isinstance(tree, ast.Module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("atomx"):
            continue
        folder = source_root.joinpath(*node.module.split("."))
        module = folder.with_suffix(".py") if folder.with_suffix(".py").exists() else folder / "__init__.py"
        check("import_parent_exists/" + node.module, module.exists())
        declared = bindings(ast.parse(module.read_text()))
        for alias in node.names:
            child = folder / (alias.name + ".py")
            child_package = folder / alias.name / "__init__.py"
            present = alias.name in declared or child.exists() or child_package.exists()
            check("import_name_exists/" + node.module + "/" + alias.name, present)
            selected = child if child.exists() else child_package if child_package.exists() else module
            imports.append({"module": node.module, "import_name": alias.name, "alias": alias.asname,
                "source_path": str(selected.relative_to(ROOT)), "source_sha256": sha(selected),
                "kind": "submodule" if selected != module else declared[alias.name]})
    dirac_file = source_root / "atomx/numerics/bspline/dirac_assembly.py"
    check("dirac_alias_has_exact_solve_dirac_API", "solve_dirac" in bindings(ast.parse(dirac_file.read_text())))
    check("old_nonexistent_submodule_still_absent", not (source_root / "atomx/numerics/bspline/dirac.py").exists())
    roles = {role: ROOT / row["path"] for role, row in contract["input_roles"].items()}

    def schema(role, fields, test):
        data = read(roles[role])
        check("JSON_fields/" + role, all(field in data for field in fields))
        check("JSON_semantics/" + role, test(data))
        schemas.append({"role": role, "path": str(roles[role].relative_to(ROOT)),
                        "fields": fields, "sha256": sha(roles[role]), "passed": True})

    schema("candidate_source_identity", ["source_map_sha256", "all_atomx_python_sha256"], lambda d: len(d["all_atomx_python_sha256"]) == 356)
    schema("literal208_checkpoint", ["global_accepted_step", "iteration", "checkpoint_qualified", "stationarity_passed",
        "source_map_sha256", "metric_sha256", "geometry_sha256", "objective_sha256", "array_sha256", "owner", "targets", "diagnostics"],
        lambda d: d["global_accepted_step"] == 208 and d["iteration"] == 128 and d["checkpoint_qualified"] is True
        and d["stationarity_passed"] is False and len(d["targets"]) == 5
        and all({"label", "UID", "selected_level", "tracking"} <= set(r) for r in d["targets"])
        and {tuple(r["UID"]) for r in d["targets"]} == {tuple(uid) for _, uid in contract["fixed"]["target_UIDs"]}
        and all({"keys", "shape", "slice"} <= set(row) for row in read(roles["saved_layout"])))
    schema("literal208_completion", ["all_completion_metadata_checks_passed", "all_actual_live_controls_passed",
        "total_global_accepted_steps", "stationarity_passed", "actual_foreground_exit_code"],
        lambda d: d["all_completion_metadata_checks_passed"] and d["all_actual_live_controls_passed"]
        and d["total_global_accepted_steps"] == 208 and not d["stationarity_passed"] and d["actual_foreground_exit_code"] == 0)
    schema("refinement_qualified_proof", ["all_independent_saved_array_checks_passed"], lambda d: d["all_independent_saved_array_checks_passed"] is True)
    schema("screened_basis", ["balanced_virtual"], lambda d: len(d["balanced_virtual"]["retained_signatures"]) == 1046)
    schema("literal208_contract", ["input_roles", "fixed", "source", "metric"], lambda d:
        d["input_roles"]["screened_models"] == contract["input_roles"]["transitive_original_screened_models"])
    schema("transitive_original_screened_models", ["models"], lambda d: "baseline" in d["models"]
        and all("config" in r for r in d["models"]["baseline"]["configs"]))
    schema("tracking_binding", ["targets"], lambda d: len(d["targets"]) == 5
        and all({"label", "twoJ", "parity", "models"} <= set(r) and "local_root" in r["models"]["DC"] for r in d["targets"])
        and {(r["label"], r["twoJ"], r["parity"], r["models"]["DC"]["local_root"]) for r in d["targets"]}
        == {(label, *uid) for label, uid in contract["fixed"]["target_UIDs"]})
    schema("nominal_NIST_rows", ["DC_rows"], lambda d: len(d["DC_rows"]) == 5
        and all({"label", "NIST_nominal_cm1"} <= set(r) for r in d["DC_rows"])
        and {r["label"] for r in d["DC_rows"]} == {label for label, _ in contract["fixed"]["target_UIDs"]})
    layout = read(roles["saved_layout"])
    check("layout7_exact_JSON_fields", len(layout) == 7 and all({"keys", "shape", "slice"} <= set(r)
          and r["shape"][0] == 119 and len(r["slice"]) == 2 for r in layout))
    model = read(HERE / "models.json")
    check("models_complete_JSON_fields", set(model["models"]) == {"baseline", "complete5f"}
          and all({"configs", "full_CSFs", "complete_target_CSFs", "full_J_parity_sectors", "complete_target_blocks", "radial_keys"}
                  <= set(row) for row in model["models"].values()))
    old_binding = read(OLD / "execution_binding.json")
    trigger = old_binding["actual_iteration_trigger"]
    proof = read(ROOT / trigger["delivery_proof_path"])
    check("root_actual_trigger_schema_example", proof["passed"] is True and proof["message_id"] == trigger["message_id"]
          and proof["chat_id"] == "oc_fb2fc627b3e98dc9555de31580878758"
          and proof["bot_open_id"] == "ou_c697e48a528ad9aa6157f62531561b59" and proof["actual_send"]["passed"] is True)
    result = {"all_static_import_JSON_and_hash_checks_passed": True, "checks_count": len(checks), "checks": checks,
        "static_import_bindings": imports, "JSON_field_schemas": schemas,
        "source_map_sha256": contract["fixed"]["source_map_sha256"], "contract_sha256": sha(HERE / "contract.json"),
        "driver_sha256": sha(HERE / "run.py"), "models_sha256": sha(HERE / "models.json"),
        "preparation_driver_sha256": sha(Path(__file__).resolve()), "failed_predecessor_receipt_sha256": sha(OLD / "root_execution.json"),
        "prepared_scientific_calls": 0, "prepared_NPZ_array_reads": 0, "driver_evaluation_or_compilation": 0,
        "numpy_scipy_Atomx_imports": 0, "automatic_retry": False,
        "import_smoke_executed": False, "scientific_execution_authorized": False,
        "scope": "Names/exports from actualfrozenPythonAST and allconsumedJSONfieldschemas verified; no dynamic import assurance or physical qualification. Original540proof/failedlaunch retained."}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks_count": len(checks), "static_imports": len(imports),
                      "preparation_sha256": sha(destination), "scientific_calls": 0}))


if __name__ == "__main__":
    main()
