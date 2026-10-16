"""Source/metadata preparation only: no consumer imports or array execution."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    checks = []

    def gate(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(name)

    contract = json.loads((HERE / "contract.json").read_text())
    old_contract = json.loads((BASE / "contract.json").read_text())
    parsed = {}
    for name in ("helper.py", "run.py", "launch.py", "prepare.py"):
        source = (HERE / name).read_text()
        compile(source, str(HERE / name), "exec")
        parsed[name] = ast.parse(source)
        gate("syntax_" + name, True)
        gate("no_runtime_exec_eval_" + name, not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("exec", "eval")
            for n in ast.walk(parsed[name])))
    helper = parsed["helper.py"]
    old = ast.parse((BASE / "run.py").read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in helper.body if isinstance(n, ast.FunctionDef)}
    gate("four_literal_consumer_functions", set(new_functions) == {
        "prepare_point", "validate_trial", "finish_checked_chain", "compare_saved_chains"})

    # Static comparison of literal nodes; never execute/unparse original source.
    def moved(name, start, stop, destination):
        nodes = [n for n in old_functions["chain"].body if start <= n.lineno <= stop]
        haystack = [dump(n) for n in new_functions[destination].body]
        needle = [dump(n) for n in nodes]
        gate(name, bool(needle) and any(haystack[i:i + len(needle)] == needle
             for i in range(len(haystack) - len(needle) + 1)),
             {"old_lines": [start, stop], "top_level_nodes": len(needle), "new_function": destination})

    moved("literal_point_setup_operation_order", 361, 421, "prepare_point")
    moved("literal_public_owner_centering_bind", 427, 433, "validate_trial")
    moved("literal_complete_CI_residual_Gram_stale_guards", 439, 453, "validate_trial")
    moved("literal_original3451_metric_and_gradient_guards", 460, 481, "finish_checked_chain")
    old_compare = old_functions["compare"]
    new_compare = new_functions["compare_saved_chains"]

    def named_loop(function, variable):
        return next(n for n in function.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == variable)

    gate("literal_full1046_signed_cluster_loop", dump(named_loop(old_compare, "row")) ==
         dump(named_loop(new_compare, "row")))
    old_limits = next(n for n in old_compare.body if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Tuple))
    new_limits = next(n for n in new_compare.body if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Tuple))
    gate("literal_whole_array_threshold_loop", dump(old_limits) == dump(new_limits))
    old_guard_functions = (old_functions["chain"], old_compare)
    new_guard_functions = tuple(new_functions.values())
    # All physical check expressions retained; explicit exclusions are the old
    # ordinary-only saved-H replay and old execution's one-chain ledger.
    old_physical_calls = [n for f in old_guard_functions for n in ast.walk(f)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                          and n.func.id in ("gate", "close", "check")
                          and not (513 <= n.lineno <= 547 or 640 <= n.lineno <= 644)]
    new_call_dumps = {dump(n) for f in new_guard_functions for n in ast.walk(f)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id in ("gate", "close", "check")}
    # File-read-only calls that changed to explicit paths are checked separately.
    path_changed = {"same_complete_sector_columns", "complete_same_physical_owner"}
    for n in old_physical_calls:
        if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in path_changed:
            continue
        gate("inherited_guard_line_" + str(n.lineno), dump(n) in new_call_dumps)
    gate("original_numeric_gate_manifest_unchanged",
         {k: v for k, v in contract["gates"].items() if k != "ordinary_H_and_roots"} ==
         {k: v for k, v in old_contract["gates"].items() if k != "ordinary_H_and_roots"})
    gate("original_physical_model_unchanged", contract["fixed"] == old_contract["fixed"])
    gate("symmetric_extra5_replay_exclusion_predeclared", "Excluded symmetrically" in
         contract["literal_refactor"]["inherited_extra5_saved_H_eigh_replay"])
    gate("no_eigensolver_in_helper", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr in ("eigh", "eigvalsh", "svd") for n in ast.walk(helper)))

    for variant in ("ordinary", "candidate"):
        row = contract[variant]
        identity = json.loads((REPO / row["identity"]).read_text())
        mapping = identity["all_atomx_python_sha256"]
        gate(variant + "_map_identity", identity["source_map_sha256"] == row["source_map_sha256"])
        gate(variant + "_356_frozen_files", len(mapping) == 356 and all(
            sha(REPO / row["archive"] / name.removeprefix("src/")) == expected
            for name, expected in mapping.items()))
    gate("all_bound_inputs_unchanged", all(sha(REPO / row["path"]) == row["sha256"]
         for row in contract["inputs"].values()), {"roles": len(contract["inputs"])})
    qualified = json.loads((REPO / contract["inputs"]["qualification003_result"]["path"]).read_text())
    shared = json.loads((REPO / contract["inputs"]["shared_adoption_completion"]["path"]).read_text())
    gate("actual003_closed_equivalence", qualified["all_equivalence_gates_passed"])
    gate("actual_shared123_regression", shared["test_counts"] ==
         {"tests": 123, "errors": 0, "failures": 0, "skipped": 0})
    for name in (10, 40):
        row = contract["inputs"][f"history32_pilot{name}_closed"]
        gate(f"history{name}_foreground_closed", json.loads((REPO / row["path"]).read_text())[
             "status"] == "foreground_closed")
    gate("actual_available_affinity_CPU0_CPU30", {0, 30} <= os.sched_getaffinity(0),
         {"permitted_CPUs": sorted(os.sched_getaffinity(0))})
    gate("exact14_chains_and70eigh", contract["schedule"]["total_chains"] == 14 and
         contract["resources"]["max_scipy_eigh"] == 70)
    gate("exact28hS62NPZ", contract["resources"]["max_total_hS"] == 28 and
         sum(contract["variant_exact_calls"][v]["npz_load"] for v in ("ordinary", "candidate")) == 62)
    result = subprocess.run([str(REPO / ".venv/bin/ruff"), "check", str(HERE)],
                            capture_output=True, text=True, check=False)
    gate("Ruff", result.returncode == 0, result.stdout + result.stderr)
    proof = {"all_passed": True, "checks_count": len(checks), "checks": checks,
             "driver_sha256": sha(HERE / "run.py"), "helper_sha256": sha(HERE / "helper.py"),
             "launcher_sha256": sha(HERE / "launch.py"), "contract_sha256": sha(HERE / "contract.json"),
             "preparer_sha256": sha(__file__), "qualification003_driver_sha256": sha(BASE / "run.py"),
             "source_only_numerical_imports_array_loads_H_CI_gradient_NumBa_compiles": 0,
             "scientific_execution_authorized": False,
             "static_refactor_mechanism": "Literal source functions, static AST equality only; no source execution."}
    (HERE / "preparation_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"all_passed": True, "checks_count": len(checks),
                      "preparation_sha256": sha(HERE / "preparation_verification.json")}))


if __name__ == "__main__":
    main()
