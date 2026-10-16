"""Stdlib-only source/input/AST preparation admission. Never run the worker."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    controls = []

    def gate(name, condition):
        controls.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    gate("execution_not_authorized", contract["root_execution_authorized"] is False)
    gate("no_actual_output", not (HERE / "actual").exists())
    for role, row in contract["inputs"].items():
        gate("input_" + role, sha(REPO / row["path"]) == row["sha256"])
    identities = {v: json.loads((REPO / contract[v]["identity"]).read_text()) for v in ("ordinary", "candidate")}
    gate("356files_each", all(len(i["all_atomx_python_sha256"]) == 356 for i in identities.values()))
    gate("same_complete_paths", identities["ordinary"]["all_atomx_python_sha256"].keys()
         == identities["candidate"]["all_atomx_python_sha256"].keys())
    changed = []
    for name, original in identities["ordinary"]["all_atomx_python_sha256"].items():
        for variant in ("ordinary", "candidate"):
            path = REPO / contract[variant]["archive"] / name.removeprefix("src/")
            expected = identities[variant]["all_atomx_python_sha256"][name]
            gate(variant + "_frozen_" + name, sha(path) == expected)
            compile(path.read_bytes(), str(path), "exec")  # syntax only; no import/exec
        gate("shared500ac_unchanged_" + name, sha(REPO / name) == original)
        if original != identities["candidate"]["all_atomx_python_sha256"][name]:
            changed.append(name)
    gate("sole_matrix_private_delta", changed == ["src/atomx/physics/csf/matrix.py"])
    for variant, identity in identities.items():
        digest = hashlib.sha256(json.dumps(identity["all_atomx_python_sha256"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        gate("canonical_map_" + variant, digest == contract[variant]["source_map_sha256"])
    original_path = REPO / contract["ordinary"]["archive"] / "atomx/physics/csf/matrix.py"
    candidate_path = HERE / "candidate_source/atomx/physics/csf/matrix.py"
    original, candidate = original_path.read_text(), candidate_path.read_text()
    diff = "".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                    fromfile="a/src/atomx/physics/csf/matrix.py", tofile="b/src/atomx/physics/csf/matrix.py"))
    gate("exact_reviewed_patch", diff == (HERE / "candidate.patch").read_text())
    trees = [ast.parse(t) for t in (original, candidate)]
    functions = [{n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} for tree in trees]
    gate("same_matrix_declarations", functions[0].keys() == functions[1].keys())
    for name in functions[0]:
        if name != "_bspline_pair_potential":
            gate("unchanged_matrix_AST_" + name,
                 ast.dump(functions[0][name], include_attributes=False) == ast.dump(functions[1][name], include_attributes=False))
    helper = ast.unparse(functions[1]["_bspline_pair_potential"])
    gate("existing_threshold_and_four_components", "ns >= SEMISEPARABLE_DIRECT_NS" in helper
         and "('p', 'q')" in helper and "_convol_case1_semiseparable_t" in helper
         and "_convol_case1_t" in helper and "value += contracted" in helper)
    runner = (HERE / "run.py").read_text()
    compile(runner, str(HERE / "run.py"), "exec")
    rt = ast.parse(runner)
    imports = [n for n in rt.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    gate("top_level_stdlib_only", all(not any(token in ast.unparse(n) for token in ("numpy", "scipy", "numba", "atomx")) for n in imports))
    gate("no_solver_optimizer_or_FD_invocation", all(not isinstance(n, ast.Call) or
        not isinstance(n.func, ast.Name) or n.func.id not in ("optimize_orbitals", "solve_mcdhf", "_polar_retraction")
        for n in ast.walk(rt)))
    gate("exact_chain_public_calls", runner.count("functional.evaluate(waves, gradient=False)") == 1
         and runner.count("functional.evaluate(waves, ci_result=ci, gradient=True)") == 1)
    gate("source_and_runtime_seal_guards", "execution_authorized_by_root" in runner
         and "read_only_owned_H" in runner and "stale_owner_rejected" in runner)
    gate("correct_original_L_mapping", "np.tril(h) + np.tril(h, -1).T" in runner
         and 'geometry[f"h_{k}"]' in runner and "same_raw_h_" in runner)
    gate("no_saved_owner_forge", all(token not in runner for token in ("object.__new__", "matrix_override", "from_arrays(", "pickle.loads")))
    gate("full_column_residual_norm", "np.max(np.linalg.norm(h @ ci.mixing_coefficients" in runner)
    gate("complete_target_UID_support", "reference_levels" in runner and "exact_five_UID_support" in runner)
    gate("canonical_phase_independent_cluster_gate", "c0 @ c0.T - c1 @ c1.T" in runner
         and "complete_sector_exact_zeros" in runner)
    gate("basis_anchor_count", contract["anchor_calls"]["basis_setup"] == 5)
    gate("all4components_3ranks_boundary", contract["anchor_setup"]["ranks"] == [0, 1, 6]
         and contract["anchor_setup"]["components"] == ["pppp", "qqqq", "pqpq", "qpqp"]
         and [r["ns"] for r in contract["anchor_setup"]["cases"]] == [95, 96, 95, 96, 4])
    gate("anchor_exact_budgets", contract["anchor_calls"] == {"basis_setup": 5,
        "density_band": 20, "direct_full_bank": 96, "direct_semiseparable": 84,
        "band_scalar": 150, "CI_source_pair": 15, "bank_get": 96, "factor_get": 84, "Gauss_setup_eigvalsh": 5})
    gate("chain_caps", all(contract["chain_exact_calls"][k] == n for k, n in
         (("H", 1), ("CI", 1), ("gradient", 1), ("initial_hS", 7), ("CI_internal_hS", 7),
          ("tracking_resolution", 2), ("tracking_SVD", 10), ("tracking_overlap", 3), ("scipy_eigh", 5))))
    gate("zero_new_spectrum_or_polar", contract["chain_exact_calls"]["Dirac_eigensolve"] ==
         contract["chain_exact_calls"]["eigvalsh"] == contract["chain_exact_calls"]["numpy_eigh"] == 0)
    gate("unrelaxed_original_gates", contract["gates"]["original_raw_and_tangent_g"] == 1e-9
         and contract["gates"]["original_metric_pg"] == 1e-10 and contract["gates"]["stationarity"] == 1e-7
         and contract["gates"]["indicator_Ha"] == 1e-8)
    for role in ("conditional_direction0", "conditional_direction1"):
        result = json.loads((REPO / contract["inputs"][role]["path"]).read_text())
        gate(role + "_conditional_scope_preserved", result["same_saved_point_conditional_gate_passed"] is True
             and result["original_rotation002_passed"] is False and result["optimizer_authorized"] is False)
    proof = {"all_passed": True, "checks_count": len(controls), "checks": controls,
             "driver_sha256": sha(HERE / "run.py"), "contract_sha256": sha(HERE / "contract.json"),
             "verifier_sha256": sha(__file__), "inputs_bound": len(contract["inputs"]),
             "source_files": 356, "private_candidate_map_sha256": contract["candidate"]["source_map_sha256"],
             "scientific_imports": 0, "array_loads": 0, "H_CI_gradient_kernel_eigen_optimizer_GPU_calls": 0,
             "numerical_qualification_executed": False, "performance_certified": False}
    (HERE / "preparation_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "checks"}))


if __name__ == "__main__":
    main()
