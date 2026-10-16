"""Read-only source/contract audit; no Atomx import or scientific evaluation."""
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    contract = json.loads((HERE / "contract.json").read_text())
    identity = json.loads((HERE / "source_identity.json").read_text())
    check("source_identity_binding", digest(HERE / "source_identity.json")
          == contract["source_identity_sha256"])
    check("proposal_binding", digest(HERE / "README.md") == contract["proposal_sha256"])
    check("complete354_source", len(identity["all_atomx_python_sha256"]) == 354)
    check("current_complete_source_file_set", {
        str(p.relative_to(ROOT)) for p in (ROOT / "src/atomx").rglob("*.py")
    } == set(identity["all_atomx_python_sha256"]))
    for path, expected in identity["all_atomx_python_sha256"].items():
        check("source_SHA/" + path, digest(ROOT / path) == expected)
    for path, expected in identity["protected_inputs_sha256"].items():
        check("historical_input_SHA/" + path, digest(ROOT / path) == expected)
    check("immutable_current_epoch", identity["source_map_sha256"]
          == "3304f5895bcfce70708f1aa9a77bca8a7cfc3eca9a9d393380035c42ed67df2d")
    check("authorization_pending", not any(contract["authorization"].values()))
    check("zero_preparation_science", all(value == 0 for key, value in
          contract["preparation_budget"].items() if key.startswith("maximum_")))
    check("finite_budget_18H13CI3g0opt", {
        key: value for key, value in contract["future_finite_qualification_budget"].items()
        if key.startswith("maximum_")
    } == {"maximum_H_builds": 18, "maximum_complete_CI_solves": 13,
          "maximum_full_gradients": 3, "maximum_optimizer_calls": 0})
    check("original_yardstick_preserved", contract["original_yb_gates_unchanged"]
          == {"residual": 1e-7, "indicator_Ha": 1e-8,
              "g_replay": 1e-9, "pg_replay": 1e-10, "Armijo": 1e-4})
    check("phase_all_associated_residuals", "r0bar=sigma*r0" in
          contract["finite_response"]["phase"] and
          "2d.T r0bar" in contract["finite_response"]["reported_root"])
    check("no_optimizer_adapter", contract["finite_response"]["production_adapter"] is False)
    check("nine_complete_configs", len(contract["same_model"]["configs"]) == 9
          and contract["same_model"]["complete_j_resolved_enumeration"] is True)
    check("fixed_unequal_weights", contract["same_model"]["weights"] == [0.25, 0.75])
    check("all_six_signed_scales", contract["same_model"]["signed_scales"]
          == [1e-4, -1e-4, 1e-6, -1e-6, 1e-8, -1e-8])

    source = ROOT / "src/atomx"
    trees = {name: ast.parse((source / name).read_text()) for name in
             ("methods/csf/variational.py", "methods/csf/solver.py",
              "numerics/bspline/orbital_optimization.py", "methods/ci.py")}
    functional = trees["methods/csf/variational.py"]
    methods = {node.name: node for node in ast.walk(functional)
               if isinstance(node, ast.FunctionDef)}
    for name in ("_fixed_density_energy_difference", "constrained_energy_difference"):
        body = ast.unparse(methods[name])
        check(name + "_still_rejects_mixed_tracked_blocks",
              "self.level_tracking is not None" in body
              and "len(columns) != 1" in body and "return None" in body)
    check("new_diagnostic_not_implemented", "mixed_ci_energy_difference" not in methods)
    arguments = methods["__init__"].args
    default_by_name = dict(zip((arg.arg for arg in arguments.kwonlyargs),
                              arguments.kw_defaults, strict=True))
    check("functional_retention_default_false", ast.literal_eval(
        default_by_name["retain_hamiltonian"]) is False)
    solve_calls = [node for node in ast.walk(trees["methods/csf/solver.py"])
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "functional_class"]
    check("public_solver_has_no_retention_propagation", bool(solve_calls) and all(
        "retain_hamiltonian" not in {kw.arg for kw in node.keywords}
        for node in solve_calls))
    optimizer_text = ast.unparse(trees["numerics/bspline/orbital_optimization.py"])
    check("accepted_trial_CI_already_reused", "evaluate(x, ci_result=trial_ci)"
          in optimizer_text)
    check("typed_checkpoint_receives_actual_CI", "checkpoint_callback(iteration, state, unpack(x), ci,"
          in optimizer_text)
    check("no_solve_mcdhf_symbol", not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "solve_mcdhf" for p in source.rglob("*.py")
        for node in ast.walk(ast.parse(p.read_text()))))

    readme = (HERE / "README.md").read_text()
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        check("local_link/" + target, (HERE / target).is_file())
    check("proposal_no_completed_scientific_claim", "proposal, not an implementation"
          in readme and "Numerical tests above are\n**planned**" in readme)
    check("bank_representation_explicit", "**same bank/banded**" in readme)
    check("phase_corrected_explicit", "r0bar = sigma*r0" in readme)
    check("legacy_guard_preservation_explicit", "their numerical path remain unchanged" in readme)
    for path in HERE.iterdir():
        if path.suffix in (".py", ".md", ".json"):
            check("no_trailing_whitespace/" + path.name,
                  all(line.rstrip() == line for line in path.read_text().splitlines()))
    result = {
        "status": "readonly_preparation_verified_only",
        "all_passed": all(row["passed"] for row in checks),
        "checks_count": len(checks), "checks": checks,
        "source_map_sha256": identity["source_map_sha256"],
        "contract_sha256": digest(HERE / "contract.json"),
        "source_identity_sha256": digest(HERE / "source_identity.json"),
        "proposal_sha256": digest(HERE / "README.md"),
        "verifier_sha256": digest(Path(__file__)),
        "actual_scientific_calls": {"H_build": 0, "CI_solve": 0,
                                   "gradient": 0, "optimizer": 0},
        "source_edit_authorized": False,
        "finite_numerical_qualification_executed": False,
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("all_passed", "checks_count",
                                                  "actual_scientific_calls")}))
    for row in checks:
        if not row["passed"]:
            print("FAIL " + row["name"])
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
