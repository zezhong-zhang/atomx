"""Metadata / AST / symbolic UID controls only; no numerical imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    contract = json.loads((HERE / "contract.json").read_text())
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Source preparation check failed: " + name)

    text = (HERE / "run.py").read_text()
    tree = ast.parse(text)
    compile(tree, str(HERE / "run.py"), "exec")
    check("compile_without_execution", True)
    check("explicit_failed_rotation_HOLD", contract["execution_ready"] is False
          and "HOLD" in contract["status"])
    check("no_execution_authorization", contract["root_authorized"] is False)
    check("qualified_rotation_missing_not_fabricated",
          contract["pending_prerequisites"]["rotation_qualified_result"] is None
          and contract["pending_prerequisites"]["rotation_independent_saved_proof"] is None)
    guard_role = contract["pending_prerequisites"]["exact_guard_qualification"]
    guard_proof = json.loads((REPO / guard_role["path"]).read_text())
    check("finite_guard_arithmetic_prerequisite_passed_only", guard_proof["all_passed"] is True
          and guard_proof["no_Yb_gradient_or_optimizer_qualification"] is True)
    check("exact_guard_helper_identity", guard_proof["consumer_sha256"]
          == contract["pending_prerequisites"]["exact_guard_helper"]["sha256"])
    for name, role in contract["input_roles"].items():
        check("input_sha/" + name, sha(REPO / role["path"]) == role["sha256"])
    for name, role in contract["hold_evidence"].items():
        check("failed_evidence_sha/" + name, sha(REPO / role["path"]) == role["sha256"])
    for name, role in contract["pending_prerequisites"].items():
        if role is not None:
            check("available_pending_source_sha/" + name, sha(REPO / role["path"]) == role["sha256"])
    failure = json.loads((REPO / contract["hold_evidence"]["rotation002_failure"]["path"]).read_text())
    check("original_failed_derivative_retained", failure["error"]
          == "Scientific gate failed: direction1_NEW_Rayleigh_derivative")
    source = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    check("new_complete_source_epoch", source["source_map_sha256"]
          == "500ac4c3c372e0eb51e6d9910ef3830bf8eb323b211b52ba3ec08ccbf7b1b794")
    for name, expected in source["all_atomx_python_sha256"].items():
        check("frozen_source_sha/" + name,
              sha(REPO / contract["source"]["candidate_archive"] / name.removeprefix("src/")) == expected)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    optimizer_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "optimize_orbitals"]
    check("one_public_optimizer_invocation", len(optimizer_calls) == 1)
    keywords = {keyword.arg: keyword.value for keyword in optimizer_calls[0].keywords}
    check("original_orbital_gate", ast.literal_eval(keywords["tolerance"]) == 1e-7)
    check("original_energy_gate", ast.literal_eval(keywords["energy_tolerance"]) == 1e-8)
    check("at_most16_iterations", ast.literal_eval(keywords["max_iter"]) == 16)
    check("no_energy_callback", ast.literal_eval(keywords["energy_difference"]) is None)
    check("original_spectral_policy", ast.literal_eval(keywords["orbital_preconditioner"]) == "spectral")
    check("original_nuclear_P_policy", ast.literal_eval(keywords["projector_source"]) == "nuclear")
    check("unchanged_history10", ast.literal_eval(keywords["history_size"]) == 10)
    check("whole_metric_resume_supplied", isinstance(keywords["resume_state"], ast.Name)
          and keywords["resume_state"].id == "initial")
    check("checkpoint_guard_supplied", isinstance(keywords["checkpoint_callback"], ast.Name)
          and keywords["checkpoint_callback"].id == "checkpoint")
    check("no_source_derived_seed_reprojection", "coordinates=saved_g[\"x\"]" in text)
    check("retained_real_H_owner", "retain_hamiltonian=True" in text and "hamiltonian_for" in text)
    check("same_point_g_original_gate", '"baseline_raw_coordinate_g_whole"' in text
          and '"baseline_original_pg_whole"' in text)
    check("owner_json_plain_normalization", 'manifest["owner"] != plain(load(path("saved_owner")))' in text)
    check("one_native_builder_forwarder", "matrix.build_ci_matrix = wrap(\"H\", matrix.build_ci_matrix)" in text)
    check("guard_remains_checkpoint_only", "reported_acceptance_decision(" in text
          and "energy_difference=None" in text)
    check("finite_quotient_not_objective", contract["objective"]["Rayleigh_production_objective"] is False)
    check("first_candidate_archived_before_gates", text.index('flush(f"checkpoint_{iteration}_forensic_saved_before_gates")')
          < text.index('check(f"point{iteration}_same_metric"'))
    check("guard_failed_point_does_not_replace_previous", text.index('if decision is not None and decision["qualified"] is not True:')
          < text.index('previous = {"certificate": cert'))
    check("resource_exception_not_swallowed_by_line_search", "class ResourceStop(RuntimeError)" in text
          and "class QualificationStop(RuntimeError)" in text)
    check("No_Atomx_numpy_scipy_top_level_import", all(
        not isinstance(node, (ast.Import, ast.ImportFrom)) or
        all(not name.startswith(("numpy", "scipy", "atomx")) for name in
            ([node.module or ""] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]))
        for node in tree.body))
    check("numerics_after_exact_admission", functions["execute"].body[0].value.func.id == "admit")
    check("ordinary_original_reported_path", "original_evaluate(orbitals, gradient=gradient" in text
          and "return result" in text)
    caps = contract["resources"]
    for name, expected in {"CPU": 31, "threads": 1, "max_H": 65, "max_CI": 65,
                           "max_full_gradient": 17, "max_checkpoints": 17, "max_new_accepted_steps": 16,
                           "max_optimizer_calls": 1, "max_validation_Dirac": 7, "max_total_hS": 462,
                           "max_geometry_eigvalsh": 36, "max_scipy_eigh": 346,
                           "optimizer_softwall_seconds": 570, "wall_seconds": 600,
                           "outer_wall_seconds": 615, "address_space_GiB": 3.95}.items():
        check("exact_resource_contract/" + name, caps[name] == expected)
    check("full1046_five_complete_blocks", contract["fixed"]["target_blocks"]
          == [[1, -1, 158], [1, 1, 150], [3, -1, 256], [3, 1, 236], [5, 1, 246]])
    check("full_support_pair_budget", sum(n * (n + 1) // 2 for _, _, n in
          contract["fixed"]["target_blocks"]) * 17 == caps["full_target_support_examined_pair_upper"]
          and caps["full_target_support_examined_pair_upper"] <= 3_000_000)
    spec = importlib.util.spec_from_file_location("prospective_optimizer_consumer", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.metadata(contract)
    except ValueError as error:
        check("current_physical_qualification_refuses_before_science", "optimizer HOLD" in str(error))
    else:
        check("current_physical_qualification_refuses_before_science", False)
    targets = json.loads((REPO / contract["input_roles"]["tracking_binding"]["path"]).read_text())["targets"]
    declared = [(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets]
    symbolic = [{"reference_levels": [declared[i]], "selected_levels": [declared[i]]} for i in (4, 2, 0, 3, 1)]
    check("UID_permutation_recovers_labels", list(module.target_bindings(targets, symbolic))
          == [declared[i] for i in (4, 2, 0, 3, 1)])
    for label, bad in (("duplicate", symbolic + symbolic[:1]), ("missing", symbolic[:-1]),
                       ("wrong_sector", [{"reference_levels": [declared[0]], "selected_levels": [[3, 1, 0]]}] + symbolic[:-1])):
        try:
            module.target_bindings(targets, bad)
        except module.QualificationStop:
            check("UID_reject/" + label, True)
        else:
            check("UID_reject/" + label, False)
    result = {"all_source_only_checks_passed": True, "checks_count": len(checks), "checks": checks,
              "runner_sha256": sha(HERE / "run.py"), "contract_sha256": sha(HERE / "contract.json"),
              "verifier_sha256": sha(__file__), "source_map_sha256": source["source_map_sha256"],
              "execution_ready": False, "rotation_qualification_passed": False,
              "exact_guard_array_qualification_passed": True,
              "numerical_imports": 0, "scientific_calls": 0,
              "scope": "prospective source consumer and current fail-closed admission; no physical or derivative qualification"}
    (HERE / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))


if __name__ == "__main__":
    verify()
