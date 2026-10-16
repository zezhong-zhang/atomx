"""Metadata / AST / symbolic UID controls only; no numerical imports."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
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
    check("separate_prospective_readiness_not_authorization", contract["execution_ready"] is True
          and contract["status"] == "prepared_requires_exact_root_execution"
          and contract["root_authorized"] is False)
    check("original_HOLD_files_preserved", all(sha(REPO / role["path"]) == role["sha256"]
          for role in contract["prospective_successor_of"].values()))
    previous_contract = json.loads((REPO / contract["prospective_successor_of"]["contract.json"]["path"]).read_text())
    check("old_contract_still_HOLD", previous_contract["execution_ready"] is False
          and "HOLD" in previous_contract["status"])
    check("new_admission_does_not_upgrade_failures", contract["admission_domain"]["old_derivative_failures_upgraded"] is False
          and contract["admission_domain"]["uniform_error_bound"] is False)
    old_tree = ast.parse((REPO / contract["prospective_successor_of"]["run.py"]["path"]).read_text())
    def get_function(t, name):
        return next(node for node in t.body if isinstance(node, ast.FunctionDef) and node.name == name)
    old_execute = get_function(old_tree, "execute")
    new_execute = get_function(tree, "execute")
    def numerical_core(t):
        nodes = [node for node in ast.walk(t) if isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Attribute) and node.func.attr in
                 ("optimize_orbitals", "evaluate", "from_arrays", "gradient_from_weights", "hamiltonian_for"))
                 or (isinstance(node.func, ast.Name) and node.func.id in
                     ("original_evaluate", "reported_acceptance_decision", "exact_reported_point")))]
        return [ast.dump(node, include_attributes=False) for node in nodes]
    check("physical_numerical_calls_identical_to_archived_consumer",
          numerical_core(old_execute) == numerical_core(new_execute))
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
    modules_before = set(sys.modules)
    check("new_samepoint_metadata_ready_only", module.metadata(contract)
          == REPO / contract["source"]["candidate_archive"])
    check("metadata_imports_no_numpy_scipy_Atomx", not any(
        n == "numpy" or n.startswith("numpy.") or n == "scipy" or n.startswith("scipy.")
        or n == "atomx" or n.startswith("atomx.") for n in set(sys.modules) - modules_before))
    evidence = {name: json.loads((REPO / role["path"]).read_text()) for name, role in
        contract["pending_prerequisites"].items() if role["path"].endswith(".json")}
    for label, name, field, value in (
        ("unclosed_baseline", "full_normalized_gradient_receipt", "returncode", 1),
        ("changed_metric", "full_normalized_gradient_control", "original_metric_sha256", "changed"),
        ("wrong_live_scope", "full_normalized_gradient_control", "completed", {"H": 1, "CI": 1, "gradient_from_weights": 1, "Dirac": 0}),
        ("failed_baseline", "full_normalized_gradient_control", "all_numeric_controls_passed", False),
        ("failed_direction", "centering_d0_result", "same_saved_point_conditional_gate_passed", False),
        ("old_failure_upgrade", "centering_d1_result", "original_rotation002_passed", True),
        ("unqualified_guard", "exact_guard_qualification", "all_passed", False)):
        bad = copy.deepcopy(evidence)
        bad[name][field] = value
        try:
            module.prerequisite_controls(contract, bad)
        except ValueError:
            check("prerequisite_reject/" + label, True)
        else:
            check("prerequisite_reject/" + label, False)
    try:
        module.admit(contract, {})
    except ValueError as error:
        check("missing_root_seal_refused_before_science", "Wrong exact root execution binding" in str(error))
    else:
        check("missing_root_seal_refused_before_science", False)
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
    symbolic_rows = [{"label": target["label"], "UID": list(uid), "energy_Ha": float(i),
        "nominal_gap_cm1": float(i)} for i, (target, uid) in enumerate(zip(targets, declared, strict=True))]
    shuffled = [symbolic_rows[i] for i in (4, 2, 0, 3, 1)]
    check("lower_selection_independent_of_target_order",
          module.physical_rows_by_label(shuffled)["6s_1/2"]["UID"] == [1, 1, 0])
    check("exact_four_upper_RMS_independent_of_target_order",
          module.nominal_four_gap_rms(shuffled) == math.sqrt(30 / 4))
    for label, bad in (("duplicate_lower", shuffled + shuffled[:1]),
                       ("wrong_lower_UID", [dict(row, UID=[3, 1, 0]) if row["label"] == "6s_1/2" else row for row in shuffled])):
        try:
            module.physical_rows_by_label(bad)
        except module.QualificationStop:
            check("physical_lower_reject/" + label, True)
        else:
            check("physical_lower_reject/" + label, False)
    result = {"all_source_only_checks_passed": True, "checks_count": len(checks), "checks": checks,
              "runner_sha256": sha(HERE / "run.py"), "contract_sha256": sha(HERE / "contract.json"),
              "verifier_sha256": sha(__file__), "source_map_sha256": source["source_map_sha256"],
              "execution_ready": True, "root_execution_authorized": False, "old_rotation_qualification_passed": False,
              "exact_guard_array_qualification_passed": True,
              "numerical_imports": 0, "scientific_calls": 0,
              "scope": "prospective source consumer with separately closed narrow baseline controls and exact-root refusal; no new physical/derivative/optimizer qualification"}
    (HERE / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))


if __name__ == "__main__":
    verify()
