"""Source/JSON-only preparation proof: no numerical imports or NPZ access."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []

    def check(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value:
            raise ValueError(name)

    text = (HERE / "run.py").read_text()
    tree = ast.parse(text)
    contract = read(HERE / "contract.json")
    roles, resources = contract["input_roles"], contract["resources"]
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    external = {"numpy", "numba", "atomx", "scipy", "cupy", "torch"}
    numeric = [n for n in ast.walk(tree) if
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] in external for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in external)]
    inside_execute = {id(n) for n in ast.walk(functions["execute"])}
    check("numeric_imports_only_inside_root_sealed_execute", all(id(n) in inside_execute for n in numeric))
    check("mandatory_root_binding_and_preparation_identity", all(token in text for token in
          ("Separate exact root execution binding required", "execution_authorized_by_root", "preparation_sha256")))
    check("implemented_but_not_execution_authorized", contract["executor_implemented"] is True
          and contract["execution_authorized"] is False)

    namespace = {}
    isolated = ast.Module(body=[functions["integer_resource_census"]], type_ignores=[])
    exec(compile(isolated, "isolated_stdlib_integer_census", "exec"), namespace)
    census = namespace["integer_resource_census"](contract)
    check("complete_support_census_271965_4623405_85_17",
          census["support_pair_upper_per_point"] == 271965
          and census["support_pair_upper_all_points"] == 4623405
          and census["target_quotients"] == 85 and census["points"] == 17)
    execution_text = ast.unparse(functions["execute"])
    check("census_before_numeric_import_and_loop", execution_text.index("integer_resource_census(contract)")
          < execution_text.index("import numpy"))
    bad = json.loads(json.dumps(contract))
    bad["resources"]["exact_guard_examined_pairs"] = 4_623_404
    try:
        namespace["integer_resource_census"](bad)
        rejected = False
    except ValueError:
        rejected = True
    check("census_refuses_insufficient_pair_cap", rejected)
    check("producer_and_complete_signed_control_caps", resources["calls"]["H"] == 65
          and resources["calls"]["CI"] == 65 and resources["calls"]["gradient_from_weights"] == 51
          and resources["calls"]["unfiltered_target_RDM"] == 85
          and resources["calls"]["optimizer"] == 1
          and resources["calls"]["gradient_pair_source"] == 51 * 7 * 31 * 32 // 2)
    check("CPU_AS_wall_literal_interpreter_caps", resources["CPU"] == 31
          and resources["AS_GiB"] == 3.95 and resources["inner_seconds"] == 600
          and resources["outer_seconds"] == 615
          and resources["interpreter"] == "/data/code/atomx/.venv/bin/python")

    inputs, deferred = {}, {}
    for role, row in roles.items():
        if row["path"].endswith(".npz"):
            deferred[role] = row
            continue
        digest = sha(REPO / row["path"])
        if digest != row["sha256"]:
            raise ValueError("Metadata input changed: " + role)
        inputs[row["path"]] = digest
    check("all_metadata_inputs_exact_all_NPZ_deferred", bool(inputs) and len(deferred) == 6)
    identity = read(REPO / roles["source_identity"]["path"])
    archive = REPO / contract["source"]["archive"]
    check("whole_frozen356_f043_source_identity", len(identity["all_atomx_python_sha256"]) == 356
          and identity["source_map_sha256"] == contract["source"]["map_sha256"]
          == "f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe"
          and all(sha(archive / name.removeprefix("src/")) == digest
                  for name, digest in identity["all_atomx_python_sha256"].items()))
    review = read(HERE / "dependency_review.json")
    check("native_CI_owner_matrix_gradient_optimizer_dependency_hashes", all(
          identity["all_atomx_python_sha256"][name] == digest
          for name, digest in review["dependency_module_hashes"].items()))
    prior = review["compared_frozen_current_proposal"]
    check("prior_current_epoch_proposal_unchanged_FAIL_not_upgraded",
          sha(REPO / prior["path"]) == prior["sha256"] and prior["currentregressionFAIL_remains"] is True)
    g, fd = (read(REPO / roles[k]["path"]) for k in ("gradient_result", "FD_result"))
    check("own438_730_live_controls_closed", len(g["checks"]) == 438 and len(fd["checks"]) == 730
          and g["all_declared_controls_passed"] is True and fd["all_declared_controls_passed"] is True
          and all(r["passed"] for result in (g, fd) for r in result["checks"]))
    check("own_single_direction_scope_not_whole_space_or_optimization_certificate",
          fd["single_direction_finite_measurement_only"] is True and fd["full31_space_certificate"] is False
          and all(r["expanded_MCDHF_optimized"] is False for r in (g, fd)))
    receipts = [read(REPO / roles[k]["path"]) for k in ("gradient_receipt", "FD_receipt")]
    check("own_receipts_foreground_closed_exit0", all(r["status"] == "foreground_closed"
          and r["exit_code"] == 0 for r in receipts))
    completion = read(REPO / roles["FD_completion"]["path"])
    check("own_FD_metadata_both_domains_closed", all(completion[k] is True for k in
          ("all_completion_metadata_checks_passed", "ordinary_both_h_passed",
           "independent_centered_both_h_both_references_passed")))

    old_guard = REPO / "docs/yb_mcdhf_rci_2026_10_11/accepted_line_search_metadata_implementation/exact_reported_guard.py"
    new_guard = REPO / roles["exact_guard"]["path"]
    restored = new_guard.read_text().replace("5-million", "3-million").replace("5_000_000", "3_000_000").replace("1595", "1046")
    check("exact_guard_only_dimension_resource_successor", restored == old_guard.read_text())
    check("single_public_default_optimizer_and_original_gates", sum(
          isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "optimize_orbitals"
          for n in ast.walk(tree)) == 1 and all(token in text for token in
          ("history_size=10, energy_difference=None", "tolerance=1e-7, energy_tolerance=1e-8",
           "allow_objective_change=False", "return_state=False")))
    check("source_package_selection_before_Atomx_import", execution_text.index("sys.path.insert")
          < execution_text.index("import atomx") and "Mutable or external Atomx import" in text)
    check("own_whole_H_E_C_owner_rawg_pg_baseline_controls", all(token in text for token in
          ("own_f043_baseline_bitwise_", "own_f043_baseline_owner", "own_f043_baseline_tracking",
           "own_f043_baseline_raw_coordinate_g", "own_f043_baseline_pg")))
    check("original_five_UID_lower_identity_and_NIST_only_reporting", "lower = by_label[\"6s_1/2\"]" in text
          and "lower_physical_UID" in text and contract["fixed"]["NIST_only_reporting"] is True)
    check("complete_unfiltered_signed_density_extra_gradients_not_replacements", all(token in text for token in
          ("contract_bilinear", "Fraction.from_float", "full original order then retained-only original order",
           "record_diagnostics=False", "no_gradient_or_energy_replacement=True",
           "full19589angularterms", "complete31x246_gradient")))
    counted = next(n for n in ast.walk(functions["execute"]) if isinstance(n, ast.FunctionDef) and n.name == "counted")
    counted_text = ast.unparse(counted)
    check("strict_pair_counters_no_perpair_disk_progress", "if name != 'gradient_pair_source':" in counted_text
          and "attempted[name] += 1" in counted_text and "completed[name] += 1" in counted_text
          and "if name == 'gradient_from_weights':" in counted_text and "resource_cap_" in counted_text)
    check("first_unqualified_candidate_and_prior_qualified_point_forensics", all(token in text for token in
          ("_before_fullsigned.npz", "accepted_candidate_before_gradient.json", "failed_CI_complete_H_",
           "first_failure_preserved_no_retry", "last_qualified_checkpoint")))
    check("no_preparation_actual_run", not (HERE / "actual").exists())
    proof = dict(all_source_only_checks_passed=True, checks=checks,
        driver_sha256=sha(HERE / "run.py"), contract_sha256=sha(HERE / "contract.json"),
        verification_script_sha256=sha(__file__), inputs_sha256=inputs,
        beforeloop_integer_resource_census=census, deferred_NPZ_inputs=deferred,
        source_map_sha256=identity["source_map_sha256"], numerical_imports=0,
        NPZ_reads_or_rehashes=0, H_CI_gradient_Dirac_optimizer_calls=0,
        no_runtime_or_convergence_claim=True)
    (HERE / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(checks=len(checks), all_passed=True, science_calls=0, NPZ_reads_or_rehashes=0)))


if __name__ == "__main__":
    main()
