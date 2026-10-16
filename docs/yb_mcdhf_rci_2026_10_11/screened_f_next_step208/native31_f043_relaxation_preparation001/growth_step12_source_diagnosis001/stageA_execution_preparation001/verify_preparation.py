"""Source and JSON admission only; never open an array or import NumPy."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    if Path(path).suffix == ".npz":
        raise ValueError("Preparation forbids NPZ reads and hashes")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    driver = (HERE / "run.py").read_text()
    tree = ast.parse(driver)
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check("new_runtime_not_executed", not (HERE / "actual").exists())
    check("source_only_prepared_not_authorized", contract["executor_implemented"] is True
          and contract["execution_authorized"] is False)
    check("all_metadata_inputs_exact_SHA", all(sha(REPO / r["path"]) == r["sha256"]
          for r in contract["metadata_inputs"].values()))
    source = (REPO / contract["metadata_inputs"]["optimizer_source"]["path"]).read_text()
    extract = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "source_sections")
    namespace = dict(ast=ast)
    exec(compile(ast.Module(body=[extract], type_ignores=[]), "source_AST_extractor_only", "exec"), namespace)
    sections = namespace["source_sections"](source)
    fingerprints = {name: hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]),
                    include_attributes=False).encode()).hexdigest()
                    for name, nodes in sections.items()}
    names = [n.name for n in sections["functions"]]
    check("exact_original_five_numerical_functions", names == ["_residual_metric_sha256",
          "_polar_retraction", "tangent", "spectral_precondition", "retract"])
    selected = ast.Module(body=[*sections["functions"], *sections["direction"], *sections["update"]],
                          type_ignores=[])
    check("selected_AST_no_import_or_physical_producer", not any(isinstance(n, (ast.Import, ast.ImportFrom))
          or isinstance(n, ast.Call) and any(token in ast.unparse(n.func)
          for token in ("evaluate", "ci_solve", "solve_dirac", "optimize_orbitals", "build_ci"))
          for n in ast.walk(selected)))
    update = ast.unparse(ast.Module(body=sections["update"], type_ignores=[]))
    direction = ast.unparse(ast.Module(body=sections["direction"], type_ignores=[]))
    check("native_chronological_transport_curvature_and_capacity", all(s in update for s in
          ("tangent(x, old_s)", "tangent(x, old_y)", "s = tangent(x, x - old_x)",
           "y = g - tangent(x, old_g)", "1e-12", "history[-history_size:]")))
    check("native_signed_two_loop_and_non_descent_reset", all(s in direction for s in
          ("reversed(history)", "precondition(q)", "np.dot(y, direction)",
           "direction = -tangent(x, direction)", "history.clear()")))
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    check("runtime_numpy_only_no_Atomx_or_SciPy", "numpy" in imported and not any(
          s.startswith(("atomx", "scipy", "numba", "cupy", "torch")) for s in imported))
    check("mandatory_separate_exact_root_binding", all(s in driver for s in
          ("execution_authorized_by_root=True", "preparation_sha256=sha", "max_H=0, max_CI=0",
           "max_gradient=0, max_Dirac=0, max_optimizer=0", 'output.mkdir(exist_ok=False)')))
    points, resource = contract["points"], contract["resources"]
    check("literal_chronology0_to13", [p["index"] for p in points] == list(range(14)))
    fields = [spec for p in points for spec in p["fields"].values()]
    decoded = sum(8 * product(spec["shape"]) for spec in fields)
    check("exact94_fields_7544472bytes_census", len(fields) == resource["fields_decoded_exact"] == 94
          and decoded == resource["decoded_bytes_exact"] == 7544472
          and decoded <= resource["decoded_bytes_max"] == 2**23)
    check("native_float64_or_int64_only", all(s["dtype"] in ("float64", "int64") for s in fields))
    core_fields = {"x", "keys", "retained_radial", "raw_coordinate_gradient", "tangent_gradient",
                   "preconditioned_gradient"}
    check("only_saved_numerical_members_and_endpoint_x", all(set(p["fields"]) == core_fields
          for p in points[1:13]) and set(points[13]["fields"]) == {"x"}
          and len(points[0]["fields"]) == 21)
    manifests = [json.loads((REPO / p["manifest_path"]).read_text()) for p in points]
    check("closed_qualified_manifest_identity_and_array_SHA_deferred", all(
          m["iteration"] == p["index"] and m["qualified"] is True and
          m["array_sha256"] == p["array_sha256"] and p["deferred_no_NPZ_read_or_rehash_in_preparation"]
          for m, p in zip(manifests, points, strict=True)))
    fixed = contract["fixed"]
    check("original_metric_geometry_and_objective_in_all_manifests", all(
          m["diagnostics"]["residual_metric_sha256"] == fixed["metric_sha256"] and
          m["diagnostics"]["geometry_sha256"] == fixed["geometry_sha256"] and
          m["diagnostics"]["objective_identity"] == fixed["objective_identity"] for m in manifests))
    check("original13_primary_no_callback_acceptances", all(
          m["diagnostics"]["accepted_line_search"]["attempt"] == 0 and
          m["diagnostics"]["accepted_line_search"]["stable_callback_used"] is False
          for m in manifests[1:]))
    check("history_starts_empty_and_no_missing_state", fixed["initial_history_empty"] is True
          and fixed["history_size"] == 10 and 'history=[], history_size=10' in driver
          and 'for index in range(13)' in driver)
    check("explicit_small_polar_budget_and_no_physical_spectrum", resource["small_coordinate_Gram_numpy_eigh_max"] == 7
          and resource["small_Gram_shape_max"] == [7, 7]
          and resource["H_CI_newgradient_Dirac_optimizer_calls"] == 0
          and 'np.linalg.eigh = small_eigh' in driver)
    check("native_pair_work_census", resource["native_transported_old_pairs_max"] == sum(min(i, 10) for i in range(12)) == 65
          and resource["native_two_loop_pair_visits_max"] == 2 * sum(min(i, 10) for i in range(13)) == 150
          and resource["chronological_new_pair_admissions_max"] == 12)
    check("declared_CPU_memory_wall_thread_caps", resource["CPU"] == 31 and resource["threads"] == 1
          and resource["AS_GiB"] == 1 and resource["inner_seconds"] == 30 and resource["outer_seconds"] == 45
          and resource["interpreter"] == "/data/code/atomx/.venv/bin/python")
    check("preserve_first_failure_and_no_stageB", all(s in driver for s in
          ("first_offline_failure_preserved_no_retry", "failed_replay_coordinates.npz",
           "future_stageB_qualified=False", "no_automatic_stageB_or_continuation=True")))
    proof = dict(status="SOURCE_JSON_AST_preparation_only", all_source_only_checks_passed=True,
                 checks=checks, checks_count=len(checks), driver_sha256=sha(HERE / "run.py"),
                 contract_sha256=sha(HERE / "contract.json"), verifier_sha256=sha(__file__),
                 exact_selected_source_AST_sha256=fingerprints, unresolved_missing_numerical_state=False,
                 numerical_imports=0, NPZ_reads_or_hashes=0, array_execution=0, physical_calls=0,
                 runtime_replay_executed=False, runtime_controls_passed=None)
    (HERE / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(status=proof["status"], checks=len(checks), all_passed=True,
                         driver_sha256=proof["driver_sha256"], contract_sha256=proof["contract_sha256"])))


def product(shape):
    count = 1
    for n in shape:
        count *= n
    return count


if __name__ == "__main__":
    main()
