"""Bound source/JSON admission; no numerical imports, arrays or science."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def sha(path):
    if Path(path).suffix == ".npz":
        raise ValueError("Preparation cannot read or hash NPZ")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    code = (HERE / "run.py").read_text()
    tree = ast.parse(code)
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check("runtime_not_executed", not (HERE / "actual").exists())
    check("concrete_not_yet_authorized", contract["executor_implemented"] is True
          and contract["execution_authorized"] is False)
    roles = contract["input_roles"]
    check("all_source_JSON_roles_SHA", all(r["deferred_numerical"] or sha(ROOT / r["path"]) == r["sha256"]
          for r in roles.values()))
    identity = json.loads((ROOT / roles["source_identity"]["path"]).read_text())
    source = ROOT / contract["source_archive"]
    check("all356_f043_source_files_bound", len(identity["all_atomx_python_sha256"]) == 356
          and identity["source_map_sha256"] == contract["fixed"]["source_map_sha256"]
          and all(sha(source / n.removeprefix("src/")) == s for n, s in identity["all_atomx_python_sha256"].items()))
    numeric_roles = {k for k, r in roles.items() if r["deferred_numerical"]}
    check("exact_five_deferred_archives", numeric_roles == set(contract["array_access"])
          and numeric_roles == {"literal13_arrays", "stageA_direction", "geometry_workspace", "screened_geometry", "tracking_ci"})
    native = json.loads((ROOT / roles["native_result"]["path"]).read_text())
    point = json.loads((ROOT / roles["literal13_manifest"]["path"]).read_text())
    stageA = json.loads((ROOT / roles["stageA_result"]["path"]).read_text())
    check("closed_literal13_actual_qualified_point", native["all_completed_controls_passed"] is True
          and point["iteration"] == 13 and point["qualified"] is True
          and roles["literal13_arrays"]["sha256"] == point["array_sha256"])
    check("closed200_replay_does_not_certify_stageB", stageA["all_declared_replay_controls_passed"] is True
          and len(stageA["checks"]) == 200 and all(r["passed"] for r in stageA["checks"])
          and stageA["summary"]["future_stageB_qualified"] is False
          and stageA["summary"]["direction_array_sha256"] == roles["stageA_direction"]["sha256"])
    check("original_physical_identity_and_nonstationarity_retained", point["metric_sha256"] == contract["fixed"]["new_metric_sha256"]
          and point["geometry_sha256"] == contract["fixed"]["new_geometry_sha256"]
          and point["objective_identity"] == contract["fixed"]["objective_identity"]
          and point["source_map_sha256"] == contract["fixed"]["source_map_sha256"]
          and point["original_stationarity_passed"] is False)
    tables = json.loads((ROOT / roles["literal13_tables"]["path"]).read_text())

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    terms, pairs = {}, {}
    for kind in ("retained", "full_minus_retained", "full_normalized"):
        weights = {tuples(k): v for k, v in tables[kind]}
        sources = set()
        for key in weights:
            if key[0] == "h":
                continue
            _, rank, a, b, c, d = key
            for x, y in ((a, b), (c, d)):
                sources.add((rank, *sorted((x, y))))
        terms[kind], pairs[kind] = len(weights), len(sources)
    check("complete_signed_literal13_terms_and_source_census", terms == contract["resources"]["baseline_gradient_terms"]
          and pairs == contract["resources"]["baseline_gradient_sources"]
          and sum(len(t["unfiltered"]) for t in tables["targets"]) == 67655)
    caps, census = contract["resources"]["calls"], contract["primitive_census"]
    check("primitive_work_admitted_before_loop", sum(pairs.values()) == census["gradient_source_pairs_total"] == caps["gradient_pair_potential"] == 2118
          and sum(pairs.values()) * 4 == census["gradient_components_total"] == caps["gradient_components"] == 8472
          and 4 * 19589 == census["finite_component_upper"] == caps["finite_component_cache_miss"]
          and caps["finite_potential_cache_miss"] == 4 * 19589 * 3
          and caps["finite_density_cache_miss"] == 4 * 19589 * 6
          and 'Before physical loops: complete primitive/resource census mismatch' in code)
    check("five_HCI_three_gradient_purpose_not_redundant", caps["H"] == caps["CI"] == 5
          and caps["gradient_from_weights"] == 3 and caps["retained_RDM"] == 1
          and caps["unfiltered_target_RDM"] == 25 and all(s in code for s in
          ('functional.evaluate(base_waves, ci_result=base_ci)',
           'functional.gradient_from_weights(orbitals, delta, record_diagnostics=False)',
           'functional.gradient_from_weights(orbitals, full, record_diagnostics=False)')))
    check("full_tracker_spectral_hS_budget", caps["tracker_resolve"] == 19 and caps["tracking_SVD"] == 95
          and caps["tracker_overlap"] == 20 and caps["validation_Dirac"] == 7
          and caps["complete_SciPy_eigh"] == 46 and caps["geometry_SPD_eigvalsh"] == 38
          and caps["hS_total"] == 70 and caps["polar_retraction"] == 28
          and caps["optimizer"] == caps["GPU"] == 0)
    old = ast.parse((ROOT / roles["frozen_FD_driver"]["path"]).read_text())
    loop_identity = dump(ast.parse("enumerate(fixed['h'])", mode="eval").body)
    old_loops = [n for n in ast.walk(old) if isinstance(n, ast.For)
                 and dump(n.iter) == loop_identity]
    new_loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and dump(n.iter) == loop_identity]
    check("both_complete_FD_loops_exact_frozen_AST", len(old_loops) == len(new_loops) == 2
          and all(dump(a) == dump(b) for a, b in zip(old_loops, new_loops, strict=True)))
    reused = {}
    for name in ("point_constraints", "matching", "fresh", "f", "mul", "dot", "vector", "sparse", "action", "difference", "add", "rational", "matrix_display"):
        a = next(n for n in ast.walk(old) if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        check("exact_frozen_consumer_AST_" + name, dump(a) == dump(b))
        reused[name] = hashlib.sha256(dump(b).encode()).hexdigest()
    nsrc = ast.parse((ROOT / roles["frozen_native_driver"]["path"]).read_text())
    a = next(n for n in ast.walk(nsrc) if isinstance(n, ast.FunctionDef) and n.name == "vectors")
    b = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "vectors")
    check("exact_native_metric_vector_consumer_AST", dump(a) == dump(b))
    check("exact_real_owner_literal13_and_signed_table_controls", all(s in code for s in
          ('own_recovered31_baseline_bitwise_', 'same_literal13_UID_selection_', 'same_own_baseline_owner',
           'whole_fresh_full_table_identity', 'fresh_literal13_rawg', 'fresh_literal13_pg')))
    check("fixed_whole_StageA_direction_no_Wolfe_claim", all(s in code for s in
          ('StageA_same_literal13_x', 'StageA_original_transport_exact', 'StageA_normalization_exact',
           'old_alpha1_path_derivative_or_Wolfe_qualified=False'))
          and contract["direction"]["old_alpha1_path_derivative_or_Wolfe_test"] is False)
    check("same_original_gates_both_h_and_domains", contract["fixed"]["h"] == [1e-5, 5e-6]
          and contract["gates"]["FD_absolute"] == 1e-8 and contract["gates"]["FD_relative"] == 1e-5
          and 'for label, reference in (("fixedLambda_ideal", centered_analytic), ("public_retained", retained_analytic))' in code)
    check("no_new_kernel_or_objective_callback", 'energy_difference=' not in code
          and 'no_energy_callback=True' in code and 'functional.mixed_ci_energy_difference(' in code
          and 'optimizer", 0' in code)
    check("native_dimensions_and_budgeted_inputs", contract["fixed"]["coordinate_dimension"] == 3689
          and contract["fixed"]["target_CSFS"] == 1595 and contract["fixed"]["radial_orbitals"] == 31
          and len(contract["array_access"]["stageA_direction"]["fields"]) == 4
          and census["five_H_C_archives_uncompressed_bytes"] == 203522000
          and contract["resources"]["maximum_total_input_decoded_bytes"] == 192 * 2**20)
    check("resource_caps_stop_not_completion_claim", contract["resources"]["CPU"] == 31
          and contract["resources"]["threads"] == 1 and contract["resources"]["AS_GiB"] == 3.95
          and contract["resources"]["inner_seconds"] == 300 and contract["resources"]["outer_seconds"] == 315)
    check("quiet_per_source_counter_and_failure_preservation", '"gradient_pair_potential", "gradient_components"' in code
          and 'failed_preserved_no_retry' in code and 'all four finite ledgers preserved' in code)
    proof = dict(status="SOURCE_JSON_AST_only_stageB_preparation", all_source_JSON_preparation_checks_passed=True,
                 checks=checks, checks_count=len(checks), driver_sha256=sha(HERE / "run.py"),
                 contract_sha256=sha(HERE / "contract.json"), verifier_sha256=sha(__file__),
                 unchanged_frozen_consumer_AST_sha256=reused, full_signed_table_census=dict(terms=terms,sources=pairs),
                 numerical_imports=0, NPZ_reads_or_hashes=0, physical_calls=0,
                 execution_authorized=False, actual_runtime=None, derivative_qualification_claimed=False)
    (HERE / "preparation.json").write_text(json.dumps(proof,indent=2) + "\n")
    print(json.dumps(dict(status=proof["status"], checks=len(checks), all_passed=True,
          driver_sha256=proof["driver_sha256"], contract_sha256=proof["contract_sha256"])))


if __name__ == "__main__":
    main()
