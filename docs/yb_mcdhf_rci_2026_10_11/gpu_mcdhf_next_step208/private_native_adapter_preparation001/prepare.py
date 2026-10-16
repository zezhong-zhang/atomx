"""SOURCE/JSON/stdlib admission only, with actual31 signed table descriptors."""

import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
REPO = next(parent for parent in HERE.parents if (parent / "pyproject.toml").exists())
FORBIDDEN = {"numpy", "numba", "cupy", "atomx", "scipy"}


class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("numerical import forbidden in source preparation")
        return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_function(tree, dotted):
    node = tree
    for part in dotted.split("."):
        node = next(v for v in node.body if isinstance(v, (ast.ClassDef, ast.FunctionDef))
                    and v.name == part)
    return node


class PublicAlias(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="self", ctx=node.ctx), node) if node.id == "functional" else node

    def visit_AugAssign(self, node):
        if (isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Attribute)
                and node.target.value.attr == "counters"):
            return None
        return self.generic_visit(node)


def canonical_body(nodes):
    module = ast.Module(body=nodes, type_ignores=[])
    copy = ast.parse(ast.unparse(module))
    return ast.dump(PublicAlias().visit(copy), include_attributes=False)


def main():
    if any(n.split(".")[0] in FORBIDDEN for n in sys.modules):
        raise RuntimeError("clean standard-library process required")
    contract = json.loads((HERE / "contract.json").read_text())
    controls = []

    def check(name, value):
        controls.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    def refusal(name, call):
        try:
            call()
        except ValueError:
            check(name, True)
        else:
            check(name, False)

    for name, row in contract["inputs_sha256"].items():
        check("input_" + name, sha(REPO / row["path"]) == row["sha256"])
    identity = json.loads((HERE / "public_source_identity.json").read_text())
    current_map = {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "src/atomx").rglob("*.py"))}
    check("entire_public_source_unchanged", current_map == identity["all_atomx_python_sha256"])
    tree = ast.parse((HERE / "native_adapter.py").read_text())
    compile((HERE / "native_adapter.py").read_text(), str(HERE / "native_adapter.py"), "exec")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]]
            check(f"stdlib_top_import_{node.lineno}", not FORBIDDEN.intersection(modules))
    check("no_AST_text_execution", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                          and n.func.id in {"eval", "exec"} for n in ast.walk(tree)))
    check("no_H_CI_eigen_gradient_optimizer_solver_call", not any(
        isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        and (n.func.id if isinstance(n.func, ast.Name) else n.func.attr) in
        {"ci_solve", "build_ci_matrix", "eigh", "eigvalsh", "optimize_orbitals", "evaluate"}
        for n in ast.walk(tree)))
    public_tree = ast.parse((REPO / "src/atomx/methods/csf/variational.py").read_text())
    public = find_function(public_tree, "CSFOrbitalFunctional.gradient_from_weights")
    private = find_function(tree, "NativeBatchGradient.gradient_from_weights")
    public_h = next(n for n in ast.walk(public) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "key[0] == 'h'")
    private_h = next(n for n in ast.walk(private) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "key[0] == 'h'")
    check("public_one_body_updates_exact_except_counters", canonical_body(public_h.body) == canonical_body(private_h.body))
    public_actions = next(n for n in ast.walk(public) if isinstance(n, ast.For)
                          and ast.unparse(n.target) == "(left, right, weight)")
    private_actions = next(n for n in ast.walk(private) if isinstance(n, ast.For)
                           and ast.unparse(n.target) == "(left, right, weight)")
    check("public_actions_signed_adds_exact_except_counters", canonical_body(public_actions.body) ==
          canonical_body(private_actions.body))
    source_collector = find_function(tree, "collect_signed_layout")
    private_orientation = next(n for n in ast.walk(source_collector) if isinstance(n, ast.For)
                               and ast.unparse(n.target) == "(left, right, source_a, source_b)")
    public_orientation = next(n for n in ast.walk(public) if isinstance(n, ast.For)
                              and ast.unparse(n.target) == "(left, right, source_a, source_b)")
    check("same_two_derivative_orientation_order", ast.dump(private_orientation.iter, include_attributes=False) ==
          ast.dump(public_orientation.iter, include_attributes=False))
    check("no_private_density_filter_or_rescale", "abs(weight)" not in ast.unparse(source_collector)
          and "1e-15" not in (HERE / "native_adapter.py").read_text())
    check("existing_public_bands_no_radial_formula_copy", "self._density_sym(" in ast.unparse(private)
          and "_banded_to_full_add(" in ast.unparse(private))

    blocker = NoNumericalImports()
    sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location("private_native31_adapter_preparation", HERE / "native_adapter.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        engine = module.frozen_engine()
        check("literal_frozen_executor_reused", contract["executor_sha256"] == module.ENGINE_SHA256)
        native_contract = json.loads((REPO / contract["inputs_sha256"]["native31_contract"]["path"]).read_text())
        keys = tuple(tuple(k) for k in native_contract["fixed"]["keys31"])
        tables = json.loads((REPO / contract["inputs_sha256"]["native31_signed_tables"]["path"]).read_text())
        layout = module.collect_signed_layout(keys, tables["full_normalized"])
        row_counts = {tag: sum(k[0] == tag for k, _ in layout.items) for tag in ("h", "r")}
        exact = {
            "radial_keys": len(keys), "weights": len(layout.items), "weight_tag_counts": row_counts,
            "negative_weight_rows": sum(w < 0 for _, w in layout.items),
            "zero_weight_rows": sum(w == 0 for _, w in layout.items),
            "source_jobs": len(layout.jobs), "four_component_jobs": 4 * len(layout.jobs),
            "unique_source_pairs": len(tuple(dict.fromkeys((a, b) for _, a, b in layout.jobs))),
            "signed_derivative_incidence": sum(len(v) for _, v in layout.terms),
            "ranks": sorted({rank for rank, _, _ in layout.jobs}),
            "weights_sha256": layout.weights_sha256, "source_union_sha256": layout.source_union_sha256,
            "source_job_order": layout.jobs,
            "scope": "actual closed JSON union/order only; no bands, vectors or derivative recomputed",
        }
        actual = json.loads((REPO / contract["inputs_sha256"]["native31_result"]["path"]).read_text())
        check("closed_native31_result_passed_controls", actual["all_declared_controls_passed"])
        check("HF_partial_finite_response_pending_not_upgraded", actual["full_gradient_is_HF_partial"]
              and not actual["finite_CI_response_qualified"] and not actual["expanded_MCDHF_optimized"])
        role_layouts = {role: module.collect_signed_layout(keys, tables[role])
                        for role in ("retained", "full_minus_retained", "full_normalized")}
        by_role = {role: len(value.jobs) for role, value in role_layouts.items()}
        check("actual_three_independent_source_unions", by_role ==
              {"retained": 706, "full_minus_retained": 688, "full_normalized": 706})
        check("actual2100_total_is_sum_not_each_mean", sum(by_role.values()) ==
              actual["completed"]["gradient_pair_potential"])
        exact["three_independent_source_unions"] = by_role
        exact["individual_table_identities"] = {
            role: {"rows": len(value.items), "source_jobs": len(value.jobs),
                   "weights_sha256": value.weights_sha256,
                   "source_union_order_sha256": value.source_union_sha256,
                   "signed_incidence_order_sha256": module.digest(value.terms)}
            for role, value in role_layouts.items()}
        check("all19589_rows_kept_in_order", layout.items == tuple(
            (module.tuple_key(k), w) for k, w in tables["full_normalized"]))
        check("all39064_incidences_two_per_r_key", exact["signed_derivative_incidence"] == 2 * row_counts["r"])
        plan = module.plan_from_layout(layout, source_sha256=actual["source_map_sha256"],
                                       workspace_sha256="2" * 64, ns=123,
                                       orders=(9, 9), extents=(123, 123))
        work = engine.resource_preflight(plan)
        check("real2824_component_jobs_not13888_mechanical", work["component_jobs"] == 2824)
        check("workspace_hash_mock_explicit_not_physical", plan.workspace_sha256 == "2" * 64)

        a, b, c, d = keys[:4]
        mock_rows = ((('h', a[0], a[1], a[0]), 2.0),
                     (('r', 2, a, b, c, d), -3.0),
                     (('r', 0, c, c, a, a), 0.0))
        mock = module.collect_signed_layout(keys, mock_rows)
        check("signed_two_orientation_order", mock.terms[0] == ((2, *sorted((c, d))), ((a, b, -3.0),))
              and mock.terms[1] == ((2, *sorted((a, b))), ((c, d, -3.0),)))
        check("explicit_zero_input_not_filtered", len(mock.jobs) == 4 and mock.items[-1][1] == 0.0)
        refusal("partial_keys_refuse", lambda: module.collect_signed_layout(keys[:-1], mock_rows))
        refusal("duplicate_table_key_refuse", lambda: module.collect_signed_layout(keys, mock_rows + mock_rows[:1]))
        refusal("complex_coefficient_refuse", lambda: module.collect_signed_layout(keys, ((mock_rows[0][0], complex(2)),)))
        refusal("unknown_operator_refuse", lambda: module.collect_signed_layout(keys, ((('breit', 0), 1.0),)))
        refusal("negative_rank_refuse", lambda: module.collect_signed_layout(keys, ((('r', -1, a, b, c, d), 1.0),)))
        refusal("metadata_normalization_not_coerced", lambda: module.quantum_plain(1.5))
        check("native_tuple_list_quantum_identity", module.quantum_plain([[1, -1, 2], 1]) == ((1, -1, 2), 1))
        check("no_numerical_modules_after_mocks", not any(n.split(".")[0] in FORBIDDEN for n in sys.modules))
        (HERE / "actual_signed_union.json").write_text(json.dumps(exact, indent=2) + "\n")
        (HERE / "real_union_mechanical_preflight.json").write_text(json.dumps({
            "status": "pure_integer_upper_bounds_using_real_JSON_source_union",
            "real_union": work, "assumed_bound_orders": [9, 9], "assumed_bound_ns": 123,
            "actual_workspace_arrays_inspected": False,
            "component_orders_and_extents": "deferred; factory validates actual data; these are bounds only",
            "mechanical_full31_source_upper": 3472,
            "no_NPZ_read_or_array_allocation": True}, indent=2) + "\n")
    finally:
        sys.meta_path.remove(blocker)
    result = {"status": "source_JSON_adapter_preparation_only", "all_passed": all(c["passed"] for c in controls),
              "checks_count": len(controls), "checks": controls,
              "adapter_sha256": sha(HERE / "native_adapter.py"),
              "adapter_AST_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
              "contract_sha256": sha(HERE / "contract.json"), "prepare_sha256": sha(HERE / "prepare.py"),
              "actual_union_sha256": sha(HERE / "actual_signed_union.json"),
              "real_union_preflight_sha256": sha(HERE / "real_union_mechanical_preflight.json"),
              "actual_calls": {"NumPy_NumBa_Atomx_CUDA_imports": 0, "arrays_NPZ": 0,
                               "H_CI_g_Dirac_optimizer_compile_GPU_benchmark": 0},
              "native31_derivative_qualified": False, "GPU_qualified": False}
    (HERE / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_passed": result["all_passed"], "checks": len(controls),
                      "actual_sources": exact["source_jobs"], "actual_components": work["component_jobs"],
                      "science": 0}, indent=2))


if __name__ == "__main__":
    main()
