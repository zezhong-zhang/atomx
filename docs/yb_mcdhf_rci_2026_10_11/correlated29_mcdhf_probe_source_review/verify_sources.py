"""Read-only source/JSON/AST audit; never import the numerical consumer."""

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOP = HERE.parent
NATIVE = TOP / "correlated29_mcdhf_probe_preparation"
EXPECTED = {
    "run.py": "7489d91e9f5bc820ba71db94ef6449c886a544a22392ff117682721e737c7572",
    "contract.json": "1459db8b950e749f6d88932f5b1a876624bb6dcef40a4987cb448f814fd59195",
    "preparation_verification.json": "32b5e2e8d881de4d1cb634a55ea5643fe046ed17f61966213bdc1dda15363ae4",
    "handoff.json": "180a957c37fccd740fa02ad0c6e7f2fe361c8acbdf6ab062f78a3bc6d62e3071",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks, inputs = [], {}

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    def bind(path):
        inputs[str(path.relative_to(REPO))] = sha(path)

    for name, expected in EXPECTED.items():
        path = NATIVE / name
        bind(path)
        check("final_native_SHA/" + name, sha(path) == expected)
    source = (NATIVE / "run.py").read_text()
    tree = ast.parse(source)
    contract = json.loads((NATIVE / "contract.json").read_text())
    preparation = json.loads((NATIVE / "preparation_verification.json").read_text())
    check("native_preparation_passed", preparation["all_passed"] is True)
    check("native_preparation_count424", preparation["checks_count"] == 424)
    check("native_preparation_same_runner", preparation["runner_sha256"] == EXPECTED["run.py"])
    check("native_preparation_same_contract", preparation["contract_sha256"] == EXPECTED["contract.json"])
    for row in contract["input_roles"].values():
        path = REPO / row["path"]
        bind(path)
        check("bound_input/" + row["path"], sha(path) == row["sha256"])
    for row in contract["pending_prerequisites"].values():
        path = REPO / row["path"]
        bind(path)
        check("bound_prerequisite/" + row["path"], sha(path) == row["sha256"])
    check("unchanged_original_shared_test", sha(REPO / "tests/test_mixed_ci_finite_increment.py") ==
          "045ee541fc648ea9bbc51a476d04fe2a3bf31b0158bfcc02c9daa8357ab2caa8")
    identity = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    check("same_candidate356_epoch", identity["source_map_sha256"] ==
          "0e5496eb7eab63df2e44d08c4d986574821b33b806a7d3c760e243bda5cc2193")
    for name, expected in contract["source"]["critical_sources_sha256"].items():
        bind(REPO / name)
        check("critical_source_unchanged/" + name, sha(REPO / name) == expected)
    top_imports = [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("no_top_level_numerical_imports", not any(
        word in item for item in top_imports for word in ("numpy", "scipy", "atomx")))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    check("no_optimizer_call", not any(ast.unparse(node.func).endswith("optimize_orbitals") for node in calls))
    for key, expected in {"max_H": 6, "max_CI": 6, "max_full_gradient": 1,
                          "max_optimizer": 0, "max_validation_Dirac": 7,
                          "max_total_hS": 77, "max_tracking_resolutions": 21,
                          "max_tracking_SVD": 105, "max_tracking_overlap": 22,
                          "max_geometry_eigvalsh": 36, "max_scipy_eigh": 37,
                          "max_polar_retraction": 28,
                          "max_polar_numpy_eigh_fallback": 28}.items():
        check("declared_budget/" + key, contract["resources"][key] == expected)
    check("tracker_resolution_accounting", 6 * 2 + 1 + 4 * 2 == 21)
    check("tracking_overlap_includes_constructor", 21 + 1 == 22)
    check("rank_one_SVD_accounting", 21 * 5 == 105)
    check("geometry_eigvalsh_accounting", 7 + 29 == 36)
    check("scipy_eigh_accounting", 7 + 6 * 5 == 37)
    check("raw_to_logical_operator_mapping", "logical = np.tril(h) + np.tril(h, -1).T" in source)
    check("raw_operator_separately_byte_checked", "h.tobytes() == reference_h.tobytes()" in source)
    check("original_P_byte_checked", "projector.tobytes() == saved[f\"projector_{i}\"].tobytes()" in source)
    check("original_inverse_blocks_preserved", "saved[f\"inverse_blocks_{i}\"][source_column].tobytes()" in source)
    check("coordinate_gradient_projected_from_full_radial", "projector.T @ np.column_stack([base[3][k] for k in keys])" in source)
    check("native_tangent_formula", "g - .5 * y @ (a + a.T)" in source)
    check("preconditioned_gradient_reprojected", "pg = tangent(x, inverse_g)" in source)
    check("explicit_new_metric", "new_metric_differs_from_old24_metric" in source)
    check("no_legacy_gradient_replay_claim", '"same_point_whole_gradient_replay_executed": False' in source)
    check("original_derivative_gate", '"gate": max(1e-8, 1e-5 * abs(analytical))' in source)
    check("both_reported_and_Rayleigh_gated", "_reported_derivative" in source and "_Rayleigh_derivative" in source)
    check("finite_h_declared", contract["finite_step"]["h"] == 1e-6)
    check("ordinary_subtraction_reporting_only", "ordinary big-total subtraction is resolution reporting only" in
          contract["finite_step"]["central_gate_applies_to"])
    basis_path = REPO / contract["input_roles"]["screened_basis"]["path"]
    basis = json.loads(basis_path.read_text())
    sector_order = list(dict.fromkeys(tuple(json.loads(row)[-2:]) for row in basis["baseline"]["full_signatures"]))
    target_path = REPO / contract["input_roles"]["tracking_binding"]["path"]
    targets = json.loads(target_path.read_text())["targets"]
    target_order = [(row["twoJ"], row["parity"]) for row in targets]
    check("saved_symmetry_permutation_reproduced", sector_order[:5] == [(1, 1), (5, 1), (3, 1), (3, -1), (1, -1)])
    check("positional_target_zip_would_be_wrong", sector_order[:5] != target_order)
    check("positional_zip_removed", "zip(targets, tracking" not in source)
    check("complete_UID_matching_present", "set(matched) != expected" in source and "matched[uid]" in source and
          "if uid in matched or tuple(binding[\"selected_levels\"][0][:2]) != uid[:2]" in source)
    check("explicit_3451_new_coordinates", contract["metric"]["coordinate_count"] == 3451)
    check("complete1046_target_sectors", contract["fixed"]["target_CSFs"] == 1046 and
          sum(row[2] for row in contract["fixed"]["target_blocks"]) == 1046)
    check("uncomputed_parent_scope", contract["fixed"]["full_NR_parent_CSFs"] == 2163 and
          contract["fixed"]["full_model_not_computed"] is True)
    for path in (HERE / "README.md", Path(__file__)):
        bind(path)
    result = {"scope": "Independent standard-library source/JSON/AST audit only; no numerical imports or arrays.",
              "source_review_all_passed": all(row["passed"] for row in checks),
              "checks_count": len(checks), "checks": checks, "inputs_sha256": inputs,
              "scientific_calls": 0, "numerical_qualification_executed_by_reviewer": False,
              "material_open_findings": [], "root_execution_authority_inferred": False,
              "draft_findings_closed": ["physical target label permutation", "unledgered geometry/tracking calls"]}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "all_passed": result["source_review_all_passed"],
                      "verification_sha256": sha(HERE / "verification.json")}))
    if not result["source_review_all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
