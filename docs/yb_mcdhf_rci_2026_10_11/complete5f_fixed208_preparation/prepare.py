"""Source/JSON/byte preparation proof only. Never imports or evaluates run.py."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE / "preparation.json"
    if out.exists():
        raise FileExistsError("Preserve preparation; use explicit successor if inputs change")
    contract = json.loads((HERE / "contract.json").read_text())
    models = json.loads((HERE / "models.json").read_text())
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    check("pending_not_executed", contract["status"] == "source_only_preparation_pending_exact_root_binding_no_scientific_execution")
    for name, expected in contract["inputs_sha256"].items():
        check("input/" + name, sha(ROOT / name) == expected)
    check("model_bytes", sha(HERE / "models.json") == contract["models_sha256"])
    identity = json.loads((ROOT / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    source_root = ROOT / contract["source"]["candidate_archive"]
    check("source356_identity", len(identity["all_atomx_python_sha256"]) == 356
          and identity["source_map_sha256"] == contract["fixed"]["source_map_sha256"])
    for name, expected in identity["all_atomx_python_sha256"].items():
        check("source/" + name, sha(source_root / Path(name).relative_to("src")) == expected)
    old = json.loads((ROOT / contract["input_roles"]["literal208_contract"]["path"]).read_text())
    manifest = json.loads((ROOT / contract["input_roles"]["literal208_checkpoint"]["path"]).read_text())
    check("literal208_no_selected_history_seed", manifest["global_accepted_step"] == contract["fixed"]["global_seed_step"] == 208)
    check("old29_exactkeys", contract["fixed"]["original29keys"] == old["fixed"]["keys29"])
    check("original_metric_exactSHA", manifest["metric_sha256"] == contract["fixed"]["checkpoint_metric_sha256"])
    check("original_two_gates_fail", manifest["stationarity_passed"] is False
          and manifest["diagnostics"]["orbital_residual"] > 1e-7
          and manifest["diagnostics"]["estimated_energy_error"] > 1e-8)
    check("full5f_91_NR", len(models["models"]["complete5f"]["configs"]) == 91)
    check("original69_prefix", models["models"]["complete5f"]["configs"][:69] == models["models"]["baseline"]["configs"])
    check("public_count_pending", models["models"]["complete5f"]["full_CSFs"] == 3789
          and models["models"]["complete5f"]["complete_target_CSFs"] == 1595)
    for row in models["models"]["complete5f"]["complete_target_blocks"]:
        check("integer_raw_within_original2000/" + str((row["twoJ"], row["parity"])), row["all_NR_patterns_M_equals_J_magnetic_dimension"] <= 2000)
    check("no_cap_relaxation", contract["resources"]["maximum_raw_determinants_per_block"] == 2000
          and contract["resources"]["maximum_signed_compatible_overlap_pairs_per_call"] == 1_000_000)
    check("all118_bounded", contract["numerical_method"]["complete_complement_dimensions"] == [118, 118]
          and contract["resources"]["numerical_method_caps"]["maximum_attempted_pair_rotations"] == 2 * 12 * (118 * 117 // 2))
    check("fullCI_and_realH_two_notfour", contract["resources"]["call_limits"]["H"] == 2
          and contract["resources"]["call_limits"]["CI"] == 2
          and contract["resources"]["call_limits"]["CI_internal_hS"] == 14)
    check("no_Dirac_Fock_g_opt", all(contract["resources"]["call_limits"][k] == 0 for k in ["Dirac", "Fock", "gradient", "optimizer"]))
    tree = ast.parse((HERE / "run.py").read_text())
    check("driver_AST_parsed_not_compiled", isinstance(tree, ast.Module))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("top_level_imports_stdlib", all(not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("numpy", "scipy", "atomx")))
          and not (isinstance(n, ast.Import) and any(a.name.startswith(("numpy", "scipy", "atomx")) for a in n.names)) for n in imports))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    check("authorize_metadata_no_science", not any(isinstance(n, (ast.Import, ast.ImportFrom)) for name in ("metadata", "authorize") for n in ast.walk(functions[name])))
    text = (HERE / "run.py").read_text()
    check("wholehelper_explicit_no_AST_extraction", 'runpy.run_path(str(roles["refinement_helper"])' in text
          and "ast.parse" not in text)
    check("phase_only_oldwaves_not_polar", "_polar_retraction" not in text and "oldwave_bytes" in text)
    check("fullH_and_signed_old_mapping", "retain_hamiltonian=True" in text and "nested_whole_signed1046_H" in text)
    check("array_reads_only_execute", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
          and n.func.attr == "load" for name in ("metadata", "authorize") for n in ast.walk(functions[name])))
    check("trigger_schema_saved_only", 'proof.get("message_id") != trigger["message_id"]' in text
          and 'proof.get("chat_id")' in text and 'proof.get("bot_open_id")' in text and "lark-cli" not in text)
    for doc in (HERE / "README.md", TOP / "README.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", doc.read_text()):
            if "://" not in target and not target.startswith("#"):
                check("local_link/" + str(doc.relative_to(ROOT)) + "/" + target, (doc.parent / target).exists())
    result = {"all_source_JSON_metadata_checks_passed": True, "checks_count": len(checks), "checks": checks,
        "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(HERE / "run.py"),
        "model_sha256": sha(HERE / "models.json"), "README_sha256": sha(HERE / "README.md"),
        "campaign_README_sha256": sha(TOP / "README.md"), "preparation_driver_sha256": sha(Path(__file__).resolve()),
        "source_map_sha256": contract["fixed"]["source_map_sha256"], "inputs_sha256": contract["inputs_sha256"],
        "Atomx_numpy_scipy_imports": 0, "scientific_producers": 0, "NPZ_array_reads": 0,
        "driver_execution_or_compilation": 0, "actual_public_CSF_preflight_executed": False,
        "resource_and_signed_overlap_numerics_pending": True, "execution_authorized": False,
        "scope": "Hashes, JSON/source AST and links only; no derived numerical object or scientific qualification"}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks_count": len(checks), "preparation_sha256": sha(out), "scientific_calls": 0}))


if __name__ == "__main__":
    main()
