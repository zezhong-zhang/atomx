"""Source/JSON checks only; never inspect NPZ bytes or import numerical code."""
import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before_modules = set(sys.modules)
    driver = HERE / "run.py"
    tree = ast.parse(driver.read_text())
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {node.module.split(".")[0] for node in imports if isinstance(node, ast.ImportFrom)}
    names |= {alias.name.split(".")[0] for node in imports if isinstance(node, ast.Import) for alias in node.names}
    check("top_level_imports_standard_library_only", names <= sys.stdlib_module_names)
    spec = importlib.util.spec_from_file_location("prospective208_preparation_only", driver)
    module = importlib.util.module_from_spec(spec)
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    sys.dont_write_bytecode = original_bytecode
    contract = module.load(HERE / "contract.json")
    module.metadata(contract, archives=False)
    check("source_JSON_admission_passed", True)
    check("no_numerical_imports", not any(name == "numpy" or name.startswith("numpy.")
          or name == "atomx" or name.startswith("atomx.") for name in set(sys.modules) - before_modules))
    mock_spec = importlib.util.spec_from_file_location("exact_capture_mock", HERE / "check_instrumentation.py")
    mock = importlib.util.module_from_spec(mock_spec)
    mock_spec.loader.exec_module(mock)
    mock_result = mock.exercise(module)
    module.write(HERE / "instrumentation_mock_verification.json", mock_result)
    check("exact_new_helper_frozen_mock_passed", mock_result["all_mock_checks_passed"] is True)
    parent = HERE.parent
    prior_contract = module.load(parent / "contract.json")
    for field in ("fixed", "metric", "point", "direction", "gates", "source", "resources", "HF_prerequisite"):
        check("inherited_unchanged_" + field, contract[field] == prior_contract[field])
    prior_result = module.load(parent / "actual/result.json")
    check("failed_original_instrumentation_not_derivative", prior_result["error"] == "FrozenInstanceError: cannot assign to field 'resolve'"
          and len(prior_result["controls"]) == 58 and all(row["passed"] for row in prior_result["controls"])
          and prior_result["completed"]["initial_hS"] == 7 and prior_result["completed"]["tracking_overlap"] == 1
          and all(prior_result["completed"][name] == 0 for name in ("H", "CI", "gradient", "scipy_eigh", "polar", "Dirac", "optimizer")))
    original_source, new_source = (parent / "run.py").read_text(), driver.read_text()
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install_resolution_capture"]
    check("one_explicit_capture_helper", len(helpers) == 1)
    helper_source = ast.get_source_segment(new_source, helpers[0])
    roundtrip = new_source.replace(helper_source + "\n\n\n", "", 1)
    roundtrip = roundtrip.replace('REPO = next(parent for parent in HERE.parents if (parent / "pyproject.toml").is_file())', 'REPO = HERE.parents[2]', 1)
    old_capture = '        old_resolve = tracker.resolve\n\n        def resolve(*args, **kwargs):\n            result = old_resolve(*args, **kwargs)\n            capture["weights"], capture["bindings"] = result\n            return result\n\n        tracker.resolve = resolve'
    roundtrip = roundtrip.replace('        install_resolution_capture(CIBasisTracking, tracker, capture)', old_capture, 1)
    check("exact_only_instrumentation_and_ancestor_diff", roundtrip == original_source)
    check("no_object_setattr_bypass", not any(isinstance(node, ast.Attribute) and node.attr == "__setattr__" for node in ast.walk(tree)))
    roles = contract["input_roles"]
    checked, inherited = {}, {}
    for name, role in roles.items():
        if role["path"].endswith(".npz"):
            inherited[name] = role
            continue
        actual = sha(REPO / role["path"])
        check("bound_source_JSON_" + name, actual == role["sha256"])
        checked[name] = actual
    check("literal208_archive_hash_from_closed_manifest_only",
          module.load(REPO / roles["literal208_manifest"]["path"])["array_sha256"]
          == roles["literal208_arrays"]["sha256"])
    check("no_scientific_output_or_attempt", not any((HERE / name).exists() for name in
          ("actual", "execution_contract.json", "once_marker.json")))
    limits = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "limits" for t in node.targets):
            limits = ast.literal_eval(node.value)
    check("native_producer_caps_frozen", limits is not None and limits["H"] == limits["CI"] == 5
          and limits["gradient"] == 2 and limits["optimizer"] == limits["Dirac"] == 0)
    check("hS_producer_caps_upper_bound_not_CI_equality", limits["initial_hS"] + limits["CI_internal_hS"] + limits["mixed_hS"] == 70)
    check("complete_replay_and_polar_caps", limits["scipy_eigh"] == 50 and limits["polar"] == 28 and limits["polar_eigh"] == 28)
    check("separate_request_producer_caps", limits["retained_H_getter_requests"] == 21 and limits["H"] == 5)
    check("complete_signed_counter_caps", limits["tracking_resolve"] == 13 and limits["tracking_SVD"] == 65
          and limits["tracking_overlap"] == 14 and limits["bilinear_RDM"] == 25)
    check("original_directional_tolerances", contract["gates"]["centered_derivative_abs"] == 1e-8
          and contract["gates"]["centered_derivative_rel"] == 1e-5)
    check("fixed_two_scales", contract["direction"]["fixed_h"] == [1e-5, 5e-6])
    check("original_stationarity_failure_preserved", contract["point"]["stationarity_passed"] is False
          and contract["point"]["original_gates"] == {"orbital_residual": 1e-7, "estimated_energy_error_Ha": 1e-8})
    check("main_campaign_frozen_source_not_new_shared_epoch", contract["source"]["current_shared_source_consumed"] is False
          and contract["source"]["map_sha256"] == "f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe")
    lint = subprocess.run([str(REPO / ".venv/bin/ruff"), "check", str(driver), str(HERE / "prepare.py"), str(HERE / "check_instrumentation.py")],
                          capture_output=True, text=True, check=False)
    check("Ruff_passed", lint.returncode == 0)
    module.write(HERE / "preparation_verification.json", {
        "all_source_and_JSON_checks_passed": True, "checks": checks,
        "driver_sha256": sha(driver), "contract_sha256": sha(HERE / "contract.json"),
        "preparation_driver_sha256": sha(__file__), "findings_sha256": sha(HERE.parent / "findings.json"),
        "source_map_sha256": contract["source"]["map_sha256"],
        "source_archive_files_checked": len(module.load(REPO / roles["candidate_source_identity"]["path"])["all_atomx_python_sha256"]),
        "source_JSON_roles_sha256": checked, "inherited_NPZ_hash_declarations_not_recomputed": inherited,
        "NPZ_bytes_read": 0, "NumPy_Atomx_imports": 0, "scientific_calls": 0,
        "syntax_AST_and_bound_literal_caps_only": True,
        "ruff": {"exit_code": lint.returncode, "stdout": lint.stdout, "stderr": lint.stderr},
        "instrumentation_mock_sha256": sha(HERE / "instrumentation_mock_verification.json"),
        "instrumentation_mock_driver_sha256": sha(HERE / "check_instrumentation.py"),
        "exact_diff_sha256": sha(HERE / "changes.diff"),
        "prior_failure_result_sha256": sha(HERE.parent / "actual/result.json"),
        "source_only_readiness_is_not_scientific_qualification": True})
    print(f"{len(checks)} source/JSON controls passed; no NPZ or scientific execution")


if __name__ == "__main__":
    main()
