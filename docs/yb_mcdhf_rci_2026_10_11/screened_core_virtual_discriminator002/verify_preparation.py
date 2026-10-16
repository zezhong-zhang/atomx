"""Metadata/AST preparation verification; zero physical imports or evaluations."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    contract = json.loads((HERE / "contract.json").read_text())
    caps = contract["resources"]
    fixed = contract["fixed_conditions"]
    check("prepared_only", contract["status"] == "prepared_only_pending_NEW_exact_root_execution_binding")
    check("not_executed", not any((HERE / name).exists() for name in
          ("result.json", "failure.json", "execution_source.py", "resource_admission.json")))
    check("exact_accepted160", fixed["checkpoint"]["manifest_sha256"] ==
          "1ce49cd2bcc49bb22ed7bf21ec8f1e21fe3f37f945a2ef5ca3a3669e1a15c93c")
    check("whole353_epoch", fixed["full_atomx_source_count"] == 353 and fixed["source_map_sha256"] ==
          "10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f")
    check("inherited_stationarity_fail", fixed["checkpoint"]["stationarity_gates_passed"] is False)
    check("remaining_complete_sizes", contract["numerical_method"]["remaining_spectrum_sizes"] == [116, 114, 113, 114])
    check("complete_pair_budget", caps["maximum_attempted_pair_rotations"] ==
          12 * sum(math.comb(n, 2) for n in [116, 114, 113, 114]) == 310560)
    check("complete_residual_budget", caps["maximum_saved_residual_evaluations"] == 4 * (12 + 1) == 52)
    check("exact_four_SVD_gvd_refinement", caps["screened_complement_SVD_calls_budget"] ==
          caps["screened_original_P_full_gvd_spectral_controls"] == caps["full_spectrum_refinement_attempts"] == 4)
    check("two_completeCI_four_H", caps["complete_CI_calls_budget"] == 2 and
          caps["total_many_electron_H_builds_budget"] == 4)
    check("raw_publicCI_hS_explicit", caps["internal_CI_onebody_builds_budget"] == 28)
    check("zero_external_field_geometry_calls", all(caps[k] == 0 for k in
          ("screened_JK_builder_calls_budget", "scalar_Slater_calls_budget", "external_geometry_hS_build_calls",
           "initial_one_body_matrix_kappa_count", "public_virtual_generator_calls_budget",
           "nuclear_Dirac_spectral_calls", "orbital_gradients_budget", "optimizer_calls_budget")))
    check("unchanged_full_spectrum_gates", contract["acceptance"]["screened_full_spectrum_constrained_residual_relative"] ==
          1e-8 and contract["acceptance"]["screened_full_spectrum_SGram_error"] == 1e-10 and
          contract["acceptance"]["screened_complement_minimum_energy_match_Ha"] == 1e-8)
    helper = HERE / "refine_spectrum.py"
    check("helper_bound_exactly", contract["inputs_sha256"][str(helper.relative_to(ROOT))] == sha(helper))
    role = contract["saved_operator_roles"]["qualified_first_original_control"]
    check("first_complement_bound", sha(ROOT / role["path"]) == role["sha256"] and
          contract["inputs_sha256"][role["path"]] == role["sha256"])
    for name in ("run.py", "refine_spectrum.py"):
        compile((HERE / name).read_bytes(), str(HERE / name), "exec")
    check("compile_only", True)
    tree = ast.parse(helper.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("helper_no_physical_or_spectral_imports", all(
          isinstance(node, ast.ImportFrom) and node.module == "__future__" or
          isinstance(node, ast.Import) and all(alias.name == "time" for alias in node.names)
          for node in imports))
    source = (HERE / "run.py").read_text()
    check("no_new_Fock_functional_constructor", "DFSolver" not in source and "CSFOrbitalFunctional" not in source)
    check("publicCI_raw_onebody_forwarded", 'matrix_module.build_dirac_matrices = observed_ci_dirac' in source
          and 'value = original_call(*args, **kwargs)' in source)
    namespace = runpy.run_path(str(HERE / "run.py"), run_name="prepared_metadata_only")
    _, _, _, _, source_root = namespace["metadata"]()
    check("all_input_and_source_hashes_via_metadata", source_root.resolve() ==
          (ROOT / fixed["frozen_source_root"]).resolve())
    check("no_Atomx_or_numpy_scipy_imports", not any(name == "atomx" or name.startswith("atomx.") or
          name in {"numpy", "scipy"} for name in sys.modules))
    refusal_path = HERE / "rejected_binding_control.json"
    refusal_path.write_text('{"execution_authorized_by_root": false}\n')
    rejected = False
    try:
        namespace["authorize"](refusal_path)
    except ValueError as error:
        rejected = "Missing/mismatched NEW root binding" in str(error)
    check("pending_binding_rejected_before_science", rejected)
    lint = subprocess.run([str(ROOT / ".venv/bin/ruff"), "check", str(HERE / "run.py"), str(helper),
                           str(HERE / "verify_preparation.py")], capture_output=True, text=True, check=False)
    check("focused_Ruff", lint.returncode == 0)
    result = dict(status="metadata_AST_preparation_verified_no_execution", all_preparation_checks_passed=True,
                  checks=checks, source_map_sha256=fixed["source_map_sha256"],
                  driver_sha256=sha(HERE / "run.py"), contract_sha256=sha(HERE / "contract.json"),
                  helper_sha256=sha(helper), verification_driver_sha256=sha(Path(__file__)),
                  physical_imports=0, scientific_calls=0, eigensolves=0,
                  Ruff_output=lint.stdout.strip(), pending_exact_new_root_binding=True)
    (HERE / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=len(checks), physical_imports=0, scientific_calls=0,
                         verification_sha256=sha(HERE / "preparation_verification.json"))))


if __name__ == "__main__":
    main()
