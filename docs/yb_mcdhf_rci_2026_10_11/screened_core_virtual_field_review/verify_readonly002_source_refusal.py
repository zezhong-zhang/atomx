"""Metadata and AST verification only: never import atomx or numerical tools."""

import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    started = time.monotonic()
    contract = json.loads((HERE / "contract.json").read_text())
    limit = contract["resources"]["address_space_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[0]})
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(name)

    source_paths = [
        "src/atomx/methods/dirac_fock.py",
        "src/atomx/angular/coefficients.py",
        "src/atomx/angular/recoupling.py",
        "src/atomx/numerics/bspline/convol.py",
        "src/atomx/physics/csf/frozen_core.py",
        "src/atomx/physics/csf/radial_snapshot.py",
        "src/atomx/physics/csf/matrix.py",
        "src/atomx/methods/mbpt.py",
        "src/atomx/config/subshell.py",
    ]
    test_paths = [
        "tests/test_closed_shell_fock.py",
        "tests/test_dbsr_common_spinors.py",
        "tests/test_review_structure_numerics.py",
        "tests/test_configuration_spectroscopy.py",
        "tests/test_angular.py",
    ]
    identity_paths = [
        "docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol005_projected_increment/source_identity.json",
        "docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/source_identity.json",
        "docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run002/checkpoints/accepted_0160.json",
        "docs/yb_mcdhf_rci_2026_10_11/virtual_only_discriminator/contract.json",
    ]
    inputs = {path: sha(ROOT / path) for path in source_paths + test_paths + identity_paths}
    before = dict(inputs)
    epochs = [json.loads((ROOT / path).read_text()) for path in identity_paths[:2]]
    identities = []
    archive_root = ROOT / "docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/source"
    for path in source_paths:
        for i, epoch in enumerate(epochs):
            check("source_byte_same_epoch_%s/%s" % (i, path),
                  inputs[path] == epoch["all_atomx_python_sha256"][path])
        archived = archive_root / path.removeprefix("src/")
        check("source_archive_bytes/" + path, sha(archived) == inputs[path])
        tree = ast.parse((ROOT / path).read_text())
        saved = ast.parse(archived.read_text())
        check("source_archive_AST/" + path,
              ast.dump(tree, include_attributes=False) == ast.dump(saved, include_attributes=False))
        identities.append({"path": path, "sha256": inputs[path], "same_original005": True,
                           "same_run003_archive": True})

    df = ast.parse((ROOT / source_paths[0]).read_text())
    cls = next(node for node in df.body if isinstance(node, ast.ClassDef) and node.name == "DFSolver")
    names = ("_precompute_angular_coefficients", "_build_hf_matrix_dbsr",
             "_get_direct_coef", "_get_exchange_coef", "_mean_field_weight", "_mean_field_divisor")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names}
    check("adapter_six_method_definitions_present", set(methods) == set(names))
    targets = {node.attr for method in methods.values() for node in ast.walk(method)
               if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
               and node.value.id == "self" and isinstance(node.ctx, ast.Store)}
    reads = {node.attr for method in methods.values() for node in ast.walk(method)
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == "self" and isinstance(node.ctx, ast.Load)}
    external = reads - targets - set(names) - {"_get_WW"}
    expected = {"subshells", "mean_field_coeffs", "mean_field_normalization", "bsp_data"}
    check("adapter_exact_external_self_attributes", external == expected, sorted(external))
    check("adapter_cache_fallback_explicit", "_convol_cache" in reads and "_convol_cache" in targets)
    pre = methods["_precompute_angular_coefficients"]
    check("angular_schedule_unused", not any(isinstance(node, ast.Name) and node.id == "orbital_schedule"
                                               for node in ast.walk(pre)))
    source_text = (ROOT / source_paths[0]).read_text()
    check("builder_full_quadrant_default", "skip_pq_exchange=False" in source_text)
    method_text = ast.get_source_segment(source_text, methods["_build_hf_matrix_dbsr"])
    check("builder_no_onebody_or_dirac_call", not any(word in method_text
          for word in ("build_dirac_matrices(", "solve_dirac(", "nuclear_matrix(")))
    check("builder_threshold_division_declared", "if norm_i > 1e-15:" in method_text)
    check("builder_missing_orbital_zero_fill_declared", "orb_list = orbitals.get(sh.kappa, [])" in method_text
          and "if k_idx < len(orb_list):" in method_text)

    functions = {
        test_paths[0]: {"test_overlapping_core_and_rotated_signed_fock", "test_full_core_self_interaction_and_fractional_rejection"},
        test_paths[1]: {"test_same_spinor_final_fock_expectations"},
        test_paths[2]: {"test_mixed_basis_fock_is_derivative_of_hf_energy"},
        test_paths[3]: {"test_statistical_p2_fock_field_agrees_between_solver_modes"},
        test_paths[4]: {"test_ckm_obeys_the_wigner_eckart_sum_rule"},
    }
    anchors = []
    for path, required in functions.items():
        tree = ast.parse((ROOT / path).read_text())
        found = {node.name: node.lineno for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        check("test_anchor_definitions/" + path, required <= found.keys())
        anchors.extend({"path": path, "function": name, "line": found[name], "executed_by_this_review": False}
                       for name in sorted(required))
    search = subprocess.run(["rg", "-l", "mean_field_coeffs|mean_field_normalization", "tests", "-g", "*.py"],
                            cwd=ROOT, capture_output=True, text=True, timeout=4, check=False)
    check("no_direct_test_seam_keyword_matches", search.returncode == 1 and not search.stdout.strip())

    virtual_contract = json.loads((ROOT / identity_paths[3]).read_text())
    old = [tuple(key) for key in virtual_contract["fixed_conditions"]["radial_keys"]]
    core = [key for key in old if key[0] <= 4 or (key[0] == 5 and key[1] in (-2, -1, 1))]
    new = [tuple(key) for key in contract["fixed_conditions"]["new_zero_population_labels"]]
    check("old24_core19_disjoint_new5", len(old) == 24 and len(core) == 19 and len(new) == 5
          and not (set(new) & set(old)))
    check("closed_core_capacities_sum68", sum(2 * abs(k) for _, k in core) == 68)
    check("five_distinct_new_kappa", len({k for _, k in new}) == 5)
    check("synthetic_probe_pair_count80", 5 * 4 * 4 == 80)
    sources_after = {path: sha(ROOT / path) for path in before}
    check("all_reviewed_inputs_unchanged", sources_after == before)
    readme = (HERE / "README.md").read_text()
    import re
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        file_part = target.split("#", 1)[0]
        if file_part in ("review.json", "verification.json"):
            continue
        check("readme_link/" + target, (HERE / file_part).resolve().is_file())
    check("no_atomx_imported", not any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    review = {
        "record_id": contract["record_id"],
        "verdict": "source_algebra_supported_conditional_on_separate_runtime_qualification",
        "physical_or_driver_qualification": False,
        "actual_calls": contract["resources"],
        "source_epochs": [{"source_map_sha256": epoch["source_map_sha256"]} for epoch in epochs],
        "reviewed_source": identities,
        "existing_anchor_definitions_not_executed": anchors,
        "input_sha256": inputs,
        "closed_core_keys_and_capacity": [{"key": list(key), "occupation": 2 * abs(key[1])} for key in core],
        "new_zero_population_keys": [list(key) for key in new],
        "adapter_external_attributes": sorted(external),
        "consumer_prebound_matching_cache_required": True,
        "no_direct_seam_test_keyword_matches": True,
        "runtime_conditions": [
            "Finite C/D exact shapes, explicit C new-target/core-source rows only; D1 or declared finite scaled D exceeds division threshold.",
            "Every ordered core radial source and every old24 remains exact; no silent missing-source zero fill, floor, self-density, or renormalization.",
            "ConvolCache and workspace exact content match; C/D do not mutate after angular precompute.",
            "Save all PP/QQ/PQ/QP; compare 80 scalar signed entries and measured nonzero cross anchors; row scaling alone is insufficient.",
            "Add saved nuclear Dirac matrix once; no second field/one-body matrix and no auxiliary field inserted into CI.",
            "Choose new roots in original nuclear P complement orthogonal to all old same-kappa orbitals, not just 19 core sources.",
            "No MCDHF stationarity or experimental accuracy certification from auxiliary eigen-residuals; accepted160 original stationarity remains failed.",
        ],
        "limitations": [
            "No Fock, Slater, H, CI, gradient, Dirac or optimizer invocation and no tests executed in this audit.",
            "Only source/test algebra and identities reviewed; final future driver is not frozen or numerically accepted here.",
            "Closed68 auxiliary field omits valence/mixed-CI offdiagonal conditional density, Breit/QED/recoil; its fixed core is not independently self-consistent closed68 DHF.",
        ],
    }
    save("review.json", review)
    save("docimpact.json", {
        "source_changes": [], "site_changes": [], "existing_frozen_evidence_changes": [],
        "updated": ["screened_core_virtual_field_review/README.md"],
        "retained": [{"topic": "Previous accepted160/320, virtual/rank and website v4 archives", "reason": "Read-only auxiliary-field source audit changes no recorded physical result."}],
        "knowledge_disposition": "Task-scoped seam/normalization and missing-source guards recorded here; no global insight promotion before actual field qualification.",
    })
    elapsed = time.monotonic() - started
    save("verification.json", {
        "all_passed": all(row["passed"] for row in checks), "checks_passed": len(checks), "checks": checks,
        "inputs_sha256": inputs,
        "owned_inputs_sha256": {name: sha(HERE / name) for name in
                                ("contract.json", "verify_readonly.py", "README.md", "review.json", "docimpact.json",
                                 "readonly_failure001.json", "verify_readonly001_failed.py")},
        "resources": {"host_affinity_before": cpus, "chosen_cpu": cpus[0], "address_space_cap_bytes": limit,
                      "wall_elapsed_seconds": elapsed, "outer_timeout_seconds": 30,
                      "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "zero_physical_calls": True, "tests_executed": 0, "atomx_imported": False,
    })
    print(json.dumps({"all_passed": True, "checks_passed": len(checks), "wall_seconds": elapsed,
                      "review_sha256": sha(HERE / "review.json"), "verification_sha256": sha(HERE / "verification.json")}))


if __name__ == "__main__":
    main()
