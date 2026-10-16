"""Replay the archived correction ledger without rerunning a physical solver."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "same_orbital001"
CONTRACT = json.loads((HERE / "contract.json").read_text())
INPUT = ROOT / CONTRACT["fixed_conditions"]["input_run"]
FROZEN = ROOT / CONTRACT["fixed_conditions"]["frozen_source_root"]
sys.path.insert(0, str(FROZEN))

from atomx.constants import HARTREE_CM1  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None


def calculation_ast(path: Path) -> str:
    tree = WithoutImports().visit(ast.parse(path.read_text()))
    return ast.dump(tree, include_attributes=False)


def main() -> None:
    result = json.loads((OUT / "result.json").read_text())
    coverage = json.loads((OUT / "coverage_correction.json").read_text())
    source = json.loads((INPUT / "source_identity.json").read_text())
    checks: list[dict] = []

    def compare(name: str, value: float, expected: float, tolerance: float) -> None:
        error = float(abs(value - expected))
        checks.append({"name": name, "absolute_error": error, "limit": tolerance,
                       "passed": bool(np.isfinite(error) and error <= tolerance)})

    def check(name: str, passed: bool) -> None:
        checks.append({"name": name, "passed": bool(passed)})

    with np.load(OUT / "correction_arrays.npz") as arrays:
        for index, level in enumerate(result["state_correction_ledger"]):
            for name, ledger in level["components"].items():
                compare(f"absolute/{level['label']}/{name}", arrays[name][index],
                        ledger["absolute_Ha"], 1e-14)
                relative = (arrays[name][index] - arrays[name][0]) * HARTREE_CM1
                compare(f"transition/{level['label']}/{name}", relative,
                        ledger["relative_to_6s_cm1"], 1e-10)
        for label, upper, lower in (("5d_fine", 2, 1), ("6p_fine", 4, 3)):
            for name, value in result["splitting_correction_cm1"][label].items():
                compare(f"splitting/{label}/{name}",
                        (arrays[name][upper] - arrays[name][lower]) * HARTREE_CM1,
                        value, 1e-10)
        for name, left, right in (
            ("static_Breit", "energies_static_Breit_Ha", "energies_DC_Ha"),
            ("dynamic_Breit_total", "energies_dynamic_Breit_Ha", "energies_DC_Ha"),
            ("dynamic_minus_static_Breit", "energies_dynamic_Breit_Ha",
             "energies_static_Breit_Ha"),
        ):
            check(f"root_difference/{name}",
                  np.array_equal(arrays[name], arrays[left] - arrays[right]))
        check("all_raw_arrays_finite", all(np.isfinite(arrays[key]).all() for key in arrays.files))
        with np.load(INPUT / "ci_arrays.npz") as native_arrays:
            check("saved_same_state_DC_amplitudes",
                  np.array_equal(arrays["mixing_DC"], native_arrays["replay_dc_mixing"]))

    source_failures = [path for path, expected in source["all_atomx_python_sha256"].items()
                       if sha256(FROZEN / Path(path).relative_to("src")) != expected]
    input_failures = [path for path, expected in CONTRACT["inputs"].items()
                      if sha256(INPUT / path) != expected]
    check("all_353_frozen_source_hashes", not source_failures)
    check("all_seven_input_hashes", not input_failures)
    check("contract_hash_replay", sha256(HERE / "contract.json") == result["contract_sha256"])
    check("raw_arrays_hash_replay", sha256(OUT / "correction_arrays.npz")
          == result["correction_arrays_sha256"])
    check("original_script_calculation_ast_unchanged_after_import_cleanup",
          calculation_ast(OUT / "run_source.py") == calculation_ast(HERE / "run.py"))
    check("archived_numerical_gates_pass", all(item["passed"] for item in result["checks"]))
    check("failed_original_stationarity_retained", result["stationarity_passed"] is False)
    check("no_production_certification", result["production_certified"] is False)
    check("no_complete_corrected_spectrum_claim", result["no_complete_corrected_spectrum_claim"] is True)
    check("original_coverage_failure_retained",
          coverage["original_coverage_description_passed"] is False)
    check("coverage_correction_preserves_raw_values", coverage["numerical_values_changed"] is False)

    broken_links = []
    for target in re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text()):
        if "://" in target or target.startswith("#"):
            continue
        path = HERE / target.split("#", 1)[0]
        if path == HERE / "verification.json":
            continue  # This invocation creates the linked artifact.
        if not path.exists():
            broken_links.append(target)
    check("documentation_links", not broken_links)

    files = [p for p in HERE.rglob("*") if p.is_file()
             and p.name != "verification.json" and "__pycache__" not in p.parts]
    verification = {
        "passed": bool(all(item["passed"] for item in checks)),
        "scope": "Raw-array ledger replay, source/input identity, calculation-AST and documentation verification; not independent validation of model operators or experimental accuracy.",
        "archived_scientific_numeric_gate_count": len(result["checks"]),
        "replay_check_count": len(checks),
        "checks": checks,
        "broken_links": broken_links,
        "source_hash_failures": source_failures,
        "input_hash_failures": input_failures,
        "production_certified": False,
        "original_stationarity_passed": False,
        "original_coverage_description_passed": False,
        "corrected_coverage_owner": "same_orbital001/coverage_correction.json",
        "files_sha256": {str(p.relative_to(HERE)): sha256(p) for p in sorted(files)},
    }
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({key: verification[key] for key in (
        "passed", "archived_scientific_numeric_gate_count", "replay_check_count",
        "production_certified", "original_stationarity_passed")}, indent=2))
    if not verification["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
