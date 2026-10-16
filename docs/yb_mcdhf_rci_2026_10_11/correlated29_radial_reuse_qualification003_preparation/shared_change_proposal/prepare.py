"""Freeze the proposed shared change using stdlib source/metadata checks only."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
QUAL = BASE.parent
PATHS = (
    "src/atomx/methods/ci.py",
    "src/atomx/physics/csf/matrix.py",
    "tests/test_ci_dense_orthogonality.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    candidate = json.loads((QUAL / "candidate_source_identity.json").read_text())
    actual = json.loads((QUAL / "actual/result.json").read_text())
    completion = json.loads((QUAL / "completion.json").read_text())
    assert actual["all_equivalence_gates_passed"] is True
    assert completion["gate_counts"] == {
        "anchors": 406, "ordinary": 1294, "candidate": 241, "saved_comparison": 1126,
    }
    changes, patch = {}, []
    for name in PATHS:
        before, proposed = BASE / "before" / name, BASE / "proposed" / name
        assert before.read_bytes() == (REPO / name).read_bytes(), name
        ast.parse(proposed.read_text())
        if name.startswith("src/"):
            assert sha(proposed) == candidate["all_atomx_python_sha256"][name]
            assert proposed.read_bytes() == (
                QUAL / "candidate_source" / name.removeprefix("src/")
            ).read_bytes()
        changes[name] = {
            "before_path": str(before.relative_to(REPO)), "before_sha256": sha(before),
            "proposed_path": str(proposed.relative_to(REPO)),
            "proposed_sha256": sha(proposed),
        }
        patch.extend(difflib.unified_diff(
            before.read_text().splitlines(True), proposed.read_text().splitlines(True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    # The complete existing six-case test body is retained byte for byte.
    old_test = (BASE / "before" / PATHS[2]).read_text()
    new_test = (BASE / "proposed" / PATHS[2]).read_text()
    assert new_test.replace("import math\n", "", 1).startswith(old_test)
    assert len([n for n in ast.parse(new_test).body if isinstance(n, ast.FunctionDef)]) == 2
    (BASE / "changes.patch").write_text("".join(patch))
    current_map = {
        str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "src/atomx").rglob("*.py"))
    }
    proposed_map = dict(current_map)
    for name in PATHS[:2]:
        proposed_map[name] = sha(BASE / "proposed" / name)
    assert proposed_map == candidate["all_atomx_python_sha256"]
    map_hash = hashlib.sha256(json.dumps(proposed_map, sort_keys=True).encode()).hexdigest()
    # The qualification's producer specifies its canonical source-map serialization.
    assert len(proposed_map) == candidate["file_count"] == 356
    references = {
        str(p.relative_to(REPO)): sha(p) for p in (
            QUAL / "candidate_source_identity.json", QUAL / "ordinary_source_identity.json",
            QUAL / "contract.json", QUAL / "actual/result.json", QUAL / "actual/comparison.json",
            QUAL / "root_execution.json", QUAL / "completion.json", QUAL / "RESULTS.md",
            REPO / "tests/data/ne_clustered_ci_block.json",
            REPO / "tests/test_convol.py",
        )
    }
    contract = {
        "status": "source_only_proposed_requires_root_apply_and_test_seal",
        "shared_apply_authorized_by_this_file": False,
        "numerical_execution_authorized_by_this_file": False,
        "changes": changes,
        "changes_patch_sha256": sha(BASE / "changes.patch"),
        "qualified_private_source_map_sha256": candidate["source_map_sha256"],
        "proposed_all_atomx_python_sha256": proposed_map,
        "proposed_source_map_json_sort_keys_sha256": map_hash,
        "reference_files_sha256": references,
        "behavior": {
            "dense_CI": "nearest-integer first real diagonal scalar subtraction, same evd/evr and restore energies; original H remains owned unchanged",
            "matrix": "reuse existing complete PP/QQ/PQ/QP semiseparable factors only for ns>=96; ordinary bank route below threshold unchanged",
            "unchanged": ["public signatures", "Davidson", "Hamiltonian/model/nucleus/CSF and target identities", "all signed terms", "gradient and optimizer", "all numerical and experimental gates"],
        },
        "epoch_disposition": {
            "old_unshifted_to_centered": "source evolution; never bitwise energy replay",
            "new_epoch": "qualified003 has same-H centered energy bitwise replay and all1046 full spans passed",
            "old_route_failures": "qualification001 analytic FAIL and qualification002 cluster296 FAIL remain unchanged",
            "native_run005": "continues from its frozen f043 source/3451 metric; no consumer switch within an existing run",
            "old_tests": "immutable before copy retained; future identity validation must select the declared epoch rather than current shared path",
            "production_stationarity_experiment_speedup": "not certified by this patch or its unit tests",
        },
        "prospective_affected_test": {
            "path": PATHS[2], "old_cases": 6, "new_independent_signed_cases": 4,
            "expected_public_CI_unit_calls": 10, "expected_stub_matrix_calls": 10,
            "expected_scipy_eigh_calls": 16,
            "native_physical_H_gradient_Dirac_optimizer_GPU_calls": 0,
            "new_oracle": "closed-form signed 2x2 angle/hypot eigenvalues and projectors, complete and subset",
            "old_tolerances_preserved": True,
            "resources": {"CPU": 30, "threads": 1, "RSS_cap_GiB": 3.95,
                          "inner_seconds": 60, "outer_seconds": 65, "kill_after_seconds": 5,
                          "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "fresh_NUMBA_CACHE_DIR": True},
            "execution": "pending exact root seal; no test executed during preparation",
        },
        "preparation_calls": {"source_AST_JSON_SHA": True, "array_load_H_CI_gradient_spectral_optimizer_GPU": 0},
    }
    write(BASE / "contract.json", contract)
    write(BASE / "source_boundary.json", {
        "all_passed": True, "shared_current_matches_all_three_before_images": True,
        "two_proposed_sources_byte_identical_to_qualified003_candidate": True,
        "complete_source356_map_equals_qualified003_candidate": True,
        "legacy_six_testcases_body_byte_unchanged": True,
        "all_three_proposed_files_parse": True,
        "shared_source_and_test_writes": 0, "numerical_calls": 0,
        "inputs_sha256": references, "changes": changes,
    })
    write(BASE / "handoff.json", {
        "status": "reviewable_source_only_shared_patch",
        "files_sha256": {
            str(p.relative_to(REPO)): sha(p) for p in (
                BASE / "prepare.py", BASE / "changes.patch", BASE / "contract.json",
                BASE / "source_boundary.json", BASE / "documentation_impact.json", BASE / "README.md",
            )
        },
        "apply_instruction": "Root must verify every current destination matches before_sha256 before applying only changes.patch; archive the complete resulting source epoch before affected tests/consumers.",
        "pending": ["root apply/review seal", "affected tests", "controlled warm benchmark"],
    })


if __name__ == "__main__":
    main()
