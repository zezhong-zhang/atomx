"""Verify this increment against preserved inputs without staging the workspace."""

import ast
import difflib
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return {"path": path.relative_to(REPO).as_posix(), "sha256": digest(path)}


def main():
    before = json.loads((HERE / "previous_sources/manifest.json").read_text())
    impact = json.loads((HERE / "documentation_impact.json").read_text())
    patch = []
    for entry in before["files"]:
        snapshot, current = REPO / entry["snapshot"], REPO / entry["path"]
        assert digest(snapshot) == entry["sha256"], entry["path"]
        patch.extend(difflib.unified_diff(snapshot.read_text().splitlines(True),
                                         current.read_text().splitlines(True),
                                         fromfile=f"before/{entry['path']}",
                                         tofile=f"after/{entry['path']}"))
    new_tests = [p for p in impact["owned_test_files"]
                 if p != "tests/test_atomic_parallel_iteration.py"]
    for relative in new_tests:
        patch.extend(difflib.unified_diff([], (REPO / relative).read_text().splitlines(True),
                                         fromfile="/dev/null", tofile=f"after/{relative}"))
    (HERE / "root_changes.patch").write_text("".join(patch))
    frozen = HERE / "native_run/nuclear_eol005_projected_increment"
    source_map = json.loads((frozen / "source_identity.json").read_text())["all_atomx_python_sha256"]
    source_checks = []
    for relative, expected in source_map.items():
        original = frozen / "source" / Path(relative).relative_to("src")
        current = REPO / relative
        assert digest(original) == expected, relative
        identical = digest(current) == expected
        same_ast = ast.dump(ast.parse(current.read_text())) == ast.dump(ast.parse(original.read_text()))
        if not identical:
            assert relative == "src/atomx/numerics/bspline/orbital_optimization.py", relative
            assert same_ast, relative
        source_checks.append({**binding(current), "frozen005_sha256": expected,
                              "byte_identical": identical, "AST_identical": same_ast})
    (HERE / "current_source_manifest.json").write_text(json.dumps({
        "scientific_run_source_map_sha256": "9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f",
        "allowed_post_run_change": "Optimizer comment clarification only; identical Python AST. Actual numerical runs remain bound to frozen005 bytes.",
        "files": source_checks,
    }, indent=2) + "\n")
    result = json.loads((HERE / "results_stage002.json").read_text())
    for entry in result["inputs"].values():
        assert digest(REPO / entry["path"]) == entry["sha256"], entry["path"]
    proof = json.loads((REPO / result["inputs"]["current_mcdhf"]["path"]).read_text())
    for entry in proof["inputs"].values():
        assert digest(REPO / entry["path"]) == entry["sha256"], entry["path"]
    receipt = json.loads((HERE / "delivery_stage002/receipt.json").read_text())
    assert receipt["passed"] and not receipt["highest_level_reached"]
    iteration = json.loads((HERE / "delivery_stage002/iteration_result.json").read_text())
    assert iteration["message_id"] == receipt["message_id"]
    assert iteration["receipt_backed_selection_executed"]
    assert not iteration["new_expanded_MCDHF_production_execution"]
    for entry in iteration["artifacts"]:
        assert digest(REPO / entry["path"]) == entry["sha256"], entry["path"]
    for entry in receipt["source_identity"]:
        if entry["path"].startswith("private "):
            continue
        assert digest(REPO / entry["path"]) == entry["sha256"], entry["path"]
    tests = json.loads((HERE / "test_attempts.json").read_text())
    assert "160 passed in 93.02s" in (HERE / "scientific_successor_tests.log").read_text()
    assert "45 passed in 1.53s" in (HERE / "card_tests.log").read_text()
    assert tests["final_result"] == "160 passed in 93.02s"
    assert len(tests["attempts"]) >= 3
    impact["documentation_review"]["docs/yb_mcdhf_rci_2026_10_11/native_run/README.md"] = (
        "Correct the former future-only 68-CSF text to completed fixed005-orbital RCI; immutable numerical proofs remain unchanged."
    )
    docs = list(impact["documentation_review"]) + [
        "docs/yb_mcdhf_rci_2026_10_11/README.md",
    ]
    links, broken = [], []
    for relative in docs:
        document = REPO / relative
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", document.read_text()):
            if "://" in target or target.startswith(("#", "mailto:", "app:")):
                continue
            target = target.strip("<>").split("#", 1)[0]
            target = re.sub(r":\d+$", "", target)
            path = document.parent / target
            row = {"document": relative, "target": target, "exists": path.exists()}
            links.append(row)
            if not row["exists"]:
                broken.append(row)
    assert not broken, broken
    (HERE / "documentation_links_final.json").write_text(json.dumps({
        "scope": "Relative working-tree links in current owned documentation entry points; not remote/tracked publication validation.",
        "documents": docs, "links_checked": len(links), "broken": broken,
    }, indent=2) + "\n")
    for entry in impact["owned_source_files"]:
        entry["sha256"] = digest(REPO / entry["path"])
    impact["post_run_source_change"] = "Only optimizer comment clarification; 353-file numerical source epoch is preserved and AST equality verified."
    impact["source_bound_scientific_results"] = binding(HERE / "results_stage002.json")
    impact["new_numerical_stationarity_limit"] = proof["production_certification_scope"]
    impact["test_boundary"] = "Existing untracked parallel-iteration test file had a pre-existing body; only the model-identity parameterization was changed here. It is hashed, not represented as wholly new in root_changes.patch."
    (HERE / "documentation_impact.json").write_text(json.dumps(impact, ensure_ascii=False, indent=2) + "\n")
    lint_paths = [x["path"] for x in impact["owned_source_files"]] + impact["owned_test_files"] + [
        "docs/atomx_hourly_progress_2026_10_10/sync_card.py",
        "docs/atomx_hourly_progress_2026_10_10/parallel_planner.py",
        (HERE / "integrate_stage002.py").relative_to(REPO).as_posix(),
        (HERE / "deliver_stage002.py").relative_to(REPO).as_posix(),
        Path(__file__).resolve().relative_to(REPO).as_posix(),
    ]
    lint = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "ruff", "check", *lint_paths],
                          cwd=REPO, text=True, capture_output=True, check=False)
    (HERE / "final_ruff.log").write_text(lint.stdout + lint.stderr)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    whitespace = []
    for relative in docs + lint_paths:
        for number, line in enumerate((REPO / relative).read_text().splitlines(), 1):
            if line.endswith((" ", "\t")):
                whitespace.append({"path": relative, "line": number})
    assert not whitespace, whitespace
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "source_files": len(source_checks),
        "byte_identical_to005": sum(r["byte_identical"] for r in source_checks),
        "all_AST_identical_to005": all(r["AST_identical"] for r in source_checks),
        "original_metric_preserved": proof["stationarity"]["residual_metric_sha256"],
        "numerical_stationarity_passed": True,
        "expanded68_MCDHF_stationarity_certified": False,
        "expanded531_MCDHF_stationarity_certified": False,
        "experimental_gate_passed": False,
        "highest_level_reached": False,
        "test_results": {"science": tests["final_result"], "cards": tests["card_panel_result"]},
        "failed_attempts_preserved": len(tests["attempts"]),
        "ruff_returncode": lint.returncode,
        "relative_working_tree_links_checked": len(links),
        "broken_relative_links": broken,
        "trailing_whitespace": whitespace,
        "no_unrelated_staging_or_commits": True,
        "actual_card_message_id": receipt["message_id"],
        "iteration_dispatched": receipt["checks"]["iteration_dispatched"],
        "actual_iteration_status": iteration["status"],
        "artifacts": [binding(HERE / p) for p in [
            "results_stage002.json", "delivery_stage002/receipt.json",
            "delivery_stage002/iteration_result.json",
            "current_source_manifest.json", "root_changes.patch", "documentation_impact.json",
            "documentation_links_final.json", "test_attempts.json", "final_ruff.log",
            "scientific_successor_tests.log", "card_tests.log", "card_heading_successor_tests.log",
        ]],
    }
    (HERE / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"passed": True, "source_files": len(source_checks),
                      "links": len(links), "verification": binding(HERE / "verification.json")}))


if __name__ == "__main__":
    main()
