"""Read-only frozen-evidence, local-link and doc-only identity verification."""

import ast
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    checked = 0
    for group in (
        "evidence_sha256",
        "setup_and_bookkeeping_failures_retained",
        "owner_documentation_sha256",
    ):
        for path, expected in summary[group].items():
            assert digest(OUT / path) == expected, (group, path)
            checked += 1
    assert digest(OUT / summary["independent_oracle"]["review_path"]) == summary[
        "independent_oracle"
    ]["review_sha256"]
    frozen_checks = 0
    for contract_name in (
        "contract.json",
        "four_core_contract.json",
        "localization_contract.json",
        "quotient_contract.json",
    ):
        contract = json.loads((OUT / contract_name).read_text())
        for path, expected in contract["source_sha256"].items():
            candidate = OUT / "source" / path
            if not candidate.exists():
                candidate = OUT / path
            assert digest(candidate) == expected, (contract_name, path)
            frozen_checks += 1
    chain = summary["doc_only_identity_chain"]
    frozen = OUT / "source" / chain["current_source"]
    current = REPO / chain["current_source"]
    assert digest(frozen) == chain["before_sha256"]
    assert digest(current) == chain["current_sha256"]
    assert digest(OUT / "core_sd_before_docstring.py") == chain["before_sha256"]
    frozen_ast = ast.dump(WithoutDocstrings().visit(ast.parse(frozen.read_text())))
    current_ast = ast.dump(WithoutDocstrings().visit(ast.parse(current.read_text())))
    assert frozen_ast == current_ast
    unchanged_current_sd_files = []
    for name in (
        "methods/monovalent_sd.py",
        "methods/reduced_monovalent_sd.py",
        "methods/linearized_sd.py",
        "angular/sd_rho.py",
    ):
        path = Path("src/atomx") / name
        assert digest(REPO / path) == digest(OUT / "source" / path), str(path)
        unchanged_current_sd_files.append(str(path))
    links = []
    for link in re.findall(r"\[[^\]]+\]\(([^)\n]+)\)", (OUT / "README.md").read_text()):
        assert (OUT / link.split("#", 1)[0]).exists(), link
        links.append(link)
    files_to_check = ["README.md", "trace.md", "verify_report.py"]
    for name in files_to_check:
        assert all(line == line.rstrip() for line in (OUT / name).read_text().splitlines()), name
    result = dict(
        pass_=True,
        read_only_verification=True,
        summary_sha256=digest(OUT / "summary.json"),
        verifier_sha256=digest(Path(__file__)),
        evidence_hash_checks=checked + 1,
        frozen_source_hash_checks=frozen_checks,
        current_SD_files_matching_frozen_sha256=unchanged_current_sd_files,
        non_docstring_AST_identical=True,
        local_link_count=len(links),
        missing_links=[],
        numerical_or_orbital_solves_in_verification=0,
        original_failure_preserved=True,
        scope="Integrity and doc-only identity; no new physics or performance certificate",
    )
    (OUT / "final_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
