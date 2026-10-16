"""SOURCE and bound JSON only; arrays and scientific imports forbidden."""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def sha(path):
    path = Path(path)
    if path.suffix == ".npz":
        raise ValueError("no NPZ preparation access")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (
        not (HERE / "preparation.json").exists()
        and not (HERE / "handoff.json").exists()
    )
    contract = json.loads((HERE / "contract.json").read_text())
    tree = ast.parse((HERE / "run.py").read_text())
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check("AST_parse", True)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name.split(".")[0] for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0]]
            )
            check(
                "top_level_stdlib/" + ast.unparse(node),
                all(name in sys.stdlib_module_names for name in names),
            )
    metadata = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "metadata"
    )
    check(
        "metadata_no_imports",
        not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(metadata)
        ),
    )
    check(
        "exact_two_fullCI_archives_no_geometry",
        {key for key, row in contract["input_roles"].items() if row["kind"] == "npz"}
        == {"baseline_arrays", "trial_arrays"},
    )
    fixed, cap = contract["fixed"], contract["resources"]
    check(
        "unchanged_complete1595_31_ns123_fiveUID",
        fixed["CSFs"] == 1595
        and fixed["orbitals"] == 31
        and fixed["ns"] == 123
        and fixed["weights"] == [0.2] * 5
        and len(fixed["five_UIDs"]) == 5,
    )
    check(
        "native_UID_order_preflight",
        fixed["five_UIDs"]
        == contract["preflight_native_order"]
        == [[1, 1, 0], [5, 1, 0], [3, 1, 0], [3, -1, 0], [1, -1, 0]],
    )
    check(
        "all19589keys_and_original_gates",
        fixed["angular_term_capacity"] == 19589
        and contract["original_gates"]["full_CI_residual_Ha"] == 1e-8
        and contract["original_gates"]["full_CI_Gram"] == 1e-10,
    )
    check(
        "CPU30_AS4_90_105caps",
        cap["CPU"] == 30
        and cap["threads"] == 1
        and cap["AS_GiB"] == 4
        and cap["inner_seconds"] == 90
        and cap["outer_seconds"] == 105
        and cap["decoded_array_bytes_max"] == 128 * 2**20,
    )
    check(
        "complete_tables_work_caps",
        cap["angular_nnz_max"] == 12000000
        and cap["angular_pairs_max"] == 2000000
        and cap["calls_exact"]["forward_table_product"] == 20
        and cap["calls_exact"]["signed_target_vectors"] == 5
        and cap["calls_exact"]["signed_row_contraction"] == 125,
    )
    check(
        "zero_producer_caps",
        all(
            cap["calls_max"][key] == 0
            for key in (
                "H",
                "CI",
                "hS",
                "radial",
                "functional",
                "gradient",
                "optimizer",
                "eigh",
            )
        ),
    )
    command = [sys.executable, "-I", str(HERE / "run.py")]
    default = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    check("SOURCE_default_fullmetadata_exit0", default.returncode == 0)
    check(
        "SOURCE_default_accessscope",
        "0 NPZ access, numerical imports or scientific calls" in default.stdout,
    )
    help_result = subprocess.run(
        command + ["--help"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    check("help_exit0", help_result.returncode == 0)
    ruff = subprocess.run(
        [
            str(ROOT / ".venv/bin/ruff"),
            "check",
            "--no-cache",
            str(HERE / "run.py"),
            str(HERE / "verify_preparation.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    check("Ruff", ruff.returncode == 0)
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", (HERE / "README.md").read_text())
    check(
        "local_links",
        all(
            link.startswith("http")
            or (HERE / link).exists()
            or link in ("preparation.json", "handoff.json")
            for link in links
        ),
    )
    check(
        "no_actual_binding_imports",
        not (HERE / "actual").exists()
        and not (HERE / "execution_binding.json").exists()
        and not any(
            name == "atomx" or name.startswith("atomx.") or name in ("numpy", "scipy")
            for name in sys.modules
        ),
    )
    proof = dict(
        status="SOURCE_JSON_only_prepared_NOT_executed",
        all_SOURCE_checks_passed=True,
        checks=checks,
        driver_sha256=sha(HERE / "run.py"),
        contract_sha256=sha(HERE / "contract.json"),
        verifier_sha256=sha(__file__),
        README_sha256=sha(HERE / "README.md"),
        NPZ_stat_hash_header_ZIP_body_access=0,
        numerical_imports_calls=0,
        prior_failures_preserved=True,
        certified_error_bound=False,
        default_stdout=default.stdout.strip(),
        Ruff=ruff.stdout.strip(),
    )
    (HERE / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
    handoff = dict(
        status="ROOT_review_exact_seal_required",
        execution_authorized=False,
        artifacts_sha256={
            name: sha(HERE / name)
            for name in (
                "run.py",
                "contract.json",
                "verify_preparation.py",
                "README.md",
                "preparation.json",
            )
        },
        resources=cap,
        input_roles=contract["input_roles"],
        NPZ_access_and_numerical_calls=0,
        root_prerequisites=[
            "Review full exact source and contract and independent SOURCE review",
            "Bind complete356f043source and every metadata/archive input hash; preserve ancestor failures",
            "Write execution binding driver/contract/preparation hashes,input_roles,resources,execution_authorized_by_root",
            "Foreground literal scientificinterpreter -I run.py --execute binding, threads1, inner90s outer105s; preserve closed receipt/failure without retry",
        ],
    )
    (HERE / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                SOURCE_checks=len(checks), passed=True, NPZ_access=0, scientific_calls=0
            )
        )
    )


if __name__ == "__main__":
    main()
