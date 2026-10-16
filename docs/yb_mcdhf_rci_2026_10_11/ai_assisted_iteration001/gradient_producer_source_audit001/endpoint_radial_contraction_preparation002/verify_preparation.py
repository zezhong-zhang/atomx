"""SOURCE/JSON only proof; no array/header/hash or numerical execution."""

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
        raise ValueError("no NPZ access during SOURCE preparation")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "preparation.json").exists() or (HERE / "handoff.json").exists():
        raise FileExistsError("SOURCE preparation single-write")
    contract = json.loads((HERE / "contract.json").read_text())
    tree = ast.parse((HERE / "run.py").read_text())
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check("parse", True)
    literal_order = [
        row["reference_level"]
        for row in json.loads(
            (ROOT / contract["input_roles"]["mixed"]["path"]).read_text()
        )["targets"]
    ]
    previous = json.loads(
        (ROOT / contract["input_roles"]["predecessor_contract"]["path"]).read_text()
    )
    check(
        "SOURCE_literal_native_order_preflight",
        literal_order
        == contract["preflight_native_order"]
        == contract["fixed"]["five_UIDs"],
    )
    check(
        "same_physical_UID_set_resources_gates",
        set(map(tuple, literal_order))
        == set(map(tuple, previous["fixed"]["five_UIDs"]))
        and contract["resources"] == previous["resources"]
        and contract["original_gates"] == previous["original_gates"]
        and contract["archive_fields"] == previous["archive_fields"]
        and contract["source_map_sha256"] == previous["source_map_sha256"],
    )
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
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    previous_tree = ast.parse(
        (ROOT / contract["input_roles"]["predecessor_driver"]["path"]).read_text()
    )
    previous_execute = next(
        node
        for node in previous_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    check(
        "same_scientific_execute_AST",
        ast.dump(functions["execute"]) == ast.dump(previous_execute),
    )
    metadata = functions["metadata"]
    check(
        "metadata_no_numerical_import",
        not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(metadata)
        ),
    )
    check(
        "11_bound_arrays_no_preparation_census",
        sum(row["kind"] == "npz" for row in contract["input_roles"].values()) == 11
        and "zip_directory_census" not in contract,
    )
    check(
        "same1595_31_ns123_fiveUIDs",
        contract["fixed"]["CSFs"] == 1595
        and contract["fixed"]["orbitals"] == 31
        and contract["fixed"]["ns"] == 123
        and len(contract["fixed"]["five_UIDs"]) == 5
        and contract["fixed"]["weights"] == [0.2] * 5,
    )
    check(
        "exact19589_allstructural_keys",
        contract["fixed"]["angular_term_capacity"] == 19589,
    )
    cap = contract["resources"]
    check(
        "bounded_resources",
        cap["CPU"] == 29
        and cap["threads"] == 1
        and cap["AS_GiB"] == 4
        and cap["inner_seconds"] == 300
        and cap["outer_seconds"] == 315
        and cap["decoded_array_bytes_max"] == 128 * 2**20,
    )
    check(
        "bounded_complete_key_requests",
        cap["calls_exact"]["ordinary_radial_evaluation"] == 39178
        and cap["calls_exact"]["polynomial_key_request"] == 19589
        and cap["calls_exact"]["angular_forward_table"] == 10
        and cap["calls_exact"]["signed_target_density"] == 10,
    )
    check(
        "no_newH_CI_orhS",
        all(
            cap["calls_max"][key] == 0 for key in ("H", "CI_eigh", "native_hS_assembly")
        ),
    )
    check(
        "original_gates_unchanged",
        contract["original_gates"]["full_CI_residual_Ha"] == 1e-8
        and contract["original_gates"]["full_CI_Gram"] == 1e-10,
    )
    check(
        "geometry_complete_prefix_not_assumed_wholearchive",
        contract["resources"]["geometry_archive_fields_max"] == 128
        and len(contract["archive_fields"]["geometry_workspace"]) == 18,
    )
    command = [sys.executable, "-I", str(HERE / "run.py")]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    check("SOURCE_default_exit0", result.returncode == 0)
    check(
        "SOURCE_default_access_scope",
        "0 NPZ hashes/header/body access; 0 numerical imports/calls" in result.stdout,
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
        "links",
        all(
            link.startswith("http")
            or (HERE / link).exists()
            or link in ("preparation.json", "handoff.json")
            for link in links
        ),
    )
    check(
        "no_binding_actual_or_numericalimports",
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
        original_two_failures_preserved=True,
        certified_error_bound=False,
        source_default_stdout=result.stdout.strip(),
        Ruff=ruff.stdout.strip(),
    )
    (HERE / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
    handoff = dict(
        status="ROOT_review_and_exact_once_seal_required",
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
        NPZ_access_and_numerical_execution=0,
        root_prerequisites=[
            "Read exactcomplete driver/contract/source-only proof",
            "Bind all356frozenAtomx source files and allmetadata/NPZinputhashes afterseal",
            "Check hostload/availablememory/CPUaffinity before bounded foreground execution",
            "Write execution_binding.json fields execution_authorized_by_root,driver_sha256,contract_sha256,preparation_sha256,input_roles,resources",
            "Execute literal scientificinterpreter -I run.py --execute binding withthreadenvironment1, outer315s cap; preserveclosedreceipt/failure no retry",
        ],
    )
    (HERE / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                SOURCE_checks=len(checks),
                passed=True,
                new_physical_calls=0,
                NPZ_access=0,
            )
        )
    )


if __name__ == "__main__":
    main()
