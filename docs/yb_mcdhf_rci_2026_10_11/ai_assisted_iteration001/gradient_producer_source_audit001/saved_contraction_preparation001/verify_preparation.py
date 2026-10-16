"""SOURCE/ZIP directory verifier; numerical branches are never called."""

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
        raise ValueError("array hashing forbidden during preparation")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "preparation.json").exists() or (HERE / "handoff.json").exists():
        raise FileExistsError("preparation is single-write")
    c = json.loads((HERE / "contract.json").read_text())
    tree = ast.parse((HERE / "run.py").read_text())
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check("AST_parse", True)
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name.split(".")[0] for a in n.names]
                if isinstance(n, ast.Import)
                else [n.module.split(".")[0]]
            )
            check(
                "top_level_stdlib/" + ast.unparse(n),
                all(x in sys.stdlib_module_names for x in names),
            )
    check(
        "no_Atomx_import_anywhere",
        all(
            not isinstance(n, ast.ImportFrom) or not n.module.startswith("atomx")
            for n in ast.walk(tree)
        ),
    )
    check(
        "two_arrays_original_manifests",
        sum(v["kind"] == "npz" for v in c["input_roles"].values()) == 2,
    )
    expected = {"H.npy", "energies.npy", "mixing.npy", "keys.npy", "P.npy", "Q.npy"}
    check(
        "six_complete_ZIP_fields",
        all(
            {x["filename"] for x in row["entries"]} == expected
            and len(row["entries"]) == 6
            for row in c["zip_directory_census"].values()
        ),
    )
    check(
        "ZIP_uncompressed_bound",
        sum(
            x["uncompressed_bytes"]
            for row in c["zip_directory_census"].values()
            for x in row["entries"]
        )
        <= c["resources"]["decoded_array_bytes_max"],
    )
    check(
        "CPU28_threads1_original_caps",
        c["resources"]["CPU"] == 28
        and c["resources"]["threads"] == 1
        and c["resources"]["AS_GiB"] == 1.5
        and c["resources"]["inner_seconds"] == 120
        and c["resources"]["outer_seconds"] == 135,
    )
    check(
        "original_residual_Gram_unchanged",
        c["original_gates"]["full_CI_residual_Ha"] == 1e-8
        and c["original_gates"]["full_CI_Gram"] == 1e-10,
    )
    check(
        "complete1595_five_UIDs",
        c["fixed"]["CSFs"] == 1595
        and len(c["fixed"]["five_UIDs"]) == 5
        and c["fixed"]["weights"] == [0.2] * 5,
    )
    check(
        "bounded_complete_arithmetic",
        c["resources"]["whole_matrix_products_max"] == 4
        and c["resources"]["full_matrix_vector_products_max"] == 20
        and c["resources"]["five_full_signed_DeltaH_contractions_exact"] == 5,
    )
    check(
        "zero_new_physical_calls",
        all(
            c["resources"][k] == 0
            for k in (
                "new_H",
                "new_CI",
                "new_gradient",
                "new_Dirac",
                "new_optimizer",
                "new_radial_key_evaluations",
            )
        ),
    )
    command = "import sys,runpy,json;p=sys.argv[1];sys.argv=[p];runpy.run_path(p,run_name='__main__');print(json.dumps({'numerical_import':any(k=='numpy' or k.startswith('numpy.') or k=='atomx' or k.startswith('atomx.') for k in sys.modules)}))"
    default = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", command, str(HERE / "run.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    check("SOURCE_default_exit0", default.returncode == 0)
    check(
        "SOURCE_default_no_numerical_import",
        not json.loads(default.stdout.splitlines()[-1])["numerical_import"],
    )
    help_result = subprocess.run(
        ["/usr/bin/python3", "-I", str(HERE / "run.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    check("help_exit0", help_result.returncode == 0)
    ruff = subprocess.run(
        [
            str(ROOT / ".venv/bin/ruff"),
            "check",
            str(HERE / "run.py"),
            str(HERE / "verify_preparation.py"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    check("Ruff", ruff.returncode == 0)
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (HERE / "README.md").read_text())
    check(
        "local_links",
        all(
            p in ("preparation.json", "handoff.json") or (HERE / p).is_file()
            for p in links
        ),
    )
    check(
        "no_actual_or_binding",
        not (HERE / "actual").exists()
        and not (HERE / "execution_binding.json").exists(),
    )
    prep = dict(
        status="SOURCE_ZIP_directory_only_prepared_NOT_executed",
        all_SOURCE_checks_passed=True,
        checks=checks,
        driver_sha256=sha(HERE / "run.py"),
        contract_sha256=sha(HERE / "contract.json"),
        verifier_sha256=sha(__file__),
        README_sha256=sha(HERE / "README.md"),
        NPZ_body_header_reads_hashes=0,
        ZIP_directory_census_only=True,
        scientific_imports_calls=0,
        original_two_failures_preserved=True,
        zero_hypothesis_certified=False,
        Ruff=ruff.stdout.strip(),
    )
    (HERE / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    handoff = dict(
        status="READY_FOR_ROOT_REVIEW_ONLY",
        driver_sha256=prep["driver_sha256"],
        contract_sha256=prep["contract_sha256"],
        preparation_sha256=sha(HERE / "preparation.json"),
        verifier_sha256=prep["verifier_sha256"],
        resources=c["resources"],
        science_executed=False,
        root_binding_created=False,
        NPZ_body_header_reads_hashes=0,
        original_two_failures_preserved=True,
        zero_hypothesis="unresolved_no_certified_arithmetic_bound",
        all_SOURCE_checks_passed=True,
    )
    (HERE / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                checks=len(checks),
                driver=prep["driver_sha256"],
                contract=prep["contract_sha256"],
                preparation=handoff["preparation_sha256"],
                handoff=sha(HERE / "handoff.json"),
            )
        )
    )


if __name__ == "__main__":
    main()
