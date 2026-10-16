"""Source/JSON/stdlib mocks only. Never invoke a numerical executor factory."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
REPO = next(parent for parent in HERE.parents if (parent / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def symbol(tree, name):
    node = tree
    for part in name.split("."):
        node = next(value for value in node.body
                    if isinstance(value, (ast.FunctionDef, ast.ClassDef)) and value.name == part)
    return node


def semantic(node):
    # Discard only a leading docstring; all numerical statements/decorators stay.
    node = ast.parse(ast.unparse(node)).body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


def main():
    forbidden = {"numpy", "numba", "cupy", "atomx", "scipy"}
    if any(name.split(".")[0] in forbidden for name in sys.modules):
        raise RuntimeError("preparation must run in a clean standard-library process")
    contract = json.loads((HERE / "contract.json").read_text())
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    for label, row in contract["inputs_sha256"].items():
        check(f"input_{label}", sha(REPO / row["path"]) == row["sha256"])
    public_map = {str(path.relative_to(REPO)): sha(path)
                  for path in sorted((REPO / "src/atomx").rglob("*.py"))}
    identity = json.loads((HERE / "public_source_identity.json").read_text())
    check("entire_public_source_epoch_unchanged", public_map == identity["all_atomx_python_sha256"])
    check("public_source_map_hash", hashlib.sha256(json.dumps(
        public_map, sort_keys=True).encode()).hexdigest() == identity["source_map_sha256"])

    f043 = REPO / "docs/yb_mcdhf_rci_2026_10_11/rank_aware_cell_discriminator_preparation/candidate_source/atomx"
    declarations = (
        ("numerics/bspline/convol.py", "_convol_case1_semiseparable_t"),
        ("numerics/bspline/convol.py", "_density_sym"),
        ("numerics/bspline/convol.py", "_banded_to_full_add"),
        ("numerics/bspline/convol.py", "ConvolCache.get_coulomb_factors"),
        ("methods/csf/variational.py", "CSFOrbitalFunctional.gradient_from_weights"),
        ("methods/csf/variational.py", "CSFOrbitalFunctional._pair_potential"),
    )
    source_symbols = []
    for relative, name in declarations:
        current = symbol(ast.parse((REPO / "src/atomx" / relative).read_text()), name)
        previous = symbol(ast.parse((f043 / relative).read_text()), name)
        check(f"current_f043_same_numerical_{name}", semantic(current) == semantic(previous))
        source_symbols.append({"path": f"src/atomx/{relative}", "symbol": name,
                               "line": current.lineno, "end_line": current.end_lineno,
                               "numerical_AST_sha256": hashlib.sha256(semantic(current).encode()).hexdigest()})

    executor = ast.parse((HERE / "batch_executor.py").read_text())
    compile((HERE / "batch_executor.py").read_text(), str(HERE / "batch_executor.py"), "exec")
    for node in executor.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]]
            check(f"stdlib_top_import_{node.lineno}", not forbidden.intersection(modules))
    text = (HERE / "batch_executor.py").read_text()
    check("no_angular_CI_or_physics_import", "from atomx" not in text and "import atomx" not in text)
    check("no_exec_AST_or_text", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                        and n.func.id in {"exec", "eval"} for n in ast.walk(executor)))
    check("no_FMA_atomic_fastmath", "atomic" not in text and "fastmath=True" not in text
          and "fma" not in text.lower())
    scalar = symbol(executor, "_ordered_scalar_functions")
    scan = symbol(scalar, "scan")
    row = symbol(scalar, "row")
    scan_text, row_text = ast.unparse(scan), ast.unparse(row)
    for label, exact in (
        ("jp_specific_end", "end = ns - jp"),
        ("ascending_prefix", "range(end)"),
        ("descending_suffix", "range(end - 1, -1, -1)"),
    ):
        check(label, exact in scan_text)
    for label, exact in (
        ("unequal_left_first_near", "first_near = i + ip - kl + 1"),
        ("right_tail_end", "end = ns - jp"),
        ("before_clamp", "before = min(end, max(0, first_near))"),
        ("after_clamp", "after = min(end, max(0, i + kr - jp))"),
        ("full_near_strip_ascending", "range(max(0, first_near), after)"),
        ("row_support_not_reduction_change", "ip < kl and i < ns - ip"),
    ):
        check(label, exact in row_text)
    row_calls = [n for n in ast.walk(row) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "add"]
    check("three_ordered_row_add_sites", len(row_calls) == 3)
    for target in ("_cpu_kernels", "_cuda_kernels"):
        function = symbol(executor, target)
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_ordered_scalar_functions"]
        check(f"{target}_uses_same_scalar_factory_once", len(calls) == 1)
        check(f"{target}_two_kernel_functions", len([n for n in function.body
                                                    if isinstance(n, ast.FunctionDef)]) == 2)
    check("cuda_explicit_IEEE_ops", "libdevice.dadd_rn" in text and "libdevice.dmul_rn" in text)
    check("no_density_total_minus_prefix", "total - prefix" not in text)
    check("inactive_scan_padding_reset", "for j in range(ns + 1):" in scan_text)

    spec = importlib.util.spec_from_file_location("native_batch_descriptor_mocks", HERE / "descriptor_mocks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mocks = module.run()
    check("descriptor_mocks_all_passed", mocks["all_passed"])
    check("no_numerical_modules_after_mocks", not any(
        name.split(".")[0] in forbidden for name in sys.modules))
    check("contract_work_preflight_matches", mocks["mechanical31_worst_case"] == contract["mechanical31_preflight"])
    write("descriptor_verification.json", mocks)
    result = {
        "status": "source_and_stdlib_descriptor_preparation_only",
        "all_passed": all(c["passed"] for c in checks), "checks_count": len(checks), "checks": checks,
        "descriptor_checks_count": mocks["checks_count"], "source_symbols": source_symbols,
        "implementation_sha256": sha(HERE / "batch_executor.py"),
        "implementation_AST_sha256": hashlib.sha256(ast.dump(executor, include_attributes=False).encode()).hexdigest(),
        "contract_sha256": sha(HERE / "contract.json"), "prepare_sha256": sha(HERE / "prepare.py"),
        "descriptor_driver_sha256": sha(HERE / "descriptor_mocks.py"),
        "descriptor_verification_sha256": sha(HERE / "descriptor_verification.json"),
        "source_identity_sha256": sha(HERE / "public_source_identity.json"),
        "actual_calls": {"NumPy_NumBa_Atomx_CUDA_imports": 0, "arrays_NPZ": 0,
                         "compile_NumBa_CUDA": 0, "GPU_context_kernel": 0,
                         "H_CI_g_Dirac_optimizer_benchmark": 0},
        "not_qualified": ["FP64_numerical_output", "CPU_prange", "CUDA_compilation_runtime",
                          "native31_gradient_metric_derivative", "performance"],
    }
    write("preparation_verification.json", result)
    print(json.dumps({"all_passed": result["all_passed"], "source_checks": len(checks),
                      "descriptor_checks": mocks["checks_count"], "science": 0}, indent=2))


if __name__ == "__main__":
    main()
