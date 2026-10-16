"""Read-only exact insertion/source-archive checks; no NumPy or Atomx import."""
import ast
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).parent
REPO = BASE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    identity_path = BASE / "source_identity.json"
    identity = json.loads(identity_path.read_text())
    old_identity = json.loads((BASE / "before/source_identity.json").read_text())
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    mapping = identity["all_atomx_python_sha256"]
    check("canonical_source_map", hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == identity["source_map_sha256"])
    check("previous_356_epoch", len(mapping) == 356 and set(mapping) == set(old_identity["all_atomx_python_sha256"]))
    changed = []
    for name, digest in mapping.items():
        check("current_" + name, sha(REPO / name) == digest)
        check("archive_" + name, sha(BASE / "source" / Path(name).relative_to("src")) == digest)
        if digest != old_identity["all_atomx_python_sha256"][name]:
            changed.append(name)
    check("only_optimizer_changed", changed == ["src/atomx/numerics/bspline/orbital_optimization.py"])
    before = (BASE / "before/orbital_optimization.py").read_text()
    after = (REPO / changed[0]).read_text()
    stripped = after
    additions = [
        "    Checkpoint diagnostics include ``accepted_line_search``: None at zero,\n"
        "    otherwise a copy of the preceding accepted numerical comparison. Its\n"
        "    step and directional derivative precede polar retraction; attempt and\n"
        "    backtrack are zero-based. ``stable_callback_used`` records an applied\n"
        "    replacement of the ordinary change, not merely a callback invocation.\n",
        "    accepted_line_search = None\n",
        "                stable_callback_used = False\n",
        "                        stable_callback_used = True\n",
        "                    accepted_line_search = {\n"
        "                        \"parent_energy_ha\": float(energy),\n"
        "                        \"trial_energy_ha\": float(trial_energy),\n"
        "                        \"step\": float(step),\n"
        "                        \"directional_derivative\": float(derivative),\n"
        "                        \"energy_change_ha\": float(change),\n"
        "                        \"armijo_rhs_ha\": float(1e-4 * step * derivative),\n"
        "                        \"attempt\": int(attempt), \"backtrack\": int(_),\n"
        "                        \"stable_callback_used\": stable_callback_used,\n"
        "                    }\n",
    ]
    for i, addition in enumerate(additions):
        check("one_exact_metadata_insertion_" + str(i), stripped.count(addition) == 1)
        stripped = stripped.replace(addition, "")
    new_field = ('                "geometry_sha256": geometry.payload_sha256,\n'
                 '                "accepted_line_search": (None if accepted_line_search is None\n'
                 '                                         else accepted_line_search.copy())}\n')
    old_field = '                "geometry_sha256": geometry.payload_sha256}\n'
    check("one_exact_checkpoint_copy_insertion", stripped.count(new_field) == 1)
    stripped = stripped.replace(new_field, old_field)
    check("all_previous_optimizer_bytes_recovered", stripped == before)
    check("unmodified_Armijo_expression", 'if np.isfinite(change) and change <= 1e-4 * step * derivative:' in before and 'if np.isfinite(change) and change <= 1e-4 * step * derivative:' in after)
    tree = ast.parse(after)
    compile(tree, changed[0], "exec")
    check("syntax_compile_without_execution", True)
    for name in ("exact_reported_guard.py", "run_array_qualification.py"):
        module = ast.parse((BASE / name).read_text())
        imports = [node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
        check(name + "_no_Atomx_import", not any(v and v.startswith("atomx") for v in imports))
        compile(module, name, "exec")
        check(name + "_syntax_only", True)
    proof = {"all_passed": all(v["passed"] for v in checks), "checks_count": len(checks), "checks": checks,
             "source_identity_sha256": sha(identity_path), "source_map_sha256": identity["source_map_sha256"],
             "optimizer_sha256": identity["optimizer_sha256"], "verifier_sha256": sha(Path(__file__)),
             "no_numerical_execution": True, "H_CI_gradient_optimizer_calls": 0,
             "metadata_runtime_validation_pending": True}
    (BASE / "source_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: proof[k] for k in ("all_passed", "checks_count", "source_map_sha256")}, indent=2))


if __name__ == "__main__":
    main()
