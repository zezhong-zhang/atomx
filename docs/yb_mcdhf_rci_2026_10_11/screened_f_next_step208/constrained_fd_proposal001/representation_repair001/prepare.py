"""SOURCE/JSON preparation; no deferred arrays or numerical imports."""
import ast
import hashlib
import json
import runpy
import sys
import tempfile
from pathlib import Path

E = Path(__file__).resolve().parent
B = E.parent
def H(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
source = (E / "run.py").read_text()
tree = ast.parse(source)
contract = json.loads((E / "contract.json").read_text())
proposal = json.loads((B / "contract.json").read_text())
checks = []


def check(name, value):
    checks.append(dict(name=name, passed=bool(value)))
    if not value:
        raise ValueError(name)


compile(source, str(E / "run.py"), "exec")
check("AST_compile_only", True)
check("frozen_proposal_contract_preserved", H(B / "contract.json") == "a4172ec08312332fc71b142ce02a52c81c2162495e275e48dc95f6034ed52025")
check("frozen_proposal_review_preserved", H(B / "source_review.json") == "982aa9e67b4105dd7b94b7730bca018d984623de235cc4a7474654ab392221f1")
check("same_2h", contract["fixed"]["h"] == proposal["fixed"]["h"] == [1e-5, 5e-6])
check("same_f043_31_model_point_metric_UIDs", all(contract["fixed"][k] == proposal["fixed"][k] for k in ("source_map_sha256", "radial_orbitals", "coordinate_dimension", "target_CSFS", "original_UIDs", "weights", "new_metric_sha256", "new_geometry_sha256", "new_state_sha256", "objective_identity")))
check("same_separate_domains", contract["independent_centered_diagnostic"] == proposal["independent_centered_diagnostic"] and contract["gates"]["FD_absolute"] == 1e-8 and contract["gates"]["FD_relative"] == 1e-5)
check("literal_venv", contract["resources"]["interpreter_literal"] == "/data/code/atomx/.venv/bin/python")
check("CPU_AS_wall_stops", all(contract["resources"][k] == v for k, v in dict(CPU=31, AS_GiB=3.95, threads=1, inner_seconds=300, outer_seconds=315, maximum_system_load1=32).items()))
check("declared_true_calls", all(contract["resources"]["calls"][k] == v for k, v in dict(H=5, CI=5, gradient_from_weights=0, optimizer=0, GPU=0, mixed_ci_finite_increment=4, unfiltered_target_RDM=20, hS_total=70, tracker_resolve=18, tracking_SVD=90, tracker_overlap=19, validation_Dirac=7, complete_SciPy_eigh=46, geometry_SPD_eigvalsh=38, polar_retraction=28).items()))
check("hour27_exact_trigger", contract["input_roles"]["hour27_delivery"]["sha256"] == "421a5604bb55a91ce5cfd12c39984f3f7b59deb6db08727e08439b7825851dfb")
execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
body = list(ast.walk(execute))
external = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        names = [a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom):
        names = [n.module]
    else:
        continue
    if any(name.startswith(("numpy", "scipy", "atomx", "numba", "cupy")) for name in names):
        external.append(n)
check("numerical_imports_only_after_execute_authorize", all(n in body for n in external) and execute.body[0].value.func.id == "authorize")
calls = [n for n in body if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
check("single_fresh_CI_call_site", sum(n.func.attr == "ci_solve" for n in calls) == 1)
check("reported_owner_evaluate_gradientFalse", all(any(k.arg == "gradient" and isinstance(k.value, ast.Constant) and k.value.value is False for k in n.keywords) for n in calls if n.func.attr == "evaluate"))
check("no_gradient_optimizer_call", not any(n.func.attr in ("gradient_from_weights", "optimize_orbitals") for n in calls))
check("no_fixeddensity_callback_substitution", all(s not in source for s in ("energy_difference=", "fixed_density_energy_difference", "constrained_energy_difference")))
check("full_signed_quadratic_reconstruction", "quadratic[i][j] = dot(total[a], sd[b])" in source and "dm[i][j] != expm[i][j] + recm[i][j]" in source and "pair = value.as_integer_ratio()" in source)
check("short_progress_failure_fullrecords", "checks_count=len(checks)" in source and "archives_count=len(stages)" in source and "derivative_domains=derivative_domains" in source)
check("complete_trials_before_domain_fail", "# Domain gates are evaluated only after both h ledgers exist." in source)
check("physical_label_lower_UID", 'lower = by_label["6s_1/2"]' in source and 'tuple(lower["UID"]) != (1, 1, 0)' in source)
ns = runpy.run_path(str(E / "run.py"), run_name="SOURCE_JSON_PREPARATION")
_, roles, _, deferred = ns["metadata"](arrays=False)
check("source_JSON_default_actualpass_6NPZdeferred", len(deferred) == 6)
check("full356_source_bound", len(json.loads(roles["source_identity"].read_text())["all_atomx_python_sha256"]) == 356)
# This temporary preseal metadata only lets the deliberately invalid binding
# reach its explicit identity refusal. Final proof is written below.
(E / "preparation.json").write_text('{"preseal_refusal_fixture": true}\n')
with tempfile.TemporaryDirectory() as td:
    a = Path(td) / "invalid_binding.json"
    a.write_text("{}")
    try:
        ns["authorize"](a)
    except ValueError as error:
        check("bad_seal_refuses_before_arrays", "Exact separate root authorization" in str(error))
    else:
        raise ValueError("Invalid seal accepted")
check("no_actual_directory", not (E / "actual").exists())
check("no_numerical_imports_or_arrays", not any(n.startswith(("numpy", "scipy", "atomx", "numba", "cupy")) for n in sys.modules))
proof = dict(status="SOURCE_JSON_preparation_passed_NOT_numerically_executed", all_source_JSON_preparation_checks_passed=True, checks=checks, controls=len(checks), driver_sha256=H(E / "run.py"), contract_sha256=H(E / "contract.json"), preparation_script_sha256=H(Path(__file__)), source_map_sha256=contract["fixed"]["source_map_sha256"], deferred_NPZ=[dict(role=k, **v) for k, v in contract["input_roles"].items() if v["path"].endswith(".npz")], actual_default_mode="SOURCE_JSON metadata pass", bad_root_binding_refused_before_NPZ=True, numerical_imports=0, NPZ_reads_or_rehashes=0, physical_calls=0, runtime_authorization=False)
(E / "preparation.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(dict(controls=len(checks), all_passed=True, driver_sha256=proof["driver_sha256"], contract_sha256=proof["contract_sha256"], preparation_sha256=H(E / "preparation.json"), science=0)))
