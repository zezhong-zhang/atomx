"""Bounded preparation controls; zero native physical invocations."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    preflight = json.loads((HERE / "preflight.json").read_text())
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Need three CPU identities before preparation")
    os.sched_setaffinity(0, {affinity[-3]})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Preparation30s cap")))
    signal.alarm(30)
    import numpy as np

    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    spec = importlib.util.spec_from_file_location("static_consumer_preparation", HERE / "run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    checkpoint, endpoint = driver.validate_identities(contract)
    check("preflight_same_driver_contract", preflight["contract_sha256"] == sha(HERE / "contract.json")
          and preflight["driver_sha256"] == sha(HERE / "run.py"))
    check("complete_model_resources", preflight["CSFs"] == 531 and len(preflight["blocks"]) == 16
          and preflight["max_raw_M_equals_J"] == 232)
    check("deterministic_cpu_third_last", preflight["resources"]["selected_CPU"] == 29
          and preflight["resources"]["original_accounting_CPUs"] == 32
          and contract["resources"]["affinity_selection"].startswith("original_available_affinity[-3]"))
    check("exact_failed_original_stationarity_gate", checkpoint["stationarity_passed"] is False
          and endpoint["all_identity_replay_checks_passed"] is True
          and endpoint["actual_original_metric_orbital_residual"] > 1e-7
          and endpoint["actual_original_metric_energy_indicator_Ha"] < 1e-8)
    for path, expected in contract["inputs_sha256"].items():
        check("bound_input:" + path, sha(ROOT / path) == expected)
    for path, expected in contract["frozen_source_files_sha256"].items():
        check("bound_source:" + path,
              sha(ROOT / contract["fixed_conditions"]["source_root"] / Path(path).relative_to("src")) == expected)
    archive = np.load(ROOT / contract["fixed_conditions"]["endpoint_arrays"])
    for name, shape in (("x", (2856,)), ("P", (24, 123)), ("Q", (24, 123)),
                        ("matrix_Ha", (531, 531)), ("energies_Ha", (531,)), ("mixing", (531, 531))):
        check("saved_point_shape_finite:" + name, archive[name].shape == shape
              and archive[name].dtype == np.float64 and np.isfinite(archive[name]).all())
    approval = {"execution_authorized_by_root": True, "driver_sha256": sha(HERE / "run.py"),
                "preparation_contract_sha256": sha(HERE / "contract.json"),
                "source_identity_sha256": sha(ROOT / contract["fixed_conditions"]["source_identity"]),
                "scientific_inputs_sha256": contract["inputs_sha256"],
                "source_map_sha256": contract["fixed_conditions"]["source_map_sha256"],
                "max_new_full_H_builds": 1, "max_new_CI_spectrum_solves": 1,
                "max_new_gradients": 0, "max_optimizer_calls": 0, "wall_cap_s": 180,
                "address_space_cap_GiB": 4, "threads": 1,
                "affinity_selection": "original_available_affinity[-3]"}
    check("synthetic_exact_root_schema_accepts", driver.execution_authorized(
        approval, HERE / "contract.json", HERE / "run.py", contract))
    check("missing_root_schema_rejects", not driver.execution_authorized({}, HERE / "contract.json", HERE / "run.py", contract))
    for name in approval:
        mutation = copy.deepcopy(approval)
        mutation[name] = None
        check("mutated_authorization_rejects:" + name, not driver.execution_authorized(
            mutation, HERE / "contract.json", HERE / "run.py", contract))
    tree = ast.parse((HERE / "run.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    check("one_public_full_H_builder_call", sum(isinstance(n.func, ast.Name)
          and n.func.id == "build_ci_matrix" for n in calls) == 1)
    check("no_public_CI_rebuild_or_optimizer_call", not any(isinstance(n.func, ast.Name)
          and n.func.id in ("ci_solve", "optimize_orbitals") for n in calls))
    check("no_functional_evaluate_call", not any(isinstance(n.func, ast.Attribute)
          and n.func.attr == "evaluate" for n in calls))
    hcall = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "build_ci_matrix")
    kwargs = {k.arg: k.value for k in hcall.keywords}
    check("complete_static_operator_no_finite_frequency", ast.literal_eval(kwargs["include_breit"]) is True
          and ast.literal_eval(kwargs["breit_orbital_energies"]) is None)
    for name in ("run.py", "verify_arrays.py"):
        ast.parse((HERE / name).read_text())
        check("parse:" + name, True)
    check("no_Atomx_imported", not any(n == "atomx" or n.startswith("atomx.") for n in sys.modules))
    check("no_physical_output_exists", not any((HERE / n).exists() for n in ("arrays.npz", "result.json", "execution_source.py")))
    proof = {"all_passed": all(row["passed"] for row in checks), "checks_count": len(checks), "checks": checks,
             "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(HERE / "run.py"),
             "verifier_sha256": sha(HERE / "verify_arrays.py"), "preflight_sha256": sha(HERE / "preflight.json"),
             "preparation_verifier_sha256": sha(Path(__file__)), "actual_CPU_affinity": sorted(os.sched_getaffinity(0)),
             "resource_budget": contract["preparation_controls_resource_budget"],
             "Atomx_imports": 0, "H_builds": 0, "CI_solves": 0, "gradients": 0, "optimizer_calls": 0,
             "synthetic_authorization_only": True, "actual_root_execution_contract_issued_by_this_script": False,
             "scope": "Read-only identity/AST/saved-point dimensions; synthetic exact contract guard controls only"}
    (HERE / "preparation_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    signal.alarm(0)
    print(json.dumps({"all_passed": proof["all_passed"], "checks_count": proof["checks_count"]}))
    if not proof["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
