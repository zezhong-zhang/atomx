"""Metadata/AST review only; import no Atomx or numerical module."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
checks = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, passed):
    checks.append({"name": name, "passed": bool(passed)})
    if not passed:
        raise ValueError(name)


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    identity = json.loads((ROOT / contract["source"]["identity_path"]).read_text())
    frozen = ROOT / contract["source"]["root"]
    check("Original immutable356 scientific source map", identity["source_map_sha256"] == contract["source"]["map_sha256"] and identity["source_file_count"] == len(identity["all_atomx_python_sha256"]) == 356 and all(sha(frozen / Path(p).relative_to("src")) == expected for p, expected in identity["all_atomx_python_sha256"].items()))
    check("Read-only supporting archive identities", all(sha(ROOT / p) == expected for p, expected in contract["read_only_inputs_sha256"].items()))
    check("Observed probe draft archived as observation only", sha(HERE / "observed_probe_contract.json") == contract["source"]["observed_probe_contract_sha256"] and contract["source"]["observed_probe_is_not_an_execution_or_final_input_binding"])
    check("No execution authorized or implied by source planning", not contract["scientific_execution_permitted"] and not contract["root_execution_authorized"] and all(v is None for v in contract["required_future_inputs"].values()))
    optimizer_path = frozen / "atomx/numerics/bspline/orbital_optimization.py"
    functional_path = frozen / "atomx/methods/csf/variational.py"
    optimizer_text, functional_text = optimizer_path.read_text(), functional_path.read_text()
    optimizer_tree, functional_tree = ast.parse(optimizer_text), ast.parse(functional_text)
    optimizer = next(n for n in optimizer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_orbitals")
    names = {n.arg for n in optimizer.args.kwonlyargs}
    check("Existing public optimizer already exposes required bounded resume", {"resume_state", "objective_identity", "allow_objective_change", "checkpoint_callback", "max_wall_time", "return_state", "max_iter", "energy_difference"}.issubset(names))
    check("Accepted point zero and allaccepted callbacks precede residual and maxiter stop", optimizer_text.index("checkpoint_callback(iteration, state") < optimizer_text.index("if residual < tolerance and energy_error < energy_tolerance:") < optimizer_text.index("if iteration == max_iter:"))
    check("Public resume does not recompute population scaling", "if not resumed:\n        # The orbital Hessian" in optimizer_text and "resume_state.validate(functional)" in optimizer_text)
    check("Accepted trial CI reused for complete HellmannFeynman gradient", "energy, g, ci = evaluate(x, ci_result=trial_ci)" in optimizer_text)
    check("Current original metric residual and energy indicator retained", "preconditioned_gradient = tangent(x, spectral_precondition(g))" in optimizer_text and "energy_error = 0.5 * abs(float(g @ preconditioned_gradient))" in optimizer_text)
    check("Ordinary original Armijo rule unchanged", "change = trial_energy - energy" in optimizer_text and "change <= 1e-4 * step * derivative" in optimizer_text)
    check("Finite callback only inside original128ULP window", "abs(change) <= 128 * np.spacing(max(abs(energy), 1.0))" in optimizer_text)
    check("Callback output consumed as scalar change", "if stable_change is not None:\n                        change = stable_change" in optimizer_text and "if np.isfinite(change)" in optimizer_text)
    check("Native line-search/fallback capped2times28", "for attempt in range(2):" in optimizer_text and "for _ in range(28):" in optimizer_text)
    functional_class = next(n for n in functional_tree.body if isinstance(n, ast.ClassDef) and n.name == "CSFOrbitalFunctional")
    methods = {n.name: n for n in functional_class.body if isinstance(n, ast.FunctionDef)}
    check("Existing public functional has retainedH optin and mixed diagnostic", {"evaluate", "mixed_ci_energy_difference"}.issubset(methods) and "retain_hamiltonian=False" in functional_text)
    mixed = methods["mixed_ci_energy_difference"]
    final = mixed.body[-1]
    flags = {kw.arg: kw.value.value for kw in final.value.keywords if isinstance(kw.value, ast.Constant)}
    check("Mixed return remains diagnostic dict withoutcertified errorbound/adapter", isinstance(final, ast.Return) and isinstance(final.value, ast.Call) and isinstance(final.value.func, ast.Name) and final.value.func.id == "dict" and flags["diagnostic_only"] is True and flags["certified_error_bound"] is False and flags["optimizer_adapter"] is False)
    check("Diagnostic reported and retainedRayleigh semantics separate", "reported_change=reported_change, rayleigh_change=rayleigh_change" in functional_text and "quotient_closure_error" in functional_text)
    check("RetainedCI supplied at exactphysicalpoint validated notrecomputed", "elif isinstance(result, CIResult) and result.hamiltonian_snapshot is not None:" in functional_text and "result.hamiltonian_for(self.csfs, orbitals" in functional_text)
    check("Tracker resolves every energy and gradient trial", "state_weights, tracking = self.level_tracking.resolve" in functional_text and functional_text.index("state_weights, tracking = self.level_tracking.resolve") < functional_text.index("if not gradient:\n            return energy, result"))
    proposal = contract["public_optimizer"]
    caps = contract["proposed_resources_pending_probe_timing"]
    check("Sixteen accepts17gradients65completeHCI worstbound", proposal["max_iter"] == caps["max_new_accepted_steps"] == 16 and caps["max_full_gradient_calls"] == 17 and caps["max_complete_CI_solves"] == caps["max_complete_H_builds"] == caps["max_energy_trial_evaluations"] + 1 == 65)
    check("Accepted gradient owner reuse addsnoHCI", caps["max_checkpoint_H_builds"] == 0 and proposal["retain_hamiltonian"])
    check("Initial7 plus65CI times7 hS separately counted", caps["max_CI_internal_hS"] == 7 * 65 and caps["max_total_hS"] == 7 + 7 * 65)
    check("Public fromarrays positive branch validated7 no extraresume spectra", caps["max_from_arrays_validation_Dirac"] == 7 and caps["optimizer_resume_validation_spectra"] == 0)
    check("Tracking complete functional and checkpoint cap", caps["max_functional_tracking_resolves"] == 17 + 64 and caps["max_checkpoint_tracking_resolves"] == 17 and caps["max_total_tracking_resolves"] == 98 and caps["max_tracking_SVD"] == 5 * 98)
    check("Actual probe metric/P/all29/objective required without reseeding", contract["fixed"]["all29_radials_active"] and contract["fixed"]["coordinate_count"] == 3451 and proposal["resume_state"].endswith("exact actualprobe recovered-coordinate point") and proposal["allow_objective_change"] is False and len(contract["objective_identity_payload"]) == 7)
    check("Unchanged stationarity gates and no experimental fit", proposal["tolerance"] == contract["gates"]["orbital_residual"] == 1e-7 and proposal["energy_tolerance_Ha"] == contract["gates"]["energy_indicator_Ha"] == 1e-8 and contract["gates"]["experimental_gate_is_report_only_not_fit_target"])
    check("Ordinary chosen with no unqualifiedcallback", proposal["energy_difference"] is None and not proposal["linear_energy_increments"] and not contract["ordinary_line_search_decision"]["mixed_diagnostic_is_optimizer_callback"])
    report = {"all_passed": True, "checks_count": len(checks), "checks": checks,
        "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(Path(__file__)),
        "source_map_sha256": identity["source_map_sha256"], "critical_source_sha256": {
            str(optimizer_path.relative_to(ROOT)): sha(optimizer_path),
            str(functional_path.relative_to(ROOT)): sha(functional_path)},
        "Atomx_imports": 0, "scientific_or_numerical_calls": 0,
        "execution_authorized": False,
        "scope": "Source-only candidate356 bounded16-step public29 plan; pending actualprobe and root exactbinding. No optimizer/Armijo numerical qualification claimed."}
    (HERE / "source_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": True, "checks_count": len(checks)}))


if __name__ == "__main__":
    main()
