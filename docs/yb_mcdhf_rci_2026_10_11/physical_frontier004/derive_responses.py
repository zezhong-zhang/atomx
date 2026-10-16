"""Difference frozen saved energies only; no Atomx physical evaluation."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
REPO = MODEL.parents[1]
LABELS = ("6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")


def read(relative):
    return json.loads((MODEL / relative).read_text())


def bylabel(rows):
    answer = {row["label"]: row for row in rows}
    assert tuple(answer) == LABELS
    return answer


def multiplets(values):
    return {"5d": {"centroid_cm1": (4 * values["5d_3/2"] + 6 * values["5d_5/2"]) / 10,
                    "splitting_cm1": values["5d_5/2"] - values["5d_3/2"]},
            "6p": {"centroid_cm1": (2 * values["6p_1/2"] + 4 * values["6p_3/2"]) / 6,
                    "splitting_cm1": values["6p_3/2"] - values["6p_1/2"]}}


def response(new, old, conversion):
    absolute = {label: new[label] - old[label] for label in LABELS}
    transitions = {label: (absolute[label] - absolute[LABELS[0]]) * conversion for label in LABELS}
    return {"absolute_energy_delta_Ha": absolute,
            "upper_minus_same_lower_response_cm1": transitions,
            "multiplet_response": multiplets(transitions)}


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    hash_checks = []
    for namespace in ("inputs_sha256", "source_sha256"):
        for path, expected in contract[namespace].items():
            actual = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            hash_checks.append({"path": path, "expected": expected, "actual": actual, "pass": actual == expected})
    assert all(row["pass"] for row in hash_checks), [row for row in hash_checks if not row["pass"]]
    bare = read("native_run/nuclear_eol005_projected_increment/result.json")
    balanced = read("balanced_cv_replay/result.json")
    mixed = read("mixed_ci_stationarity_probe/result.json")
    corrections = read("correction_budget/same_orbital001/result.json")
    coverage = read("correction_budget/same_orbital001/coverage_correction.json")
    bare_dc = bylabel(bare["rows_dc"])
    conversion = bare_dc[LABELS[1]]["computed_cm1"] / (bare_dc[LABELS[1]]["energy_Ha"] - bare_dc[LABELS[0]]["energy_Ha"])
    nominal = {label: bare_dc[label]["experimental_cm1"] for label in LABELS}
    checks = []
    previous = {"DC": {k: r["energy_Ha"] for k, r in bare_dc.items()},
                "static_Breit": {k: r["energy_Ha"] for k, r in bylabel(bare["rows_rci"]).items()}}
    initial = {model: dict(rows) for model, rows in previous.items()}
    stages = []
    for stage in balanced["stages"]:
        output = {"name": stage["name"], "CSFs": stage["CSFs"], "same_orbitals": "native005 24-radial common spinors, frozen; no expanded MCDHF stationarity claim", "models": {}}
        for model, source_rows in stage["tracked_levels"].items():
            rows = bylabel(source_rows)
            energies = {label: rows[label]["energy_Ha"] for label in LABELS}
            exc = {label: (energies[label] - energies[LABELS[0]]) * conversion for label in LABELS}
            gaps = {label: exc[label] - nominal[label] for label in LABELS}
            increment = response(energies, previous[model], conversion)
            total = response(energies, initial[model], conversion)
            for label in LABELS:
                checks.append({"name": f"{stage['name']}/{model}/{label}/excitation", "error_cm1": abs(exc[label] - rows[label]["excitation_from_tracked_6s_cm1"]), "pass": abs(exc[label] - rows[label]["excitation_from_tracked_6s_cm1"]) < 1e-7})
                checks.append({"name": f"{stage['name']}/{model}/{label}/increment", "error_cm1": abs(increment["upper_minus_same_lower_response_cm1"][label] - rows[label]["transition_delta_from_previous_cm1"]), "pass": abs(increment["upper_minus_same_lower_response_cm1"][label] - rows[label]["transition_delta_from_previous_cm1"]) < 1e-6})
            output["models"][model] = {"absolute_energies_Ha": energies, "excitation_cm1": exc,
                                        "NIST_nominal_gap_cm1": gaps,
                                        "multiplet_values": multiplets(exc),
                                        "multiplet_NIST_gap": multiplets(gaps),
                                        "response_to_previous": increment,
                                        "response_to_bare005": total,
                                        "original_1percent_per_line_gate_passed": all(abs(gaps[label]) <= .01 * nominal[label] for label in LABELS[1:])}
            previous[model] = energies
        output["same_stage_static_minus_DC"] = response(previous["static_Breit"], previous["DC"], conversion)
        stages.append(output)
    one_step = bylabel(mixed["one_step"]["lower_upper_difference_rows"])
    step = response({k: r["accepted_energy_Ha"] for k, r in one_step.items()},
                    {k: r["baseline_energy_Ha"] for k, r in one_step.items()}, conversion)
    for label in LABELS:
        checks.append({"name": f"mixedstep/DC/{label}/response", "pass": abs(step["upper_minus_same_lower_response_cm1"][label] - one_step[label]["transition_delta_from_same_lower_cm1"]) < 1e-6})
    independent_correction_rows = bylabel(corrections["state_correction_ledger"])
    independent_corrections = {}
    for component in independent_correction_rows[LABELS[0]]["components"]:
        values = {label: independent_correction_rows[label]["components"][component]["absolute_Ha"] for label in LABELS}
        relative = {label: (values[label] - values[LABELS[0]]) * conversion for label in LABELS}
        for label in LABELS:
            checks.append({"name": f"same003only/{component}/{label}/response", "pass": abs(relative[label] - independent_correction_rows[label]["components"][component]["relative_to_6s_cm1"]) < 1e-7})
        independent_corrections[component] = {"absolute_expectation_Ha": values,
                                             "upper_minus_same_lower_cm1": relative,
                                             "multiplet_response": multiplets(relative)}
    assert all(c["pass"] for c in checks), [c for c in checks if not c["pass"]]
    output = {
        "status": "read_only_derivation_from_saved_energies_no_new_physics",
        "contract_sha256": hashlib.sha256((HERE / "contract.json").read_bytes()).hexdigest(),
        "hash_checks": hash_checks, "all_hash_checks_pass": True,
        "conversion_cm1_per_Ha": conversion,
        "conversion_provenance": "Recovered exactly from archived005 energy/transition pair; all005/531/mixed003 correction saved differences crosschecked with their stored cm-1 values; no new constants or solver imported.",
        "NIST_nominal_excitation_cm1": nominal,
        "NIST_nominal_multiplets": multiplets(nominal),
        "bare005": {"stationarity_passed": bare["stationarity_passed"],
                    "orbital_residual": bare["orbital_optimization"]["orbital_residual"],
                    "gate_scope": "Five reference CSFs only, not531 mixed-CI orbital objective",
                    "seed_alternative_projector_leakage_max": max(bare["orbital_optimization"]["no_pair_projector_leakage"].values()),
                    "leakage_scope": "Seed-only S-norm yardstick; not final531 no-pair differential-energy bound",
                    "static_minus_DC": response(initial["static_Breit"], initial["DC"], conversion)},
        "balanced_frozen005_orbital_stages": stages,
        "one_original_metric_mixed_DC_orbital_step": {
            "response": step,
            "EOL_objective_delta_Ha": mixed["one_step"]["EOL_delta_Ha"],
            "original_metric_residual_before": mixed["baseline"]["original_metric_orbital_residual"],
            "original_metric_residual_after": mixed["one_step"]["original_metric_orbital_residual"],
            "stationarity_passed": mixed["one_step"]["stationarity_passed"],
            "interpretation": "Complete mixed-CI DC response on one accepted trial, far above original1e-7 residual gate; not converged MCDHF or missing-physics bound, cannot add DC shifts to static531/003 corrections."},
        "independent_same003_only_correction_evidence": {
            "reference": corrections["input_run"], "original_stationarity": corrections["original_stationarity"],
            "components": independent_corrections,
            "actual_operator_coverage": coverage["component_coverage"],
            "self_energy_model_differential_spread_cm1": coverage["SE_model_differential_spread_cm1"],
            "scope": "Different failed003 five-CSF orbitals; isolated ledger only. Not a531 additive correction, uncertainty bound or531 hierarchy estimate."},
        "arithmetic_checks": checks, "all_arithmetic_checks_pass": True,
        "missing_quantities": [
            "Converged531 mixed-CI EOL DC energies and same-state static-Breit RCI on that exact endpoint (current root run is independent and not consumed here).",
            "Complete5p/4f/two-family core-double finite-space lower/upper responses.",
            "Balanced radial virtual layer and higher-l layer lower/upper responses;31/24 orbital stationarity cannot certify virtual completeness.",
            "Final mixed531 screened-vs-nuclear positive-boundary overlaps and matched no-pair excitation response.",
            "Same531 frequency-dependent Breit complete-energy-map response, point/finite-size VP and matrix SE/recoil expectations with exact CI density including off-diagonal terms.",
            "Full nonlocal/screened QED and independent external matched-Hamiltonian reference for these exact endpoints.",
        ],
    }
    (HERE / "same_model_responses.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"hash_controls": len(hash_checks), "arithmetic_controls": len(checks), "final531_DC_gaps": stages[-1]["models"]["DC"]["NIST_nominal_gap_cm1"], "final531_static_gaps": stages[-1]["models"]["static_Breit"]["NIST_nominal_gap_cm1"], "final531_static_multiplet_gaps": stages[-1]["models"]["static_Breit"]["multiplet_NIST_gap"], "same531_static_minus_DC": stages[-1]["same_stage_static_minus_DC"], "mixed_one_step_multiplet_response": step["multiplet_response"]}, indent=2))


if __name__ == "__main__":
    main()
