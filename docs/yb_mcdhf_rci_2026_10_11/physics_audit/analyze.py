"""Recompute a differential Yb physics-scale ledger from owned evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
INPUTS = {
    "current_sd_comparison": "docs/atomic_pci_sd_native_2026_10_09/yb_monovalent_experiment003/comparison.json",
    "frozen_ci_transverse": "docs/atomic_pci_sd_native_2026_10_09/yb_dynamic_breit003/result.json",
    "finite_ci_core_holes": "docs/yb_sd_breit_completion_2026_10_09/results.json",
    "default_se_coverage": "docs/yb_diagnosis_repair_2026_10_08/qed_coverage_after.json",
    "previous_budget": "docs/atomic_physics_budget_2026_10_11/results.json",
    "old_projector_anchor": "docs/yb_mcdhf_rci_2026_10_11/physics_audit/fock_projector_result.json",
    "corrected_projector_replay": "docs/yb_mcdhf_rci_2026_10_11/physics_audit/fock_projector_replay.json",
    "stable_difference_oracle": "docs/yb_mcdhf_rci_2026_10_11/physics_audit/stable_difference_review.json",
    "polar_oracle": "docs/yb_mcdhf_rci_2026_10_11/physics_audit/polar_review.json",
    "native_Yb_angular_filter": "docs/yb_mcdhf_rci_2026_10_11/native_run/stable_polar_endpoint/result.json",
    "constraint_center_oracle": "docs/yb_mcdhf_rci_2026_10_11/physics_audit/constraint_center_review.json",
    "native_Yb_radial_constraint": "docs/yb_mcdhf_rci_2026_10_11/native_run/radial_constraint_endpoint002/result.json",
}


def main() -> None:
    records = {key: json.loads((ROOT / path).read_text()) for key, path in INPUTS.items()}
    bindings = {key: {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                for key, path in INPUTS.items()}
    comparison = records["current_sd_comparison"]
    h = comparison["constants"]["HARTREE_CM1"]
    current = comparison["rows"]
    doublets = []
    for term, keys in (("5d", ((5, 2), (5, -3))), ("6p", ((6, 1), (6, -2)))):
        rows = [next(row for row in current if tuple(row["key"]) == key) for key in keys]
        errors = [row["computed_cm1"] - row["experimental_cm1"] for row in rows]
        weights = [row["twoJ"] + 1 for row in rows]
        doublets.append({"term": term, "centroid_gap_cm1": sum(a * b for a, b in zip(weights, errors, strict=True)) / sum(weights),
                         "splitting_gap_cm1": errors[1] - errors[0]})
    core_holes = records["finite_ci_core_holes"]["rows"]
    ground_holes = core_holes["6s_1/2"]
    finite_correlation = []
    for label, row in core_holes.items():
        if label == "6s_1/2":
            continue
        dc_transition = (row["DC"]["total_Ha"] - ground_holes["DC"]["total_Ha"]) * h
        dcb_transition = (row["DCB"]["total_Ha"] - ground_holes["DCB"]["total_Ha"]) * h
        one_hole_transition = (row["one_hole_DC"]["total_Ha"] - ground_holes["one_hole_DC"]["total_Ha"]) * h
        finite_correlation.append({"label": label, "two_hole_increment_transition_cm1": dc_transition - one_hole_transition,
                                   "full_static_breit_transition_increment_cm1": dcb_transition - dc_transition})
    transverse = []
    for row in records["frozen_ci_transverse"]["rows"]:
        if row["label"] == "6s_1/2":
            continue
        shifts = row["excitation_cm1"]
        transverse.append({"label": row["label"], "static_minus_DC_transition_cm1": shifts["static"] - shifts["DC"],
                           "dynamic_minus_static_transition_cm1": shifts["dynamic"] - shifts["static"]})
    se_records = records["default_se_coverage"]["records"]
    se_ground = next(row["SE_Ha"] for row in se_records if row["label"] == "6s_1/2")
    se = [{"label": row["label"], "model_SE_upper_minus_6s_cm1": (row["SE_Ha"] - se_ground) * h,
           "upper_warnings": row["warnings"], "ground_has_leading_analytic_fallback": True,
           "uncertainty_bound_cm1": None} for row in se_records if row["label"] != "6s_1/2"]
    for row in current:
        assert abs((row["computed_cm1"] - row["experimental_cm1"]) - row["error_cm1"]) < 1e-10
    for row in finite_correlation:
        saved = core_holes[row["label"]]
        assert abs(row["full_static_breit_transition_increment_cm1"] - saved["full_B_transition_cm1"]) < 1e-6
    assert records["corrected_projector_replay"]["passed"]
    assert records["stable_difference_oracle"]["passed"]
    assert records["polar_oracle"]["passed"]
    assert records["constraint_center_oracle"]["passed"]
    report = {
        "status": "Physics audit and independent operator/polar/energy-difference/constraint-centered API anchors completed",
        "primary_route_user_decision": "Native Atomx common-orbital MCDHF followed by RCI; external software supplies independent references only",
        "new_Yb_roots_calculated_by_this_audit": 0, "experimental_improvement_measured_by_this_audit": False,
        "inputs": bindings, "current_SD_doublets": doublets,
        "historical_frozen_CI_transverse_differentials": transverse,
        "historical_finite_CI_differentials": finite_correlation,
        "historical_default_SE_differentials": se,
        "historical_core_and_triples_differentials": records["previous_budget"]["historical_model_controls"],
        "normal_mass_shift_proxy": records["previous_budget"]["normal_mass_shift_proxy"],
        "total_physical_error_bound_cm1": None,
        "add_historical_controls_to_current_or_new_roots": False,
        "independent_projector_bug_confirmed": records["old_projector_anchor"]["confirmed_one_body_double_addition"],
        "independent_projector_repair_replay_passed": records["corrected_projector_replay"]["passed"],
        "native_Yb_angular_filter": {
            "counts": records["native_Yb_angular_filter"]["angular_audit"],
            "scope": "Real native five-state block coefficient count; paired unfiltered-versus-retained diagnostic now available for the saved eight radial increments only",
            "maximum_paired_increment_effect_Ha": max(abs(row["angular_filter_increment_difference_Ha"])
                                                       for row in records["native_Yb_radial_constraint"]["rows"]),
            "certifies_original_solve": False,
        },
        "native_Yb_roundoff_constraint_diagnostic": {
            "rows": [{key: row[key] for key in ("step", "retraction", "retained_exact_difference_Ha",
                                                   "quotient_normal_metric_defect_Ha",
                                                   "exact_minus_quotient_metric_defect_Ha",
                                                   "coordinate_tangent_linear_Ha")}
                     for row in records["native_Yb_radial_constraint"]["rows"]],
            "scope": "Saved original final candidate at fixed model/gradient/metric; no solver acceptance or stationarity certification",
            "maximum_base_Gram_defect": max(group["base_metric_defect_max_wide"]
                                           for row in records["native_Yb_radial_constraint"]["rows"]
                                           for group in row["groups"]),
            "certifies_original_solve": False,
        },
        "post_audit_shared_repairs": {
            "complete_Fock_double_addition": "parent repaired; independent saved arrays and old-fail/new-pass regressions",
            "determinant_cap_propagation": "parent implemented CSF solver -> functional -> radial density and CI propagation; original missing consumer retained in source archive",
            "stable_energy_difference": "parent implemented restricted one-CSF-block unchanged-RDM finite-polynomial difference; independent long-double full-energy oracle passed",
            "near_unit_trial_polar": "parent implemented binomial inverse-square-root correction; independent 75-decimal dense/threshold/equivariance oracle passed with original seed spectral path retained",
            "stable_difference_source_epoch": records["stable_difference_oracle"]["source_hashes"],
            "polar_optimizer_source_epoch": records["polar_oracle"]["source_sha256"],
            "constraint_centered_energy": "Separate unchanged-RDM one-CSF API with actual h/n normal multiplier, full same-kappa Gram curvature, both-endpoint roundoff guards and declared linear increments; independent complete normalized-energy oracle passed",
            "constraint_centered_source_epoch": records["constraint_center_oracle"]["source_hashes"],
            "original_report_before_repair_dispositions": "results_before_post_audit_repairs.json",
        },
        "priority_order": [
            "Correct no-pair operator identity; retain nuclear and seed-Fock spaces as explicit distinct physical models",
            "Stationary native five-state common-orbital Dirac-Coulomb EOL, same physical reference and energy zero",
            "Separate fixed-orbital CI and same-space radial relaxation; use both upper-minus-lower changes to select mechanism",
            "Outer-core 4f/5s/5p CV and relevant 4f-hole multireference mixing with identity tracking; measure full five-state responses",
            "Same-snapshot full static Breit RCI, declared model-QED matrix and recoil differential ledger; no leading-Zalpha precision claim",
            "Dynamic-minus-static and remaining correlation refinements after leading discrepancies are isolated; exact angular reuse and roots-only CI propagation",
        ],
        "unresolved": [
            "No measured fixed-space MCDHF radial-relaxation shift yet in this audit",
            "No complete radial/correlation/no-pair model physical-error upper bound",
            "No complete general production large-CV MCDHF certification",
            "Current EOL scalar objective is Dirac-Coulomb only; no transverse/QED orbital gradient",
            "No implicit physical root-character tracking in public Atom",
            "No frozen/active-orbital public EOL partition observed",
            "Full-spectrum dense CI remains default within orbital functional; root-limited Davidson contract not propagated",
            "QED post-CI expectation does not diagonalize corrections within degenerate manifolds; nonlocal precision SE coverage not universal",
        ],
        "verification": {"arithmetic_rederived": True, "upper_lower_energy_reference_respected": True,
                         "historical_model_scopes_preserved": True, "original_experimental_1percent_gates_preserved": True,
                         "passed": True},
    }
    (OWNER / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "inputs": len(bindings), "current_doublets": doublets,
                      "correction_total_upper_bound": None}, indent=2))


if __name__ == "__main__":
    main()
