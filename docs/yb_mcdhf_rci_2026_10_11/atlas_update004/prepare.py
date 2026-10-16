"""Prepare two fixed-point virtual controls from qualified saved JSON only."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / "docs/yb_mcdhf_rci_2026_10_11"
LABELS = ("5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    contract = load(HERE / "contract.json")
    site = Path(contract["site"])
    if contract["status"] != "prospective_before_local_site_edits":
        raise ValueError("Prospective preimage contract required")
    if sha(HERE / "prechange_manifest.json") != contract["prechange_manifest_sha256"]:
        raise ValueError("Site preimage identity changed")
    for relative, item in load(HERE / "prechange_manifest.json")["files"].items():
        if sha(site / relative) != item["sha256"]:
            raise ValueError("Site preimage changed: " + relative)
    for relative, expected in contract["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Saved scientific input changed: " + relative)
    frozen = ROOT / contract["frozen_source"]
    source = load(CAMPAIGN / "public_mixed_resume_run002/source_identity.json")
    if source["source_map_sha256"] != contract["source_map_sha256"]:
        raise ValueError("Native scientific source epoch changed")
    for relative, expected in source["all_atomx_python_sha256"].items():
        if sha(frozen / Path(relative).relative_to("src")) != expected:
            raise ValueError("Frozen scientific source changed: " + relative)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(frozen))
    from atomx.reference_data.nist import normalise_configuration
    from tools.export_nist_atlas import level_record

    payload = json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))

    def eligible(configuration, twoj, parity):
        return [r for r in payload["levels"]
                if normalise_configuration(r["configuration"]) == configuration
                and level_record(r).two_j == twoj and level_record(r).parity == parity
                and r["energy_kind"] == "evaluated" and r["energy_cm1"] is not None
                and not r["bracketed"] and not level_record(r).flags
                and "?" not in (r["configuration_nist"] or "") + (r["term"] or "")]

    zeros = eligible("4f14 6s1", 1, 1)
    if len(zeros) != 1 or zeros[0]["energy_cm1"] != 0:
        raise ValueError("Unique evaluated same6s zero required")
    zero = zeros[0]
    point = load(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
    models = []
    baseline_rows = None
    specifications = (
        ("virtual_only_discriminator", "nuclear", "固定160步种子 · 核场虚层 · DC", 3112),
        ("screened_core_virtual_discriminator002", "screened_closed68", "固定160步种子 · 闭核68屏蔽虚层 · DC", 586),
    )
    for directory, field, title, expected_checks in specifications:
        result = load(CAMPAIGN / directory / "result.json")
        audit = load(CAMPAIGN / directory / "saved_array_audit/verification.json")
        tracking = load(CAMPAIGN / directory / "balanced_virtual_tracking.json")
        if not (result["numeric_identity_gates_passed"]
                and audit["all_independent_saved_array_checks_passed"]
                and audit["checks_count"] == expected_checks
                and result["old24_wave_bytes_preserved"]
                and not result["original_tracker_rebased"]
                and not result["original_stationarity_passed"]
                and not result["expanded_stationarity_evaluated"]
                and not result["QED_recoil_dynamic_or_static_Breit_added"]
                and not result["full_spectrum_or_globalground_certified"]
                and result["source_map_sha256"] == contract["source_map_sha256"]
                and result["checkpoint_manifest_sha256"] == sha(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
                and result["checkpoint_arrays_sha256"] == point["array_sha256"]
                and result["original_metric_sha256"] == point["original_metric_sha256"]
                and not tracking["original_reference_changed"]):
            raise ValueError("Qualified samepoint virtual-control admission failed")
        baseline = next(s for s in result["stages"] if s["name"] == "baseline")
        stage = next(s for s in result["stages"] if s["name"] == "balanced_virtual")
        if (baseline["full_model_CSFs"], baseline["retained_target_CSFs"],
                stage["full_model_CSFs"], stage["retained_target_CSFs"]) != (531, 261, 2163, 1046):
            raise ValueError("Complete finite target-sector scope changed")
        if baseline_rows is not None and baseline_rows != baseline["rows"]:
            raise ValueError("Two virtual families do not share exact baseline")
        baseline_rows = baseline["rows"]
        original = {r["label"]: r for r in stage["rows"]}
        rows = []
        for label in LABELS:
            src = original[label]
            configuration = "4f14 " + label.split("_")[0] + "1"
            matches = eligible(configuration, src["twoJ"], src["parity"])
            if len(matches) != 1 or matches[0]["energy_cm1"] != src["NIST_nominal_cm1"]:
                raise ValueError("Unique nominal physical ASD branch changed")
            ref = matches[0]
            matches_bound = [t for t in tracking["physical_tracking"]
                             if t["selected_levels"] == [[src["twoJ"], src["parity"], src["local_root"]]]]
            if len(matches_bound) != 1:
                raise ValueError("Physical target assignment is ambiguous")
            bound = matches_bound[0]
            if bound["per_level_weight"] != .2 or bound["principal_overlap_squared"] != src["original_reference_principal_squared"]:
                raise ValueError("Original signed physical tracker binding changed")
            error = src["excitation_cm1"] - ref["energy_cm1"]
            rows.append({"label": label, "configuration": configuration,
                "two_j": src["twoJ"], "parity": src["parity"],
                "calculated_cm1": src["excitation_cm1"], "matching": "unique_nominal_branch",
                "candidates": 1, "level_id": ref["level_id"], "nist_row": ref["row_index"],
                "configuration_nist": ref["configuration_nist"], "term_nist": ref["term"],
                "j_text_nist": ref["j_text"], "reference_cm1": ref["energy_cm1"],
                "uncertainty_cm1": ref["uncertainty_cm1"], "reference_kind": ref["energy_kind"],
                "reference_code": ref["reference"], "error_cm1": error,
                "error_percent": 100 * error / ref["energy_cm1"], "relative_goal": .01,
                "goal_pass": abs(error / ref["energy_cm1"]) < .01,
                "handbook_cm1": src["NIST_nominal_cm1"], "asd_minus_handbook_cm1": 0,
                "reference_zero_level_id": zero["level_id"], "source_energy_Ha": src["energy_Ha"],
                "source_physical_UID": {"twoJ": src["twoJ"], "parity": src["parity"], "local_root": src["local_root"]},
                "source_column": src["column"], "exact_J_local_root": src["local_root"],
                "physical_reference_overlap_squared": bound["principal_overlap_squared"][0],
                "physical_reference_overlap_scope": "Original full531 baselineDC tracker and original005 waves; not bare CSF weight",
                "state_average_weight": .2, "state_binding_unambiguous": True,
                "absolute_upper_virtual_minus_samepoint_baseline_Ha": src["absolute_delta_upper_Ha"],
                "same_lower_virtual_minus_samepoint_baseline_Ha": src["absolute_delta_lower_Ha"],
                "virtual_minus_samepoint_baseline_transition_cm1": src["transition_delta_upper_minus_lower_cm1"]})
        moments = []
        for group, js in (("5d", (3, 5)), ("6p", (1, 3))):
            a, b = [next(r for r in rows if r["label"] == f"{group}_{j}/2") for j in js]
            weights = [j + 1 for j in js]
            center = sum(r["calculated_cm1"] * w for r, w in zip((a, b), weights, strict=True)) / sum(weights)
            refcenter = sum(r["reference_cm1"] * w for r, w in zip((a, b), weights, strict=True)) / sum(weights)
            split = b["calculated_cm1"] - a["calculated_cm1"]
            refsplit = b["reference_cm1"] - a["reference_cm1"]
            measured = next(m for m in stage["multiplets"] if m["label"] == group)
            moments.append({"label": group, "degeneracy_weights": weights,
                "centroid_cm1": center, "reference_centroid_cm1": refcenter,
                "centroid_gap_cm1": center - refcenter, "splitting_cm1": split,
                "reference_splitting_cm1": refsplit, "splitting_gap_cm1": split - refsplit,
                "reference_uncertainty_cm1": None,
                "delta_centroid_cm1": measured["delta_centroid_cm1"],
                "delta_splitting_cm1": measured["delta_splitting_cm1"]})
        rms = math.sqrt(sum(r["error_cm1"] ** 2 for r in rows) / 4)
        if abs(rms - stage["nominal_four_gap_RMS_cm1"]) > 1e-9:
            raise ValueError("Same-model nominal gap RMS inconsistent")
        model = {"id": "fixed160_" + field + "_virtual_DC", "title": title,
            "model_kind": "fixed_nonstationary_virtual_rci", "virtual_field": field,
            "saved_run": directory + "/balanced_virtual", "operator": "DC",
            "fixed_orbital_point_total_accepted_steps": 160,
            "rci_csf_count": 1046, "declared_full_model_csf_count": 2163,
            "target_sector_csf_count": 1046, "symmetry_block_count": 5,
            "target_sectors": contract["fixed_conditions"]["target_sectors"],
            "orbital_optimization_objective_csf_count": 531,
            "radial_orbital_count": 29, "fixed_original_radial_count": 24,
            "new_unoptimized_radial_count": 5, "electron_count": 69,
            "stationarity_passed": False, "expanded_model_stationarity_certified": False,
            "expanded_stationarity_evaluated": False, "orbital_optimizer_calls": 0,
            "inherited_orbital_residual": result["inherited_orbital_residual"], "orbital_tolerance": 1e-7,
            "stationarity_note": "旧24轨道固定于accepted160；继承原残差4.717047646600387e−5 > 1e−7。新增5轨道只作虚层构造，29轨道扩展模型尚无MCDHF梯度或自洽优化。",
            "scope": "原24轨道逐字保留，加平衡s/p/d五虚轨道，共29径向轨道；原电子no-pair子空间与physical tracker不变。声明全模型2163 CSF，仅精确保留五个完整目标J/宇称扇区1046 CSF及全部根；原同点基线261 CSF。无组态或振幅剪裁，无全谱或全局基态认证。",
            "source": str((CAMPAIGN / directory / "result.json").relative_to(ROOT)),
            "sha256": sha(CAMPAIGN / directory / "result.json"),
            "source_asset": "evidence/yb-fixed160-virtual-fields/" + field + "-result.json",
            "verification_asset": "evidence/yb-fixed160-virtual-fields/" + field + "-verification.json",
            "verification_sha256": sha(CAMPAIGN / directory / "saved_array_audit/verification.json"),
            "independent_saved_array_checks": expected_checks,
            "point_manifest_sha256": result["checkpoint_manifest_sha256"],
            "point_arrays_sha256": result["checkpoint_arrays_sha256"],
            "source_map_sha256": result["source_map_sha256"], "original_metric_sha256": result["original_metric_sha256"],
            "original_tracker_rebased": False, "no_pair_projector_changed": False,
            "ground": original["6s_1/2"], "rows": rows, "multiplets": moments, "is_current": True,
            "samepoint_baseline_nominal_gap_RMS_cm1": baseline["nominal_four_gap_RMS_cm1"],
            "nominal_gap_RMS_cm1": rms, "EOL_Ha": stage["EOL_Ha"],
            "samepoint_baseline_EOL_Ha": baseline["EOL_Ha"],
            "EOL_delta_Ha": stage["EOL_Ha"] - baseline["EOL_Ha"],
            "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
            "boundary_note": "有限固定轨道DC对照。核场与屏蔽场仅用于新增轨道构造，不改变原no-pair边界；屏蔽hN+J/K不叠加到多电子DC。总能降低不代表实验gap降低；原四条1%比较均失败。无Breit/QED/反冲或完整缺失物理界，不与320点或旧修正拼接。"}
        models.append(model)
    evidence = site / contract["new_evidence_dir"]
    evidence.mkdir(parents=True, exist_ok=False)
    assets = []
    for relative in contract["inputs_sha256"]:
        p = ROOT / relative
        local = str(p.relative_to(CAMPAIGN))
        if local.startswith("virtual_only_discriminator/"):
            prefix, rest = "nuclear", local.split("/", 1)[1]
        elif local.startswith("screened_core_virtual_discriminator002/"):
            prefix, rest = "screened_closed68", local.split("/", 1)[1]
        else:
            prefix, rest = "original160", local.split("/", 1)[1]
        suffix = "verification.json" if rest == "saved_array_audit/verification.json" else rest.replace("/", "-")
        name = prefix + "-" + suffix
        shutil.copyfile(p, evidence / name)
        assets.append({"path": relative, "asset": "evidence/yb-fixed160-virtual-fields/" + name, "sha256": sha(p)})
    overlay = {"status": "fixed160_nuclear_vs_closed68_screened_virtual_DC_controls",
        "species": {"z": 70, "charge": 1, "symbol": "Yb", "label": "Yb II"},
        "reference_zero": zero, "nist_payload_sha256": sha(site / "dist/ions/70-1.json.gz"),
        "source_map_sha256": contract["source_map_sha256"],
        "point_arrays_sha256": point["array_sha256"], "original_metric_sha256": point["original_metric_sha256"],
        "stationarity": {"stationarity_passed": False, "orbital_residual": contract["fixed_conditions"]["inherited_residual"],
                         "orbital_tolerance": 1e-7, "expanded_stationarity_evaluated": False},
        "models": models, "source_files": assets, "fixed_original_point_accepted_steps": 160,
        "old_models_preserved": 14, "old_rows_preserved": 56, "append_only_overlay": True,
        "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
        "performance_comparison_valid": False,
        "scope": "Same fixed160 seed/original P/DC/tracker, separate nuclear and closed68 screened constructions in the same finite1046 family. Neither expanded model is stationary; no correction or crosspoint splice."}
    write(site / contract["new_overlay"], overlay)
    write(HERE / "prepared.json", {"contract_sha256": sha(HERE / "contract.json"),
        "new_model_count": 2, "new_rows": 8, "total_models": 16, "total_rows": 64,
        "assets_sha256": {str(p.relative_to(site)): sha(p) for p in [site / contract["new_overlay"], *sorted(evidence.iterdir())]},
        "science_inputs_sha256": contract["inputs_sha256"], "script_sha256": sha(Path(__file__)),
        "publication_performed": False, "new_scientific_calls": 0})


if __name__ == "__main__":
    main()
