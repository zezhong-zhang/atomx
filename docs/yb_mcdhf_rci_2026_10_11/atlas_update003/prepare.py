"""Append two separate native DC diagnostics from frozen proofs; local-only."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / "docs/yb_mcdhf_rci_2026_10_11"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    contract = load(HERE / "contract.json")
    if contract["status"] != "prospective_before_local_site_edits":
        raise ValueError("Require frozen prospective site preimage/input contract")
    site = Path(contract["site"])
    for relative, expected in contract["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen science input changed: " + relative)
    for relative, item in load(HERE / "prechange_manifest.json")["files"].items():
        if sha(site / relative) != item["sha256"]:
            raise ValueError("Site preimage changed: " + relative)
    frozen = CAMPAIGN / "public_mixed_resume_run003/source"
    source = load(CAMPAIGN / "public_mixed_resume_run003/source_identity.json")
    for relative, expected in source["all_atomx_python_sha256"].items():
        if sha(frozen / Path(relative).relative_to("src")) != expected:
            raise ValueError("Frozen native source changed: " + relative)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(frozen))
    from atomx.reference_data.nist import normalise_configuration
    from tools.export_nist_atlas import level_record

    proof = load(CAMPAIGN / "public_mixed_resume_run003/independent_endpoint_verification.json")
    point = load(CAMPAIGN / "public_mixed_resume_run003/checkpoints/accepted_0160.json")
    completed = load(CAMPAIGN / "public_mixed_resume_run003/completion/completion.json")
    audit = load(CAMPAIGN / "public_mixed_resume_run003/completion/verification.json")
    if not (proof["all_identity_replay_checks_passed"] and proof["checks_count"] == 1011
            and point["accepted_complete_point"] and point["iteration"] == 160
            and point["parent_accepted_iteration"] == 160 and audit["all_passed"]):
        raise ValueError("Exact native accepted320 proof/admission failed")
    if (proof["original_stationarity_gates_passed"] or proof["production_certified"]
            or proof["actual_original_metric_orbital_residual"] <= 1e-7):
        raise ValueError("This point must remain explicitly nonstationary")
    payload = json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))

    def eligible(configuration, twoj, parity):
        return [row for row in payload["levels"]
                if normalise_configuration(row["configuration"]) == configuration
                and level_record(row).two_j == twoj and level_record(row).parity == parity
                and row["energy_kind"] == "evaluated" and row["energy_cm1"] is not None
                and not row["bracketed"] and not level_record(row).flags
                and "?" not in (row["configuration_nist"] or "") + (row["term"] or "")]

    zeros = eligible("4f14 6s1", 1, 1)
    if len(zeros) != 1 or zeros[0]["energy_cm1"] != 0:
        raise ValueError("Require unique evaluated same6s reference zero")
    zero = zeros[0]
    original = {row["label"]: row for row in proof["DC_lower_upper_rows"]}
    identity = {row["label"]: row for row in proof["independently_rebuilt_objective_payload"]["target_identity"]}
    labels = ("5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    rows = []
    for label in labels:
        src, uid = original[label], identity[label]
        configuration = "4f14 " + label.split("_")[0] + "1"
        matches = eligible(configuration, uid["twoJ"], uid["parity"])
        if len(matches) != 1 or matches[0]["energy_cm1"] != src["NIST_nominal_cm1"]:
            raise ValueError("Require exact unique nominal physical ASD branch")
        reference = matches[0]
        bound = [entry for entry in proof["physical_tracking"]
                 if entry["selected_levels"] == [[uid["twoJ"], uid["parity"], uid["root"]]]]
        if len(bound) != 1 or bound[0]["per_level_weight"] != .2:
            raise ValueError("Original full531 physical tracker target not bound")
        error = src["excitation_cm1"] - reference["energy_cm1"]
        row = {"label": label, "configuration": configuration, "two_j": uid["twoJ"],
            "parity": uid["parity"], "calculated_cm1": src["excitation_cm1"],
            "matching": "unique_nominal_branch", "candidates": 1, "level_id": reference["level_id"],
            "nist_row": reference["row_index"], "configuration_nist": reference["configuration_nist"],
            "term_nist": reference["term"], "j_text_nist": reference["j_text"],
            "reference_cm1": reference["energy_cm1"], "uncertainty_cm1": reference["uncertainty_cm1"],
            "reference_kind": reference["energy_kind"], "reference_code": reference["reference"],
            "error_cm1": error, "error_percent": 100 * error / reference["energy_cm1"],
            "relative_goal": .01, "goal_pass": abs(error / reference["energy_cm1"]) < .01,
            "handbook_cm1": src["NIST_nominal_cm1"], "asd_minus_handbook_cm1": 0,
            "reference_zero_level_id": zero["level_id"], "source_energy_Ha": src["energy_Ha"],
            "source_physical_UID": {"twoJ": uid["twoJ"], "parity": uid["parity"], "local_root": uid["root"]},
            "exact_J_local_root": uid["root"],
            "physical_reference_overlap_squared": bound[0]["principal_overlap_squared"][0],
            "physical_reference_overlap_scope": "Original full531 baselineDC tracker and original005 waves; not bare CSF weight",
            "state_average_weight": .2, "state_binding_unambiguous": True,
            "absolute_upper_delta_from_parent160_Ha": src["absolute_upper_delta_from_parent160_Ha"],
            "same_lower_delta_from_parent160_Ha": src["same_lower_delta_from_parent160_Ha"],
            "same_lower_upper_transition_delta_from_parent160_cm1": src["same_lower_upper_transition_delta_from_parent160_cm1"]}
        rows.append(row)
    multiplets = []
    for label, jvalues in (("5d", (3, 5)), ("6p", (1, 3))):
        a, b = [next(row for row in rows if row["label"] == f"{label}_{j}/2") for j in jvalues]
        weights = [j + 1 for j in jvalues]
        center = (a["calculated_cm1"] * weights[0] + b["calculated_cm1"] * weights[1]) / sum(weights)
        reference_center = (a["reference_cm1"] * weights[0] + b["reference_cm1"] * weights[1]) / sum(weights)
        split = b["calculated_cm1"] - a["calculated_cm1"]
        reference_split = b["reference_cm1"] - a["reference_cm1"]
        multiplets.append({"label": label, "degeneracy_weights": weights, "centroid_cm1": center,
            "reference_centroid_cm1": reference_center, "centroid_gap_cm1": center - reference_center,
            "splitting_cm1": split, "reference_splitting_cm1": reference_split,
            "splitting_gap_cm1": split - reference_split, "reference_uncertainty_cm1": None})
    stationarity = {"stationarity_passed": False,
        "orbital_residual": proof["actual_original_metric_orbital_residual"], "orbital_tolerance": 1e-7,
        "estimated_energy_error_Ha": proof["actual_original_metric_energy_indicator_Ha"],
        "energy_tolerance_Ha": 1e-8}
    model = {"id": "public531_endpoint320_DC", "title": "公共531 · 累计320步 · DC（轨道驻值未过）",
        "saved_run": "public_mixed_resume_run003/accepted_0160", "native_total_accepted_steps": 320,
        "parent_accepted_steps": 160, "new_accepted_steps": 160,
        "model_kind": "nonstationary_native_mcdhf_endpoint", "operator": "DC",
        "rci_csf_count": 531, "orbital_optimization_objective_csf_count": 531,
        "radial_orbital_count": 24, "electron_count": 69, "symmetry_block_count": 16,
        "stationarity_passed": False, "expanded_model_stationarity_certified": False,
        "stationarity_note": "原DC轨道驻值残差1.9434848235489216e−6 > 1e−7；累计320接受步仍未过原门。能量指示量9.225371955806855e−12 Ha通过1e−8 Ha，不能代替驻值。",
        "scope": "完整531-CSF DC等权五态EOL公共共同轨道优化的累计320步非驻值端点；两个160步公共续算段，原projector/inverse metric与原physical tracker保持。无新的静态Breit计算，不把此前160点Breit谱拼入这里。",
        "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
        "source": "docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/independent_endpoint_verification.json",
        "sha256": sha(CAMPAIGN / "public_mixed_resume_run003/independent_endpoint_verification.json"),
        "source_asset": "evidence/yb-public531-endpoint320/DC-endpoint-verification.json",
        "verification_asset": "evidence/yb-public531-endpoint320/trajectory-verification.json",
        "verification_sha256": sha(CAMPAIGN / "public_mixed_resume_run003/completion/verification.json"),
        "ground": original["6s_1/2"], "rows": rows, "multiplets": multiplets, "is_current": True,
        "boundary_note": "独立同点全H/CI/原g与pg重放通过，但原驻值失败；残差下降不能替代物理完备或实验认证。四条原1%名义谱比较仍失败。无静态或动态Breit/QED/反冲增量，不能与旧点拼接。",
        "residual_reduction_from_parent160": completed["trajectory_summary"]["residual_reduction_factor"],
        "peripheral_note": "继承的screened Fock诊断文字不重定义真正恢复的原inverse metric，未重建原Fock。"}
    rank = load(CAMPAIGN / "rank_only_discriminator/result.json")
    rank_audit = load(CAMPAIGN / "rank_only_discriminator/saved_array_audit/verification.json")
    tracking = load(CAMPAIGN / "rank_only_discriminator/rank_5p_double_tracking.json")
    if not (rank["numeric_identity_gates_passed"]
            and rank_audit["all_independent_saved_array_checks_passed"]
            and rank_audit["checks_count"] == 3247
            and rank["old24_wave_bytes_preserved"] and not rank["original_tracker_rebased"]
            and not rank["original_stationarity_passed"] and not rank["expanded_stationarity_evaluated"]
            and not rank["QED_recoil_dynamic_or_static_Breit_added"]
            and not rank["full_spectrum_or_globalground_certified"]
            and rank["checkpoint_manifest_sha256"] == sha(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
            and rank["source_map_sha256"] == source["source_map_sha256"]
            and rank["original_metric_sha256"] == point["original_metric_sha256"]
            and not tracking["original_reference_changed"]):
        raise ValueError("Independent fixed160 rank-only admission failed")
    stage = next(x for x in rank["stages"] if x["name"] == "rank_5p_double")
    baseline = next(x for x in rank["stages"] if x["name"] == "baseline")
    if stage["full_model_CSFs"] != 2483 or stage["retained_target_CSFs"] != 1240:
        raise ValueError("Rank model target-sector identity changed")
    rank_by = {x["label"]: x for x in stage["rows"]}
    rank_rows = []
    for label in labels:
        src = rank_by[label]
        configuration = "4f14 " + label.split("_")[0] + "1"
        matches = eligible(configuration, src["twoJ"], src["parity"])
        if len(matches) != 1 or matches[0]["energy_cm1"] != src["NIST_nominal_cm1"]:
            raise ValueError("Unique rank same6s ASD target changed")
        ref = matches[0]
        bound = next(x for x in tracking["physical_tracking"]
                     if x["selected_levels"] == [[src["twoJ"], src["parity"], src["local_root"]]])
        if bound["per_level_weight"] != .2 or bound["principal_overlap_squared"] != src["original_reference_principal_squared"]:
            raise ValueError("Rank original signed tracker binding changed")
        error = src["excitation_cm1"] - ref["energy_cm1"]
        rank_rows.append({"label": label, "configuration": configuration, "two_j": src["twoJ"],
            "parity": src["parity"], "calculated_cm1": src["excitation_cm1"],
            "matching": "unique_nominal_branch", "candidates": 1, "level_id": ref["level_id"],
            "nist_row": ref["row_index"], "configuration_nist": ref["configuration_nist"],
            "term_nist": ref["term"], "j_text_nist": ref["j_text"],
            "reference_cm1": ref["energy_cm1"], "uncertainty_cm1": ref["uncertainty_cm1"],
            "reference_kind": ref["energy_kind"], "reference_code": ref["reference"],
            "error_cm1": error, "error_percent": 100 * error / ref["energy_cm1"],
            "relative_goal": .01, "goal_pass": abs(error / ref["energy_cm1"]) < .01,
            "handbook_cm1": src["NIST_nominal_cm1"], "asd_minus_handbook_cm1": 0,
            "reference_zero_level_id": zero["level_id"], "source_energy_Ha": src["energy_Ha"],
            "source_physical_UID": {"twoJ": src["twoJ"], "parity": src["parity"], "local_root": src["local_root"]},
            "source_column": src["column"], "exact_J_local_root": src["local_root"],
            "physical_reference_overlap_squared": bound["principal_overlap_squared"][0],
            "physical_reference_overlap_scope": "Original full531 baselineDC tracker and original005 waves; not bare CSF weight",
            "state_average_weight": .2, "state_binding_unambiguous": True,
            "absolute_upper_rank_minus_samepoint_baseline_Ha": src["absolute_delta_upper_Ha"],
            "same_lower_rank_minus_samepoint_baseline_Ha": src["absolute_delta_lower_Ha"],
            "rank_minus_samepoint_baseline_transition_cm1": src["transition_delta_upper_minus_lower_cm1"]})
    rank_moments = []
    for group, js in (("5d", (3, 5)), ("6p", (1, 3))):
        a, b = (next(x for x in rank_rows if x["label"] == f"{group}_{j}/2") for j in js)
        w = [j + 1 for j in js]
        center = sum(x["calculated_cm1"] * weight for x, weight in zip((a, b), w, strict=True)) / sum(w)
        refcenter = sum(x["reference_cm1"] * weight for x, weight in zip((a, b), w, strict=True)) / sum(w)
        split, refsplit = b["calculated_cm1"] - a["calculated_cm1"], b["reference_cm1"] - a["reference_cm1"]
        rank_moments.append({"label": group, "degeneracy_weights": w, "centroid_cm1": center,
            "reference_centroid_cm1": refcenter, "centroid_gap_cm1": center - refcenter,
            "splitting_cm1": split, "reference_splitting_cm1": refsplit,
            "splitting_gap_cm1": split - refsplit, "reference_uncertainty_cm1": None})
    rank_model = {"id": "fixed160_5p_double_DC", "title": "固定160步轨道 · 完整5p双激发目标扇区 · DC",
        "model_kind": "fixed_nonstationary_rank_rci", "saved_run": "rank_only_discriminator/rank_5p_double",
        "operator": "DC", "fixed_orbital_point_total_accepted_steps": 160,
        "rci_csf_count": 1240, "declared_full_model_csf_count": 2483,
        "target_sector_csf_count": 1240, "symmetry_block_count": 5,
        "target_sectors": [[1, -1], [1, 1], [3, -1], [3, 1], [5, 1]],
        "orbital_optimization_objective_csf_count": 531, "radial_orbital_count": 24, "electron_count": 69,
        "stationarity_passed": False, "expanded_model_stationarity_certified": False,
        "expanded_stationarity_evaluated": False, "orbital_optimizer_calls": 0,
        "inherited_orbital_residual": rank["inherited_orbital_residual"], "orbital_tolerance": 1e-7,
        "stationarity_note": "固定accepted160的24轨道；继承原残差4.717047646600387e−5 > 1e−7。扩展模型没有计算轨道梯度或优化，不能称为收敛MCDHF。",
        "scope": "在原accepted160固定24轨道上加入完整有限5p双空穴家族。全声明模型2483 CSF，仅精确保留五个完整目标J/宇称扇区1240 CSF及其全部根（原基线261）；无组态/振幅剪裁，无全谱或全局基态认证。",
        "source": "docs/yb_mcdhf_rci_2026_10_11/rank_only_discriminator/result.json",
        "sha256": sha(CAMPAIGN / "rank_only_discriminator/result.json"),
        "source_asset": "evidence/yb-public531-endpoint320/rank-fixed160-result.json",
        "verification_asset": "evidence/yb-public531-endpoint320/rank-fixed160-verification.json",
        "verification_sha256": sha(CAMPAIGN / "rank_only_discriminator/saved_array_audit/verification.json"),
        "point_manifest_sha256": rank["checkpoint_manifest_sha256"], "point_arrays_sha256": rank["checkpoint_arrays_sha256"],
        "source_map_sha256": rank["source_map_sha256"], "original_metric_sha256": rank["original_metric_sha256"],
        "ground": rank_by["6s_1/2"], "rows": rank_rows, "multiplets": rank_moments, "is_current": True,
        "samepoint_baseline_nominal_gap_RMS_cm1": baseline["nominal_four_gap_RMS_cm1"],
        "nominal_gap_RMS_cm1": stage["nominal_four_gap_RMS_cm1"],
        "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
        "boundary_note": "独立保存数组3247项通过；该固定点有限5p双激发对照使D差距增大、P差距缩小，四线RMS由2022.83增到2225.97 cm⁻¹。无虚轨道扩展、Breit/QED/反冲或完整物理误差界；不与320点或旧静态修正相加。"}
    copies = {"DC-endpoint-verification.json": "public_mixed_resume_run003/independent_endpoint_verification.json",
        "accepted320-checkpoint.json": "public_mixed_resume_run003/checkpoints/accepted_0160.json",
        "source-identity.json": "public_mixed_resume_run003/source_identity.json",
        "objective-identity.json": "public_mixed_resume_run003/objective_identity.json",
        "trajectory-completion.json": "public_mixed_resume_run003/completion/completion.json",
        "trajectory-verification.json": "public_mixed_resume_run003/completion/verification.json",
        "native-contract.json": "public_mixed_resume_run003/contract.json",
        "native-execution-contract.json": "public_mixed_resume_run003/execution_contract.json",
        "rank-fixed160-result.json": "rank_only_discriminator/result.json",
        "rank-fixed160-verification.json": "rank_only_discriminator/saved_array_audit/verification.json",
        "rank-fixed160-contract.json": "rank_only_discriminator/contract.json",
        "rank-fixed160-execution-binding.json": "rank_only_discriminator/execution_binding.json",
        "rank-fixed160-tracking.json": "rank_only_discriminator/rank_5p_double_tracking.json"}
    evidence = site / contract["new_evidence_dir"]
    evidence.mkdir(parents=True, exist_ok=False)
    assets = []
    for name, relative in copies.items():
        original_path = CAMPAIGN / relative
        shutil.copyfile(original_path, evidence / name)
        assets.append({"path": str(original_path.relative_to(ROOT)),
                      "asset": "evidence/yb-public531-endpoint320/" + name, "sha256": sha(original_path)})
    overlay = {"status": "public531_accepted320_nonstationary_DC_endpoint",
        "species": {"z": 70, "charge": 1, "symbol": "Yb", "label": "Yb II"},
        "reference_zero": zero, "nist_payload_sha256": sha(site / "dist/ions/70-1.json.gz"),
        "source_map_sha256": source["source_map_sha256"],
        "point_arrays_sha256": point["array_sha256"], "original_metric_sha256": point["original_metric_sha256"],
        "stationarity": stationarity, "models": [model, rank_model], "source_files": assets,
        "native_total_accepted_steps": 320, "parent_point_accepted_steps": 160,
        "old_models_preserved": 12, "old_rows_preserved": 48, "append_only_overlay": True,
        "independent_endpoint_checks": 1011, "saved_trajectory_checks": audit["checks_count"],
        "independent_rank_checks": 3247, "rank_fixed_point_accepted_steps": 160,
        "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
        "performance_comparison_valid": False,
        "scope": "Separate accepted320 full531 DC endpoint and fixed160 finite5p-double target-sector DC control; all earlier48rows retained. No point/correction splice."}
    write(site / contract["new_overlay"], overlay)
    write(HERE / "prepared.json", {"contract_sha256": sha(HERE / "contract.json"),
        "new_model_count": 2, "new_rows": 8, "total_models": 14, "total_rows": 56,
        "assets_sha256": {str(q.relative_to(site)): sha(q) for q in
                          [site / contract["new_overlay"], *sorted(evidence.iterdir())]},
        "science_inputs_sha256": contract["inputs_sha256"], "script_sha256": sha(Path(__file__)),
        "publication_performed": False, "new_scientific_calls": 0})


if __name__ == "__main__":
    main()
