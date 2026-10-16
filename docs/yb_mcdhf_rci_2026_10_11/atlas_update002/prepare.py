"""Append exact public531 same-point models to the local immutable atlas."""

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
CONTRACT = json.loads((HERE / "contract.json").read_text())
SITE = Path(CONTRACT["site"])
FROZEN = CAMPAIGN / "public_mixed_resume_run002/source"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FROZEN))

from atomx.reference_data.nist import normalise_configuration  # noqa: E402
from tools.export_nist_atlas import level_record  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def eligible_match(levels, configuration, two_j, parity):
    return [row for row in levels
            if normalise_configuration(row["configuration"]) == configuration
            and level_record(row).two_j == two_j and level_record(row).parity == parity
            and row["energy_kind"] == "evaluated" and row["energy_cm1"] is not None
            and not row["bracketed"] and not level_record(row).flags
            and "?" not in (row["configuration_nist"] or "") + (row["term"] or "")]


def main():
    for relative, expected in CONTRACT["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("science input changed: " + relative)
    pre = load(HERE / "prechange_manifest.json")
    for relative, value in pre["files"].items():
        if sha(SITE / relative) != value["sha256"]:
            raise ValueError("site preimage changed: " + relative)
    source = load(CAMPAIGN / "public_mixed_resume_run002/source_identity.json")
    for relative, expected in source["all_atomx_python_sha256"].items():
        if sha(FROZEN / Path(relative).relative_to("src")) != expected:
            raise ValueError("frozen matcher source changed: " + relative)
    result = load(CAMPAIGN / "public531_endpoint_static_rci/result.json")
    saved = load(CAMPAIGN / "public531_endpoint_static_rci/saved_array_verification.json")
    endpoint = load(CAMPAIGN / "public_mixed_resume_run002/independent_endpoint_verification.json")
    if not (saved["all_passed"] and endpoint["all_identity_replay_checks_passed"]
            and result["static_RCI_numeric_integrity_passed"]):
        raise ValueError("independent full spectrum/endpoint integrity failed")
    if endpoint["original_stationarity_gates_passed"] or result["DC_original_stationarity_passed"]:
        raise ValueError("this contract requires the explicit original stationarity failure")
    if saved["result_sha256"] != sha(CAMPAIGN / "public531_endpoint_static_rci/result.json"):
        raise ValueError("saved-array proof does not bind the actual result")
    payload = json.loads(gzip.decompress((SITE / "dist/ions/70-1.json.gz").read_bytes()))
    zero = eligible_match(payload["levels"], "4f14 6s1", 1, 1)
    if len(zero) != 1 or zero[0]["energy_cm1"] != 0:
        raise ValueError("unique evaluated same6s zero required")
    zero = zero[0]
    labels = ("5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    models = []
    for operator, key, title, tracking_key, moment_key in (
        ("DC", "DC_rows", "公共531 · accepted160 · DC（轨道驻值未过）", "DC_tracking", "DC"),
        ("static_Breit", "static_rows", "公共531 · accepted160 同点 · 静态 Breit RCI", "static_tracking", "static_Breit"),
    ):
        target_rows = result[key]
        if tuple(row["label"] for row in target_rows[1:]) != labels:
            raise ValueError("exact declared physical target label order changed")
        rows = []
        for original in target_rows[1:]:
            config = "4f14 " + original["label"].split("_")[0] + "1"
            selected = eligible_match(payload["levels"], config, original["twoJ"], original["parity"])
            if len(selected) != 1:
                raise ValueError("unique physical ASD branch required")
            ref = selected[0]
            if ref["energy_cm1"] != original["NIST_nominal_cm1"]:
                raise ValueError("retained ASD nominal differs from scientific reference")
            binding = [x for x in result[tracking_key]
                       if x["selected_levels"] == [[original["twoJ"], original["parity"], original["local_root"]]]]
            if len(binding) != 1 or binding[0]["per_level_weight"] != .2:
                raise ValueError("declared equalweight public tracker identity absent")
            calc = original["excitation_cm1"]
            diff = calc - ref["energy_cm1"]
            row = {"label": original["label"], "configuration": config,
                   "two_j": original["twoJ"], "parity": original["parity"],
                   "calculated_cm1": calc, "matching": "unique_nominal_branch", "candidates": 1,
                   "level_id": ref["level_id"], "nist_row": ref["row_index"],
                   "configuration_nist": ref["configuration_nist"], "term_nist": ref["term"],
                   "j_text_nist": ref["j_text"], "reference_cm1": ref["energy_cm1"],
                   "uncertainty_cm1": ref["uncertainty_cm1"], "reference_kind": ref["energy_kind"],
                   "reference_code": ref["reference"], "error_cm1": diff,
                   "error_percent": 100 * diff / ref["energy_cm1"], "relative_goal": .01,
                   "goal_pass": abs(diff / ref["energy_cm1"]) < .01,
                   "handbook_cm1": original["NIST_nominal_cm1"], "asd_minus_handbook_cm1": 0,
                   "reference_zero_level_id": zero["level_id"], "source_energy_Ha": original["energy_Ha"],
                   "source_column": original["column"], "exact_J_local_root": original["local_root"],
                   "physical_reference_overlap_squared": binding[0]["principal_overlap_squared"][0],
                   "physical_reference_overlap_scope": "Original full531 baselineDC physical tracker with original005 waves; not bare CSF weight",
                   "state_average_weight": .2, "state_binding_unambiguous": True}
            if operator == "static_Breit":
                for name in ("absolute_upper_static_minus_DC_Ha", "same_lower_static_minus_DC_Ha", "static_minus_DC_transition_cm1"):
                    row[name] = original[name]
            rows.append(row)
        moments = []
        for moment in result["multiplets"]:
            moments.append({"label": moment["label"], "degeneracy_weights": moment["degeneracy_weights"],
                "centroid_cm1": moment[moment_key]["centroid_cm1"],
                "reference_centroid_cm1": moment["NIST_nominal"]["centroid_cm1"],
                "centroid_gap_cm1": moment["DC_gap_centroid_cm1" if operator == "DC" else "static_gap_centroid_cm1"],
                "splitting_cm1": moment[moment_key]["splitting_cm1"],
                "reference_splitting_cm1": moment["NIST_nominal"]["splitting_cm1"],
                "splitting_gap_cm1": moment["DC_gap_splitting_cm1" if operator == "DC" else "static_gap_splitting_cm1"],
                "reference_uncertainty_cm1": None})
        models.append({"id": "public531_endpoint160_" + operator, "title": title,
            "saved_run": "public_mixed_resume_run002/accepted_0160",
            "model_kind": "nonstationary_endpoint_rci", "operator": operator,
            "rci_csf_count": 531, "orbital_optimization_objective_csf_count": 531,
            "radial_orbital_count": 24, "electron_count": 69, "symmetry_block_count": 16,
            "stationarity_passed": False, "expanded_model_stationarity_certified": False,
            "stationarity_note": "原DC轨道驻值残差4.717047646600387e−5 > 1e−7；160步后仍未过门。能量指示量6.8897239137767995e−9 Ha通过1e−8 Ha，不能替代轨道驻值。静态RCI仅在这个非驻值DC端点固定轨道求谱。",
            "scope": "公共531-CSF DC EOL共同轨道真实优化的accepted160端点；同24轨道、69电子、完整16个J/宇称块与全531根。DC与完整Gaunt+零频迟滞RCI均按原physical tracker匹配，非已收敛MCDHF。",
            "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
            "source": "docs/yb_mcdhf_rci_2026_10_11/public531_endpoint_static_rci/result.json",
            "sha256": sha(CAMPAIGN / "public531_endpoint_static_rci/result.json"),
            "source_asset": "evidence/yb-public531-endpoint160/static-result.json",
            "verification_asset": "evidence/yb-public531-endpoint160/saved-array-verification.json",
            "verification_sha256": sha(CAMPAIGN / "public531_endpoint_static_rci/saved_array_verification.json"),
            "ground": target_rows[0], "rows": rows, "multiplets": moments,
            "is_current": operator == "static_Breit",
            "boundary_note": "同点差能与全根矩阵重放通过，但原轨道驻值失败。不是完整CV关联/动态Breit/QED/反冲；名义ASD比较未认证实验精度、寿命、强度或最高水平。"})
    copies = {"static-result.json": "public531_endpoint_static_rci/result.json",
              "static-contract.json": "public531_endpoint_static_rci/contract.json",
              "static-execution-contract.json": "public531_endpoint_static_rci/execution_contract.json",
              "saved-array-verification.json": "public531_endpoint_static_rci/saved_array_verification.json",
              "static-completion.json": "public531_endpoint_static_rci/completion.json",
              "verified-card-evidence.json": "public531_endpoint_static_rci/card_evidence_verified.json",
              "DC-endpoint-verification.json": "public_mixed_resume_run002/independent_endpoint_verification.json",
              "accepted160-checkpoint.json": "public_mixed_resume_run002/checkpoints/accepted_0160.json",
              "source-identity.json": "public_mixed_resume_run002/source_identity.json",
              "objective-identity.json": "public_mixed_resume_run002/objective_identity.json"}
    evidence = SITE / CONTRACT["new_evidence_dir"]
    evidence.mkdir(parents=True, exist_ok=False)
    sources = []
    for filename, relative in copies.items():
        original = CAMPAIGN / relative
        shutil.copyfile(original, evidence / filename)
        sources.append({"path": str(original.relative_to(ROOT)),
            "asset": f"evidence/yb-public531-endpoint160/{filename}", "sha256": sha(original)})
    overlay = {"status": "public531_accepted160_nonstationary_samepoint_RCI_diagnostic",
        "species": {"z": 70, "charge": 1, "symbol": "Yb", "label": "Yb II"},
        "reference_zero": zero, "nist_payload_sha256": sha(SITE / "dist/ions/70-1.json.gz"),
        "source_map_sha256": result["source_map_sha256"],
        "original_metric_sha256": "aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe",
        "stationarity": result["inherited_DC_stationarity_evidence"], "models": models,
        "static_minus_DC_multiplets": result["multiplets"],
        "same_lower_static_minus_DC_Ha": result["static_minus_DC_same_lower_Ha"],
        "source_files": sources, "append_only_overlay": True,
        "old_models_preserved": 10, "old_rows_preserved": 40,
        "independent_saved_array_checks": saved["checks_count"],
        "production_certified": False, "highest_level_reached": False, "experimental_gate_passed": False,
        "performance_comparison_valid": False,
        "scope": "Separate accepted160 endpoint; no splice into prior fixed005 or one-step stage003 spectrum",
        "comparison_engine": result["CI_provenance"]}
    write(SITE / CONTRACT["new_overlay"], overlay)
    write(HERE / "prepared.json", {"contract_sha256": sha(HERE / "contract.json"),
        "new_model_count": 2, "new_rows": 8, "total_models": 12, "total_rows": 48,
        "assets_sha256": {str(p.relative_to(SITE)): sha(p) for p in [SITE / CONTRACT["new_overlay"], *sorted(evidence.iterdir())]},
        "science_inputs_sha256": CONTRACT["inputs_sha256"], "script_sha256": sha(Path(__file__))})
    print(json.dumps({"new_models": 2, "new_rows": 8, "old_models": 10, "old_rows": 40,
                      "science_calls": 0, "publication_performed": False}))


if __name__ == "__main__":
    main()
