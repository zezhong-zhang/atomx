"""Independently verify immutable corpus and same-point model presentation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
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


def main():
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    def num(name, actual, expected, tolerance=1e-8):
        error = abs(actual - expected)
        checks.append({"name": name, "error": error, "limit": tolerance,
                       "passed": bool(math.isfinite(error) and error <= tolerance)})

    pre = load(HERE / "prechange_manifest.json")
    for relative, value in pre["files"].items():
        if relative != "dist/index.html":
            check("original_site_unchanged:" + relative, sha(SITE / relative) == value["sha256"])
    for relative, expected in pre["old_atlas_update_files_sha256"].items():
        check("old_archive_unchanged:" + relative, sha(ROOT / relative) == expected)
    for relative, expected in CONTRACT["inputs_sha256"].items():
        check("science_input_hash:" + relative, sha(ROOT / relative) == expected)
    prepared = load(HERE / "prepared.json")
    check("prospective_contract_bound", prepared["contract_sha256"] == sha(HERE / "contract.json"))
    check("prepare_script_bound", prepared["script_sha256"] == sha(HERE / "prepare.py"))
    for relative, expected in prepared["assets_sha256"].items():
        check("new_exact_asset:" + relative, sha(SITE / relative) == expected)
    source = load(CAMPAIGN / "public_mixed_resume_run002/source_identity.json")
    for relative, expected in source["all_atomx_python_sha256"].items():
        check("frozen_science_source:" + relative, sha(FROZEN / Path(relative).relative_to("src")) == expected)
    index = load(SITE / "dist/index.json")
    check("118_elements_and_original_species", len(index["elements"]) == 118
          and sum(len(x["ions"]) for x in index["elements"]) == 7021)
    check("6019_original_gzip_payloads", len(list((SITE / "dist/ions").glob("*.json.gz"))) == 6019)
    check("original_NIST_counts", index["counts"]["levels"] == 119394
          and index["counts"]["lines"] == 296998 and index["counts"]["nist_ionization"] == 6019)
    old = load(SITE / "dist/updates/yb-mcdhf-stage002.json")
    payload = json.loads(gzip.decompress((SITE / "dist/ions/70-1.json.gz").read_bytes()))
    current = load(SITE / CONTRACT["new_overlay"])
    result = load(CAMPAIGN / "public531_endpoint_static_rci/result.json")
    saved = load(CAMPAIGN / "public531_endpoint_static_rci/saved_array_verification.json")
    endpoint = load(CAMPAIGN / "public_mixed_resume_run002/independent_endpoint_verification.json")
    checkpoint = load(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
    check("checkpoint_original_metric", current["original_metric_sha256"] == checkpoint["original_metric_sha256"])
    check("all_independent_science_integrity_pass", saved["all_passed"]
          and saved["checks_count"] == 3710 and endpoint["all_identity_replay_checks_passed"])
    check("2new_models_8rows_and_12total48", len(current["models"]) == 2
          and sum(len(x["rows"]) for x in current["models"]) == 8
          and len(old["models"]) == 8 and sum(len(x["rows"]) for x in old["models"]) == 32
          and len(payload["models"]) == 2 and sum(len(x["rows"]) for x in payload["models"]) == 8)
    check("original_source_epoch_bound", current["source_map_sha256"] == result["source_map_sha256"]
          == source["source_map_sha256"] == "10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f")
    check("original_metric_bound", current["original_metric_sha256"]
          == "aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe")
    check("same_accepted_point_bound", endpoint["accepted_checkpoint_sha256"]
          == sha(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
          and checkpoint["iteration"] == 160 and checkpoint["accepted_complete_point"])
    check("original_gate_FAILURE_explicit", not current["stationarity"]["stationarity_passed"]
          and not result["DC_original_stationarity_passed"] and not endpoint["original_stationarity_gates_passed"])
    num("actual_original_metric_residual", current["stationarity"]["orbital_residual"],
        endpoint["actual_original_metric_orbital_residual"], 0)
    check("original1e7_1e8_gates", current["stationarity"]["orbital_tolerance"] == 1e-7
          and current["stationarity"]["energy_tolerance_Ha"] == 1e-8)
    eligible = [x for x in payload["levels"] if x["energy_kind"] == "evaluated"
                and x["energy_cm1"] is not None and not x["bracketed"] and not level_record(x).flags
                and "?" not in (x["configuration_nist"] or "") + (x["term"] or "")]
    zeros = [x for x in eligible if normalise_configuration(x["configuration"]) == "4f14 6s1"
             and level_record(x).two_j == 1 and level_record(x).parity == 1]
    check("unique_same6s_zero", len(zeros) == 1 and zeros[0] == current["reference_zero"]
          and zeros[0]["energy_cm1"] == 0)
    for entry in current["source_files"]:
        check("science_asset_copy:" + entry["asset"], sha(SITE / "dist" / entry["asset"])
              == sha(ROOT / entry["path"]) == entry["sha256"])
    for model in current["models"]:
        name = model["id"]
        is_static = model["operator"] == "static_Breit"
        science = result["static_rows" if is_static else "DC_rows"]
        lookup = {x["label"]: x for x in science}
        check(name + "/531fullspace", model["rci_csf_count"] == 531
              and model["orbital_optimization_objective_csf_count"] == 531
              and model["electron_count"] == 69 and model["radial_orbital_count"] == 24
              and model["symmetry_block_count"] == 16)
        check(name + "/nonstationary_no_certification", model["stationarity_passed"] is False
              and model["expanded_model_stationarity_certified"] is False
              and model["production_certified"] is False and model["experimental_gate_passed"] is False
              and model["highest_level_reached"] is False)
        check(name + "/fixed_four_physical_labels", [x["label"] for x in model["rows"]]
              == ["5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"])
        check(name + "/source_not_stage003_splice", model["saved_run"] == "public_mixed_resume_run002/accepted_0160"
              and model["sha256"] == sha(CAMPAIGN / "public531_endpoint_static_rci/result.json"))
        for row in model["rows"]:
            ident = name + "/" + row["label"]
            matched = [x for x in eligible if normalise_configuration(x["configuration"]) == row["configuration"]
                       and level_record(x).two_j == row["two_j"] and level_record(x).parity == row["parity"]]
            check(ident + "/unique_physical_ASD_identity", len(matched) == 1)
            ref, src = matched[0], lookup[row["label"]]
            check(ident + "/ASDstableIDandnullsigma", row["level_id"] == ref["level_id"]
                  and row["uncertainty_cm1"] == ref["uncertainty_cm1"] is None
                  and row["reference_zero_level_id"] == zeros[0]["level_id"])
            num(ident + "/actual_source_energy", row["source_energy_Ha"], src["energy_Ha"], 0)
            num(ident + "/actual_source_excitation", row["calculated_cm1"], src["excitation_cm1"], 0)
            num(ident + "/ASDreference", row["reference_cm1"], ref["energy_cm1"], 0)
            num(ident + "/same6s_ASD_residual", row["error_cm1"], row["calculated_cm1"] - ref["energy_cm1"], 0)
            num(ident + "/relative_percent", row["error_percent"], 100 * row["error_cm1"] / ref["energy_cm1"], 0)
            check(ident + "/original1percent", row["relative_goal"] == .01
                  and row["goal_pass"] == (abs(row["error_cm1"] / ref["energy_cm1"]) < .01))
            check(ident + "/trackercolumn_localrank", row["source_column"] == src["column"]
                  and row["exact_J_local_root"] == src["local_root"] and row["state_average_weight"] == .2)
            tracked = next(x for x in result["static_tracking" if is_static else "DC_tracking"]
                           if x["selected_levels"] == [[row["two_j"], row["parity"], row["exact_J_local_root"]]])
            num(ident + "/actual_physical_reference_overlap", row["physical_reference_overlap_squared"],
                tracked["principal_overlap_squared"][0], 0)
            check(ident + "/overlap_not_bare_weight", "bare_character_weight" not in row
                  and "not bare CSF weight" in row["physical_reference_overlap_scope"])
            if is_static:
                for key in ("absolute_upper_static_minus_DC_Ha", "same_lower_static_minus_DC_Ha", "static_minus_DC_transition_cm1"):
                    num(ident + "/" + key, row[key], src[key], 0)
                dc = next(x for x in result["DC_rows"] if x["label"] == row["label"])
                num(ident + "/same_point_transition_difference", row["static_minus_DC_transition_cm1"],
                    row["calculated_cm1"] - dc["excitation_cm1"], 1e-7)
        by = {x["label"]: x for x in model["rows"]}
        for moment in model["multiplets"]:
            twoj = (3, 5) if moment["label"] == "5d" else (1, 3)
            a, b = (by[f"{moment['label']}_{j}/2"] for j in twoj)
            weights = [j + 1 for j in twoj]
            center = (a["calculated_cm1"] * weights[0] + b["calculated_cm1"] * weights[1]) / sum(weights)
            refcenter = (a["reference_cm1"] * weights[0] + b["reference_cm1"] * weights[1]) / sum(weights)
            split = b["calculated_cm1"] - a["calculated_cm1"]
            refsplit = b["reference_cm1"] - a["reference_cm1"]
            check(name + "/statisticalweights", weights == moment["degeneracy_weights"])
            for key, expected in (("centroid_cm1", center), ("reference_centroid_cm1", refcenter),
                                  ("centroid_gap_cm1", center - refcenter), ("splitting_cm1", split),
                                  ("reference_splitting_cm1", refsplit), ("splitting_gap_cm1", split - refsplit)):
                num(name + "/" + moment["label"] + "/" + key, moment[key], expected)
    check("same_point_multipletledger", current["static_minus_DC_multiplets"] == result["multiplets"])
    html = (SITE / "dist/index.html").read_text()
    check("explicit_nonstationary_status", "原轨道驻值仍未通过" in html and "不是已收敛MCDHF" in html)
    check("visible_absolute_cancellation_and_gap", "该绝对修正不是谱线改变量" in html and "1708–2313" in html)
    check("separate_newoverlay_and_failure", "endpoint_update_error" in html and "公共531端点对照读取失败" in html
          and "updates/yb-public531-endpoint160.json" in html)
    check("oldoverlay_corpus_loader", "updates/yb-mcdhf-stage002.json" in html and "DecompressionStream('gzip')" in html)
    check("copy_on_write_overlay_not_cachemutation", "models:[...endpoint.models,...(payload.models||[])]" in html)
    output = {"all_passed": all(x["passed"] for x in checks), "checks_count": len(checks), "checks": checks,
        "contract_sha256": sha(HERE / "contract.json"), "prepared_sha256": sha(HERE / "prepared.json"),
        "page_sha256": sha(SITE / "dist/index.html"), "overlay_sha256": sha(SITE / CONTRACT["new_overlay"]),
        "original_gzip_files_byte_verified": 6019, "old_atlas_archive_files_verified": len(pre["old_atlas_update_files_sha256"]),
        "new_matched_rows": 8, "prior_rows_unchanged": 40,
        "uncompressed_static_bytes": sum(p.stat().st_size for p in (SITE / "dist").rglob("*") if p.is_file()),
        "publication_performed_here": False, "scientific_calls": 0, "script_sha256": sha(Path(__file__))}
    (HERE / "data_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"all_passed": output["all_passed"], "checks_count": len(checks),
        "failed": [x for x in checks if not x["passed"]], "static_bytes": output["uncompressed_static_bytes"]}))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
