"""Independent saved-data/NIST presentation check; no scientific evaluation."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "docs/yb_mcdhf_rci_2026_10_11"
CONTRACT = json.loads((HERE / "contract.json").read_text())
SITE = Path(CONTRACT["site"])
SOURCE = BASE / "public_mixed_resume_run003/source"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from atomx.reference_data.nist import normalise_configuration
from tools.export_nist_atlas import level_record

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())

def main():
    checks = []
    def check(name, passed): checks.append({"name": name, "passed": bool(passed)})
    def num(name, actual, expected, tolerance=0):
        error = abs(actual - expected)
        checks.append({"name": name, "error": error, "limit": tolerance, "passed": math.isfinite(error) and error <= tolerance})
    pre = load(HERE / "prechange_manifest.json")
    check("contract_preimage_manifest", sha(HERE / "prechange_manifest.json") == CONTRACT["prechange_manifest_sha256"])
    for path, value in pre["files"].items():
        if path != "dist/index.html": check("old_site_exact:" + path, sha(SITE / path) == value["sha256"])
    for path, expected in pre["old_atlas_update_files_sha256"].items():
        check("old_proof_exact:" + path, sha(ROOT / path) == expected)
    for path, expected in CONTRACT["inputs_sha256"].items(): check("science_input_exact:" + path, sha(ROOT / path) == expected)
    for path, expected in CONTRACT["preparation_scripts_sha256"].items(): check("prep_source_exact:" + path, sha(HERE / path) == expected)
    source = load(BASE / "public_mixed_resume_run003/source_identity.json")
    for path, expected in source["all_atomx_python_sha256"].items():
        check("frozen_NIST_epoch:" + path, sha(SOURCE / Path(path).relative_to("src")) == expected)
    prepared = load(HERE / "prepared.json")
    check("producer_contract", prepared["contract_sha256"] == sha(HERE / "contract.json"))
    for path, expected in prepared["assets_sha256"].items(): check("new_asset_exact:" + path, sha(SITE / path) == expected)
    index = load(SITE / "dist/index.json")
    check("118elements_7021species", len(index["elements"]) == 118 and sum(len(x["ions"]) for x in index["elements"]) == 7021)
    check("6019original_gz", len(list((SITE / "dist/ions").glob("*.json.gz"))) == 6019)
    check("original_record_counts", index["counts"]["levels"] == 119394 and index["counts"]["lines"] == 296998 and index["counts"]["nist_ionization"] == 6019)
    payload = json.loads(gzip.decompress((SITE / "dist/ions/70-1.json.gz").read_bytes()))
    oldmodels = [*load(SITE / "dist/updates/yb-public531-endpoint160.json")["models"], *load(SITE / "dist/updates/yb-mcdhf-stage002.json")["models"], *payload["models"]]
    update = load(SITE / CONTRACT["new_overlay"])
    allmodels = [*update["models"], *oldmodels]
    check("old12models48rows", len(oldmodels) == 12 and sum(len(m["rows"]) for m in oldmodels) == 48)
    check("new2models8rows_total14_56", len(update["models"]) == 2 and len(allmodels) == 14 and sum(len(m["rows"]) for m in allmodels) == 56)
    check("unique_models", len({m.get("id", m["saved_run"]) for m in allmodels}) == 14)
    check("old_models_not_rewritten", update["old_models_preserved"] == 12 and update["old_rows_preserved"] == 48 and update["append_only_overlay"])
    endpoint = load(BASE / "public_mixed_resume_run003/independent_endpoint_verification.json")
    checkpoint = load(BASE / "public_mixed_resume_run003/checkpoints/accepted_0160.json")
    completion = load(BASE / "public_mixed_resume_run003/completion/completion.json")
    saved = load(BASE / "public_mixed_resume_run003/completion/verification.json")
    rank = load(BASE / "rank_only_discriminator/result.json")
    rankproof = load(BASE / "rank_only_discriminator/saved_array_audit/verification.json")
    rankstage = next(x for x in rank["stages"] if x["name"] == "rank_5p_double")
    rankbase = next(x for x in rank["stages"] if x["name"] == "baseline")
    ranktracking = load(BASE / "rank_only_discriminator/rank_5p_double_tracking.json")["physical_tracking"]
    check("actual1011_and4963proofs", endpoint["all_identity_replay_checks_passed"] and endpoint["checks_count"] == 1011 and saved["all_passed"] and saved["checks_count"] == 4963)
    check("actual3247rankproof", rankproof["all_independent_saved_array_checks_passed"] and rankproof["checks_count"] == 3247)
    check("rankproof_binds_actual_result", rankproof["bound_input_sha256"][str((BASE / "rank_only_discriminator/result.json").relative_to(ROOT))] == sha(BASE / "rank_only_discriminator/result.json"))
    check("320_point_not160", checkpoint["iteration"] == 160 and checkpoint["parent_accepted_iteration"] == 160 and update["point_arrays_sha256"] == checkpoint["array_sha256"] != rank["checkpoint_arrays_sha256"])
    check("same_original_source_metric", source["source_map_sha256"] == rank["source_map_sha256"] == update["source_map_sha256"] == "10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f" and update["original_metric_sha256"] == rank["original_metric_sha256"] == "aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe")
    check("320_stationarity_failure", not endpoint["original_stationarity_gates_passed"] and not update["stationarity"]["stationarity_passed"] and update["stationarity"]["orbital_tolerance"] == 1e-7 and update["stationarity"]["energy_tolerance_Ha"] == 1e-8)
    num("320_originalresidual", update["stationarity"]["orbital_residual"], endpoint["actual_original_metric_orbital_residual"])
    num("320_originalindicator", update["stationarity"]["estimated_energy_error_Ha"], endpoint["actual_original_metric_energy_indicator_Ha"])
    eligible = [x for x in payload["levels"] if x["energy_kind"] == "evaluated" and x["energy_cm1"] is not None and not x["bracketed"] and not level_record(x).flags and "?" not in (x["configuration_nist"] or "") + (x["term"] or "")]
    zeros = [x for x in eligible if normalise_configuration(x["configuration"]) == "4f14 6s1" and level_record(x).two_j == 1 and level_record(x).parity == 1]
    check("unique_same6s_zero", len(zeros) == 1 and zeros[0] == update["reference_zero"] and zeros[0]["energy_cm1"] == 0)
    for item in update["source_files"]: check("exact_copy:" + item["asset"], sha(SITE / "dist" / item["asset"]) == sha(ROOT / item["path"]) == item["sha256"])
    for model in update["models"]:
        name = model["id"]; isrank = name == "fixed160_5p_double_DC"
        raw = {r["label"]: r for r in (rankstage["rows"] if isrank else endpoint["DC_lower_upper_rows"])}
        tracking = ranktracking if isrank else endpoint["physical_tracking"]
        check(name + "/DC_no_cert", model["operator"] == "DC" and not model["stationarity_passed"] and not model["production_certified"] and not model["highest_level_reached"] and not model["experimental_gate_passed"])
        check(name + "/fixed4labels", [r["label"] for r in model["rows"]] == ["5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"])
        check(name + "/old24_69e", model["radial_orbital_count"] == 24 and model["electron_count"] == 69)
        check(name + "/no_bare_weights", all("bare_character_weight" not in r for r in model["rows"]))
        num(name + "/ground_energy", model["ground"]["energy_Ha"], raw["6s_1/2"]["energy_Ha"])
        if isrank:
            check(name + "/complete5targetsectors", model["rci_csf_count"] == model["target_sector_csf_count"] == 1240 and model["declared_full_model_csf_count"] == 2483 and model["symmetry_block_count"] == 5 and model["target_sectors"] == [[1, -1], [1, 1], [3, -1], [3, 1], [5, 1]])
            check(name + "/fixed160_no_opt_gradient", model["fixed_orbital_point_total_accepted_steps"] == 160 and model["orbital_optimizer_calls"] == 0 and not model["expanded_stationarity_evaluated"] and not rank["expanded_stationarity_evaluated"] and not rank["full_spectrum_or_globalground_certified"])
            check(name + "/rank_source_point", model["sha256"] == sha(BASE / "rank_only_discriminator/result.json") and model["point_arrays_sha256"] == rank["checkpoint_arrays_sha256"])
            num(name + "/inheritedFAIL", model["inherited_orbital_residual"], rank["inherited_orbital_residual"])
            for key,value in (("samepoint_baseline_nominal_gap_RMS_cm1",rankbase["nominal_four_gap_RMS_cm1"]),("nominal_gap_RMS_cm1",rankstage["nominal_four_gap_RMS_cm1"])):num(name+"/"+key,model[key],value)
            num(name+"/independent_RMS",model["nominal_gap_RMS_cm1"],math.sqrt(sum(r["nominal_gap_cm1"]**2 for r in rankstage["rows"] if r["label"] != "6s_1/2")/4),1e-8)
        else:
            check(name + "/full531_16blocks_320steps", model["rci_csf_count"] == 531 and model["symmetry_block_count"] == 16 and model["native_total_accepted_steps"] == 320 and model["orbital_optimization_objective_csf_count"] == 531)
            num(name + "/actualresidualreduction", model["residual_reduction_from_parent160"], completion["trajectory_summary"]["residual_reduction_factor"])
        for row in model["rows"]:
            ident = name + "/" + row["label"]
            matched = [r for r in eligible if normalise_configuration(r["configuration"]) == row["configuration"] and level_record(r).two_j == row["two_j"] and level_record(r).parity == row["parity"]]
            check(ident+"/unique_ASD", len(matched) == 1); src,ref = raw[row["label"]],matched[0]
            check(ident+"/stable_ASD_UID_nullsigma",row["level_id"]==ref["level_id"] and row["nist_row"]==ref["row_index"] and row["uncertainty_cm1"]==ref["uncertainty_cm1"] is None and row["reference_zero_level_id"]==zeros[0]["level_id"])
            for key,value in (("source_energy_Ha",src["energy_Ha"]),("calculated_cm1",src["excitation_cm1"]),("reference_cm1",ref["energy_cm1"]),("error_cm1",src["excitation_cm1"]-ref["energy_cm1"]),("error_percent",100*(src["excitation_cm1"]-ref["energy_cm1"])/ref["energy_cm1"])):num(ident+"/"+key,row[key],value)
            check(ident+"/original1percentFAIL",row["relative_goal"]==.01 and row["goal_pass"] is False and abs(row["error_cm1"]/row["reference_cm1"])>=.01)
            tracked=next(x for x in tracking if x["selected_levels"]==[[row["two_j"],row["parity"],row["exact_J_local_root"]]])
            num(ident+"/signedphysicaloverlap",row["physical_reference_overlap_squared"],tracked["principal_overlap_squared"][0])
            check(ident+"/originalweight_UID",row["state_average_weight"]==tracked["per_level_weight"]==.2 and row["source_physical_UID"]=={"twoJ":row["two_j"],"parity":row["parity"],"local_root":row["exact_J_local_root"]})
            pairs=(("absolute_upper_rank_minus_samepoint_baseline_Ha","absolute_delta_upper_Ha"),("same_lower_rank_minus_samepoint_baseline_Ha","absolute_delta_lower_Ha"),("rank_minus_samepoint_baseline_transition_cm1","transition_delta_upper_minus_lower_cm1")) if isrank else ((k,k) for k in ("absolute_upper_delta_from_parent160_Ha","same_lower_delta_from_parent160_Ha","same_lower_upper_transition_delta_from_parent160_cm1"))
            for key,sourcekey in pairs:num(ident+"/"+key,row[key],src[sourcekey])
        by={r["label"]:r for r in model["rows"]}
        for m in model["multiplets"]:
            js=(3,5) if m["label"]=="5d" else (1,3);a,b=(by[f"{m['label']}_{j}/2"] for j in js);w=[j+1 for j in js]
            check(name+"/"+m["label"]+"/2Jplus1",m["degeneracy_weights"]==w)
            center=(a["calculated_cm1"]*w[0]+b["calculated_cm1"]*w[1])/sum(w);refcenter=(a["reference_cm1"]*w[0]+b["reference_cm1"]*w[1])/sum(w)
            split=b["calculated_cm1"]-a["calculated_cm1"];refsplit=b["reference_cm1"]-a["reference_cm1"]
            for key,value in (("centroid_cm1",center),("reference_centroid_cm1",refcenter),("centroid_gap_cm1",center-refcenter),("splitting_cm1",split),("reference_splitting_cm1",refsplit),("splitting_gap_cm1",split-refsplit)):num(name+"/"+m["label"]+"/"+key,m[key],value,1e-8)
    html=(SITE/"dist/index.html").read_text()
    for name,fragment in (("optionaloverlay","updates/yb-public531-endpoint320.json"),("fallback","此前12模型48行仍保留"),("scope","五个完整目标J/宇称扇区1240"),("points","两个新模型分别属于320点和160点"),("negative_rank","四线名义gap RMS由"),("oldloader","DecompressionStream('gzip')"),("oldpoint","此前160点：DC → 同点静态 Breit RCI"),("copyonwrite","models:[...iteration.models,...(payload.models||[])]")):check("render/"+name,fragment in html)
    result={"all_passed":all(c["passed"] for c in checks),"checks_count":len(checks),"checks":checks,"contract_sha256":sha(HERE/"contract.json"),"verifier_sha256":sha(Path(__file__)),"page_sha256":sha(SITE/"dist/index.html"),"overlay_sha256":sha(SITE/CONTRACT["new_overlay"]),"prior_models":12,"prior_rows":48,"models":14,"rows":56,"old_6019_gzip_files_byte_verified":6019,"scientific_calls":0,"publication_performed":False,"uncompressed_static_bytes":sum(p.stat().st_size for p in (SITE/"dist").rglob("*") if p.is_file())}
    (HERE/"data_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("all_passed","checks_count","models","rows","uncompressed_static_bytes")}))
    if not result["all_passed"]:print(json.dumps([x for x in checks if not x["passed"]]));raise SystemExit(1)

if __name__=="__main__":main()
