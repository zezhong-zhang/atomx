"""Independent local JSON/NIST/samepoint arithmetic and preservation review."""
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
checks = []


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, passed):
    checks.append({"name": name, "passed": bool(passed)})
    if not passed:
        raise ValueError(name)


def close(a, b, atol=1e-8):
    return math.isclose(a, b, rel_tol=0, abs_tol=atol)


def main():
    contract = load(HERE / "contract.json")
    site = Path(contract["site"])
    before = load(HERE / "prechange_manifest.json")
    preserved = [p for p in before["files"] if p != "dist/index.html"]
    check("Every old non-page file byte-preserved", all(sha(site / p) == before["files"][p]["sha256"] for p in preserved))
    gz = list((site / "dist/ions").glob("*.json.gz"))
    check("All6019 old NIST ionpayloads preserved", len(gz) == 6019 and all("dist/ions/" + p.name in before["files"] for p in gz))
    check("Hosting identity and audience unchanged", sha(site / ".openai/hosting.json") == before["files"][".openai/hosting.json"]["sha256"])
    check("Every scientific JSON input unchanged", all(sha(ROOT / p) == expected for p, expected in contract["inputs_sha256"].items()))
    prepared = load(HERE / "prepared.json")
    check("Prepared overlay and source assets intact", all(sha(site / p) == expected for p, expected in prepared["assets_sha256"].items()))
    frozen = ROOT / contract["frozen_source"]
    source = load(CAMPAIGN / "public_mixed_resume_run002/source_identity.json")
    check("Full353 original native source epoch unchanged", len(source["all_atomx_python_sha256"]) == 353 and all(sha(frozen / Path(p).relative_to("src")) == expected for p, expected in source["all_atomx_python_sha256"].items()))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(frozen))
    from atomx.reference_data.nist import normalise_configuration
    from tools.export_nist_atlas import level_record

    payload = json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))
    overlay = load(site / contract["new_overlay"])
    old = payload["models"] + load(site / "dist/updates/yb-mcdhf-stage002.json")["models"] + load(site / "dist/updates/yb-public531-endpoint160.json")["models"] + load(site / "dist/updates/yb-public531-endpoint320.json")["models"]
    check("Original14models56rows intact", len(old) == 14 and sum(len(m["rows"]) for m in old) == 56)
    for asset in ("dist/updates/yb-mcdhf-stage002.json", "dist/updates/yb-public531-endpoint160.json", "dist/updates/yb-public531-endpoint320.json"):
        check("Existing overlay byte-identical/" + asset, sha(site / asset) == before["files"][asset]["sha256"])
    models = overlay["models"]
    check("Exactly16models64rows with2newmodels8rows", len(models) == 2 and sum(len(m["rows"]) for m in models) == 8 and len(old + models) == 16 and sum(len(m["rows"]) for m in old + models) == 64)
    check("Unique model identities", len({m.get("id", m["saved_run"]) for m in old + models}) == 16)
    point = load(CAMPAIGN / "public_mixed_resume_run002/checkpoints/accepted_0160.json")
    check("Original160 point metric/source identity", overlay["point_arrays_sha256"] == point["array_sha256"] and overlay["original_metric_sha256"] == point["original_metric_sha256"] and overlay["source_map_sha256"] == source["source_map_sha256"] and overlay["fixed_original_point_accepted_steps"] == 160)
    check("No expanded stationarity claim", not overlay["stationarity"]["stationarity_passed"] and not overlay["stationarity"]["expanded_stationarity_evaluated"] and overlay["stationarity"]["orbital_residual"] > 1e-7)
    check("No certification/timing claims", not any(overlay[k] for k in ["production_certified", "highest_level_reached", "experimental_gate_passed", "performance_comparison_valid"]))
    labels = ["5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"]
    for model, directory, count in zip(models, ("virtual_only_discriminator", "screened_core_virtual_discriminator002"), (3112, 586), strict=True):
        r = load(CAMPAIGN / directory / "result.json")
        a = load(CAMPAIGN / directory / "saved_array_audit/verification.json")
        t = load(CAMPAIGN / directory / "balanced_virtual_tracking.json")
        stage = next(s for s in r["stages"] if s["name"] == "balanced_virtual")
        baseline = next(s for s in r["stages"] if s["name"] == "baseline")
        own = {x["label"]: x for x in stage["rows"]}
        base = {x["label"]: x for x in baseline["rows"]}
        tag = model["id"]
        check(tag + "/qualified complete saved audit", a["all_independent_saved_array_checks_passed"] and a["checks_count"] == count and r["numeric_identity_gates_passed"])
        check(tag + "/whole finite space scope", model["rci_csf_count"] == 1046 == stage["retained_target_CSFs"] and model["declared_full_model_csf_count"] == 2163 == stage["full_model_CSFs"] and model["symmetry_block_count"] == 5 and model["radial_orbital_count"] == 29 and model["fixed_original_radial_count"] == 24 and model["new_unoptimized_radial_count"] == 5 and model["electron_count"] == 69)
        check(tag + "/original tracker projectors old24 and DC unchanged", r["old24_wave_bytes_preserved"] and not r["original_tracker_rebased"] and not t["original_reference_changed"] and not model["no_pair_projector_changed"] and model["operator"] == "DC" and not r["QED_recoil_dynamic_or_static_Breit_added"])
        check(tag + "/expanded objective not certified", not model["stationarity_passed"] and not model["expanded_model_stationarity_certified"] and not model["expanded_stationarity_evaluated"] and model["orbital_optimizer_calls"] == r["calls"]["optimizer_calls"] == r["calls"]["gradients"] == 0 and model["inherited_orbital_residual"] == r["inherited_orbital_residual"] > model["orbital_tolerance"] == 1e-7)
        check(tag + "/exact scientific result and audit source", model["sha256"] == sha(CAMPAIGN / directory / "result.json") == sha(site / "dist" / model["source_asset"]) and model["verification_sha256"] == sha(CAMPAIGN / directory / "saved_array_audit/verification.json") == sha(site / "dist" / model["verification_asset"]))
        check(tag + "/exact samepoint lower", model["ground"] == own["6s_1/2"] and close(own["6s_1/2"]["absolute_delta_lower_Ha"], own["6s_1/2"]["energy_Ha"] - base["6s_1/2"]["energy_Ha"], 1e-14))
        check(tag + "/fixed labels and unique references", [x["label"] for x in model["rows"]] == labels and len({x["level_id"] for x in model["rows"]}) == 4)
        for row in model["rows"]:
            src = own[row["label"]]
            references = [n for n in payload["levels"] if normalise_configuration(n["configuration"]) == row["configuration"] and level_record(n).two_j == row["two_j"] and level_record(n).parity == row["parity"] and n["energy_kind"] == "evaluated" and n["energy_cm1"] is not None and not n["bracketed"] and not level_record(n).flags and "?" not in (n["configuration_nist"] or "") + (n["term"] or "")]
            check(tag + "/uniqueNIST/" + row["label"], len(references) == 1 and references[0]["level_id"] == row["level_id"] and references[0]["energy_cm1"] == row["reference_cm1"] == src["NIST_nominal_cm1"] and row["uncertainty_cm1"] is None and row["reference_zero_level_id"] == overlay["reference_zero"]["level_id"])
            # The campaign converts its independently paired Ha differences with
            # its original frozen conversion, not a rounded total-energy shift.
            check(tag + "/same upper lower differences/" + row["label"], row["source_energy_Ha"] == src["energy_Ha"] and row["calculated_cm1"] == src["excitation_cm1"] and close(row["absolute_upper_virtual_minus_samepoint_baseline_Ha"], src["energy_Ha"] - base[row["label"]]["energy_Ha"], 1e-14) and row["same_lower_virtual_minus_samepoint_baseline_Ha"] == own["6s_1/2"]["absolute_delta_lower_Ha"] and row["virtual_minus_samepoint_baseline_transition_cm1"] == src["transition_delta_upper_minus_lower_cm1"] and close(row["calculated_cm1"] - base[row["label"]]["excitation_cm1"], row["virtual_minus_samepoint_baseline_transition_cm1"], 5e-7))
            bounds = [x for x in t["physical_tracking"] if x["selected_levels"] == [[row["two_j"], row["parity"], row["exact_J_local_root"]]]]
            check(tag + "/physical signed tracker/" + row["label"], len(bounds) == 1 and bounds[0]["per_level_weight"] == row["state_average_weight"] == .2 and bounds[0]["principal_overlap_squared"][0] == row["physical_reference_overlap_squared"] and row["source_column"] == src["column"] and row["source_physical_UID"] == {"twoJ": src["twoJ"], "parity": src["parity"], "local_root": src["local_root"]})
            error = row["calculated_cm1"] - row["reference_cm1"]
            check(tag + "/original1percentFAIL/" + row["label"], row["error_cm1"] == error and close(row["error_percent"], 100 * error / row["reference_cm1"], 1e-12) and not row["goal_pass"] and row["relative_goal"] == .01 and abs(error / row["reference_cm1"]) > .01)
        rms = math.sqrt(sum(x["error_cm1"] ** 2 for x in model["rows"]) / 4)
        check(tag + "/independent RMS and EOL separate", close(model["nominal_gap_RMS_cm1"], rms) and close(rms, stage["nominal_four_gap_RMS_cm1"]) and model["samepoint_baseline_nominal_gap_RMS_cm1"] == baseline["nominal_four_gap_RMS_cm1"] and model["EOL_delta_Ha"] == stage["EOL_Ha"] - baseline["EOL_Ha"] < 0)
        for moment in model["multiplets"]:
            pair = [x for x in model["rows"] if x["label"].startswith(moment["label"] + "_")]
            weights = [x["two_j"] + 1 for x in pair]
            center = sum(x["calculated_cm1"] * w for x, w in zip(pair, weights, strict=True)) / sum(weights)
            refcenter = sum(x["reference_cm1"] * w for x, w in zip(pair, weights, strict=True)) / sum(weights)
            split = pair[1]["calculated_cm1"] - pair[0]["calculated_cm1"]
            refsplit = pair[1]["reference_cm1"] - pair[0]["reference_cm1"]
            check(tag + "/independent moments/" + moment["label"], moment["degeneracy_weights"] == weights and close(moment["centroid_cm1"], center) and close(moment["reference_centroid_cm1"], refcenter) and close(moment["centroid_gap_cm1"], center - refcenter) and close(moment["splitting_cm1"], split) and close(moment["reference_splitting_cm1"], refsplit) and close(moment["splitting_gap_cm1"], split - refsplit))
    check("Nuclear slight improvement versus screened aggregate worsening", models[0]["nominal_gap_RMS_cm1"] < models[0]["samepoint_baseline_nominal_gap_RMS_cm1"] < models[1]["nominal_gap_RMS_cm1"])
    check("Every proof asset exact byte-copy with local source link", all(sha(site / "dist" / x["asset"]) == x["sha256"] == sha(ROOT / x["path"]) for x in overlay["source_files"]))
    page = (site / "dist/index.html").read_text()
    check("Visible accurate latest scope and old-stage qualification", all(x in page for x in ["固定轨道RCI对照", "29轨道不是已收敛MCDHF", "2163 CSF", "1046 CSF", "当前共16模型64行", "该归档阶段共14模型56行", "屏蔽场只用于构造新增轨道", "最新虚层对照读取失败"]))
    check("Preparation zero science/publication", prepared["new_scientific_calls"] == 0 and not prepared["publication_performed"])
    (HERE / "data_verification.json").write_text(json.dumps({"all_passed": True, "checks_count": len(checks), "checks": checks,
        "models": 16, "rows": 64, "old_models_preserved": 14, "old_rows_preserved": 56,
        "all_old_nonpage_files_preserved": len(preserved), "old_gzip_payloads_preserved": 6019,
        "page_sha256": sha(site / "dist/index.html"), "overlay_sha256": sha(site / contract["new_overlay"]),
        "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(Path(__file__)),
        "science_calls": 0, "publication_performed": False}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"all_passed": True, "checks_count": len(checks)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (HERE / "data_failure.json").write_text(json.dumps({"all_passed": False, "error": repr(error), "checks": checks}, indent=2) + "\n")
        raise
