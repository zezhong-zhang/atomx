"""Check byte-preserved corpus, independent state rematching and source ledger."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = json.loads((HERE / "contract.json").read_text())
SITE = Path(CONTRACT["site"])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol005_projected_increment/source"))

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
        difference = abs(actual - expected)
        checks.append({"name": name, "error": difference, "limit": tolerance,
                       "passed": bool(math.isfinite(difference) and difference <= tolerance)})

    preimage, prepared = load(HERE / "prechange_manifest.json"), load(HERE / "prepared.json")
    check("preparation_script_identity", prepared["script_sha256"] == sha(HERE / "prepare.py"))
    check("amendment_script_identity", prepared["addendum_script_sha256"] == sha(HERE / "amend_probe.py"))
    for relative, record in preimage["files"].items():
        if relative != "dist/index.html":
            check("original_payload_unchanged:" + relative, sha(SITE / relative) == record["sha256"])
    check("original_gzip_count_6019", preimage["gzip_files"] == 6019
          and len(list((SITE / "dist/ions").glob("*.json.gz"))) == 6019)
    check("hosting_manifest_unchanged", sha(SITE / ".openai/hosting.json") == preimage["files"][".openai/hosting.json"]["sha256"])
    for relative, expected in CONTRACT["inputs_sha256"].items():
        check("scientific_input_sha:" + relative, sha(ROOT / relative) == expected)
    addendum = load(HERE / "contract_addendum_stage003.json")
    check("addendum_preceded_display_change", prepared["addendum_contract_sha256"] == sha(HERE / "contract_addendum_stage003.json")
          and addendum["previous_prepared_sha256"] == sha(HERE / "prepared_stage002.json"))
    for relative, expected in addendum["inputs_sha256"].items():
        check("addendum_scientific_input_sha:" + relative, sha(ROOT / relative) == expected)
    for relative, expected in prepared["new_assets_sha256"].items():
        check("prepared_asset_sha:" + relative, sha(SITE / relative) == expected)
    index = load(SITE / "dist/index.json")
    check("118_elements_unchanged", len(index["elements"]) == 118 and index["highest_level_reached"] is False)
    check("NIST_record_counts", index["counts"]["levels"] == 119394 and index["counts"]["lines"] == 296998
          and index["counts"]["nist_ionization"] == 6019)
    old = json.loads(gzip.decompress((SITE / "dist/ions/70-1.json.gz").read_bytes()))
    current = load(SITE / "dist/updates/yb-mcdhf-stage002.json")
    probe = load(ROOT / "docs/yb_mcdhf_rci_2026_10_11/mixed_ci_stationarity_probe/result.json")
    probe_status = current["latest_stationarity_probe"]
    num("mixed_probe_actual_baseline_residual", probe_status["baseline_residual"], probe["baseline"]["original_metric_orbital_residual"], 1e-15)
    num("mixed_probe_actual_one_step_residual", probe_status["one_step_residual"], probe["one_step"]["original_metric_orbital_residual"], 1e-15)
    num("mixed_probe_actual_average_energy_change", probe_status["EOL_delta_Ha"], probe["one_step"]["EOL_delta_Ha"], 1e-15)
    check("mixed_probe_has_no_stationarity_or_spectral_promotion", probe_status["baseline_stationarity_passed"] is False
          and probe_status["one_step_stationarity_passed"] is False and probe_status["four_NIST_gaps_improved"] is False
          and probe_status["full_expanded_MCDHF_completed"] is False and probe_status["static_Breit_spectra_changed"] is False)
    native = load(ROOT / "docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol005_projected_increment/result.json")
    balanced = load(ROOT / "docs/yb_mcdhf_rci_2026_10_11/balanced_cv_replay/result.json")
    lookup = {s["name"]: s for s in balanced["stages"]}
    check("historical_models_preserved_byte_exact", len(old["models"]) == 2 and current["historical_models_preserved"] == 2
          and sha(SITE / "dist/ions/70-1.json.gz") == current["nist_payload_sha256"])
    check("eight_new_models_32_rows", len(current["models"]) == 8 and sum(len(m["rows"]) for m in current["models"]) == 32)
    check("new_model_ids_unique", len({m["id"] for m in current["models"]}) == 8)
    eligible = [r for r in old["levels"] if r["energy_kind"] == "evaluated" and r["energy_cm1"] is not None
                and not level_record(r).flags and not r["bracketed"]
                and "?" not in (r["configuration_nist"] or "") + (r["term"] or "")]
    zeros = [r for r in eligible if normalise_configuration(r["configuration"]) == "4f14 6s1"
             and level_record(r).two_j == 1 and level_record(r).parity == 1]
    check("unique_same6s_zero", len(zeros) == 1 and zeros[0]["energy_cm1"] == 0 and zeros[0] == current["reference_zero"])
    for source in current["source_files"]:
        check("exact_evidence_copy:" + source["asset"], sha(SITE / "dist" / source["asset"]) == sha(ROOT / source["path"]) == source["sha256"])
    for model in current["models"]:
        prefix = model["id"]
        is_balanced = model["model_kind"] == "fixed_orbital_rci"
        source_rows = lookup[model["saved_run"]]["tracked_levels"][model["operator"]] if is_balanced else native[
            "rows_dc" if model["operator"] == "DC" else "rows_rci"]
        by_source = {row["label"]: row for row in source_rows}
        check(prefix + "/no_physical_promotion", model["highest_level_reached"] is False
              and model["production_certified"] is False and model["experimental_gate_passed"] is False)
        check(prefix + "/only5CSF_orbitals_optimized", model["orbital_optimized_csf_count"] == 5)
        if is_balanced:
            check(prefix + "/expanded_stationarity_not_certified", model["stationarity_passed"] is None
                  and model["expanded_model_stationarity_certified"] is False)
        else:
            check(prefix + "/original_stationarity_certified", model["stationarity_passed"] is True)
        for row in model["rows"]:
            name = prefix + "/" + row["label"]
            matched = [r for r in eligible if normalise_configuration(r["configuration"]) == row["configuration"]
                       and level_record(r).two_j == row["two_j"] and level_record(r).parity == row["parity"]]
            check(name + "/unique_identity_not_row_number", len(matched) == 1)
            reference, source = matched[0], by_source[row["label"]]
            check(name + "/stable_ASD_state_id", row["level_id"] == reference["level_id"])
            check(name + "/null_uncertainty_not_zero", row["uncertainty_cm1"] == reference["uncertainty_cm1"] is None)
            check(name + "/common_zero_identity", row["reference_zero_level_id"] == zeros[0]["level_id"])
            calculated = source["excitation_from_tracked_6s_cm1"] if is_balanced else source["computed_cm1"]
            num(name + "/scientific_energy_cm1", row["calculated_cm1"], calculated)
            num(name + "/scientific_absolute_energy_Ha", row["source_energy_Ha"], source["energy_Ha"], 1e-12)
            num(name + "/matched_reference_cm1", row["reference_cm1"], reference["energy_cm1"])
            error = calculated - reference["energy_cm1"]
            num(name + "/residual_cm1", row["error_cm1"], error)
            num(name + "/relative_percent", row["error_percent"], 100 * error / reference["energy_cm1"])
            check(name + "/original1percent_gate", row["relative_goal"] == .01
                  and row["goal_pass"] == (abs(error / reference["energy_cm1"]) < .01))
            if is_balanced:
                check(name + "/actual_unambiguous_state_binding", row["state_binding_unambiguous"] == source["binding_unambiguous"] is True)
                num(name + "/actual_signed_bare_character", row["bare_character_weight"], source["bare_character_weight"], 1e-12)
                for key in ("energy_delta_from_previous_Ha", "energy_delta_from_bare_Ha", "transition_delta_from_previous_cm1", "transition_delta_from_bare_cm1"):
                    num(name + "/" + key, row[key], source[key])
        by_label = {row["label"]: row for row in model["rows"]}
        for moment in model["multiplets"]:
            orbital = moment["label"]
            js = (3, 5) if orbital == "5d" else (1, 3)
            a, b = (by_label[f"{orbital}_{j}/2"] for j in js)
            denominator = sum(j + 1 for j in js)
            center = (a["calculated_cm1"] * (js[0]+1) + b["calculated_cm1"] * (js[1]+1)) / denominator
            ref_center = (a["reference_cm1"] * (js[0]+1) + b["reference_cm1"] * (js[1]+1)) / denominator
            split = b["calculated_cm1"] - a["calculated_cm1"]
            ref_split = b["reference_cm1"] - a["reference_cm1"]
            for key, expected in (("centroid_cm1", center), ("reference_centroid_cm1", ref_center),
                                  ("centroid_gap_cm1", center-ref_center), ("splitting_cm1", split),
                                  ("reference_splitting_cm1", ref_split), ("splitting_gap_cm1", split-ref_split)):
                num(prefix + "/" + orbital + "/" + key, moment[key], expected)
            if is_balanced:
                for key in ("centroid_cm1", "splitting_cm1"):
                    num(prefix + "/" + orbital + "/scientific_" + key, moment[key], lookup[model["saved_run"]]["multiplets"][model["operator"]][orbital][key])
    html = (SITE / "dist/index.html").read_text()
    check("explicit_expanded_MCDHF_boundary_visible", "扩展空间的 MCDHF 驻值尚未认证" in html)
    check("negative_probe_visible_without_ninth_model", "扩展混合CI驻值探针未过" in html
          and "不修改这里的固定005轨道静态Breit谱" in html and len(current["models"]) == 8)
    check("overlay_only_for_Yb_II", "element.z===70&&ion.charge===1" in html)
    check("original_gzip_loader_preserved", "DecompressionStream('gzip')" in html)
    check("no_repeated_overlay_mutation", "models:[...current.models,...(payload.models||[])]" in html and "payload={...payload" in html)
    check("evidence_links_and_download_present", "下载本轮计算对照 JSON" in html and "currentSources" in html)
    new_assets = sorted(p for p in (SITE / "dist").rglob("*") if p.is_file() and str(p.relative_to(SITE)) not in preimage["files"])
    out = {"all_passed": all(c["passed"] for c in checks), "checks_count": len(checks), "checks": checks,
           "contract_sha256": sha(HERE / "contract.json"), "prepared_sha256": sha(HERE / "prepared.json"),
           "page_sha256": sha(SITE / "dist/index.html"), "overlay_sha256": sha(SITE / "dist/updates/yb-mcdhf-stage002.json"),
           "changed_original_files": ["dist/index.html"], "new_site_files": [str(p.relative_to(SITE)) for p in new_assets],
           "original_gzip_files_verified": 6019, "uncompressed_static_bytes": sum(p.stat().st_size for p in (SITE / "dist").rglob("*") if p.is_file()),
           "publication_performed_here": False, "scientific_source_changed_here": False, "script_sha256": sha(Path(__file__))}
    (HERE / "data_verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"all_passed": out["all_passed"], "checks_count": len(checks), "static_bytes": out["uncompressed_static_bytes"],
                      "failed": [c for c in checks if not c["passed"]]}))
    if not out["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
