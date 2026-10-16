"""Independent saved-JSON/site identity checks; no physical or array numerical replay."""
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / "contract.json").read_text())
site = Path(contract["site"])
overlay = json.loads((site / contract["new_overlay"]).read_text())
checks = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, passed):
    checks.append({"name": name, "passed": bool(passed)})
    if not passed:
        raise ValueError(name)


preimage = json.loads((HERE / "prechange_manifest.json").read_text())
changed = [r for r, value in preimage["files"].items() if sha(site / r) != value["sha256"]]
check("sole_changed_v5_asset_is_page", changed == ["dist/index.html"])
check("all6019_gzip_payloads_exact", len(list((site / "dist/ions").glob("*.gz"))) == 6019
      and all(sha(site / r) == v["sha256"] for r, v in preimage["files"].items() if r.endswith(".gz")))
models = []
for relative in ("dist/ions/70-1.json.gz", "dist/updates/yb-mcdhf-stage002.json",
                 "dist/updates/yb-public531-endpoint160.json", "dist/updates/yb-public531-endpoint320.json",
                 "dist/updates/yb-fixed160-virtual-fields.json"):
    path = site / relative
    value = json.loads(gzip.decompress(path.read_bytes())) if relative.endswith(".gz") else json.loads(path.read_text())
    models.extend(value["models"])
check("all16_old_model_objects64_rows_unchanged", models == json.loads((HERE / "old_models.json").read_text())
      and len(models) == 16 and sum(len(m["rows"]) for m in models) == 64)
prior = json.loads((HERE / "append_run004/original_overlay.json").read_text())
check("prior_native003_model_and17point_run_unchanged", overlay["models"][1:] == prior["models"] and overlay["runs"][1:] == prior["runs"])
models = overlay["models"] + models
check("two_terminal_models_total18_rows72_unique_ids", len(models) == 18 and sum(len(m["rows"]) for m in models) == 72
      and len({m.get("id", m["saved_run"]) for m in models}) == 18)
check("predeclared_local16_and_local64_global80_endpoints", [m["id"] for m in overlay["models"]]
      == ["correlated29_run004_endpoint80_DC", "correlated29_run003_endpoint16_DC"]
      and [m["accepted_steps"] for m in overlay["models"]] == [64, 16])
ids = {r["level_id"]: r for r in json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))["levels"]}
HACM = overlay["Hartree_cm1"]
outputs = []
for model in overlay["models"]:
    run = next(r for r in overlay["runs"] if r["model_id"] == model["id"])
    trajectory = run["trajectory"]
    key = model["id"] + "/"
    check(key + "all_points_no_selection", [p["iteration"] for p in trajectory] == list(range(model["accepted_steps"] + 1)))
    check(key + "original_source_metric_identity", model["source_map_sha256"] == contract["source_map_sha256"]
          and model["metric_sha256"] == contract["metric_sha256"]
          and model["geometry_sha256"] == contract["geometry_sha256"]
          and model["objective_sha256"] == contract["objective_sha256"])
    check(key + "native_target_space_DC_original_P_tracker", model["rci_csf_count"] == model["target_sector_csf_count"] == 1046
          and model["radial_orbital_count"] == 29 and model["active_coordinate_count"] == 3451
          and model["state_average_weights"] == [.2] * 5 and not model["full_parent_computed"]
          and not model["original_tracker_rebased"] and not model["no_pair_projector_changed"] and model["operator"] == "DC")
    check(key + "allFAIL_certifications_retained", not model["stationarity_passed"] and not model["experimental_gate_passed"]
          and not model["physical_completeness_certified"] and model["orbital_residual"] > 1e-7
          and model["energy_indicator_Ha"] > 1e-8 and all(not r["goal_pass"] for r in model["rows"]))
    max_line_error = max_rms_error = max_moment_error = 0.
    for point in trajectory:
        target = {r["label"]: r for r in point["targets"]}
        lower = target["6s_1/2"]["energy_Ha"]
        gaps = []
        for label, t in target.items():
            max_line_error = max(max_line_error, abs((t["energy_Ha"] - lower) * HACM - t["excitation_cm1"]))
            if label != "6s_1/2":
                gaps.append(t["excitation_cm1"] - t["NIST_nominal_cm1"])
        rms = math.sqrt(sum(x * x for x in gaps) / 4)
        max_rms_error = max(max_rms_error, abs(rms - point["nominal_four_gap_RMS_cm1"]))
        for moment in point["multiplets"]:
            js = (3, 5) if moment["label"] == "5d" else (1, 3)
            a, b = [target[f'{moment["label"]}_{j}/2'] for j in js]
            center = ((js[0] + 1) * a["excitation_cm1"] + (js[1] + 1) * b["excitation_cm1"]) / sum(j + 1 for j in js)
            max_moment_error = max(max_moment_error, abs(center - moment["centroid_cm1"]),
                                   abs((b["excitation_cm1"] - a["excitation_cm1"]) - moment["splitting_cm1"]))
    check(key + "same_lower_transitions_RMS_and_moments", max_line_error <= 1e-8 and max_rms_error <= 1e-9 and max_moment_error <= 1e-9)
    check(key + "ASD_ids_missing_uncertainty_exact", all(r["reference_cm1"] == ids[r["level_id"]]["energy_cm1"]
          and r["uncertainty_cm1"] is None and ids[r["level_id"]]["uncertainty_cm1"] is None
          and r["reference_zero_level_id"] == overlay["reference_zero_level_id"] for r in model["rows"]))
    first, last = trajectory[0], trajectory[-1]
    check(key + "model_is_exact_terminal", model["targets"] == last["targets"] and model["multiplets"] == last["multiplets"]
          and model["EOL_Ha"] == last["reported_EOL_Ha"] and model["nominal_gap_RMS_cm1"] == last["nominal_four_gap_RMS_cm1"])
    check(key + "all_step_acceptance_scope", first["exact_stored_H_acceptance_qualified"] is None
          and all(p["exact_stored_H_acceptance_qualified"] for p in trajectory[1:]))
    for asset in (model["source_asset"], model["completion_asset"]):
        check(key + "asset/" + asset, (site / "dist" / asset).is_file())
    outputs.append({"model": model["id"], "points": len(trajectory), "RMS_initial_cm1": first["nominal_four_gap_RMS_cm1"],
                    "RMS_terminal_cm1": last["nominal_four_gap_RMS_cm1"], "EOL_delta_Ha": last["reported_EOL_Ha"] - first["reported_EOL_Ha"],
                    "orbital_residual": last["orbital_residual"], "energy_indicator_Ha": last["energy_indicator_Ha"],
                    "max_saved_transition_reconstruction_error_cm1": max_line_error,
                    "max_RMS_reconstruction_error_cm1": max_rms_error, "max_moment_reconstruction_error_cm1": max_moment_error})
shared004 = next(r for r in overlay["runs"] if r["model_id"] == "correlated29_run004_endpoint80_DC")["trajectory"][0]
end003 = next(r for r in overlay["runs"] if r["model_id"] == "correlated29_run003_endpoint16_DC")["trajectory"][-1]
check("shared16_exact_arrays_and_energy_not_extra_update", shared004["source_arrays_sha256"] == end003["source_arrays_sha256"]
      and shared004["reported_EOL_Ha"] == end003["reported_EOL_Ha"]
      and [t["energy_Ha"] for t in shared004["targets"]] == [t["energy_Ha"] for t in end003["targets"]]
      and overlay["shared_global16_point_counted_as_new_update"] is False)
check("all14_exact_scientific_source_assets", len(overlay["source_files"]) == 14 and all(sha(site / "dist" / a["asset"]) == a["sha256"] for a in overlay["source_files"]))
check("no_speedup_or_physics_certification_claim", not overlay["performance_comparison_valid"] and not overlay["stationarity_passed"]
      and not overlay["experimental_gate_passed"] and not overlay["physical_completeness_certified"])
report = {"all_saved_data_and_preservation_checks_passed": all(c["passed"] for c in checks), "checks_count": len(checks),
          "checks": checks, "models": 18, "rows": 72, "original_models_preserved": 16, "original_rows_preserved": 64,
          "unchanged_gzip_payloads": 6019, "trajectory_points_by_run": [17, 65], "distinct_global_points": 81,
          "outputs": outputs, "contract_sha256": sha(HERE / "contract.json"), "append_contract_sha256": sha(HERE / "append_run004/contract.json"),
          "overlay_sha256": sha(site / contract["new_overlay"]), "page_sha256": sha(site / "dist/index.html"),
          "verifier_sha256": sha(__file__), "science_calls": 0, "physical_array_replay": False, "publication_performed": False}
(HERE / "data_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"all_passed": True, "checks": len(checks), "models": 18, "rows": 72, "gzip": 6019, "new_science_calls": 0}))
