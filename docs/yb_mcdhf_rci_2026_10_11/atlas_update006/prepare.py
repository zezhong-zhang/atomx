"""Thin append-only saved-JSON wrapper; reuse atlas005 utilities and validated model template."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UTILITY = HERE.with_name("atlas_update005") / "prepare.py"
spec = importlib.util.spec_from_file_location("atlas005_saved_utilities", UTILITY)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha, load, write, LABELS = shared.sha, shared.load, shared.write, shared.LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", required=True)
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    completion_path = (ROOT / args.completion).resolve()
    assert contract["status"] == "prospective_before_local_site_edits"
    assert completion_path == (ROOT / contract["completion_path"]).resolve()
    assert sha(UTILITY) == contract["shared_utilities_sha256"]
    assert sha(HERE / "prechange_manifest.json") == contract["prechange_manifest_sha256"]
    site = Path(contract["site"])
    preimage = load(HERE / "prechange_manifest.json")
    for relative, item in preimage["files"].items():
        assert sha(site / relative) == item["sha256"], relative
    for relative, expected in contract["inputs_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(HERE / "original_overlay.json") == contract["old_overlay_sha256"]
    assert len(contract["new_source_filenames"]) == 6
    original = load(HERE / "original_overlay.json")
    model = copy.deepcopy(original["models"][0])
    assert model["id"] == "correlated29_run004_endpoint80_DC"
    run = completion_path.parent
    closed, actual, receipt, scientific = [load(run / p) for p in
                                          ("completion.json", "actual/result.json", "root_execution.json", "contract.json")]
    assert receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0
    assert closed["all_completion_metadata_checks_passed"] and closed["scientific_calls_in_completion"] == 0
    assert closed["actual_live_controls"] == len(actual["controls"]) == 8233 and all(c["passed"] for c in actual["controls"])
    assert closed["checks"] == 11 and not closed["stationarity_passed"]
    assert actual["source_map_sha256"] == scientific["source"]["candidate356_map_sha256"] == model["source_map_sha256"] == contract["source_map_sha256"]
    assert scientific["fixed"]["target_CSFs"] == 1046 and len(scientific["fixed"]["keys29"]) == 29
    assert scientific["fixed"]["five_weights"] == [.2] * 5 and scientific["fixed"]["full_model_not_computed"] and scientific["fixed"]["all29_allowed_to_relax"]
    assert not actual["energy_callback_used"] and not actual["ordinary_reported_objective_changed"]
    assert not actual["stationarity_passed"] and not actual["experimental_certification"] and not actual["physical_completeness_certified"]
    assert scientific["gates"]["orbital_residual"] == 1e-7 and scientific["gates"]["energy_indicator_Ha"] == 1e-8
    assert closed["actual_new_accepted_steps"] == 128 and closed["total_global_accepted_steps"] == 208
    assert closed["all_qualified_finite_stored_H_acceptances_qualified"]
    assert [p["iteration"] for p in actual["trajectory"]] == [p["iteration"] for p in closed["trajectory"]] == list(range(129))
    nist = load_nist(site)
    references = {r["label"]: r for r in model["rows"]}
    for ref in references.values():
        matches = [r for r in nist["levels"] if r["level_id"] == ref["level_id"]]
        assert len(matches) == 1 and matches[0]["energy_cm1"] == ref["reference_cm1"]
        assert ref["uncertainty_cm1"] is matches[0]["uncertainty_cm1"] is None
    uids = {t["label"]: t["UID"] for t in model["targets"]}
    trajectory = []
    for summary, ledger in zip(actual["trajectory"], closed["trajectory"], strict=True):
        manifest = load(ROOT / summary["manifest_path"])
        assert sha(ROOT / summary["manifest_path"]) == summary["manifest_sha256"]
        assert manifest["array_sha256"] == summary["array_sha256"] and manifest["checkpoint_qualified"]
        for key in ("metric", "geometry", "objective", "source_map"):
            assert manifest[key + "_sha256"] == contract[key + "_sha256"]
        targets = copy.deepcopy(manifest["targets"])
        by = {t["label"]: t for t in targets}
        assert set(by) == set(LABELS) and all(by[label]["UID"] == uids[label] for label in LABELS)
        lower = by["6s_1/2"]["energy_Ha"]
        first = {t["label"]: t for t in trajectory[0]["targets"]} if trajectory else copy.deepcopy(by)
        for t in targets:
            track = t["tracking"]
            assert track["reference_levels"] == [t["UID"]] and track["selected_levels"] == [t["selected_level"]]
            assert t["selected_level"][:2] == t["UID"][:2] and track["per_level_weight"] == .2
            assert abs((t["energy_Ha"] - lower) * original["Hartree_cm1"] - t["excitation_cm1"]) <= 1e-8
            assert t["NIST_uncertainty_cm1"] is None and t["nominal_gap_cm1"] == t["excitation_cm1"] - t["NIST_nominal_cm1"]
            if t["label"] in references:
                assert t["NIST_nominal_cm1"] == references[t["label"]]["reference_cm1"]
            t.update(delta_upper_from_initial_Ha=t["energy_Ha"] - first[t["label"]]["energy_Ha"],
                     delta_lower_from_initial_Ha=lower - first["6s_1/2"]["energy_Ha"],
                     transition_shift_from_initial_cm1=t["excitation_cm1"] - first[t["label"]]["excitation_cm1"], state_average_weight=.2)
        moments = copy.deepcopy(model["multiplets"])
        for moment in moments:
            js = (3, 5) if moment["label"] == "5d" else (1, 3)
            a, b = [by[f'{moment["label"]}_{j}/2'] for j in js]
            center = sum((j + 1) * t["excitation_cm1"] for j, t in zip(js, (a, b), strict=True)) / sum(j + 1 for j in js)
            split = b["excitation_cm1"] - a["excitation_cm1"]
            saved = next(m for m in manifest["multiplets"] if m["label"] == moment["label"])
            assert abs(center - saved["centroid_cm1"]) <= 1e-9 and abs(split - saved["splitting_cm1"]) <= 1e-9
            moment.update(centroid_cm1=center, splitting_cm1=split, centroid_gap_cm1=center - moment["reference_centroid_cm1"], splitting_gap_cm1=split - moment["reference_splitting_cm1"])
        rms = math.sqrt(sum(by[label]["nominal_gap_cm1"] ** 2 for label in LABELS[1:]) / 4)
        diag = manifest["diagnostics"]
        assert abs(rms - ledger["nominal_four_gap_RMS_cm1"]) <= 1e-9 and diag["energy_ha"] == ledger["reported_EO_Ha"]
        assert abs(sum(t["energy_Ha"] * .2 for t in targets) - diag["energy_ha"]) <= 2e-11
        assert summary["iteration"] == manifest["iteration"] == ledger["iteration"]
        assert ledger["global_accepted_step"] == 80 + summary["iteration"]
        assert not manifest["stationarity_passed"] and not manifest["experimental_certification"] and not manifest["physical_completeness_certification"]
        trajectory.append({"iteration": summary["iteration"], "global_accepted_step": ledger["global_accepted_step"],
                           "reported_EOL_Ha": diag["energy_ha"], "orbital_residual": diag["orbital_residual"], "energy_indicator_Ha": diag["estimated_energy_error"],
                           "nominal_four_gap_RMS_cm1": rms, "targets": targets, "multiplets": moments,
                           "source_manifest": summary["manifest_path"], "source_manifest_sha256": summary["manifest_sha256"], "source_arrays_sha256": summary["array_sha256"],
                           "stationarity_passed": False, "experimental_certification": False, "exact_stored_H_acceptance_qualified": ledger["exact_acceptance_qualified"]})
    first, last = trajectory[0], trajectory[-1]
    assert same_shared_point(first, model)
    by = {t["label"]: t for t in last["targets"]}
    for row in model["rows"]:
        t = by[row["label"]]
        row.update(calculated_cm1=t["excitation_cm1"], error_cm1=t["nominal_gap_cm1"], error_percent=100 * t["nominal_gap_cm1"] / row["reference_cm1"], goal_pass=abs(t["nominal_gap_cm1"] / row["reference_cm1"]) < .01,
                   source_energy_Ha=t["energy_Ha"], source_column=t["column"], exact_J_local_root=t["selected_level"][2], physical_reference_overlap_squared=t["tracking"]["principal_overlap_squared"][0],
                   **{k: t[k] for k in ("delta_upper_from_initial_Ha", "delta_lower_from_initial_Ha", "transition_shift_from_initial_cm1")})
    assert all(not row["goal_pass"] for row in model["rows"])
    asset_prefix = contract["new_evidence_dir"].removeprefix("dist/") + "/"
    model.update(id=contract["model_id"], title="原生29轨道相关MCDHF · run005末态累计208步 · DC", saved_run=run.name + "/accepted_0128",
                 accepted_steps=128, global_initial_step=80, global_accepted_steps=208, ground=by["6s_1/2"], targets=last["targets"], multiplets=last["multiplets"],
                 EOL_Ha=last["reported_EOL_Ha"], initial_EOL_Ha=first["reported_EOL_Ha"], EOL_delta_Ha=last["reported_EOL_Ha"] - first["reported_EOL_Ha"],
                 nominal_gap_RMS_cm1=last["nominal_four_gap_RMS_cm1"], initial_nominal_gap_RMS_cm1=first["nominal_four_gap_RMS_cm1"], initial_orbital_residual=first["orbital_residual"],
                 orbital_residual=last["orbital_residual"], energy_indicator_Ha=last["energy_indicator_Ha"], point_manifest_sha256=last["source_manifest_sha256"], point_arrays_sha256=last["source_arrays_sha256"],
                 elapsed_uncontrolled_s=receipt["elapsed_uncontrolled_seconds"], peak_RSS_MiB=actual["peak_RSS_MiB"], runtime_controls=8233, completion_metadata_checks=11, completed_calls=actual["completed_calls"],
                 source=str((run / "actual/result.json").relative_to(ROOT)), sha256=sha(run / "actual/result.json"), source_asset=asset_prefix + "actual-result.json", completion_asset=asset_prefix + "completion.json",
                 stationarity_note="续128个共同轨道更新、累计208步后，原残差与能量指示量仍未通过1e−7 / 1e−8 Ha门。步数上限结束；能量下降不代表驻值或实验精度。",
                 boundary_note="只取预声明run005末态local128/global208；全部129点保留，shared80是起点，不算新增更新。不按最小gap选点。DC，未加Breit、QED、反冲或经验修正，f043源，不拼接旧修正。858秒是未受控墙钟，不是性能比较。8233运行控制及11元数据闭环不替代独立末态HCIg重放或完整物理认证。")
    evidence = site / contract["new_evidence_dir"]
    evidence.mkdir(exist_ok=False)
    assets = []
    for filename in contract["new_source_filenames"]:
        name = filename.replace("/", "-")
        shutil.copyfile(run / filename, evidence / name)
        assets.append({"path": str((run / filename).relative_to(ROOT)), "asset": asset_prefix + name, "sha256": sha(run / filename)})
    overlay = copy.deepcopy(original)
    overlay["models"].insert(0, model)
    overlay["runs"].insert(0, {"model_id": model["id"], "terminal_iteration": 128, "global_initial_step": 80, "global_terminal_step": 208, "trajectory": trajectory,
                               "receipt": {k: receipt[k] for k in ("started_utc", "finished_utc", "exit_code", "elapsed_uncontrolled_seconds", "timing_scope")}})
    overlay["source_files"] = assets + overlay["source_files"]
    overlay.update(total_models=19, total_rows=76, previous_v6_models_preserved=18, previous_v6_rows_preserved=72,
                   shared_global80_point_counted_as_new_update=False, distinct_global_points=209, global_accepted_updates=208,
                   endpoint_selection="Predeclared terminal local128/global208 and original16/80 endpoints; no minimum-gap selection")
    assert overlay["models"][1:] == original["models"] and overlay["runs"][1:] == original["runs"] and overlay["source_files"][6:] == original["source_files"]
    write(site / contract["new_overlay"], overlay)
    write(HERE / "prepared.json", {"all_saved_data_preparation_checks_passed": True, "models": 19, "rows": 76, "new_trajectory_points": 129, "distinct_global_points": 209,
                                  "contract_sha256": sha(HERE / "contract.json"), "builder_sha256": sha(__file__), "shared_utilities_sha256": sha(UTILITY), "overlay_sha256": sha(site / contract["new_overlay"]),
                                  "new_assets_sha256": {str(p.relative_to(site)): sha(p) for p in sorted(evidence.iterdir())}, "science_calls": 0, "publication_performed": False})
    print({"models": 19, "rows": 76, "new_points": 129, "distinct_global_points": 209, "science_calls": 0})


def load_nist(site):
    return shared.json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))


def same_shared_point(first, model):
    return (first["source_arrays_sha256"] == model["point_arrays_sha256"] and first["reported_EOL_Ha"] == model["EOL_Ha"]
            and first["orbital_residual"] == model["orbital_residual"]
            and [t["energy_Ha"] for t in first["targets"]] == [t["energy_Ha"] for t in model["targets"]])


if __name__ == "__main__":
    main()
