"""Append a closed native endpoint from explicitly bound saved JSON; no Atomx imports."""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ("6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", required=True, help="Exact closed completion path bound in this contract")
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    completion_path = (ROOT / args.completion).resolve()
    if completion_path != (ROOT / contract["completion_path"]).resolve():
        raise ValueError("An explicit successor contract is required for another run")
    site = Path(contract["site"])
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    check("prospective_contract", contract["status"] == "prospective_before_local_site_edits")
    check("exact_preimage_manifest", sha(HERE / "prechange_manifest.json") == contract["prechange_manifest_sha256"])
    preimage = load(HERE / "prechange_manifest.json")
    for relative, item in preimage["files"].items():
        if sha(site / relative) != item["sha256"]:
            raise ValueError("Preimage changed: " + relative)
    check("all_old_assets_and_hosting_exact", True)
    for relative, expected in contract["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Saved scientific input changed: " + relative)
    check("all_closed_input_bytes_exact", True)
    run = completion_path.parent
    completion = load(completion_path)
    actual = load(run / "actual/result.json")
    receipt = load(run / "root_execution.json")
    scientific = load(run / "contract.json")
    check("closed_successful_runtime", receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0)
    check("closed_metadata_scope", completion["all_completion_metadata_checks_passed"]
          and completion["actual_live_controls"] == 1175 and completion["checks"] == 11
          and completion["scientific_calls_in_completion"] == 0)
    check("actual_source_f043", actual["source_map_sha256"] == scientific["source"]["candidate356_map_sha256"] == contract["source_map_sha256"])
    check("complete1046_all29_same_five_weights", scientific["fixed"]["target_CSFs"] == 1046
          and len(scientific["fixed"]["keys29"]) == 29 and scientific["fixed"]["five_weights"] == [.2] * 5
          and scientific["fixed"]["full_model_not_computed"] and scientific["fixed"]["all29_allowed_to_relax"])
    check("ordinary_reported_EOL_no_callback", not actual["ordinary_reported_objective_changed"]
          and not actual["energy_callback_used"] and scientific["objective"]["energy_callback"] is None)
    check("FAIL_flags_preserved", not actual["stationarity_passed"] and not actual["experimental_certification"]
          and not actual["physical_completeness_certified"] and not completion["stationarity_passed"])
    check("original_gates", scientific["gates"]["orbital_residual"] == 1e-7 and scientific["gates"]["energy_indicator_Ha"] == 1e-8)

    # Only parse literal constants from the frozen source; never import or execute it.
    constants_path = next(ROOT / p for p in contract["inputs_sha256"] if p.endswith("/atomx/constants.py"))
    literals = {}
    for node in ast.parse(constants_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("CODATA", "DEFAULT_CODATA_VERSION"):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    codata = literals["CODATA"][literals["DEFAULT_CODATA_VERSION"]]
    hartree_cm1 = codata["_Ha_J"] / (100 * codata["_hplanck"] * codata["_c"])
    old = load(HERE / "old_models.json")
    prior = next(m for m in old if m.get("id") == "fixed160_screened_closed68_virtual_DC")
    nist = json.loads(gzip.decompress((site / "dist/ions/70-1.json.gz").read_bytes()))
    nist_by_id = {r["level_id"]: r for r in nist["levels"]}
    references = {r["label"]: r for r in prior["rows"]}
    zero_id = references[LABELS[1]]["reference_zero_level_id"]
    check("unique_same6s_ASD_zero", nist_by_id[zero_id]["energy_cm1"] == 0
          and sum(r["level_id"] == zero_id for r in nist["levels"]) == 1)
    for label, ref in references.items():
        record = nist_by_id[ref["level_id"]]
        check("unique_unchanged_ASD/" + label,
              sum(r["level_id"] == ref["level_id"] for r in nist["levels"]) == 1
              and ref["reference_cm1"] == record["energy_cm1"]
              and ref["uncertainty_cm1"] is None and record["uncertainty_cm1"] is None
              and record["energy_kind"] == "evaluated")

    trajectory = []
    expected_iterations = list(range(contract["trajectory_points"]))
    check("all17_ordered_points", [r["iteration"] for r in actual["trajectory"]] == expected_iterations
          and [r["iteration"] for r in completion["trajectory"]] == expected_iterations)
    initial = None
    for summary, closed in zip(actual["trajectory"], completion["trajectory"], strict=True):
        manifest_path = ROOT / summary["manifest_path"]
        manifest = load(manifest_path)
        i = summary["iteration"]
        check(f"point{i}/manifest_array_identity", sha(manifest_path) == summary["manifest_sha256"]
              and manifest["array_sha256"] == summary["array_sha256"]
              and manifest["checkpoint_qualified"] and manifest["iteration"] == i)
        check(f"point{i}/metric_geometry_objective_source", manifest["metric_sha256"] == contract["metric_sha256"]
              and manifest["geometry_sha256"] == contract["geometry_sha256"]
              and manifest["objective_sha256"] == contract["objective_sha256"]
              and manifest["source_map_sha256"] == contract["source_map_sha256"])
        targets = {r["label"]: copy.deepcopy(r) for r in manifest["targets"]}
        check(f"point{i}/five_UIDs", set(targets) == set(LABELS)
              and all(targets[label]["UID"] == contract["fixed"]["reference_UIDs"][label] for label in LABELS))
        lower = targets["6s_1/2"]["energy_Ha"]
        if initial is None:
            initial = copy.deepcopy(targets)
        for label in LABELS:
            target = targets[label]
            check(f"point{i}/signed_tracker/" + label, target["tracking"]["reference_levels"] == [target["UID"]]
                  and target["tracking"]["selected_levels"] == [target["selected_level"]]
                  and target["selected_level"][:2] == target["UID"][:2]
                  and target["tracking"]["per_level_weight"] == .2)
            check(f"point{i}/same_lower_gap/" + label,
                  abs((target["energy_Ha"] - lower) * hartree_cm1 - target["excitation_cm1"]) <= 1e-8
                  and target["NIST_uncertainty_cm1"] is None
                  and target["nominal_gap_cm1"] == target["excitation_cm1"] - target["NIST_nominal_cm1"])
            # Existing manifest delta fields describe the preceding accepted step.
            # This named cumulative increment is recomputed against point0, never reused as a full-run delta.
            target["delta_upper_from_initial_Ha"] = target["energy_Ha"] - initial[label]["energy_Ha"]
            target["delta_lower_from_initial_Ha"] = lower - initial["6s_1/2"]["energy_Ha"]
            target["transition_shift_from_initial_cm1"] = target["excitation_cm1"] - initial[label]["excitation_cm1"]
            target["state_average_weight"] = .2
            if label != "6s_1/2":
                ref = references[label]
                check(f"point{i}/same_ASD_UID/" + label,
                      target["NIST_nominal_cm1"] == ref["reference_cm1"]
                      and target["UID"] == [ref["source_physical_UID"][q] for q in ("twoJ", "parity", "local_root")])
        moments = []
        for label, js in (("5d", (3, 5)), ("6p", (1, 3))):
            a, b = [targets[f"{label}_{j}/2"] for j in js]
            weights = [j + 1 for j in js]
            center = sum(r["excitation_cm1"] * w for r, w in zip((a, b), weights, strict=True)) / sum(weights)
            refcenter = sum(r["NIST_nominal_cm1"] * w for r, w in zip((a, b), weights, strict=True)) / sum(weights)
            splitting = b["excitation_cm1"] - a["excitation_cm1"]
            refsplit = b["NIST_nominal_cm1"] - a["NIST_nominal_cm1"]
            saved = next(m for m in manifest["multiplets"] if m["label"] == label)
            check(f"point{i}/moments/" + label, abs(center - saved["centroid_cm1"]) <= 1e-9
                  and abs(splitting - saved["splitting_cm1"]) <= 1e-9)
            moments.append({"label": label, "degeneracy_weights": weights, "centroid_cm1": center,
                            "reference_centroid_cm1": refcenter, "centroid_gap_cm1": center - refcenter,
                            "splitting_cm1": splitting, "reference_splitting_cm1": refsplit,
                            "splitting_gap_cm1": splitting - refsplit, "reference_uncertainty_cm1": None})
        rms = math.sqrt(sum(targets[label]["nominal_gap_cm1"] ** 2 for label in LABELS[1:]) / 4)
        diag = manifest["diagnostics"]
        check(f"point{i}/RMS_and_reported_EOL", abs(rms - closed["nominal_four_gap_RMS_cm1"]) <= 1e-9
              and diag["energy_ha"] == closed["reported_EO_Ha"]
              and abs(sum(targets[label]["energy_Ha"] * .2 for label in LABELS) - diag["energy_ha"]) <= 2e-11)
        check(f"point{i}/original_failures", not manifest["stationarity_passed"]
              and not manifest["experimental_certification"] and not manifest["physical_completeness_certification"])
        trajectory.append({"iteration": i, "reported_EOL_Ha": diag["energy_ha"],
                           "orbital_residual": diag["orbital_residual"], "energy_indicator_Ha": diag["estimated_energy_error"],
                           "nominal_four_gap_RMS_cm1": rms, "targets": [targets[label] for label in LABELS],
                           "multiplets": moments, "source_manifest": summary["manifest_path"],
                           "source_manifest_sha256": summary["manifest_sha256"], "source_arrays_sha256": summary["array_sha256"],
                           "stationarity_passed": False, "experimental_certification": False,
                           "exact_stored_H_acceptance_qualified": closed["exact_acceptance_qualified"]})

    terminal = trajectory[-1]
    check("terminal_only_no_best_gap_selection", terminal["iteration"] == contract["terminal_iteration"] == 16)
    terminal_targets = {r["label"]: r for r in terminal["targets"]}
    reference_fields = ("configuration", "two_j", "parity", "matching", "candidates", "level_id", "nist_row",
                        "configuration_nist", "term_nist", "j_text_nist", "reference_cm1", "uncertainty_cm1",
                        "reference_kind", "reference_code", "handbook_cm1", "asd_minus_handbook_cm1", "reference_zero_level_id")
    rows = []
    for label in LABELS[1:]:
        ref, target = references[label], terminal_targets[label]
        error = target["nominal_gap_cm1"]
        rows.append({**{k: ref[k] for k in reference_fields}, "label": label,
                     "calculated_cm1": target["excitation_cm1"], "error_cm1": error,
                     "error_percent": 100 * error / ref["reference_cm1"], "relative_goal": .01,
                     "goal_pass": abs(error / ref["reference_cm1"]) < .01,
                     "source_energy_Ha": target["energy_Ha"], "source_physical_UID": dict(zip(("twoJ", "parity", "local_root"), target["UID"], strict=True)),
                     "source_column": target["column"], "exact_J_local_root": target["selected_level"][2],
                     "physical_reference_overlap_squared": target["tracking"]["principal_overlap_squared"][0],
                     "physical_reference_overlap_scope": "Frozen original balanced531 DC tracker/native005 reference; not bare weight",
                     "state_average_weight": .2, "state_binding_unambiguous": True,
                     **{k: target[k] for k in ("delta_upper_from_initial_Ha", "delta_lower_from_initial_Ha", "transition_shift_from_initial_cm1")}})

    evidence = site / contract["new_evidence_dir"]
    evidence.mkdir(exist_ok=False)
    assets = []
    for filename in ("completion.json", "actual/result.json", "root_execution.json", "contract.json", "execution_contract.json", "RESULTS.md"):
        path = run / filename
        name = filename.replace("/", "-")
        shutil.copyfile(path, evidence / name)
        assets.append({"path": str(path.relative_to(ROOT)), "asset": "evidence/yb-correlated29-run003-endpoint16/" + name, "sha256": sha(path)})
    model = {"id": contract["model_id"], "title": "原生29轨道相关MCDHF · run003末态16步 · DC",
             "model_kind": "nonstationary_correlated29_mcdhf_endpoint", "saved_run": run.name + "/accepted_0016",
             "operator": "DC", "rci_csf_count": 1046, "target_sector_csf_count": 1046,
             "declared_full_model_csf_count": 2163, "full_parent_computed": False, "symmetry_block_count": 5,
             "radial_orbital_count": 29, "active_coordinate_count": 3451, "electron_count": 69,
             "accepted_steps": 16, "all29_allowed_to_relax": True, "state_average_weights": [.2] * 5,
             "orbital_residual": terminal["orbital_residual"], "energy_indicator_Ha": terminal["energy_indicator_Ha"],
             "orbital_tolerance": 1e-7, "energy_indicator_tolerance_Ha": 1e-8,
             "stationarity_passed": False, "experimental_gate_passed": False, "production_certified": False,
             "highest_level_reached": False, "physical_completeness_certified": False,
             "source_map_sha256": contract["source_map_sha256"], "metric_sha256": contract["metric_sha256"],
             "geometry_sha256": contract["geometry_sha256"], "objective_sha256": contract["objective_sha256"],
             "point_manifest_sha256": terminal["source_manifest_sha256"], "point_arrays_sha256": terminal["source_arrays_sha256"],
             "original_tracker_rebased": False, "no_pair_projector_changed": False,
             "ground": terminal_targets["6s_1/2"], "targets": terminal["targets"], "rows": rows,
             "multiplets": terminal["multiplets"], "EOL_Ha": terminal["reported_EOL_Ha"],
             "initial_EOL_Ha": trajectory[0]["reported_EOL_Ha"], "EOL_delta_Ha": terminal["reported_EOL_Ha"] - trajectory[0]["reported_EOL_Ha"],
             "nominal_gap_RMS_cm1": terminal["nominal_four_gap_RMS_cm1"],
             "initial_nominal_gap_RMS_cm1": trajectory[0]["nominal_four_gap_RMS_cm1"],
             "initial_orbital_residual": trajectory[0]["orbital_residual"],
             "elapsed_uncontrolled_s": receipt["elapsed_uncontrolled_seconds"], "peak_RSS_MiB": actual["peak_RSS_MiB"],
             "runtime_controls": 1175, "completion_metadata_checks": 11, "completed_calls": actual["completed_calls"],
             "source": str((run / "actual/result.json").relative_to(ROOT)), "sha256": sha(run / "actual/result.json"),
             "source_asset": "evidence/yb-correlated29-run003-endpoint16/actual-result.json",
             "completion_asset": "evidence/yb-correlated29-run003-endpoint16/completion.json", "is_current": True,
             "stationarity_note": "16个共同轨道更新后，原残差与能量指示量均未通过1e−7 / 1e−8 Ha门。停止原因是声明的步数上限；能量下降不代表驻值或实验精度。",
             "scope": "原生MCDHF共同优化全部29轨道与完整五目标J/宇称扇区1046有符号CSF；3451新空间坐标度量、原七个核no-pair子空间及原physical tracker固定，五态等权0.2。声明父空间2163 CSF，其余扇区未计算，未认证全谱或全局基态。",
             "boundary_note": "只取预声明run003的末态16，不按最小gap挑中间点；全部17点保留。DC，未加静态/动态Breit、QED、反冲或经验修正。f043源；不与旧修正拼接。117秒是本次未受控墙钟时间，不是并行或软件间性能比较。1175运行控制与11元数据闭环不替代物理完备性/独立末态HCIg重放。"}
    overlay = {"status": "closed_bounded_native_correlated29_MCDHF_nonstationary_endpoint",
               "species": {"z": 70, "charge": 1, "symbol": "Yb", "label": "Yb II"},
               "models": [model], "runs": [{"model_id": model["id"], "terminal_iteration": 16, "trajectory": trajectory,
                                            "receipt": {k: receipt[k] for k in ("started_utc", "finished_utc", "exit_code", "elapsed_uncontrolled_seconds", "timing_scope")}}],
               "source_files": assets, "nist_payload_sha256": sha(site / "dist/ions/70-1.json.gz"),
               "reference_zero_level_id": zero_id, "source_map_sha256": contract["source_map_sha256"],
               "Hartree_cm1": hartree_cm1, "CODATA": literals["DEFAULT_CODATA_VERSION"],
               "old_models_preserved": 16, "old_rows_preserved": 64, "total_models": 17, "total_rows": 68,
               "append_only_overlay": True, "endpoint_selection": "predeclared terminal accepted16; no minimum-gap selection",
               "stationarity_passed": False, "experimental_gate_passed": False, "physical_completeness_certified": False,
               "performance_comparison_valid": False, "new_scientific_calls": 0, "publication_performed": False,
               "future_append_policy": "Another run requires explicit closed completion, new contract and separate terminal model; never infer running state."}
    write(site / contract["new_overlay"], overlay)
    write(HERE / "prepared.json", {"all_data_preparation_checks_passed": all(c["passed"] for c in checks),
                                  "checks_count": len(checks), "checks": checks, "new_models": 1, "total_models": 17,
                                  "total_rows": 68, "trajectory_points": 17, "contract_sha256": sha(HERE / "contract.json"),
                                  "builder_sha256": sha(__file__), "overlay_sha256": sha(site / contract["new_overlay"]),
                                  "assets_sha256": {str(p.relative_to(site)): sha(p) for p in sorted(evidence.iterdir())},
                                  "completion_path": contract["completion_path"], "science_calls": 0, "publication_performed": False})
    print(json.dumps({"checks": len(checks), "models": 17, "rows": 68, "trajectory_points": 17,
                      "overlay_sha256": sha(site / contract["new_overlay"]), "science_calls": 0}))


if __name__ == "__main__":
    main()
