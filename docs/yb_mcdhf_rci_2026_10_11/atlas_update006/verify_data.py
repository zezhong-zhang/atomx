"""Independent scalar/read-only preservation audit of the single saved-JSON append."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    contract = load(HERE / "contract.json")
    site = Path(contract["site"])
    original = load(HERE / "original_overlay.json")
    overlay = load(site / contract["new_overlay"])
    old_models = load(HERE / "old_models.json")
    changed = [p for p, v in load(HERE / "prechange_manifest.json")["files"].items() if sha(site / p) != v["sha256"]]
    assert changed == [contract["new_overlay"]]
    assert len(list((site / "dist/ions").glob("*.gz"))) == 6019
    assert overlay["models"][1:] == original["models"] and overlay["runs"][1:] == original["runs"]
    assert overlay["source_files"][6:] == original["source_files"]
    models = overlay["models"][:1] + old_models
    assert len(models) == 19 and sum(len(m["rows"]) for m in models) == 76
    assert len({m.get("id", m["saved_run"]) for m in models}) == 19
    model, run = overlay["models"][0], overlay["runs"][0]
    assert model["id"] == contract["model_id"] and run["model_id"] == model["id"]
    assert [p["iteration"] for p in run["trajectory"]] == list(range(129))
    assert [p["global_accepted_step"] for p in run["trajectory"]] == list(range(80, 209))
    assert model["accepted_steps"] == 128 and model["global_accepted_steps"] == 208
    assert model["rci_csf_count"] == 1046 and model["radial_orbital_count"] == 29 and model["active_coordinate_count"] == 3451
    assert not model["full_parent_computed"] and model["state_average_weights"] == [.2] * 5
    for key in ("source_map", "metric", "geometry", "objective"):
        assert model[key + "_sha256"] == contract[key + "_sha256"]
    assert not model["stationarity_passed"] and not model["experimental_gate_passed"] and not model["physical_completeness_certified"]
    assert model["orbital_residual"] > 1e-7 and model["energy_indicator_Ha"] > 1e-8 and all(not r["goal_pass"] for r in model["rows"])
    first, last = run["trajectory"][0], run["trajectory"][-1]
    assert model["targets"] == last["targets"] and model["ground"] == next(t for t in last["targets"] if t["label"] == "6s_1/2")
    assert model["multiplets"] == last["multiplets"] and model["EOL_Ha"] == last["reported_EOL_Ha"]
    assert first["exact_stored_H_acceptance_qualified"] is None and all(p["exact_stored_H_acceptance_qualified"] for p in run["trajectory"][1:])
    max_line = max_rms = max_moment = 0.
    for point in run["trajectory"]:
        targets = {t["label"]: t for t in point["targets"]}
        lower = targets["6s_1/2"]["energy_Ha"]
        for target in targets.values():
            max_line = max(max_line, abs((target["energy_Ha"] - lower) * overlay["Hartree_cm1"] - target["excitation_cm1"]))
            assert target["NIST_uncertainty_cm1"] is None and target["state_average_weight"] == .2
        rms = math.sqrt(sum(t["nominal_gap_cm1"] ** 2 for label, t in targets.items() if label != "6s_1/2") / 4)
        max_rms = max(max_rms, abs(rms - point["nominal_four_gap_RMS_cm1"]))
        for moment in point["multiplets"]:
            js = (3, 5) if moment["label"] == "5d" else (1, 3)
            a, b = [targets[f'{moment["label"]}_{j}/2'] for j in js]
            center = ((js[0] + 1) * a["excitation_cm1"] + (js[1] + 1) * b["excitation_cm1"]) / sum(j + 1 for j in js)
            max_moment = max(max_moment, abs(center - moment["centroid_cm1"]), abs(b["excitation_cm1"] - a["excitation_cm1"] - moment["splitting_cm1"]))
    assert max_line <= 1e-8 and max_rms <= 1e-9 and max_moment <= 1e-9
    all_points = []
    unique = {}
    for retained_run in overlay["runs"]:
        initial = retained_run.get("global_initial_step", 0)
        for point in retained_run["trajectory"]:
            index = point.get("global_accepted_step", initial + point["iteration"])
            identity = (point["source_arrays_sha256"], point["reported_EOL_Ha"], tuple(t["energy_Ha"] for t in point["targets"]))
            if index in unique:
                assert unique[index] == identity
            unique[index] = identity
            all_points.append(index)
    assert len(all_points) == 211 and sorted(unique) == list(range(209))
    assert len(overlay["source_files"]) == 20
    assert all(sha(site / "dist" / a["asset"]) == a["sha256"] for a in overlay["source_files"])
    report = {"all_saved_data_and_preservation_checks_passed": True, "models": 19, "rows": 76,
              "old_models_preserved": 18, "old_rows_preserved": 72, "unchanged_gzip": 6019, "unchanged_page_and_hosting": True,
              "new_points": 129, "per_run_point_records": 211, "distinct_global_points": 209, "global_updates": 208,
              "max_transition_reconstruction_error_cm1": max_line, "max_RMS_reconstruction_error_cm1": max_rms,
              "max_moment_reconstruction_error_cm1": max_moment,
              "overlay_sha256": sha(site / contract["new_overlay"]), "page_sha256": sha(site / "dist/index.html"),
              "contract_sha256": sha(HERE / "contract.json"), "verifier_sha256": sha(__file__),
              "new_science_calls": 0, "physical_array_replay": False, "publication_performed": False}
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_passed": True, "models": 19, "rows": 76, "new_points": 129, "distinct_global_points": 209, "new_science_calls": 0}))


if __name__ == "__main__":
    main()
