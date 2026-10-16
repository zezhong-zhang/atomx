"""Prepare a source-bound Yb overlay; preserve every original atlas payload."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = json.loads((HERE / "contract.json").read_text())
SITE = Path(CONTRACT["site"])
CAMPAIGN = ROOT / "docs/yb_mcdhf_rci_2026_10_11"
NATIVE = CAMPAIGN / "native_run/nuclear_eol005_projected_increment"
FROZEN = NATIVE / "source"
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


def candidates(levels, configuration, two_j, parity):
    """Selectors are physical identity, never persisted ASD row number."""
    return [r for r in levels if normalise_configuration(r["configuration"]) == configuration
            and level_record(r).two_j == two_j and level_record(r).parity == parity
            and r["energy_kind"] == "evaluated" and r["energy_cm1"] is not None
            and not r["bracketed"] and not level_record(r).flags
            and "?" not in (r["configuration_nist"] or "") + (r["term"] or "")]


def match(row, levels, zero, *, balanced):
    configuration = "4f14 " + row["label"].split("_")[0] + "1"
    selected = candidates(levels, configuration, row["twoJ"], row["parity"])
    if len(selected) != 1:
        raise ValueError("unique physical nominal ASD branch required, not an old row selector")
    reference = selected[0]
    value = row["excitation_from_tracked_6s_cm1"] if balanced else row["computed_cm1"]
    nominal = row["NIST_nominal_cm1"] if balanced else row["experimental_cm1"]
    if reference["energy_cm1"] != nominal or zero["energy_cm1"] != 0:
        raise ValueError("ASD and original nominal same-zero reference changed")
    error = value - (reference["energy_cm1"] - zero["energy_cm1"])
    result = {"label": row["label"], "configuration": configuration, "two_j": row["twoJ"],
              "parity": row["parity"], "calculated_cm1": value, "matching": "unique_nominal_branch",
              "candidates": len(selected), "level_id": reference["level_id"], "nist_row": reference["row_index"],
              "configuration_nist": reference["configuration_nist"], "term_nist": reference["term"],
              "j_text_nist": reference["j_text"], "reference_cm1": reference["energy_cm1"],
              "uncertainty_cm1": reference["uncertainty_cm1"], "reference_kind": reference["energy_kind"],
              "reference_code": reference["reference"], "error_cm1": error,
              "error_percent": 100 * error / reference["energy_cm1"], "relative_goal": .01,
              "goal_pass": abs(error / reference["energy_cm1"]) < .01, "handbook_cm1": nominal,
              "asd_minus_handbook_cm1": reference["energy_cm1"] - nominal,
              "reference_zero_level_id": zero["level_id"], "source_energy_Ha": row["energy_Ha"],
              "bare_character_weight": row["bare_character_weight"] if balanced else row["dominant_CSF_weight"],
              "state_binding_unambiguous": row["binding_unambiguous"] if balanced else True,
              "state_binding_scope": "Saved signed-CI overlap and full same-J spectrum" if balanced else "Five distinct J/parity single-CSF blocks"}
    if balanced:
        for key in ("energy_delta_from_previous_Ha", "energy_delta_from_bare_Ha", "transition_delta_from_previous_cm1",
                    "transition_delta_from_bare_cm1", "column", "local_root", "family_weights"):
            result[key] = row[key]
    return result


def multiplets(rows):
    by_label = {row["label"]: row for row in rows}
    out = []
    for orbital, two_js in (("5d", (3, 5)), ("6p", (1, 3))):
        selected = [by_label[f"{orbital}_{j}/2"] for j in two_js]
        weights = [j + 1 for j in two_js]
        calc_center = sum(row["calculated_cm1"] * w for row, w in zip(selected, weights, strict=True)) / sum(weights)
        ref_center = sum(row["reference_cm1"] * w for row, w in zip(selected, weights, strict=True)) / sum(weights)
        calc_split = selected[1]["calculated_cm1"] - selected[0]["calculated_cm1"]
        ref_split = selected[1]["reference_cm1"] - selected[0]["reference_cm1"]
        out.append({"label": orbital, "degeneracy_weights": weights, "centroid_cm1": calc_center,
                    "reference_centroid_cm1": ref_center, "centroid_gap_cm1": calc_center - ref_center,
                    "splitting_cm1": calc_split, "reference_splitting_cm1": ref_split,
                    "splitting_gap_cm1": calc_split - ref_split, "reference_uncertainty_cm1": None})
    return out


def main():
    for relative, expected in CONTRACT["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("declared immutable science input changed: " + relative)
    prechange = load(HERE / "prechange_manifest.json")
    for relative, record in prechange["files"].items():
        if sha(SITE / relative) != record["sha256"]:
            raise ValueError("site preimage changed before data preparation: " + relative)
    source = load(NATIVE / "source_identity.json")
    if any(sha(FROZEN / Path(p).relative_to("src")) != h for p, h in source["all_atomx_python_sha256"].items()):
        raise ValueError("frozen scientific matcher convention source changed")
    native, proof, balanced, verification, stage002, receipt = [load(path) for path in (
        NATIVE / "result.json", CAMPAIGN / "native_run/card_evidence_stationary_baseline.json",
        CAMPAIGN / "balanced_cv_replay/result.json", CAMPAIGN / "balanced_cv_replay/verification.json",
        CAMPAIGN / "results_stage002.json", CAMPAIGN / "delivery_stage002/receipt.json")]
    if not native["stationarity_passed"] or not verification["all_passed"] or not receipt["passed"]:
        raise ValueError("required native stationarity, balanced replay or delivered-stage proof failed")
    if verification["artifacts_sha256"]["result.json"] != sha(CAMPAIGN / "balanced_cv_replay/result.json"):
        raise ValueError("RCI replay identity mismatch")
    identities = {i["path"]: i["sha256"] for i in receipt["source_identity"]}
    required = ["docs/yb_mcdhf_rci_2026_10_11/" + rel for rel in (
        "results_stage002.json", "native_run/card_evidence_stationary_baseline.json")]
    if any(identities.get(relative) != sha(ROOT / relative) for relative in required):
        raise ValueError("actual stage002 delivery did not bind these exact science inputs")
    for key in ("balanced_cv_result", "balanced_cv_verification"):
        binding = stage002["inputs"][key]
        if sha(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError("delivered stage002 nested RCI input hash chain failed")
    payload = json.loads(gzip.decompress((SITE / "dist/ions/70-1.json.gz").read_bytes()))
    levels = payload["levels"]
    zeros = candidates(levels, "4f14 6s1", 1, 1)
    if len(zeros) != 1 or zeros[0]["energy_cm1"] != 0:
        raise ValueError("unique evaluated common 6s reference required")
    zero = zeros[0]
    copies = {"stage002.json": CAMPAIGN / "results_stage002.json",
              "delivery-proof.json": CAMPAIGN / "delivery_stage002/receipt.json",
              "baseline-proof.json": CAMPAIGN / "native_run/card_evidence_stationary_baseline.json",
              "baseline-result.json": NATIVE / "result.json",
              "balanced-result.json": CAMPAIGN / "balanced_cv_replay/result.json",
              "balanced-verification.json": CAMPAIGN / "balanced_cv_replay/verification.json"}
    evidence = SITE / "dist/evidence/yb-mcdhf-stage002"
    evidence.mkdir(parents=True, exist_ok=False)
    source_files = []
    for filename, original in copies.items():
        shutil.copyfile(original, evidence / filename)
        source_files.append({"path": str(original.relative_to(ROOT)), "asset": f"evidence/yb-mcdhf-stage002/{filename}", "sha256": sha(original)})
    models = []
    for model, key, title in (("DC", "rows_dc", "五参考 MCDHF · Dirac–Coulomb"),
                              ("static_Breit", "rows_rci", "五参考 MCDHF → 静态 Breit RCI")):
        rows = [match(row, levels, zero, balanced=False) for row in native[key] if row["label"] != "6s_1/2"]
        models.append({"id": "native005_" + model, "title": title, "saved_run": "nuclear_eol005_projected_increment",
            "model_kind": "native_mcdhf_baseline", "operator": model, "rci_csf_count": 5,
            "orbital_optimized_csf_count": 5, "stationarity_passed": True,
            "stationarity_note": "五参考共同轨道 MCDHF 驻值已通过；仅认证这个5-CSF数值模型。",
            "scope": "69电子、同24共同径向轨道；五个不同J/宇称块的5个CSF，无核空穴CI混合。静态模型包含完整Gaunt与零频迟滞；未叠加QED、反冲或动态Breit。",
            "expanded_model_stationarity_certified": None, "production_certified": False,
            "highest_level_reached": False, "experimental_gate_passed": False,
            "source": str((NATIVE / "result.json").relative_to(ROOT)), "sha256": sha(NATIVE / "result.json"),
            "source_asset": "evidence/yb-mcdhf-stage002/baseline-result.json", "ground": native[key][0],
            "rows": rows, "multiplets": multiplets(rows),
            "boundary_note": "匹配的是共同6s零点下的唯一评估ASD名义分支；参考不确定度缺失不补为零。数值驻值通过不等于实验精度或关联空间完备。"})
    for stage in balanced["stages"]:
        family = "+".join(stage["families"][1:])
        for model, title in (("DC", "Dirac–Coulomb RCI"), ("static_Breit", "静态 Breit RCI")):
            original = stage["tracked_levels"][model]
            rows = [match(row, levels, zero, balanced=True) for row in original if row["label"] != "6s_1/2"]
            if not stage["tracked_6s_is_actual_ground"][model] or not all(row["state_binding_unambiguous"] for row in rows):
                raise ValueError("expanded-space ground/state identity does not support this presentation")
            ground = next(row for row in original if row["label"] == "6s_1/2")
            models.append({"id": stage["name"] + "_" + model,
                "title": f"固定驻值轨道 · {family} · {stage['CSFs']} CSF · {title}",
                "saved_run": stage["name"], "model_kind": "fixed_orbital_rci", "operator": model,
                "rci_csf_count": stage["CSFs"], "orbital_optimized_csf_count": 5,
                "families": stage["families"], "stationarity_passed": None,
                "stationarity_note": "仅使用005已驻值的五参考MCDHF轨道；本扩展空间的MCDHF驻值未评估。",
                "scope": "同24轨道、完整允许pair族、全部J/宇称块与全部根；这是固定轨道的balanced-family RCI判别，不是完整CV或扩展空间MCDHF优化。",
                "expanded_model_stationarity_certified": False, "production_certified": False,
                "highest_level_reached": False, "experimental_gate_passed": False,
                "source": str((CAMPAIGN / "balanced_cv_replay/result.json").relative_to(ROOT)),
                "sha256": sha(CAMPAIGN / "balanced_cv_replay/result.json"),
                "source_asset": "evidence/yb-mcdhf-stage002/balanced-result.json",
                "verification_asset": "evidence/yb-mcdhf-stage002/balanced-verification.json",
                "verification_sha256": sha(CAMPAIGN / "balanced_cv_replay/verification.json"),
                "previous_stage": stage["previous_stage"], "ground": ground,
                "rows": rows, "multiplets": multiplets(rows),
                "is_current": stage["name"] == "stage_4f5p5s" and model == "static_Breit",
                "boundary_note": "有符号CI与同J全谱已重放，bare连接权重和上下态能移分别保留；名义ASD分支仍不认证完整CV、dressed强度、寿命或最高精度。"})
    update = {"generated_at": stage002["created_at"], "species": {"z": 70, "charge": 1, "symbol": "Yb", "label": "Yb II"},
              "route": "Atomx native MCDHF → RCI", "stationarity": proof["stationarity"],
              "native005_source_map_sha256": source["source_map_sha256"],
              "radial_orbitals_sha256": proof["inputs"]["orbitals"]["sha256"],
              "nuclear_scope": "原默认Fermi A=175经验rms半径；未认证同位素精细核尺寸/质量。",
              "scope": balanced["scope"], "summary": stage002["card_summary"],
              "reference_zero": zero, "nist_payload_sha256": sha(SITE / "dist/ions/70-1.json.gz"),
              "models": models, "scientific_proof": {"execution_checks": len(balanced["checks"]),
                 "independent_replay_checks": verification["checks_count"], "all_passed": verification["all_passed"],
                 "production_certified": False, "expanded_MCDHF_stationarity_certified": False,
                 "experimental_gate_passed": False, "highest_level_reached": False},
              "performance": balanced["resource"], "source_files": source_files,
              "append_only_overlay": True, "historical_models_preserved": len(payload["models"])}
    write(SITE / "dist/updates/yb-mcdhf-stage002.json", update)
    write(HERE / "prepared.json", {"contract_sha256": sha(HERE / "contract.json"),
          "new_model_count": len(models), "new_matched_rows": sum(len(model["rows"]) for model in models),
          "reference_binding": [{key: row[key] for key in ("label", "configuration", "two_j", "parity", "level_id", "nist_row", "reference_cm1")}
                                for row in models[0]["rows"]],
          "new_assets_sha256": {str(path.relative_to(SITE)): sha(path)
                                for path in [SITE / "dist/updates/yb-mcdhf-stage002.json", *sorted(evidence.iterdir())]},
          "scientific_inputs_sha256": CONTRACT["inputs_sha256"], "script_sha256": sha(Path(__file__)),
          "delivery_binding": "Receipt directly binds stage002/baseline; stage002.inputs transitively binds balanced result/verification. All links verified."})
    print(json.dumps({"models": len(models), "new_rows": sum(len(m["rows"]) for m in models),
                      "original_gzip_payloads_untouched": True}))


if __name__ == "__main__":
    main()
