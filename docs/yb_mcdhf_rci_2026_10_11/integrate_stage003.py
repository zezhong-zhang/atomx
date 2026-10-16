"""Append the mixed-CI stationarity diagnostic without replacing the RCI spectrum."""

import json
import math
from datetime import datetime, timezone

from integrate_stage002 import HERE, REPO, binding, read


def main():
    prefix = HERE.relative_to(REPO).as_posix()
    parent_path = f"{prefix}/results_stage002.json"
    probe_path = f"{prefix}/mixed_ci_stationarity_probe/result.json"
    verification_path = f"{prefix}/mixed_ci_stationarity_probe/verification.json"
    independent_path = f"{prefix}/mixed_ci_stationarity_probe/independent_verification.json"
    result = read(parent_path)
    for item in result["inputs"].values():
        assert binding(item["path"])["sha256"] == item["sha256"]
    probe = read(probe_path)
    verification = read(verification_path)
    independent = read(independent_path)
    assert independent["all_passed"] and independent["checks_count"] == 432
    assert binding(probe_path)["sha256"] == independent["result_sha256"]
    assert verification["scientific_identity_and_derivative_gates_passed"]
    assert binding(probe_path)["sha256"] == verification["result_sha256"]
    for path, expected in probe["inputs_sha256"].items():
        assert binding(path)["sha256"] == expected, path
    assert probe["scientific_identity_and_derivative_gates_passed"]
    assert not probe["expanded_MCDHF_converged"] and not probe["production_certified"]
    assert not probe["experimental_gate_passed"] and not probe["highest_level_reached"]
    baseline, step = probe["baseline"], probe["one_step"]
    assert probe["accepted_one_step"] and step["EOL_delta_Ha"] < 0
    assert not baseline["stationarity_passed"] and not step["stationarity_passed"]
    assert baseline["original_orbital_tolerance"] == result["stationarity"]["orbital_tolerance"]
    assert probe["original_metric_sha256"] == result["stationarity"]["residual_metric_sha256"]
    assert step["same_original_metric_SHA"] == probe["original_metric_sha256"]
    balanced = read(result["inputs"]["balanced_cv_result"]["path"])
    dc = {r["label"]: r for r in balanced["stages"][-1]["tracked_levels"]["DC"]}
    rows = []
    for row in step["lower_upper_difference_rows"]:
        before = dc[row["label"]]
        assert (row["twoJ"], row["parity"]) == (before["twoJ"], before["parity"])
        assert math.isclose(row["baseline_transition_cm1"],
                            before["excitation_from_tracked_6s_cm1"], abs_tol=2e-6,
                            rel_tol=0)
        assert math.isclose(row["accepted_transition_cm1"], row["baseline_transition_cm1"] +
                            row["transition_delta_from_same_lower_cm1"], abs_tol=2e-6,
                            rel_tol=0)
        rows.append({**row, "NIST_nominal_cm1": before["NIST_nominal_cm1"],
                     "baseline_DC_gap_cm1": row["baseline_transition_cm1"] - before["NIST_nominal_cm1"],
                     "one_step_DC_gap_cm1": row["accepted_transition_cm1"] - before["NIST_nominal_cm1"]})
    changes = "/".join(f"{r['transition_delta_from_same_lower_cm1']:+.3f}" for r in rows
                       if r["label"] != "6s_1/2")
    now = datetime.now(timezone.utc).isoformat()
    result.update(created_at=now, production_certified=False, experimental_gate_passed=False,
                  highest_level_reached=False)
    result["inputs"].update(previous_stage002=binding(parent_path),
                            mixed_CI_probe=binding(probe_path),
                            mixed_CI_probe_verification=binding(verification_path),
                            mixed_CI_independent_replay=binding(independent_path))
    result["mixed_CI_stationarity"] = {
        "scope": probe["scope"], "operator": "Dirac-Coulomb EOL objective and diagnostic spectra",
        "baseline": baseline, "one_step": step, "matched_DC_rows": rows,
        "expanded_MCDHF_converged": False, "production_certified": False,
        "experimental_gate_passed": False, "original_metric_sha256": probe["original_metric_sha256"],
        "static_Breit_spectrum_replaced_or_corrected": False,
        "complete_public_MCDHF_solve": False, "resource_observation": probe["resource"],
    }
    result["card_summary"] = (
        "Yb⁺ 原生五参考MCDHF驻值基线5.143e-08 <1e-7已闭合，认证仅限5-CSF。"
        "同轨道平衡294→468→531固定轨道静态RCI已实算；"
        "D3/D5/P1/P3的NIST gap +2082.9/+1302.9/-2027.3/-2129.1 cm⁻¹，均未过原1%门。"
        f"扩展531混合CI等权EOL沿原metric的真实残差{baseline['original_metric_orbital_residual']:.5f}，"
        f"一个原下降门接受步降至{step['original_metric_orbital_residual']:.5f}，仍远高于1e-7。"
        f"该步DC上下态差增量{changes} cm⁻¹；平均总能量下降却没有改善四线实验差距。"
        "独立双边完整CI导数核验通过；这只是有界驻值诊断，非完整扩展MCDHF。"
        "DC一步不能加到静态531或独立003的QED账。下一步完整混合CI轨道优化与稳健相对能量，"
        "并估算核双激发/高阶关联，再选择虚轨道扩展。实验和最高精度门仍失败。"
    )
    result["selection_reason"] = (
        "Actual mixed-CI EOL gradient shows the 531 fixed-orbital RCI is not stationary in the original metric. "
        "One accepted same-target DC orbital step lowers the EOL objective but worsens all four nominal transition gaps; "
        "complete original-gate orbital optimization and missing-channel differential estimates remain separate requirements."
    )
    result["next_execution_status"] = (
        "有界531混合CI驻值/单步诊断已完成；完整扩展MCDHF及其生产executor尚未完成。"
        "小时选择器依实际源变化运行已有复验，不把实现前沿标成新生产计算。"
    )
    target = HERE / "results_stage003.json"
    if target.exists():
        raise FileExistsError("Immutable stage result already exists")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    registry_path = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    registry = json.loads(registry_path.read_text())
    registry["updated_at"] = now
    registry["physics_budget_result"] = target.relative_to(REPO).as_posix()
    registry["tasks"][0].update(
        status="五参考原驻值门已过；扩展531真实混合CI驻值未过",
        detail="原metric不变：531残差0.03676→单步0.02499，均≫1e−7；双边CI重解导数核验通过。只完成有界诊断，不能称完整扩展MCDHF。")
    registry["tasks"][1].update(
        status="平衡531固定轨道RCI及混合CI轨道响应已实算",
        detail=f"静态RCI四线仍未过1%；轨道一步DC差能增量{changes} cm⁻¹，EOL下降却不改善gap。下一步完整混合CI优化及缺失核双激发量级，不拼不同算符/模型账。")
    registry["physics_next_setups"][0] = (
        "Native 531 mixed-CI MCDHF: retain original metric/projectors/weights and physical CI tracking; "
        "establish full orbital optimization and exact relative-energy/CI-response comparison with independent derivative gates. "
        "One accepted step is not stationary or a production spectrum."
    )
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(target.relative_to(REPO).as_posix()),
                      "registry": binding(registry_path.relative_to(REPO).as_posix())}))


if __name__ == "__main__":
    main()
