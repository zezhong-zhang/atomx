"""Publish saved341 and the two closed rotation failures; no physics calls."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
PROGRESS = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path.relative_to(REPO)), "sha256": sha(path)}


def write_once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    contract = read(HERE / "contract.json")
    assert contract["driver_sha256"] == sha(Path(__file__))
    assert contract["before_progress_sha256"] == sha(PROGRESS)
    previous = read(TOP / "results_stage014.json")
    assert sha(TOP / "results_stage014.json") == contract["previous_stage_sha256"]
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("previous_input/" + name, sha(REPO / row["path"]) == row["sha256"])
    for path, digest in contract["evidence_sha256"].items():
        check("new_input/" + path, sha(REPO / path) == digest)
    saved = read(TOP / "correlated29_mcdhf_probe_preparation/saved_array_audit/verification.json")
    decomposition = read(TOP / "correlated29_mcdhf_probe_preparation/saved_array_audit/decomposition.json")
    runs = []
    for directory, expected_count, expected_gate in [
        ("correlated29_rotation_rayleigh_qualification", 143, "fresh_baseline_real_owner_identity"),
        ("correlated29_rotation_rayleigh_qualification002", 223, "direction1_NEW_Rayleigh_derivative"),
    ]:
        progress = read(TOP / directory / "actual/progress.json")
        execution = read(TOP / directory / "root_execution.json")
        controls = progress["controls"]
        check(directory + "/closed_failure", execution["exit_code"] == 1)
        check(directory + "/controls", len(controls) == expected_count
              and sum(c["passed"] for c in controls) == expected_count - 1
              and controls[-1]["name"] == expected_gate and not controls[-1]["passed"])
        check(directory + "/complete_calls", progress["started"] == progress["completed"])
        calls = progress["completed"]
        runs.append({"execution": binding(TOP / directory / "root_execution.json"),
                     "progress": binding(TOP / directory / "actual/progress.json"),
                     "controls_recorded": len(controls), "controls_passed": expected_count - 1,
                     "first_failed_gate": controls[-1], "completed": calls,
                     "completed_hS": sum(calls[k] for k in ["initial_hS", "CI_internal_hS", "mixed_hS"]),
                     "optimizer_calls": 0, "qualified": False, "cause_resolved": False})
    central = read(TOP / "correlated29_rotation_rayleigh_qualification002/actual/direction1_central.json")
    check("saved341_scope", saved["all_saved_array_checks_passed"] and saved["checks_count"] == 341
          and saved["audit_calls"]["public_H_CI_gradient_Dirac_optimizer"] == 0)
    check("new_metric_both_fail", not saved["saved_newmetric_stationarity_gates_passed"]
          and saved["reconstructed_orbital_residual"] > saved["original_orbital_tolerance"]
          and saved["reconstructed_energy_indicator_Ha"] > saved["original_energy_tolerance_Ha"])
    check("rotation_true_fd_failure", abs(central["Rayleigh_central"] - central["analytic_saved_g"])
          > central["gate"] and not central["optimizer_authorized"])
    check("actual_call_counts", runs[0]["completed"]["H"] == runs[0]["completed"]["CI"] == 1
          and runs[0]["completed_hS"] == 14 and runs[1]["completed"]["H"] == runs[1]["completed"]["CI"] == 3
          and runs[1]["completed_hS"] == 28 and runs[0]["completed"]["gradient"] == runs[1]["completed"]["gradient"] == 0)
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage014_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage014"] = binding(TOP / "results_stage014.json")
    for path, digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path": path, "sha256": digest}
    result["correlated29_saved_array_review"] = {
        "verification": binding(TOP / "correlated29_mcdhf_probe_preparation/saved_array_audit/verification.json"),
        "checks_passed": 341, "new_metric_orbital_residual": saved["reconstructed_orbital_residual"],
        "new_metric_energy_indicator_Ha": saved["reconstructed_energy_indicator_Ha"],
        "new_metric_sha256": saved["new_metric_sha256"], "stationarity_passed": False,
        "Decimal_direction0_normalized_H_quotient_derivative_error": decomposition["Decimal_normalized_Rayleigh_vs_analytic_error"],
        "original_probe_upgraded": False, "original320_metric_replaced": False,
        "not_a_complete_radial_energy_or_angular_rebuild": True}
    result["correlated29_rotation_failures"] = runs
    result["correlated29_rotation_failures"][0]["cause_resolved"] = True
    result["correlated29_rotation_failures"][0]["cause_scope"] = "Tuple versus JSON list metadata comparison; saved complete owner bytes identical. Only canonical comparison changed in separate002."
    result["correlated29_rotation_failures"][1]["central_derivative_ledger"] = central
    result["expanded29_optimizer_status"] = "HOLD: actual rotation derivative gate FAILED; zero optimizer calls."
    result["selection_reason"] = (
        "Resolve the actual same-kappa old/new orbital rotation derivative mismatch before expanded29 MCDHF. "
        "Separate saved one-body logical/raw representation, normalization, explicit versus reconstructed polar motion, "
        "signed density accumulation/cutoff, and two-body kernel derivative contributions. Choose a smallest bounded "
        "causal control; do not scan larger spaces, relax gates, change weights, or infer stationarity from acceptance.")
    result["next_execution_status"] = result["expanded29_optimizer_status"]
    result["numerical_certification_scope"] = (
        previous["numerical_certification_scope"] + " Later independent saved341 review passes its replay scope "
        "but both new3451 metric stationarity gates fail. Separate rotation001 closes at metadata identity;002 "
        "closes after222PASS/1FAIL at the explicit normalized finite-increment derivative. No expanded optimizer ran.")
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI继续。完整531累计320步原驻值仍FAIL，实验千cm⁻¹gap保留。"
        "29轨道/完整1046目标扇区的341项保存数组核验通过；新3451坐标驻值残差0.0214991、"
        "指标1.20703e−4Ha，两门FAIL，不能与原320度量混比。原径向方向失败不变。"
        "旋转资格001在tuple/list身份比较处失败，仅修序列化比较后的002实际3H/3CI/0g、"
        "7Dirac/28hS；222项通过后归一化能量导数误差2.09137e−7>1.49438e−8失败，"
        "0优化。先量化梯度/径向能量与表示舍入的贡献再修复，性能实现独立推进，未认证最高水平。"
        "远程周期表实际版本5、16模型64行；小时卡片保留历史谱与实验对照。")
    check("old_stationarity_preserved", result["stationarity"] == previous["stationarity"])
    check("old_inputs_preserved", all(result["inputs"][k] == v for k, v in previous["inputs"].items()))
    progress = read(PROGRESS)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str((TOP / "results_stage015.json").relative_to(REPO))
    progress["tasks"][0]["status"] = "531原驻值FAIL；29/1046旋转导数资格FAIL，优化暂停待修复"
    progress["tasks"][0]["detail"] = result["card_summary"].split("远程周期表")[0]
    progress["tasks"][1]["status"] = "固定160屏蔽虚层判别保留；先修原生轨道梯度/能量一致性"
    progress["tasks"][1]["detail"] = (
        "完整1046/全模型2163，29共同轨道、五态等权与原P保留。屏蔽固定轨道RMS2022.830→2048.365，"
        "尚未改善。新旋转门原阈值失败；仅做可判别的同点源/保存数组诊断，无空间扫描或权重调整。")
    progress["tasks"][4]["status"] = "远程周期表版本5已实际发布；16模型64行，NIST与历史保留"
    progress["tasks"][4]["detail"] = (
        "版本5另列固定160核场/屏蔽虚层两个完整RCI模型及旧531模型；未称扩展轨道已优化。"
        "原始NIST、Ne/Eu与全部历史不变。当前卡片补充29/1046导数失败，网页仍为版本5发布快照。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(PROGRESS.read_bytes())
    write_once(TOP / "results_stage015.json", result)
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": binding(TOP / "results_stage015.json"), "progress": binding(PROGRESS),
             "driver": binding(Path(__file__)), "contract": binding(HERE / "contract.json"),
             "checks_count": len(checks), "checks": checks, "all_passed": True,
             "new_physics_calls": 0, "new_optimizer_calls": 0, "cards_sent": 0}
    write_once(HERE / "integration_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ["result", "progress", "checks_count", "all_passed"]}))


if __name__ == "__main__":
    main()
