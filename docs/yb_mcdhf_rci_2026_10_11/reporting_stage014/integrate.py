"""Integrate the actual failed29 probe; never execute physical consumers."""

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


def write_once(path, payload):
    with path.open("x") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    contract = read(HERE / "contract.json")
    assert contract["root_authorized_reporting_integration"] is True
    assert contract["driver_sha256"] == sha(Path(__file__))
    assert contract["before_progress_sha256"] == sha(PROGRESS)
    previous_path = TOP / "results_stage013.json"
    assert sha(previous_path) == contract["previous_stage_sha256"]
    previous = read(previous_path)
    before_bytes = PROGRESS.read_bytes()
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting identity gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("old_stage013_input/" + name, sha(REPO / row["path"]) == row["sha256"])
    proposal_path = TOP / "reporting_stage014_preparation/proposal.json"
    check("reviewed_failure_proposal", sha(proposal_path) == contract["proposal_sha256"])
    proposal = read(proposal_path)
    actual = proposal["actual_probe"]
    for path, expected in proposal["actual_probe_evidence_sha256"].items():
        check("actual_probe_input/" + path, sha(REPO / path) == expected)
    publication_path = REPO / proposal["website_publication"]["path"]
    check("actual_v5_publication_SHA", sha(publication_path) == proposal["website_publication"]["sha256"])
    publication = read(publication_path)
    check("actual_v5_16models64rows", publication["version"] == 5 and publication["models"] == 16
          and publication["rows"] == 64 and publication["deployment_status"] == "succeeded")
    check("actual_failure_not_success", actual["root_exit_code"] == 1 and
          actual["derivative_diagnostic_qualified"] is False)
    check("279passed_one_failed", actual["checks_passed_before_stop"] == 279 and actual["checks_recorded"] == 280)
    check("exact_failed_gate", actual["first_failed_gate"] == {
        "name": "direction0_reported_derivative", "error": 1.2648386528285568e-6,
        "limit": 1e-8, "passed": False})
    check("actual_4H4CI1g28hS", actual["completed"]["H"] == actual["completed"]["CI"] == 4
          and actual["completed"]["gradient"] == 1 and actual["total_completed_native_hS"] == 28)
    check("actual_started_completed_equal", actual["started"] == actual["completed"])
    check("Rayleigh_gate_unexecuted", actual["direction0"]["Rayleigh_derivative_gate_executed"] is False)
    check("direction1_unexecuted", actual["direction1_executed"] is False)
    check("stationarity_unavailable", actual["new_metric_orbital_residual"] is None
          and actual["new_metric_energy_indicator_Ha"] is None)
    check("zero_optimizer", actual["optimizer_calls"] == 0)
    check("no_final_probe_result", not (TOP / "correlated29_mcdhf_probe_preparation/actual/result.json").exists())
    result = copy.deepcopy(previous)
    result["historical_stage013_card_summary"] = previous["card_summary"]
    result["historical_website_before_stage014"] = copy.deepcopy(previous["website_publication"])
    result["inputs"]["previous_stage013"] = binding(previous_path)
    result["inputs"]["reporting_stage014_failure_proposal"] = binding(proposal_path)
    result["inputs"]["atlas_actual_version5_publication"] = binding(publication_path)
    for path, expected in proposal["actual_probe_evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {
            "path": path, "sha256": expected}
    result["website_publication"] = publication
    result["correlated29_derivative_probe_failed"] = copy.deepcopy(actual)
    result["correlated29_derivative_probe_failed"]["independent_saved_array_review_status"] = "pending"
    result["correlated29_derivative_probe_failed"]["independent_saved_array_qualification_claimed"] = False
    result["correlated29_derivative_probe_failed"]["physics_cause_resolved"] = False
    result["correlated29_derivative_probe_failed"]["source_map_sha256"] = proposal["probe_scope"]["native_source_map_sha256"]
    result["correlated29_derivative_probe_failed"]["new_metric_sha256"] = proposal["probe_scope"]["metric_sha256"]
    result["correlated29_derivative_probe_failed"]["old320_metric_replaced"] = False
    result["numerical_certification_scope"] = previous["numerical_certification_scope"] + (
        " Separate source356 expanded29/1046 derivative probe closed exit1 at its first reported-root central derivative; "
        "279 runtime checks passed and one failed. Saved Rayleigh comparison is within the scalar gate but its assertion was "
        "not executed; direction1 was not evaluated. Independent29 saved-array review is pending, not qualification.")
    result["card_summary"] = proposal["proposed_card_summary"]
    result["selection_reason"] = (
        "The actual expanded29/1046 five-state DC derivative probe failed its first reported-root central derivative "
        "(error1.2648386528285568e-6>1e-8) after4H/4CI/1fullgradient/7Dirac/28hS and zero optimizer calls. "
        "Keep current full531 accepted320 originalmetric stationarity failure unchanged. Audit the four saved complete "
        "points and full gradient, distinguish reported-root versus retained-Rayleigh/norm/residual/reconstruction "
        "contributions, and establish the cause before a new bounded numerical or optimizer successor. No new space scan, "
        "weight/target change, gate relaxation or physical omission is inferred from this failure.")
    result["next_execution_status"] = (
        "Actual29/1046 probe FAILED/PARTIAL; first reported derivative gate failed, Rayleigh gate and direction1 unexecuted. "
        "Native independent saved-data audit pending. New3451-coordinate metric stationarity scalars unavailable until "
        "that separate audit; no expanded MCDHF optimizer has run. Website actual version5/16models64rows succeeded.")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    check("current320_originalmetric_result_unchanged", result["stationarity"] == previous["stationarity"])
    check("all_previous_inputs_preserved", all(result["inputs"][k] == v for k, v in previous["inputs"].items()))
    progress = read(PROGRESS)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str((TOP / "results_stage014.json").relative_to(REPO))
    progress["tasks"][0]["status"] = "完整531累计320步原驻值FAIL；29/1046梯度探针首方向失败，独立保存审核待完成"
    progress["tasks"][0]["detail"] = (
        "完整531原P/逆度量累计320步，残差1.94348e−6仍FAIL1e−7；指标9.22537e−12Ha通过，实验千cm⁻¹gap保留。"
        "新29/1046探针实际4H/4CI/1完整梯度、7Dirac、28hS、0优化；279项通过后reported中心导数误差"
        "1.26484e−6>1e−8失败。Rayleigh保存差4.574e−10但门未执行，第二方向未算。"
        "独立保存数组审核待闭合，不借用He资格；新metric驻值指标暂不可用，不改原320状态。")
    progress["tasks"][1]["status"] = "固定160屏蔽虚层判别保留；29共同轨道导数资格失败，先审同点数值语义"
    progress["tasks"][1]["detail"] = (
        "原屏蔽虚层完整1046/全模型2163，1546运行及586独立保存检查通过；四线"
        "+274.483/+237.067/+240.135/+240.724cm⁻¹，P改善D恶化，RMS2022.830→2048.365。"
        "后继在新3451坐标/新metric中只做完整梯度与方向诊断，在首门停止，0优化。"
        "先独立审计已保存四点H/CI/完整g与reported/Rayleigh响应，未启动29自洽MCDHF；原态/权重/门不变。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = result["next_execution_status"]
    output = TOP / "results_stage014.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(before_bytes)
    write_once(output, result)
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": binding(output), "progress": binding(PROGRESS), "previous": binding(previous_path),
             "before_progress": binding(HERE / "before_progress.json"), "driver": binding(Path(__file__)),
             "contract": binding(HERE / "contract.json"), "checks_count": len(checks), "checks": checks,
             "all_passed": all(c["passed"] for c in checks), "old_input_count": len(previous["inputs"]),
             "actual_probe_inputs_added": len(proposal["actual_probe_evidence_sha256"]),
             "new_science_calls": 0, "cards_sent": 0, "independent29_saved_qualification_claimed": False,
             "scope": "Authorized reporting integration only; FAILED/PARTIAL actual status, pending saved audit."}
    write_once(HERE / "integration_proof.json", proof)
    print(json.dumps({"result": proof["result"], "progress": proof["progress"],
                      "checks_count": len(checks), "all_passed": proof["all_passed"]}))


if __name__ == "__main__":
    main()
