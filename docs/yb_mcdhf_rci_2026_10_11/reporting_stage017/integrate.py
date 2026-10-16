"""Publish the closed same-point full-density gradient control; no science."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
PROGRESS = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
CONTROL = TOP / "correlated29_full_normalized_gradient_control"


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
    assert contract["previous_stage_sha256"] == sha(TOP / "results_stage016.json")
    previous = read(TOP / "results_stage016.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("previous/" + name, sha(REPO / row["path"]) == row["sha256"])
    for path, digest in contract["evidence_sha256"].items():
        check("evidence/" + path, sha(REPO / path) == digest)
    control = read(CONTROL / "actual/result.json")
    execution = read(CONTROL / "root_execution.json")
    check("closed_once", execution["returncode"] == 0 and execution["retry_count"] == 0
          and execution["foreground_single_launch"] and execution["frozen_files_unchanged"])
    check("closed_result_binding", execution["result_sha256"] == sha(CONTROL / "actual/result.json"))
    check("whole_controls", control["samepoint_full_normalized_gradient_control_completed"]
          and control["all_numeric_controls_passed"] and len(control["controls"]) == 104
          and all(row["passed"] for row in control["controls"]))
    check("physical_identity", control["source_map_sha256"] == "500ac4c3c372e0eb51e6d9910ef3830bf8eb323b211b52ba3ec08ccbf7b1b794"
          and control["original_metric_sha256"] == "a96f1aecd3e18316376121d8255c156a5f5b4c01a21fd0ba86ca91131ae66d73")
    check("actual_complete_calls", control["started"] == control["completed"]
          and all(control["completed"][k] == v for k, v in {
              "H": 1, "CI": 1, "gradient_from_weights": 2, "Dirac": 0,
              "initial_hS": 7, "CI_hS": 7, "retained_RDM": 1,
              "unfiltered_target_RDM": 5, "gradient_pair_potential": 1186,
              "tracker_resolve": 2, "tracker_SVD": 10, "tracker_overlap": 3,
              "CI_eigh": 5}.items()))
    check("scope_preserved", control["historical_derivative_failures_unchanged"]
          and not any(control[k] for k in ["optimizer_authorized",
              "partial_gradient_contains_finite_CI_eigenresponse",
              "table_angular_and_aggregation_roundoff_error_bound",
              "production_callback_or_uniform_error_bound", "full_MCDHF_or_experimental_certified"]))
    check("complete5weights", len(control["targets"]) == 5
          and len({tuple(t["UID"]) for t in control["targets"]}) == 5
          and all(t["weight"] == .2 for t in control["targets"]))
    for row, gate in zip(control["samepoint_direction_diagnostics"], (1e-8, 1.494376612201439e-8), strict=True):
        check("direction_" + str(row["direction"]), row["old_original_gate"] == gate
              and abs(row["fresh_full_partial_centered_error"]) <= gate
              and not row["old_gate_status_upgraded"])
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage016_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage016"] = binding(TOP / "results_stage016.json")
    for path, digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path": path, "sha256": digest}
    result["full_normalized_gradient_control"] = {
        "result": binding(CONTROL / "actual/result.json"),
        "execution": binding(CONTROL / "root_execution.json"),
        "scope": control["scope"], "numeric_controls_passed": 104,
        "calls": control["completed"], "optimizer_calls": 0,
        "gradient_metrics": control["gradient_metrics"],
        "whole_vector_differences": control["whole_vector_differences"],
        "samepoint_direction_diagnostics": control["samepoint_direction_diagnostics"],
        "uncontrolled_seconds": execution["elapsed_uncontrolled_s"],
        "peak_RSS_KiB": control["peak_RSS_KiB"],
        "interpretation": "Measured normalized/unfiltered density contribution is negligible at this exact baseline in the original metric. This does not certify other points, finite CI response, radial closure, production stationarity or experiment.",
        "old_failures_upgraded": False, "new_spectral_improvement": False}
    result["expanded29_optimizer_status"] = "Same-point complete density-gradient control PASS; separate prospective ordinary-reported bounded MCDHF optimizer preparing, not executed. Historical optimizer contract remains HOLD."
    result["selection_reason"] = (
        "Proceed to a separately reviewed bounded native ordinary-reported MCDHF optimization of all29 radials/1046 target CSFs with original3451 metric, five UID weights and exact finite-H accepted-step guard. "
        "The measured full-density correction cannot explain the present baseline residual. Retain stationarity and experimental failures; qualify the private radial-kernel candidate on the same full problem before performance claims.")
    result["next_execution_status"] = result["expanded29_optimizer_status"]
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI主线。29轨道/1046目标CSF同点完整归一化密度梯度实算完成：1H、1CI、2次梯度，104项数值检查通过。"
        "完整密度差的原度量范数4.515e−15，不能解释当前误差；两方向中心化误差1.093e−10/1.182e−10，均在原门内。"
        "原驻值残差0.0214991、指标1.20703e−4Ha仍FAIL，实验千cm⁻¹gap保留；未产生新谱。"
        "下一步独立准入29轨道原生MCDHF有界优化，保持五态等权、原度量与接受步保护；旧失败不升级。"
        "性能线准备同问题两体径向核等价性与阶段profile，尚无29轨道加速比。远程周期表版本5，保留NIST及Ne/Eu历史。")
    check("previous_science_preserved", result["stationarity"] == previous["stationarity"]
          and result["conditional_constraint_controls"] == previous["conditional_constraint_controls"]
          and result["experimental_gate_passed"] == previous["experimental_gate_passed"])
    check("previous_inputs_preserved", all(result["inputs"][k] == v for k, v in previous["inputs"].items()))
    progress = read(PROGRESS)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str((TOP / "results_stage017.json").relative_to(REPO))
    progress["tasks"][0].update(status="完整密度梯度同点控制PASS；原驻值/实验仍FAIL，准备实际优化", detail=result["card_summary"].split("性能线")[0])
    progress["tasks"][1].update(status="准备完整29轨道相关MCDHF有界优化；保持固定轨道负结果", detail=result["selection_reason"])
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(PROGRESS.read_bytes())
    write_once(TOP / "results_stage017.json", result)
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": binding(TOP / "results_stage017.json"), "progress": binding(PROGRESS),
             "driver": binding(Path(__file__)), "contract": binding(HERE / "contract.json"),
             "checks_count": len(checks), "checks": checks, "all_passed": True,
             "new_physics_calls": 0, "new_optimizer_calls": 0, "cards_sent": 0}
    write_once(HERE / "integration_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ("result", "progress", "checks_count", "all_passed")}))


if __name__ == "__main__":
    main()
