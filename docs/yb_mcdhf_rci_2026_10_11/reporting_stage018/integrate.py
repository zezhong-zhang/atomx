"""Preserve the independent quadrature failure and select a physical scale test."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
REGISTRY = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
QUAL = TOP / "correlated29_radial_reuse_qualification_preparation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def bound(path):
    return {"path": str(path.relative_to(REPO)), "sha256": sha(path)}


def write_once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    contract = read(HERE / "contract.json")
    assert contract["driver_sha256"] == sha(Path(__file__))
    assert contract["before_registry_sha256"] == sha(REGISTRY)
    assert contract["previous_stage_sha256"] == sha(TOP / "results_stage017.json")
    previous = read(TOP / "results_stage017.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("previous/" + name, sha(REPO / row["path"]) == row["sha256"])
    for name, digest in contract["evidence_sha256"].items():
        check("evidence/" + name, sha(REPO / name) == digest)
    actual = read(QUAL / "actual/result.json")
    completion = read(QUAL / "completion.json")
    receipt = read(QUAL / "root_execution.json")
    check("closed_failed_process", receipt["exit_code"] == 1)
    check("failure_preserved", actual["status"] == "FAILED_closed_no_retry"
          and not actual["all_equivalence_gates_passed"]
          and completion["status"] == "FAILED_closed_preserved"
          and not completion["scientific_qualification_passed"])
    check("only_anchor_scope", all(actual[k] == 0 for k in (
        "actual_H", "actual_CI", "actual_gradient", "actual_Dirac", "actual_optimizer", "actual_GPU"))
        and not actual["ordinary_and_candidate_chains_executed"]
        and not actual["source_identity_final_comparison_executed"])
    check("original_analytic_failure", actual["anchor_gates_passed"] == 225
          and len(actual["anchor_gate_failure"]) == 1
          and actual["anchor_gate_failure"][0]["name"] == "analytic_6_pppp"
          and actual["anchor_gate_failure"][0]["error"] > actual["anchor_gate_failure"][0]["limit"])
    check("no_performance_certification", not actual["controlled_performance_certified"])
    check("result_receipt_binding", actual["root_execution_sha256"] == sha(QUAL / "root_execution.json"))
    delivery = read(TOP / "delivery_hour22/delivery_proof.json")
    check("hour22_actual_identity", "om_x100b638d288ee13cc4302107d4fb072" in json.dumps(delivery)
          and "oc_fb2fc627b3e98dc9555de31580878758" in json.dumps(delivery))
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage017_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage017"] = bound(TOP / "results_stage017.json")
    for path, digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path": path, "sha256": digest}
    result["radial_reuse_analytic_failure"] = {
        "result": bound(QUAL / "actual/result.json"), "execution": bound(QUAL / "root_execution.json"),
        "completion": bound(QUAL / "completion.json"),
        "failed_gate": actual["anchor_gate_failure"][0],
        "actual_H_CI_gradient_optimizer_GPU": [0, 0, 0, 0, 0],
        "numerical_failure_domain": actual["numerical_failure_domain"],
        "source_causal_inference": actual["source_causal_inference"],
        "root_review": "The shared dense/semiseparable inner rule uses order points although nonnegative rank k raises polynomial degree. For k>=0, order+k//2 covers the inner polynomial. Outer negative powers and real-Yb impact need separate checks.",
        "real_Yb_gap_measured": False, "candidate_source_applied": False,
        "ordinary_and_candidate_chains_executed": False, "speedup_certified": False}
    result["latest_hourly_delivery"] = bound(TOP / "delivery_hour22/delivery_proof.json")
    result["expanded29_optimizer_status"] = "Same-point complete density-gradient controls PASS. Prospective ordinary optimizer prepares only; first estimate the newly proven shared quadrature defect on real Yb before spending optimization steps. Historical HOLD/failures remain unchanged."
    result["selection_reason"] = (
        "Prioritize physical correctness: independently qualify a private inner-quadrature order fix, separated from semiseparable routing. "
        "Estimate its complete signed normalized expectation shifts for all five real-Yb states, lower/upper and transition shifts, at the identical123 basis/29radials/1046sector point for all affected nonnegative ranks. "
        "The toy1/4900 error is not a measured Yb error. Use the real scale to choose corrected-source qualification and MCDHF optimization, preserving original gates and shared operator reuse.")
    result["next_execution_status"] = result["expanded29_optimizer_status"]
    result["card_summary"] = (
        "原生MCDHF→RCI主线。完整密度梯度同点104项过门，原驻值与实验gap仍FAIL。"
        "性能候选独立解析锚225项通过后，在原点rank6积分失败；原核和候选共同内层积分欠阶，误差3.65867e−5Ha与解析1/4900相符。"
        "后续Yb H/CI/梯度对比均未执行，没有新加速比。先独立修复内积分阶数，并测真实五态全部受影响rank的能移与跃迁改变量，"
        "再选择有界MCDHF优化；不能把玩具偏差当Yb实验gap。22:00小时卡已实际送达读回；远程周期表版本5与Ne/Eu历史保留。")
    check("old_science_preserved", result["stationarity"] == previous["stationarity"]
          and result["full_normalized_gradient_control"] == previous["full_normalized_gradient_control"]
          and result["experimental_gate_passed"] == previous["experimental_gate_passed"])
    registry = read(REGISTRY)
    registry["updated_at"] = result["created_at"]
    registry["physics_budget_result"] = str((TOP / "results_stage018.json").relative_to(REPO))
    registry["tasks"][0].update(status="完整密度梯度控制PASS；先量化真实Yb积分欠阶，再优化", detail=result["card_summary"])
    registry["tasks"][1].update(status="保持29轨道/1046扇区，先积分物理修复量级判别", detail=result["selection_reason"])
    registry["tasks"][5].update(status="独立解析锚FAIL已保留；实际Yb效率对比未执行", detail=result["card_summary"])
    registry["physics_next_setups"][1] = result["selection_reason"]
    registry["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_registry.json").open("xb") as stream:
        stream.write(REGISTRY.read_bytes())
    write_once(TOP / "results_stage018.json", result)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": bound(TOP / "results_stage018.json"), "registry": bound(REGISTRY),
             "driver": bound(Path(__file__)), "contract": bound(HERE / "contract.json"),
             "checks_count": len(checks), "checks": checks, "all_passed": True,
             "new_science_calls": 0, "new_optimizer_calls": 0, "cards_sent": 0}
    write_once(HERE / "integration_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ("result", "registry", "checks_count", "all_passed")}))


if __name__ == "__main__":
    main()
