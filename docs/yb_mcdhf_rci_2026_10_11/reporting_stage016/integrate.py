"""Report conditional saved-point controls and arithmetic guard; no physics."""
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
    previous = read(TOP / "results_stage015.json")
    assert contract["previous_stage_sha256"] == sha(TOP / "results_stage015.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("previous/" + name, sha(REPO / row["path"]) == row["sha256"])
    for path, digest in contract["evidence_sha256"].items():
        check("evidence/" + path, sha(REPO / path) == digest)
    controls = []
    for directory, limit in [("correlated29_constraint_centering_diagnosis", 1.494376612201439e-8),
                             ("correlated29_constraint_centering_diagnosis002", 1e-8)]:
        result = read(TOP / directory / "actual/result.json")
        execution = read(TOP / directory / "root_execution.json")
        check(directory + "/closed", execution["exit_code"] == 0 and result["saved_only_diagnosis_completed"])
        check(directory + "/scope", result["physical_producer_calls"] == result["eigen_SVD_calls"]
              == result["new_public_RDM_contracts"] == result["two_body_cells_or_potentials"] == 0
              and not result["optimizer_authorized"] and not result["original_rotation002_passed"])
        check(directory + "/original_gate", result["unchanged_derivative_gate"] == limit)
        check(directory + "/conditional_pass", result["same_saved_point_conditional_gate_passed"]
              and abs(result["centered_error_vs_original"]) <= limit
              and abs(result["centered_error_vs_conditional_analytic"]) <= limit)
        check(directory + "/complete_Gram", len(result["complete_same_kappa_blocks"]) == 7
              and all(t["all_exact_Gram_identities_passed"] for b in result["complete_same_kappa_blocks"] for t in b["trials"]))
        controls.append({"result": binding(TOP / directory / "actual/result.json"),
                         "execution": binding(TOP / directory / "root_execution.json"),
                         "centered_derivative": result["centered_Rayleigh_central"],
                         "error_vs_original_analytic": result["centered_error_vs_original"],
                         "original_gate": limit, "conditional_gate_passed": True,
                         "physical_producer_calls": 0, "old_failures_upgraded": False,
                         "scope": result["scope"], "known_unresolved": result["known_unresolved"]})
    guard = read(TOP / "accepted_line_search_metadata_implementation/array_qualification002/result.json")
    check("guard19", guard["all_passed"] and guard["checks_count"] == 19
          and all(v == 0 for v in guard["actual_calls"].values())
          and guard["no_Yb_gradient_or_optimizer_qualification"])
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage015_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage015"] = binding(TOP / "results_stage015.json")
    for path, digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path": path, "sha256": digest}
    result["conditional_constraint_controls"] = controls
    result["accepted_line_search_metadata"] = {
        "source_identity": binding(TOP / "accepted_line_search_metadata_implementation/source_identity.json"),
        "objective_and_acceptance_unchanged": True, "public_Yb_optimizer_executed": False,
        "exact_finite_H_guard_checks": 19,
        "guard_scope": "Exact supplied binary64 H/columns/roots/weights only; not a radial or gradient error bound."}
    result["expanded29_optimizer_status"] = "HOLD: conditional constrained controls pass; full normalized/unfiltered density gradient control pending."
    result["selection_reason"] = (
        "At the same real-owner baseline measure the complete normalized, unfiltered signed density gradient "
        "against the production retained gradient, keeping all29 radials/1046 CSFs/original3451 metric and target UIDs. "
        "Then review a separate bounded native ordinary-reported MCDHF successor with exact finite-H accepted-step guard. "
        "Preserve original failures; profile the same full problem before performance claims.")
    result["next_execution_status"] = result["expanded29_optimizer_status"]
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI主线不变。531累计320步原驻值FAIL，实验千cm⁻¹gap保留。"
        "29轨道/1046目标CSF的原reported与Rayleigh失败不改。完整signed Gram中心化的保存点检验："
        f"d0误差{controls[1]['error_vs_original_analytic']:.6g}、d1误差{controls[0]['error_vs_original_analytic']:.6g}，"
        "均在原门内，支持端点法向运动解释；零新增H/CI/梯度，不认证生产驻值。"
        "原3451度量残差0.0214991、指标1.20703e−4Ha仍FAIL。下一实測完整归一化密度梯度差，"
        "再准入有界原生轨道优化。接受步元数据已实现、有限保存H保护19项算术资格通过；"
        "新优化器仍零调用。性能线准备复用两体semiseparable核，补丁未应用，尚无29轨道加速比。"
        "远程周期表实际版本5、16模型64行，NIST与Ne/Eu历史保留。")
    check("old_stationarity", result["stationarity"] == previous["stationarity"])
    check("old_inputs", all(result["inputs"][k] == v for k, v in previous["inputs"].items()))
    progress = read(PROGRESS)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str((TOP / "results_stage016.json").relative_to(REPO))
    progress["tasks"][0].update(status="原驻值/实验门仍FAIL；约束保存点控制过门，补测完整密度梯度", detail=result["card_summary"].split("远程周期表")[0])
    progress["tasks"][1].update(status="固定轨道判别保留；推进完整29轨道相关MCDHF准入", detail=result["selection_reason"])
    progress["tasks"][5]["detail"] += " 两体势semiseparable路由仅完成源提案，原29实际阶段profile与收益待测。"
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(PROGRESS.read_bytes())
    write_once(TOP / "results_stage016.json", result)
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": binding(TOP / "results_stage016.json"), "progress": binding(PROGRESS),
             "driver": binding(Path(__file__)), "contract": binding(HERE / "contract.json"),
             "checks_count": len(checks), "checks": checks, "all_passed": True,
             "new_physics_calls": 0, "new_optimizer_calls": 0, "cards_sent": 0}
    write_once(HERE / "integration_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ["result", "progress", "checks_count", "all_passed"]}))


if __name__ == "__main__":
    main()
