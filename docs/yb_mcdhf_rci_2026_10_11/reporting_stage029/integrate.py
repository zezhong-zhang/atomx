"""Report actual native screened-f RCI and original208 directional qualification."""
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
T = B.parent
R = T.parents[1]
H = R / "docs/atomx_hourly_progress_2026_10_10"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bound(p):
    return {"path": str(p.relative_to(R)), "sha256": sha(p)}


def once(p, value):
    with p.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    contract = read(B / "contract.json")
    registry = H / "progress.json"
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        assert passed, name

    check("exact_driver", sha(Path(__file__)) == contract["driver_sha256"])
    check("live_registry_before", sha(registry) == contract["registry_before_sha256"])
    for p, digest in contract["inputs"].items():
        check("input/" + p, sha(R / p) == digest)
    fd = T / "stationarity_next_step208/instrumentation_repair001"
    native = T / "screened_f_next_step208/execution001"
    f = read(fd / "actual/result.json")
    n = read(native / "actual/result.json")
    delivery = T / "delivery_hour26/delivery_authorization_proof.json"
    d = read(delivery)
    check("FD_real_once_closed", read(fd / "root_execution.json")["exit_code"] == 0)
    check("FD7053_actual_controls", len(f["controls"]) == 7053 and all(v["passed"] for v in f["controls"]))
    check("FD_one_direction_two_scales", [v["h"] for v in f["directions"]] == [1e-5, 5e-6] and all(abs(v["error"]) <= v["original_derivative_gate"] for v in f["directions"]))
    check("FD_unchanged_stationarity_FAIL", f["stationarity_passed"] is False and f["completed"]["H"] == 5 and f["completed"]["gradient"] == 2 and f["completed"]["optimizer"] == 0)
    check("screened941_actual_controls", read(native / "root_execution.json")["exit_code"] == 0 and len(n["checks"]) == 941 and all(v["passed"] for v in n["checks"]))
    check("screened_full_scope", n["signed_scalar_controls"]["ordered_bilinears"] == 32 and all(v["whole_spectrum_rows"] == 118 for v in n["virtuals"]) and n["models"]["complete5f"]["complete_target_CSFs"] == 1595 and n["completed"]["H"] == 2 and n["completed"]["CI"] == 2)
    check("screened_no_stationarity_upgrade", not n["expanded_MCDHF_executed"] and not n["original_stationarity_passed"] and not n["physical_completeness_certified"] and not n["models"]["complete5f"]["nominal_four_1percent_goal_passed"])
    check("hS_requests_and_producers_distinct", n["hS_request_ledger"]["completed"] == 74 and n["hS_request_ledger"]["cache_hits"] == 67 and n["hS_request_ledger"]["cache_misses"] == n["completed"]["CI_internal_hS"] == 7)
    check("actual_hour26_delivery", d["passed"] and d["actual_send"]["passed"] and d["service_iteration"]["automatic_jobs"] == 0)
    check("FD_metadata_closure", read(fd / "completion.json")["all_closure_metadata_checks_passed"])
    check("native_metadata_closure", read(native / "completion.json")["all_completion_metadata_checks_passed"])
    previous = read(T / "results_stage028.json")
    result = copy.deepcopy(previous)
    panel = copy.deepcopy(read(T / "reporting_stage028/current_card_evidence.json"))
    record = read(registry)
    now = datetime.now(timezone.utc)
    summary = "Atomx原生MCDHF→RCI。真实208梯度同一切向两尺度7053门PASS，导数误差约6e−11；原驻值FAIL保持。原生closed68屏蔽f固定31轨道/1595目标RCI完成941门PASS，先32独立有符号矩阵元、再两完整118谱与1046→1595嵌套CI。D3/D5位移−207.30/−202.41、P1/P3+15.94/+19.77cm⁻¹，四线RMS2054.32→1915.98，实验门仍FAIL；新f平均半径约9a0，旧核场约0.3a0。证明有限径向种子影响，不能据旧小增量排除f关联。下一先核验31轨道相关Euler全符号梯度与新几何/度量，closed68场仅作种子；再按同点物理量级选择核关联及修正。旧失败保留，最高水平未达；暖态负载FAIL，无端到端/GPU/并行加速认证，网页v8已有21模型84行。"
    next_step = "先31/1595原生相关Euler全符号梯度/约束/新几何与度量；依据同点实证选择优化、4d/多核孔与Breit/QED，避免盲目扩截断。小时服务0自动科学作业，本小时root两项有界前台实算分别记录。"
    result.update(created_at=now.isoformat(), card_summary=summary, selection_reason=next_step, next_execution_status=next_step)
    fd_summary = {"result": bound(fd / "actual/result.json"), "receipt": bound(fd / "root_execution.json"), "completion": bound(fd / "completion.json"), "actual_controls_PASS": 7053, "independent_tangent_directions": 1, "fixed_scales": f["directions"], "completed": f["completed"], "stationarity_passed": False, "physical_completeness_certified": False}
    native_summary = {"result": bound(native / "actual/result.json"), "receipt": bound(native / "root_execution.json"), "completion": bound(native / "completion.json"), "actual_controls_PASS": 941, "models": n["models"], "virtuals": n["virtuals"], "radial_support": bound(native / "actual/radial_support.json"), "completed": n["completed"], "fixed_point_RCI_only": True, "expanded_MCDHF_executed": False, "physical_completeness_certified": False, "no_NIST_selection": True}
    result["literal208_directional_qualification"] = fd_summary
    result["screened_f_fixed208_native_RCI"] = native_summary
    result["historical_latest_hourly_delivery_stage028"] = previous["latest_hourly_delivery"]
    result["latest_hourly_delivery"] = dict(bound(delivery), source_stage=28, message_id=d["message_id"], actual_send_passed=True, actual_native_iteration_started_by_service=False, separately_admitted_root_iterations=[fd_summary["receipt"], native_summary["receipt"]])
    panel["literal208_directional_qualification"] = fd_summary
    panel["screened_f_fixed208_native_RCI"] = native_summary
    base = {v["label"]: v for v in n["models"]["baseline"]["rows"]}
    new = {v["label"]: v for v in n["models"]["complete5f"]["rows"]}
    nuclear = read(T / "complete5f_fixed208_import_repair_preparation/actual/complete5f_same_point_evidence.json")
    oldf = {v["label"]: v for v in nuclear["rows"]}
    lines = ["上述驻值量属于原29轨道208点；新增31固定轨道RCI尚未优化。单位cm⁻¹，实验为NIST名义值。", "|谱线|同208基线29|核场f固定31|屏蔽f固定31|NIST|", "|---|---:|---:|---:|---:|"]
    for label in ("5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"):
        check("same_NIST_and_UID/" + label, base[label]["reference_UID"] == new[label]["reference_UID"] == oldf[label]["reference_UID"] and base[label]["NIST_nominal_cm1"] == new[label]["NIST_nominal_cm1"])
        lines.append(f"|{label}|{base[label]['excitation_cm1']:.2f}|{oldf[label]['excitation_cm1']:.2f}|{new[label]['excitation_cm1']:.2f}|{new[label]['NIST_nominal_cm1']:.2f}|")
    lines.append("两尺度约束导数PASS仅支持原208一个切向，原驻值与实验门仍FAIL。屏蔽f全32符号矩阵元及完整118谱/嵌套CI共941门PASS；新f约9a0，种子响应不等于空间收敛。旧核场14vs7计数与首轮导数观测失败原样保留；核场保存数组另有389门资格。")
    panel["card_summary"] = "\n".join(lines)
    check("baseline_panel_physics_unchanged", panel["rows"] == read(T / "reporting_stage028/current_card_evidence.json")["rows"] and panel["stationarity"] == read(T / "reporting_stage028/current_card_evidence.json")["stationarity"] and not panel["production_certified"])
    for p, digest in contract["inputs"].items():
        result["inputs"][p] = {"path": p, "sha256": digest}
    for p in (fd / "actual/result.json", fd / "root_execution.json", fd / "completion.json", native / "actual/result.json", native / "root_execution.json", native / "completion.json", delivery):
        panel["inputs"][str(p.relative_to(R))] = bound(p)
    once(B / "current_card_evidence.json", panel)
    result["inputs"]["historical_current_mcdhf_stage028"] = previous["inputs"]["current_mcdhf"]
    result["inputs"]["current_mcdhf"] = bound(B / "current_card_evidence.json")
    result["current_native_card_evidence"] = bound(B / "current_card_evidence.json")
    check("inherited_identity_manifest_preserved", all(result["inputs"][key] == value for key, value in previous["inputs"].items() if key != "current_mcdhf"))
    check("current_panel_identity", result["inputs"]["current_mcdhf"] == bound(B / "current_card_evidence.json"))
    once(T / "results_stage029.json", result)
    record.update(physics_budget_result=str((T / "results_stage029.json").relative_to(R)), yb_mcdhf_result=str((B / "current_card_evidence.json").relative_to(R)), updated_at=result["created_at"])
    record["tasks"][0].update(status="原208梯度7053门PASS；驻值FAIL", detail=summary)
    record["tasks"][1].update(status="屏蔽f RCI941门PASS；实验FAIL", detail=next_step)
    record["tasks"][5].update(status="下一31相关Euler核验；暖态负载FAIL", detail=next_step)
    record["physics_next_setups"][1] = next_step
    record["physics_next_setups"][5] = next_step
    once(B / "registry_before.json", read(registry))
    (B / "planner_scope").mkdir()
    once(B / "planner_scope/progress.json", record)
    sys.path.insert(0, str(H))
    spec = importlib.util.spec_from_file_location("stage029_planner", H / "parallel_planner.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    decision = {"status": "implementation_frontier", "jobs": [], "frontier": [], "evidence": []}
    planner.annotate_physics_priority(R, B / "planner_scope", decision)
    check("public_proposed_binding", decision["direction"] == summary and decision["evidence"][-1]["sha256"] == sha(T / "results_stage029.json"))
    import sync_card
    card, citations = sync_card.render(record, now)
    check("actual_public_card_render", card["schema"] == "2.0" and bool(citations))
    once(B / "preview_card.json", card)
    once(B / "preview_citations.json", citations)
    check("registry_atomic_before", sha(registry) == contract["registry_before_sha256"])
    temp = registry.with_suffix(".stage029.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(registry)
    decision = {"status": "implementation_frontier", "jobs": [], "frontier": [], "evidence": []}
    planner.annotate_physics_priority(R, H, decision)
    check("public_live_binding", decision["direction"] == summary)
    once(B / "integration_proof.json", {"all_passed": True, "checks_count": len(checks), "checks": checks, "driver": bound(Path(__file__)), "contract": bound(B / "contract.json"), "result": bound(T / "results_stage029.json"), "panel": bound(B / "current_card_evidence.json"), "registry": bound(registry), "new_science_API_cards": 0, "historical_artifact_bytes_rehashed": False, "historical_identity_manifest_preserved_from_bound_stage028": True, "original_failures_and_model_limits_preserved": True})
    print(json.dumps({"PASS": len(checks), "result": bound(T / "results_stage029.json"), "registry": bound(registry), "card_elements": len(card["body"]["elements"]), "citations": len(citations)}))


if __name__ == "__main__":
    main()
