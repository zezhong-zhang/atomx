"""Stdlib-only stage020: closed baseline/fix plus immutable optimizer start."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
REGISTRY = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
BASE = TOP / "corrected29_baseline_qualification_preparation"
FIX = TOP / "rank_aware_cell_discriminator_preparation/shared_fix_implementation"
OPT = TOP / "correlated29_optimization_run003_preparation"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text())
def bound(p):
    return {"path": str(p.relative_to(REPO)), "sha256": sha(p)}
def once(p, v):
    with p.open("x") as f:
        f.write(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("Prepared only; explicit --execute required")
    contract = read(HERE / "contract.json")
    assert not contract["pending_evidence"]
    assert sha(Path(__file__)) == contract["driver_sha256"]
    assert sha(REGISTRY) == contract["before_registry_sha256"]
    assert sha(TOP / "results_stage019.json") == contract["previous_stage_sha256"]
    previous = read(TOP / "results_stage019.json")
    checks = []
    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)
    for name, row in previous["inputs"].items():
        check("previous/" + name, sha(REPO / row["path"]) == row["sha256"])
    for path, digest in contract["evidence_sha256"].items():
        check("evidence/" + path, sha(REPO / path) == digest)
    actual, closed = read(BASE / "actual/result.json"), read(BASE / "root_execution.json")
    check("baseline_closed", closed["status"] == "foreground_closed" and closed["exit_code"] == 0)
    check("baseline_186_controls", len(actual["controls"]) == 186 and actual["all_numeric_controls_passed"] and all(c["passed"] for c in actual["controls"]))
    calls = actual["completed"]
    check("baseline_actual_calls", actual["started"] == calls and (calls["H"], calls["CI"], calls["gradient_from_weights"], calls["initial_hS"] + calls["CI_hS"] + calls["mixed_hS"], calls["Dirac"]) == (5, 5, 3, 42, 0))
    check("two_original_conditional_directions", [d["archived_seed"] for d in actual["samepoint_direction_diagnostics"]] == [20261011,20261012] and all(abs(d["new_centered_error"]) <= d["original_abs_rel_gate"] and d["historical_failed_gates_unchanged"] and not d["optimizer_authorized"] for d in actual["samepoint_direction_diagnostics"]))
    check("no_global_gradient_or_experimental_claim", not actual["production_callback_or_uniform_error_bound"] and not actual["full_MCDHF_or_experimental_certified"] and not actual["partial_gradient_contains_finite_CI_eigenresponse"])
    fix, fixrun = read(FIX / "completion.json"), read(FIX / "execution_record.json")
    check("shared_fix_closed3", fixrun["status"] == "foreground_closed" and fixrun["exit_code"] == 0 and fix["tests_passed"] == 3)
    check("shared_numeric_AST_lineage", fix["source_map_sha256"] == contract["shared_source_map_sha256"] and fix["numeric_AST_matches_private_f043_ignoring_docs"] and fix["H_CI_gradient_hS_Dirac_optimizer_GPU"] == 0)
    start, seal = read(OPT / "once_marker.json"), read(OPT / "execution_contract.json")
    prep = read(OPT / "preparation_verification.json")
    check("optimizer_immutable_once_start", start["exactly_one_launch"] and start["execution_contract_sha256"] == sha(OPT / "execution_contract.json") and seal["source_map_sha256"] == actual["source_map_sha256"])
    check("optimizer_prepared_bound", prep["all_source_only_checks_passed"] and prep["runner_sha256"] == sha(OPT / "run.py") and prep["contract_sha256"] == sha(OPT / "contract.json"))
    opt, optclosed = read(OPT / "actual/result.json"), read(OPT / "root_execution.json")
    optcompletion = read(OPT / "completion.json")
    check("optimizer_closed", optclosed["status"] == "foreground_closed" and optclosed["exit_code"] == 0 and opt["status"] == "bounded_public_optimizer_returned" and opt["error"] is None)
    check("optimizer_all1175_controls", len(opt["controls"]) == 1175 and all(c["passed"] for c in opt["controls"]))
    check("optimizer_actual_call_counts", opt["attempted_calls"] == opt["completed_calls"] and all(opt["completed_calls"][k] == n for k,n in {"H":27,"CI":27,"gradient":17,"optimizer":1,"validation_Dirac":7,"initial_hS":7,"CI_internal_hS":7}.items()))
    check("optimizer_original_gates_targets", not opt["stationarity_passed"] and not opt["experimental_certification"] and not opt["physical_completeness_certified"] and not opt["ordinary_reported_objective_changed"] and not opt["energy_callback_used"] and opt["original_reported_derivative_failure_preserved"])
    trajectory = opt["trajectory"]
    check("optimizer_17accepted_points", [r["iteration"] for r in trajectory] == list(range(17)) and opt["optimizer_final_diagnostics"]["iterations"] == 16 and opt["optimizer_final_diagnostics"]["termination_reason"] == "maximum_iterations")
    point_rows = []
    for row in trajectory:
        manifest = REPO / row["manifest_path"]
        check("checkpoint_manifest/" + str(row["iteration"]), sha(manifest) == row["manifest_sha256"])
        m = read(manifest)
        check("checkpoint_state/" + str(row["iteration"]), m["checkpoint_qualified"] and m["source_map_sha256"] == opt["source_map_sha256"] and m["metric_sha256"] == actual["original_metric_sha256"] and m["array_sha256"] == row["array_sha256"])
        if row["iteration"] > 0:
            check("stored_H_acceptance/" + str(row["iteration"]), m["exact_acceptance"]["qualified"] and m["exact_acceptance"]["actual_optimizer_comparison_passed"])
        point_rows.append({"iteration":row["iteration"],"checkpoint":bound(manifest),"diagnostics":row["diagnostics"],"nominal_four_gap_RMS_cm1":row["nominal_four_gap_RMS_cm1"],"targets":m["targets"],"multiplets":m["multiplets"],"stationarity_passed":m["stationarity_passed"],"experimental_certification":m["experimental_certification"]})
    check("closed_receipt_result_binding", optclosed["actual_files_sha256"]["actual/result.json"] == sha(OPT / "actual/result.json"))
    check("completion_bound_actual", sha(OPT / "actual/result.json") in json.dumps(optcompletion) and sha(OPT / "root_execution.json") in json.dumps(optcompletion))
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage019_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage019"] = bound(TOP / "results_stage019.json")
    for path,digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path":path,"sha256":digest}
    result["corrected29_baseline_qualification"] = {"result":bound(BASE / "actual/result.json"), "execution":bound(BASE / "root_execution.json"), "completion":bound(BASE / "completion.json"), "controls_passed":186, "actual_calls":calls, "source_map_sha256":actual["source_map_sha256"], "gradient_metrics":actual["gradient_metrics"], "directions":actual["samepoint_direction_diagnostics"], "scope":"Fresh corrected-source samepoint29/1046 HF-partial plus two fixed conditional Q-Lagrange controls; no global bound, optimizer callback, stationarity or experiment certificate."}
    result["shared_inner_quadrature_fix_implementation"] = {"completion":bound(FIX / "completion.json"), "execution":bound(FIX / "execution_record.json"), "shared_source_map_sha256":fix["source_map_sha256"], "numeric_AST_matches_private_f043_ignoring_docs":True, "tests_passed":3, "new_science_calls":0, "broader_suite_or_numerical_replay":False}
    result["correlated29_optimizer_run003"] = {"status":"closed_16_updates_maximum_iterations_not_stationary", "result":bound(OPT / "actual/result.json"),"execution":bound(OPT / "root_execution.json"),"completion":bound(OPT / "completion.json"),"execution_contract":bound(OPT / "execution_contract.json"),"once_marker":bound(OPT / "once_marker.json"),"source_map_sha256":opt["source_map_sha256"],"controls_passed":1175,"actual_calls":opt["completed_calls"],"points":point_rows,"optimizer_final_diagnostics":opt["optimizer_final_diagnostics"],"exact_stored_H_guard_budget":opt["exact_guard_budget"],"elapsed_uncontrolled_seconds":optclosed["elapsed_uncontrolled_seconds"],"stationarity_passed":False,"experimental_certification":False,"physical_completeness_certified":False,"energy_callback_used":False,"scope":"All29/common1046 native ordinary reported-EOL updates under original P/inverse metric/UID weights. Saved-H acceptance and retained-state controls pass; they do not certify physical derivative, stationarity or experiment accuracy."}
    result["expanded29_optimizer_status"] = "run003已闭合：17个接受点/16次共同轨道更新；达到有界步数上限，驻值及实验门仍失败。"
    result["next_execution_status"] = "后续同空间续算须独立root准入；本报告不执行或授权新的数值。"
    result["selection_reason"] = "原生29/1046共同轨道已开始真正相关自洽；16步EOL下降0.012996621Ha，但原metric残差由0.021499升至0.040910，四线RMS由2048.365升至2203.862cm⁻¹，证明总能量下降不能作为精度通过。保持空间/目标/物理门，下一续算以驻值为必要目标。"
    result["card_summary"] = "原生MCDHF→RCI：修正源基线186控制闭合（5H/5CI/3梯度、42h/S），共享核修复3解析pytest通过。run003真实闭合17点/16次共同轨道更新，27H/27CI/17梯度，1175控制与全部16步stored-H验收通过，前台116.92s；并非加速认证。EOL下降0.012997Ha，但残差0.021499→0.040910，四线RMS gap2048.36→2203.86cm⁻¹，驻值及实验门仍失败。下一同空间续算须独立准入；旧科学、旧失败、周期表和Ne/Eu历史保留。本卡送达和小时服务数值工作另记，不因本报告零数值而称服务无job。"
    check("old_scientific_fields_unchanged", all(result[k] == v for k,v in previous.items() if k not in contract["mutable_report_fields"]))
    registry = read(REGISTRY)
    registry["updated_at"] = result["created_at"]
    registry["physics_budget_result"] = str((TOP / "results_stage020.json").relative_to(REPO))
    registry["tasks"][0].update(status="原生29共同轨道16步闭合；驻值及实验门仍失败", detail=result["card_summary"])
    registry["tasks"][1].update(status="run003达到16步上限；同空间续算待准入",detail=result["selection_reason"])
    registry["tasks"][5].update(status="共享核修复3解析测试通过；无新加速认证",detail=result["card_summary"])
    registry["physics_next_setups"][1] = result["selection_reason"]
    registry["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_registry.json").open("xb") as f:
        f.write(REGISTRY.read_bytes())
    once(TOP / "results_stage020.json",result)
    REGISTRY.write_text(json.dumps(registry,ensure_ascii=False,indent=2,allow_nan=False) + "\n")
    proof = {"result":bound(TOP / "results_stage020.json"),"registry":bound(REGISTRY),"driver":bound(Path(__file__)),"contract":bound(HERE / "contract.json"),"checks_count":len(checks),"checks":checks,"all_passed":True,"new_science_calls":0,"cards_sent":0,"websites_changed":0}
    once(HERE / "integration_proof.json",proof)
    print(json.dumps({k:proof[k] for k in ("result","registry","checks_count","all_passed")}))
if __name__ == "__main__":
    main()
