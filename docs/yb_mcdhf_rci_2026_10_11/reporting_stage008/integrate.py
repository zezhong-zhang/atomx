"""Integrate completed native qualification and physical controls; no science."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    previous = TOP / "results_stage007.json"
    result = read(previous)
    for row in result["inputs"].values():
        assert binding(ROOT / row["path"])["sha256"] == row["sha256"]
    qdir = TOP / "hamiltonian_snapshot_qualification002"
    q = read(qdir / "completion.json")
    assert q["qualification_complete"] and q["producer_checks"] == 1491
    assert q["independent_checks"] == 3217 and not q["original_stationarity_passed"]
    handoff = read(qdir / "completion_handoff.json")
    for name, expected in handoff["files_sha256"].items():
        assert binding(qdir / name)["sha256"] == expected, name
    rdir = TOP / "rank_only_discriminator"
    rank = read(rdir / "result.json")
    rankproof = read(rdir / "saved_array_audit/verification.json")
    assert rank["numeric_identity_gates_passed"]
    assert rankproof["all_independent_saved_array_checks_passed"]
    assert rankproof["checks_count"] == 3247 and not rank["original_stationarity_passed"]
    diagdir = TOP / "rank_vs_virtual_discriminator/minimum_root_diagnostic"
    diag = read(diagdir / "result.json")
    diagproof = read(diagdir / "saved_array_audit/verification.json")
    assert diagproof["all_independent_saved_array_checks_passed"]
    pdir = TOP / "retained_hamiltonian_performance"
    panel = read(pdir / "actual/result.json")
    for name in [qdir, rdir, diagdir, pdir]:
        execution = read(name / "root_execution.json")
        assert execution["exit_code"] == 0 and not execution["outer_wall_cap_stopped"]
    result["inputs"]["previous_stage007"] = binding(previous)
    paths = {
        "qualification": (qdir, ["completion_handoff.json", "completion.json", "result.json",
                                   "verification.json", "root_execution.json"]),
        "rank": (rdir, ["result.json", "saved_array_audit/verification.json", "decision.json",
                          "root_execution.json"]),
        "minimum_control": (diagdir, ["result.json", "saved_array_audit/verification.json",
                                        "causal_summary.json", "root_execution.json"]),
        "performance": (pdir, ["contract.json", "execution_contract.json", "actual/result.json",
                                  "root_execution.json"]),
    }
    for role, (directory, names) in paths.items():
        for name in names:
            result["inputs"][role + "/" + name] = binding(directory / name)
    result["retained_H_implementation"].update(
        full531_qualified=True, actual_full531_checks=1491,
        independent_full531_checks=3217,
        controlled_speedup_measured=panel["controlled_warm_chain_speedup_supported"],
        original_stationarity_passed=False)
    result["full531_retained_H_qualification"] = q
    result["fixed160_complete5p_double_response"] = {
        "live_numeric_checks": len(rank["checks"]), "independent_checks": 3247,
        "stages": rank["stages"], "actual_calls": rank["calls"],
        "original_stationarity_passed": False, "expanded_orbital_optimization": False,
        "scope": rank["scope"], "experiment_gap_improved_overall": False}
    result["failed_virtual_minimum_diagnostic"] = {
        "independent_checks": len(diagproof["checks"]),
        "original_failed_gate_preserved": True, "fixed_kappa": -3,
        "same_old_and_fresh_projector_complement": True,
        "controls": diag["complements"], "other_four_minima_checked": False,
        "virtual_arm_CI_completed": False}
    result["retained_H_chain_performance"] = panel
    result["historical_five_reference_correlation_scope"] = result["correlation_scope"]
    result["correlation_scope"] = (
        "Current optimized model: full531/21 NR configurations and complete one-core-hole "
        "families in old24 radial pool. Complete5p double-hole RCI is a separate fixed160 "
        "nonstationary-point diagnostic (full2483/complete1240 target sectors), not "
        "an optimized model or complete CV/radial-correlation certificate.")
    result["numerical_certification_scope"] = (
        "Current full531 accepted320 source353 point has independent H/CI/g/pg/metric "
        "replay. Source354 retained-H path passes1491 live and3217 independent controls "
        "at the same point. Original1e-7 stationarity and1% experiment gates remain failed.")
    result["selection_reason"] = (
        "Measured complete5p doubles increase D gaps and lower total experimental accuracy; "
        "do not promote rank-only as main repair. Resolve accurately measured virtual "
        "minimum controls with unchanged gates before same160 balanced-virtual response; "
        "optimize expanded physical model only after its finite-space identity is proven.")
    result["next_execution_status"] = (
        "Complete531 reuse qualification and same160 complete5p-double discriminator "
        "finished. Balanced-virtual successor pending exact root contract and minimum controls.")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage008.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    progress_path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(progress_path.read_bytes())
    progress = read(progress_path)
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["updated_at"] = result["created_at"]
    progress["tasks"][1]["status"] = "完整5p双空穴同点实算并独立3247项核验；总体实验偏差增加"
    progress["tasks"][1]["detail"] = (
        "固定accepted160旧24轨道，完整目标块261→1240（全模型531→2483），"
        "D3/D5/P1/P3线增量+461.246/+596.713/+194.486/+148.672cm⁻¹；"
        "RMS gap2022.830→2225.974，D更差、P略改善，不能视为主gap补救。"
        "扩展模型未优化；下一步保持原门测平衡虚层，原虚层最低根报值失败另存。")
    controlled = panel["controlled_admission_passed"]
    speedup = panel["controlled_warm_chain_speedup_supported"]
    progress["tasks"][5]["status"] = (
        "完整531 H复用通过1491运行/3217独立核验；五组暖态配对计时已完成"
        if panel["scientific_parity_passed"] else
        "完整531 H复用已核验；性能判别未通过，失败与已完成测量保留")
    progress["tasks"][5]["detail"] = (
        f"默认关闭，完整dense DC；链含trial→完整g→检查与归档，"
        f"实际{panel['actual_calls']}。五组比值中位数{panel['median_warm_ratio']}，"
        f"受控准入{controlled}、预设加速门{speedup}；"
        "仅限此链，未认证整个优化器/GPU/并行效率。原驻值与实验FAIL保留。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "progress": binding(progress_path),
                      "new_physical_calls": 0}))


if __name__ == "__main__":
    main()
