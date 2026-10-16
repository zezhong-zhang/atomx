"""Register closed refinement and finite decrement executions without promotion."""

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
    previous = TOP / "results_stage010.json"
    result = read(previous)
    for row in result["inputs"].values():
        assert binding(ROOT / row["path"])["sha256"] == row["sha256"]
    refined = TOP / "screened_core_virtual_refinement001"
    refinement = read(refined / "result.json")
    assert read(refined / "root_execution.json")["exit_code"] == 0
    assert refinement["numerical_first_kappa_qualification_passed"]
    assert refinement["first_kappa_full_root_count"] == 116
    assert all(row["passed"] for row in refinement["checks"])
    assert refinement["calls"]["new_Atomx_Fock_Slater_Dirac_SCF_hS_H_CI_gradient_optimizer_calls"] == 0
    mixed = TOP / "mixed_ci_finite_increment_implementation"
    failed = read(mixed / "actual/result.json")
    assert read(mixed / "root_execution.json")["exit_code"] == 1
    assert failed["numerical_qualification_passed"] is False
    calls = read(mixed / "actual/candidate_calls.json")
    old_calls = read(mixed / "actual/frozen_calls.json")
    assert calls["completed"]["H_build"] == calls["completed"]["CI_solve"] == 13
    assert old_calls["completed"]["H_build"] == 5
    assert calls["completed"]["gradient"] + old_calls["completed"]["gradient"] == 3
    log = (mixed / "root_execution.log").read_text()
    assert "26 passed" in log and "1 failed" in log
    assert "test_signed_central_directional_derivative_in_same_constrained_space" in log
    for directory, names in ((refined, ("contract.json", "run.py", "execution_binding.json",
            "root_execution.json", "result.json", "final_complete_spectrum.npz")),
            (mixed, ("qualification_contract.json", "execution_contract.json", "prepared_handoff.json",
            "root_execution.json", "root_execution.log", "actual/result.json", "actual/candidate_calls.json",
            "actual/frozen_calls.json", "actual/finite_diagnostics.json", "actual/geometry_inputs.npz"))):
        for name in names:
            result["inputs"][directory.name + "/" + name] = binding(directory / name)
    result["inputs"]["previous_stage010"] = binding(previous)
    result["inputs"]["actual_stage010_card"] = binding(TOP / "delivery_stage010/delivery_proof.json")
    result["inputs"]["mixed_independent_source_review"] = binding(
        TOP / "mixed_ci_finite_increment_source_review/verification.json")
    result["screened_first_kappa_numerical_refinement"] = {
        key: refinement[key] for key in ("numerical_first_kappa_qualification_passed", "calls",
            "first_kappa_full_root_count", "final_full_residual_max", "lowest_actual_Rayleigh_Ha",
            "first_actual_gap_Ha", "raw_old_root_overlap_squared", "screened_CI_response_available",
            "other_four_kappas_qualified", "original_stationarity_passed", "experimental_validation_passed")}
    result["screened_first_kappa_numerical_refinement"]["independent_saved_array_audit_completed"] = False
    result["mixed_ci_finite_increment_work"] = {
        "source_edit_root_sealed": True, "source_map_sha256": failed["source_map_sha256"],
        "numerical_qualification_executed": True, "numerical_qualification_passed": False,
        "tests_passed": 26, "tests_failed": 1,
        "failed_test": "central constrained directional derivative at saved h=1e-4",
        "measured_derivative": -0.008720635851431197,
        "gradient_derivative": -0.00872046743791691,
        "candidate_calls": calls, "frozen_calls": old_calls,
        "remaining_scope_rejection_tests_executed": False,
        "saved_three_scale_causal_audit_completed": False, "optimizer_adopted": False,
        "scope": "All thirteen finite Decimal radial/CI response checks and five frozen-H plus gradient parity passed; overall finite qualification failed its central derivative check. Later guards skipped by fail-fast. No changed gate or optimizer/performance certificate."}
    result["historical_stage010_card_summary"] = result["card_summary"]
    result["selection_reason"] = (
        "The original saved first-kappa operator now passes all116 original residual/Gram/no-pair gates after one deterministic complete Jacobi sweep. "
        "Independently replay this result, then separately admit all5 spectra and full1046 CI using saved fields. "
        "Mixed-CI finite radial/response and frozen-path parity checks pass, but the complete qualification fails a central derivative gate; "
        "analyze all saved signed three-scale data before choosing an implementation change. Preserve failures and unexecuted guards.")
    result["next_execution_status"] = (
        "Saved-array independent refinement and central-step causal audits are in progress. All5 screened spectra/1046 CI need a separate bounded contract; "
        "no screened line/RMS response, mixed optimizer adoption, improved experiment or highest-level claim.")
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI主线不变。完整531累计320步，原残差1.94348e−6仍FAIL，实验千cm⁻¹gap仍在。"
        "固定160点5p双空穴RMS2022.830→2225.974；核场虚层→2019.201。闭核屏蔽势80个signed矩阵锚通过，"
        "首轮116根残差3.11309e−8失败保留；613项数组审核定位求解器向量误差。后继同h/S/P全116根经一轮6670个Jacobi旋转，"
        "最大原物理相对残差2.04606e−14，Gram3.10732e−16，最低态身份保留。此项仅首κ数值通过，独立数组重验待闭合；"
        "其余四κ和完整1046屏蔽CI未算，尚无屏蔽谱线或RMS。混合CI增量实际13CI/18H/3g，26项通过后中心导数门失败；"
        "完整Decimal80径向/系数响应及旧H/梯度逐位一致通过，后续非法输入测试未执行。分析已保存三个步长，未接优化器。"
        "已测受检H归档链中位1.24259倍/用时降19.52%，非全优化器/GPU/并行加速。远程周期表版本4，14模型56行。")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage011.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(path.read_bytes())
    progress = read(path)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["tasks"][0]["detail"] += (
        " 有限混合CI增量26项通过后中心导数门失败；13CI/18H/3g已实算，三个已保存步长因果分析待闭合，后续拒绝非法输入门未执行。")
    progress["tasks"][1]["status"] = "首κ全部116根数值精化通过；完整屏蔽CI响应待算"
    progress["tasks"][1]["detail"] = (
        "固定accepted160/原P/旧24。原80矩阵锚/613数组审核及全谱失败保留；同h/S/P全部116根一轮Jacobi后原相对残差2.04606e−14、"
        "Gram3.10732e−16通过，最低根重叠平方约1。独立数组重验待闭合；其他四κ和完整1046屏蔽响应仍未实算，不报告谱线/RMS改善。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "progress": binding(path), "new_science_calls": 0}))


if __name__ == "__main__":
    main()
