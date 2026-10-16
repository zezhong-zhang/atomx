"""Record the actual screened-field failure and its independent causal audit."""

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
    previous = TOP / "results_stage009.json"
    result = read(previous)
    for row in result["inputs"].values():
        assert binding(ROOT / row["path"])["sha256"] == row["sha256"]
    run = TOP / "screened_core_virtual_discriminator"
    failure = read(run / "failure.json")
    execution = read(run / "root_execution.json")
    audit = read(run / "saved_failure_audit/verification.json")
    assert execution["exit_code"] == 1 and not execution["outer_wall_cap_stopped"]
    assert execution["driver_unchanged"] and execution["execution_contract_unchanged"]
    assert failure["scientific_success_claim"] is False
    assert audit["all_independent_saved_evidence_checks_passed"]
    assert audit["original_failed_gate_remains_failed"]
    assert audit["new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves"] == 0
    for path, digest in audit["bound_inputs_sha256"].items():
        assert binding(ROOT / path)["sha256"] == digest
    failed = [row for row in failure["checks"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["name"] == "virtual_full_spectrum_constrained_residual/(6, -3)"
    assert failed[0]["limit"] == 1e-8 and failed[0]["error"] > 1e-8
    calls = failure["calls"]
    assert calls["completed_auxiliary_JK_builders"] == 10
    assert calls["completed_scalar_Slater"] == 3584
    assert calls["completed_complete_CI"] == calls["completed_implicit_H"] == 0
    assert calls["completed_explicit_H"] == calls["gradients"] == calls["optimizer_calls"] == 0
    assert audit["screened_CI_transition_or_RMS_response_available"] is False
    result["inputs"]["previous_stage009"] = binding(previous)
    for name in ("contract.json", "run.py", "execution_binding.json", "failure.json",
                 "root_execution.json", "virtual_construction_controls.json",
                 "auxiliary_scalar_control.json", "saved_failure_audit/contract.json",
                 "saved_failure_audit/verify_arrays.py", "saved_failure_audit/verification.json",
                 "saved_failure_audit/diagnostic_arrays.npz"):
        result["inputs"]["screened_failure/" + name] = binding(run / name)
    for name in ("root_source_edit_authorization002.json", "contract.json", "verification.json"):
        result["inputs"]["mixed_increment_preparation/" + name] = binding(
            TOP / "mixed_ci_finite_increment_proposal" / name)
    result["screened_auxiliary_virtual_failure"] = {
        "actual_calls": calls,
        "original_failed_gate": failed[0],
        "independent_saved_checks": audit["checks_count"],
        "saved_scalar_quadrant_controls": audit["scalar_saved_quadrant_controls"],
        "projection_and_conditioning": audit["projection_and_conditioning"],
        "lowest_root_causal_decomposition": audit["rootwise_diagnostics"][0],
        "maximum_relative_root_causal_decomposition": next(
            row for row in audit["rootwise_diagnostics"]
            if row["root"] == audit["projection_and_conditioning"]["maximum_relative_failed_root"]),
        "screened_transition_or_RMS_response_available": False,
        "original_stationarity_passed": False,
        "auxiliary_field_is_not_physical_CI_Hamiltonian": True,
        "original_gate_relaxed": False,
        "scope": audit["scope"],
    }
    result["mixed_ci_finite_increment_work"] = {
        "source_edit_root_sealed": True,
        "source_edit_only_original_authorization": True,
        "numerical_qualification_executed": False,
        "optimizer_adopted": False,
        "scope": "Opt-in complete real rank-one DC diagnostic; independent source review and finite Decimal qualification pending. Existing methods/default solver remain separate.",
    }
    result["historical_stage009_card_summary"] = result["card_summary"]
    result["selection_reason"] = (
        "The actual screened closed68 auxiliary fields pass all80 signed independent scalar "
        "quadrants. The original first116-vector full-spectrum residual gate fails before CI; "
        "613 independent saved-array checks reproduce it. The lower-triangle solver's own "
        "returned vector residual dominates the lowest and maximum-relative failures; "
        "projection/mapping contributions for those vectors are orders smaller, with S well "
        "conditioned. Next qualify deterministic full-spectrum numerical refinement on the "
        "same saved kappa-3 operator/P/complement, preserving every original gate. All5 spectra "
        "and complete1046 physical CI need a later separate contract. In parallel freeze and "
        "independently qualify the mixed-CI finite increment. No screened spectrum is available.")
    result["next_execution_status"] = (
        "Numerical refinement and finite-increment qualification are prospective. "
        "No screened CI response, improved experimental accuracy, full optimizer speedup "
        "or highest-level certificate is asserted.")
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI主线不变。完整531累计320步，原残差1.94348e−6仍FAIL，"
        "实验千cm⁻¹gap仍在。固定160点5p双空穴RMS2022.830→2225.974；"
        "核场虚层→2019.201，四线只变几cm⁻¹。新闭核屏蔽势全部80个有符号PP/QQ/PQ/QP"
        "独立矩阵锚通过；首个虚轨道116根全谱残差最大3.11309e−8超过原1e−8，"
        "在CI前停止。613项保存数组独立审核确认主要为本征求解器向量残差，"
        "最低根投影/映射贡献仅约3e−14Ha，S非病态。未得到屏蔽谱线或RMS，不能宣称改善。"
        "下轮在原P/原hS/全116根做确定性数值精化，原门与失败保留；"
        "并行开发混合CI有限增量，数值资格待跑，未接优化器。"
        "已测完整H受检归档链中位1.24259倍、用时降19.52%，非全优化器/GPU/并行加速。"
        "远程周期表版本4，14模型56行，Ne/Eu与历史保留。")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage010.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(path.read_bytes())
    progress = read(path)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["tasks"][0]["detail"] += (
        " 混合CI有限增量诊断正在实现和独立审查，有限数值资格尚未执行，未接入优化器。")
    progress["tasks"][1]["status"] = "屏蔽势80矩阵锚通过；首个虚轨道全谱残差失败，CI尚未执行"
    progress["tasks"][1]["detail"] = (
        "固定accepted160/原P/旧24/同完整1046家族。10J/K与3584标量Slater完成；"
        "首个116根全谱最大相对残差3.11309e−8>原1e−8，49根未过，在CI前停止。"
        "613项保存数组复核：最低根本征求解项3.02129e−8Ha，投影/映射项约3e−14Ha。"
        "先在同保存算符全谱做确定性数值精化；未有屏蔽线位移/RMS，不放宽门或选择性取根。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = (
        "Strict native-H identity and complete signed finite mixed-CI response; freeze "
        "candidate first, then independent Decimal80/ordinary-path parity qualification. "
        "No optimizer adapter or speedup claim before qualification. Existing retained-H "
        "archival-chain performance remains its own measured scope.")
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "progress": binding(path), "new_science_calls": 0}))


if __name__ == "__main__":
    main()
