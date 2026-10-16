"""Publish closed screened-virtual physics without changing archived evidence."""

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
    previous = TOP / "results_stage012.json"
    result = read(previous)
    for item in result["inputs"].values():
        assert binding(ROOT / item["path"])["sha256"] == item["sha256"]
    screened = TOP / "screened_core_virtual_discriminator002"
    actual = read(screened / "result.json")
    audit = read(screened / "saved_array_audit/verification.json")
    assert actual["numeric_identity_gates_passed"]
    assert all(x["passed"] for x in actual["checks"])
    assert audit["all_independent_saved_array_checks_passed"]
    assert all(x["passed"] for x in audit["checks"])
    assert audit["new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves"] == 0
    assert read(screened / "saved_array_audit/actual_execution.json")["exit_code"] == 0
    assert actual["calls"]["completed_complete_CI"] == 2
    assert actual["calls"]["completed_implicit_H"] + actual["calls"]["completed_explicit_H"] == 4
    assert actual["calls"]["gradients"] == actual["calls"]["optimizer_calls"] == 0
    assert not actual["original_stationarity_passed"]
    assert not actual["expanded_stationarity_evaluated"]
    assert not actual["experimental_gate_passed"]
    assert actual["old24_wave_bytes_preserved"] and not actual["original_tracker_rebased"]
    baseline = next(x for x in actual["stages"] if x["name"] == "baseline")
    expanded = next(x for x in actual["stages"] if x["name"] == "balanced_virtual")
    assert baseline["retained_target_CSFs"] == 261
    assert expanded["retained_target_CSFs"] == 1046
    assert expanded["full_model_CSFs"] == 2163
    labels = ("5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    transitions = [x for x in audit["transition_controls"] if x["label"] in labels]
    assert tuple(x["label"] for x in transitions) == labels
    for directory, names in (
        (screened, ("contract.json", "run.py", "execution_binding.json", "root_execution.json",
                    "root_execution.log", "result.json", "basis_rows.json", "canonical29_orbitals.npz",
                    "baseline_H.npz", "baseline_CI.npz", "baseline_tracking.json",
                    "balanced_virtual_H.npz", "balanced_virtual_CI.npz", "balanced_virtual_tracking.json",
                    "saved_array_audit/contract.json", "saved_array_audit/verify_arrays.py",
                    "saved_array_audit/completed_input_binding.json",
                    "saved_array_audit/root_review_authorization.json",
                    "saved_array_audit/actual_execution.json", "saved_array_audit/verification.json")),
        (TOP / "mixed_ci_finite_increment_guard_qualification002",
         ("contract.json", "run.py", "test_saved_guards.py", "execution_contract.json",
          "root_execution.json", "root_execution.log", "actual/result.json", "actual/calls.json")),
        (TOP / "mixed_ci_finite_increment_implementation/saved_array_audit",
         ("contract.json", "verification.json")),
        (TOP / "successor_consumer_source_review002/screened002", ("verification.json",)),
        (TOP / "mixed_ci_one_body_mapping_audit",
         ("contract.json", "run.py", "execution_contract.json", "execution_record.json",
          "launcher_successor001/execution_record.json", "actual/result.json")),
        (TOP / "mixed_ci_finite_increment_guard_qualification003",
         ("contract.json", "run.py", "test_saved_guards.py", "execution_contract.json",
          "root_execution.json", "root_execution.log", "actual/result.json", "actual/calls.json",
          "actual/baseline_owner_validation.json", "actual/captured_native_raw_hS.npz")),
    ):
        for name in names:
            path = directory / name
            result["inputs"][str(path.relative_to(TOP))] = binding(path)
    result["inputs"]["previous_stage012"] = binding(previous)
    result["inputs"]["actual_hour2000_card"] = binding(TOP / "delivery_hour2000/delivery_proof.json")
    result["historical_stage012_card_summary"] = result["card_summary"]
    result["fixed160_screened_virtual_response"] = {
        "numeric_identity_gates_passed": True,
        "runtime_checks_count": len(actual["checks"]),
        "independent_saved_array_checks_count": audit["checks_count"],
        "original24_seed_preserved": True, "original_P_preserved": True,
        "tracker_rebased": False, "lower_upper_transition_controls": transitions,
        "baseline": {k: baseline[k] for k in ("full_model_CSFs", "retained_target_CSFs",
                      "EOL_Ha", "nominal_four_gap_RMS_cm1", "rows", "multiplets")},
        "screened_virtual": {k: expanded[k] for k in ("full_model_CSFs", "retained_target_CSFs",
                             "EOL_Ha", "nominal_four_gap_RMS_cm1", "rows", "multiplets")},
        "all573_auxiliary_roots_audited": True, "calls": actual["calls"],
        "original_stationarity_passed": False, "expanded_stationarity_evaluated": False,
        "experimental_gate_passed": False, "physical_completeness_bound": None,
        "independent_scope_limits": audit["limits"],
        "scope": actual["scope"],
    }
    mixed = result["mixed_ci_finite_increment_work"]
    saved = read(TOP / "mixed_ci_finite_increment_implementation/saved_array_audit/verification.json")
    assert saved["all_passed"] and saved["passed_count"] == 1341
    mixed["independent_saved_array_checks_count"] = 1341
    mixed["independent_saved_array_audit_completed"] = True
    mixed["guard_successor002"] = {
        "executed": True, "passed": False, "fixture_error": "raw h compared with L(h)",
        "hS_calls": 3, "H_calls": 0, "CI_calls": 0, "gradient_calls": 0,
        "tests_qualified": 0, "planned_test_cases": 19,
        "scope": "Immutable failed setup; source-defined lower-triangle completion is not raw matrix equality.",
    }
    mapping = read(TOP / "mixed_ci_one_body_mapping_audit/actual/result.json")
    assert mapping["all_passed"] and mapping["checks_count"] == 138
    assert mapping["H_CI_gradient_radial_builders"] == 0
    mixed["one_body_mapping_audit"] = {
        "all_passed": True, "checks_count": 138, "new_scientific_builders": 0,
        "mapping": "L(h)=tril(h)+tril(h,-1).T; no averaging",
        "original_P_max_pair_defect_Ha": max(x["original_P"]["maximum_absolute_Ha"]
                                             for x in mapping["kappa_rows"]),
        "full_space_equivalence_certified": False, "scope": mapping["scope"],
        "environment_failed_launch_preserved": True,
    }
    guard = read(TOP / "mixed_ci_finite_increment_guard_qualification003/actual/result.json")
    owner = read(TOP / "mixed_ci_finite_increment_guard_qualification003/actual/baseline_owner_validation.json")
    assert guard["guard_qualification_passed"] and guard["pytest_exit_code"] == 0
    assert owner["genuine_public_producer"] and owner["H_bitwise_saved_baseline"]
    assert owner["all6_native_raw_h_S_bitwise_saved"]
    assert owner["functional_H_bitwise_lower_completed_saved_raw"]
    mixed["guard_successor003"] = {
        "executed": True, "passed": True, "pytest_cases_passed": 19,
        "actual_calls": guard["actual_calls"], "strict_raw_L_H_bitwise_parity": True,
        "original_coarse_qualification_status": guard["original_coarse_qualification_status"],
        "scope": guard["scope"], "Yb_derivative_qualified": False,
    }
    mixed["optimizer_adopted"] = False
    result["screened_first_kappa_numerical_refinement"]["other_four_kappas_qualified"] = True
    result["screened_first_kappa_numerical_refinement"]["screened_CI_response_available"] = True
    result["selection_reason"] = (
        "At the same nonstationary accepted160 point, screened virtual orbitals shift all four lines upward by237-274cm^-1; "
        "P nominal gaps improve while D gaps worsen and RMS rises2022.830->2048.365. "
        "Lower-state correlation lowers energy more than upper-state correlation, so EOL decrease is not a gap metric. "
        "Next qualify the full29 constrained correlated gradient in the same complete1046 target sectors and "
        "original no-pair P/physical tracker, then attempt a separately bounded native self-consistent MCDHF run. "
        "Declare expanded coordinates/metric as a new epoch; preserve original24 seed evidence and all failures."
    )
    result["next_execution_status"] = (
        "Public29/1046 correlated MCDHF source/driver preparation in progress; source-defined raw-h to L(h) mapping "
        "and finite guards now qualify in their saved He model; separately qualify Yb before any cancellation-safe optimizer adoption. "
        "No expanded stationarity, experimental accuracy, complete physics or whole-optimizer/GPU claim."
    )
    result["card_summary"] = (
        "Atomx原生MCDHF→RCI主线。完整531累计320步，原残差1.94348e−6仍FAIL，实验千cm⁻¹gap仍在。"
        "固定160点完整1046屏蔽虚层实算完成，1546项运行检查与586项独立保存数组审核通过；"
        "复用五个场及首κ，实际2CI/4H/内部7次hS，0新增J/K/梯度/优化。"
        "D3/D5/P1/P3移动+274.483/+237.067/+240.135/+240.724cm⁻¹；P gap改善、D变差，"
        "RMS2022.830→2048.365。下态降0.002370Ha多于上态，因此总/EOL下降不能代替谱线改善。"
        "核场虚层同点RMS2019.201、5p双空穴2225.974分别保留，增量不相加。"
        "下一步同P、同五态、完整1046中让全部29轨道自洽松弛，先验完整梯度和切向导数，显式新空间/metric epoch。"
        "混CI原资格26通过后粗步导数FAIL保留；113项尺度审计支持粗步截断，1341项保存复核通过。"
        "保护002因raw h与L(h)表示误比在setup失败保留；138项表示审计与后继003的19项保护/导数检查通过，"
        "1H/6hS/0CIg且旧H逐位一致，尚未给Yb导数或优化器资格。"
        "受检H归档链中位1.24259倍/用时降19.52%，非全优化器/GPU/并行加速。远程周期表版本4，14模型56行，新虚层谱准备追加。"
    )
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage013.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(path.read_bytes())
    progress = read(path)
    progress["updated_at"] = result["created_at"]
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["tasks"][0]["detail"] = (
        "完整531原P/逆度量累计320步；残差1.94348e−6仍FAIL1e−7，指标9.22537e−12Ha通过。"
        "同点独立H/CI/g/pg与161点证据保留，实验千cm⁻¹gap未闭合。混CI原资格26pass后粗步导数fail；"
        "三尺度113项支持中心截断，1341项完整保存复核通过。保护002因raw h/L(h)误比在setup失败，"
        "3hS/0HCIg；138项表示审计与003的19项原门检查通过，1H/6hS/0CIg且旧H逐位一致。"
        "未接优化器；准备Yb29/1046首完整梯度与切向导数。"
    )
    progress["tasks"][1]["status"] = "完整1046屏蔽虚层已算并独立复核；下一步全部29轨道自洽MCDHF"
    progress["tasks"][1]["detail"] = (
        "同accepted160/原P/旧24seed/原五态tracker，完整1046、全模型2163；1546运行及586独立数组检查通过。"
        "四线+274.483/+237.067/+240.135/+240.724cm⁻¹，P改善D恶化，RMS2022.830→2048.365。"
        "下态下降大于上态，必须看上下差能；未算扩展驻值、无收敛界。相关29共同轨道优化须显式新metric/space epoch，原门不放宽。"
    )
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress["physics_next_setups"][5] = result["next_execution_status"]
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": binding(output), "progress": binding(path), "previous": binding(previous),
             "before_progress": binding(HERE / "before_progress.json"), "new_science_calls": 0,
             "scope": "Current reporting only; immutable actual and historical proof bytes unchanged."}
    with (HERE / "integration_proof.json").open("x") as stream:
        stream.write(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
