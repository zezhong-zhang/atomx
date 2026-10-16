"""Publish a source-bound numerical baseline and distinct CV diagnostic."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(relative):
    return json.loads((REPO / relative).read_text())


def binding(relative):
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError("Evidence escaped repository")
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    prefix = HERE.relative_to(REPO).as_posix()
    paths = {
        "current_mcdhf": f"{prefix}/native_run/card_evidence_stationary_baseline.json",
        "actual_yb": f"{prefix}/native_run/nuclear_eol005_projected_increment/result.json",
        "original_metric_replay": f"{prefix}/native_run/nuclear_eol005_projected_increment/stationarity_replay002/result.json",
        "cv_discriminator": f"{prefix}/cv_discriminator/same005_orbitals001/result.json",
        "cv_verification": f"{prefix}/cv_discriminator/verification.json",
        "balanced_frontier": f"{prefix}/balanced_cv_frontier/result.json",
        "balanced_frontier_verification": f"{prefix}/balanced_cv_frontier/verification.json",
        "balanced_cv_result": f"{prefix}/balanced_cv_replay/result.json",
        "balanced_cv_verification": f"{prefix}/balanced_cv_replay/verification.json",
        "physics_independent_audit": f"{prefix}/physics_audit/verification.json",
        "same003_correction_ledger": f"{prefix}/correction_budget/same_orbital001/result.json",
        "correction_verification": f"{prefix}/correction_budget/verification.json",
        "gradient_reuse": f"{prefix}/performance/gradient_probe001/result.json",
        "previous_stage": f"{prefix}/results_stage001.json",
    }
    proof = read(paths["current_mcdhf"])
    cv = read(paths["cv_discriminator"])
    correction = read(paths["same003_correction_ledger"])
    frontier = read(paths["balanced_frontier"])
    balanced = read(paths["balanced_cv_result"])
    for item in proof["inputs"].values():
        if binding(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("MCDHF proof input changed")
    gate = proof["stationarity"]
    assert gate["converged"] and gate["orbital_residual"] < gate["orbital_tolerance"]
    assert gate["estimated_energy_error"] < gate["energy_tolerance"]
    assert not proof["experimental_gate_passed"] and not proof["highest_level_reached"]
    assert cv["numerical_gates_passed"] and cv["diagnostic_only"]
    assert cv["expanded68_MCDHF_stationarity_passed"] is None
    assert cv["tracked_6s_is_finite_model_global_ground"]["static_Breit"]
    assert read(paths["cv_verification"])["passed"]
    assert read(paths["physics_independent_audit"])["passed"]
    assert read(paths["correction_verification"])["passed"]
    assert read(paths["balanced_frontier_verification"])["all_passed"]
    assert not frontier["computed_spectrum"] and not frontier["expanded_MCDHF_run"]
    assert sum(frontier["family_counts"].values()) == 531
    assert read(paths["balanced_cv_verification"])["all_passed"]
    assert balanced["numerical_gates_passed"] and balanced["diagnostic_only"]
    assert balanced["bare005_MCDHF_stationarity_passed"]
    assert balanced["expanded_MCDHF_stationarity_passed"] is None
    assert not balanced["production_certified"] and not balanced["complete_CV_claim"]
    assert not balanced["experimental_gate_passed"]
    bare = {r["label"]: r for r in proof["rows"]}
    rows = cv["tracked_levels"]["static_Breit"]
    for row in rows:
        original = bare[row["label"]]
        assert (row["twoJ"], row["parity"]) == (original["twoJ"], original["parity"])
        assert row["binding_unambiguous"]
        assert math.isclose(row["excitation_from_tracked_6s_cm1"],
                            original["computed_cm1"] + row["delta_excitation_from_bare_cm1"],
                            rel_tol=0, abs_tol=2e-6)
        assert math.isclose(row["gap_cm1"],
                            row["excitation_from_tracked_6s_cm1"] - row["NIST_nominal_cm1"],
                            rel_tol=0, abs_tol=1e-6)
    previous = {name: r["computed_cm1"] for name, r in bare.items()}
    stages = balanced["stages"]
    assert [s["CSFs"] for s in stages] == [294, 468, 531]
    for stage in stages:
        assert stage["tracked_6s_is_actual_ground"]["static_Breit"]
        assert stage["all_target_bindings_unambiguous"]
        current = stage["tracked_levels"]["static_Breit"]
        for row in current:
            original = bare[row["label"]]
            assert (row["twoJ"], row["parity"]) == (original["twoJ"], original["parity"])
            assert row["binding_unambiguous"]
            excitation = row["excitation_from_tracked_6s_cm1"]
            assert math.isclose(excitation, original["computed_cm1"] +
                                row["transition_delta_from_bare_cm1"],
                                rel_tol=0, abs_tol=2e-6)
            assert math.isclose(excitation, previous[row["label"]] +
                                row["transition_delta_from_previous_cm1"],
                                rel_tol=0, abs_tol=2e-6)
            assert math.isclose(row["gap_cm1"], excitation - row["NIST_nominal_cm1"],
                                rel_tol=0, abs_tol=1e-6)
        previous = {r["label"]: r["excitation_from_tracked_6s_cm1"] for r in current}
    final_rows = stages[-1]["tracked_levels"]["static_Breit"]
    final_gaps = "/".join(f"{r['gap_cm1']:+.1f}" for r in final_rows if r["label"] != "6s_1/2")
    ledger = [r for r in correction["state_correction_ledger"] if r["label"] != "6s_1/2"]
    dynamic = [abs(r["components"]["dynamic_minus_static_Breit"]["relative_to_6s_cm1"])
               for r in ledger]
    se_difference = [abs(r["components"]["SE_indelicato_overlap_model"]["relative_to_6s_cm1"]
                         - r["components"]["SE_shabaev_local_partial"]["relative_to_6s_cm1"])
                     for r in ledger]
    now = datetime.now(timezone.utc).isoformat()
    summary = (
        f"Yb⁺ 原生五态共同轨道 MCDHF 从原初始化68步闭合原驻值门："
        f"{gate['orbital_residual']:.3e} < {gate['orbital_tolerance']:.1e}；"
        "独立原metric重放通过，仅认证5-CSF数值基线。"
        "同驻值轨道平衡4f/5p/5s允许家族已实算294→468→531 CSF，"
        "完整4f恢复此前68族缺失的6s基态关联能移，全部目标绑定无歧义。"
        f"531静态RCI的D3/D5/P1/P3实验gap为 {final_gaps} cm⁻¹，四条均未过原1%门。"
        "5p对差能增量约+304–808 cm⁻¹，5s约−34至+16 cm⁻¹；P较五参考改善、D变差。"
        "固定轨道RCI不认证扩展MCDHF驻值；下一步检验混合CI响应与轨道松弛，"
        "并估算缺失核双激发/高阶关联，不据此宣称某项主导。"
        f"另一个003轨道诊断账：动态−静态Breit约{min(dynamic):.2f}–{max(dynamic):.2f} cm⁻¹，"
        f"两个现有SE模型差{min(se_difference):.0f}–{max(se_difference):.0f} cm⁻¹；"
        "模型差不是误差界，不能拼入005/68预测，也不能拟合选模型。实验及最高精度门仍失败。"
    )
    result = {
        "created_at": now,
        "primary_route": "Atomx native MCDHF -> RCI",
        "computed_run": proof["computed_run"],
        "fitted_corrections": False,
        "highest_level_reached": False,
        "production_certified": False,
        "five_reference_numerical_stationarity_certified": True,
        "inputs": {name: binding(path) for name, path in paths.items()},
        "stationarity": gate,
        "numerical_certification_scope": proof["production_certification_scope"],
        "correlation_scope": proof["correlation_scope"],
        "card_summary": summary,
        "selection_reason": "Balanced fixed-orbital families restore missing lower-state coupling, but final P and D gaps respond differently and all fail the original gate. Establish mixed-CI orbital stationarity and estimate missing higher-rank channels before attributing the remaining gap or scanning radial cutoffs.",
        "next_execution_status": "平衡294/468/531有限家族RCI已完成；小时服务执行已注册有限复验，扩展混合CI MCDHF驻值和生产executor尚未完成。",
        "cv_discriminator": {
            "scope": cv["scope"],
            "stationarity_of_expanded_model": "not evaluated",
            "static_Breit_rows": rows,
            "static_Breit_multiplets": cv["multiplets"]["static_Breit"],
            "experimental_gate_passed": False,
            "complete_correlation_claim": False,
        },
        "balanced_frontier": {
            "scope": "Same005 orbitals, complete allowed one-core-hole/two-particle families in the finite {6s,5d,6p} active pool; no core doubles or full CV completion.",
            "preflight_summary": frontier["card_summary"],
        },
        "balanced_family_RCI": {
            "scope": balanced["scope"],
            "model_sizes": [s["CSFs"] for s in stages],
            "stages": [{"name": s["name"], "CSFs": s["CSFs"],
                        "static_Breit_rows": s["tracked_levels"]["static_Breit"],
                        "static_Breit_multiplets": s["multiplets"]["static_Breit"]}
                       for s in stages],
            "final_static_Breit_rows": final_rows,
            "expanded_MCDHF_stationarity_passed": None,
            "production_certified": False,
            "experimental_gate_passed": False,
            "complete_CV_claim": False,
            "resource_observation": balanced["resource"],
        },
        "correction_budget": {
            "input_run": correction["input_run"],
            "role": "Separate same003-orbital diagnostic; not a correction to 005 or expanded68 predictions",
            "dynamic_minus_static_abs_range_cm1": [min(dynamic), max(dynamic)],
            "SE_model_difference_abs_range_cm1": [min(se_difference), max(se_difference)],
            "model_difference_is_error_bound": False,
            "complete_nonlocal_or_screened_QED_claim": False,
        },
        "performance": {
            "component_matvecs_before": 1728,
            "component_matvecs_after": 224,
            "signed_full_gradient_bitwise_unchanged": True,
            "end_to_end_speedup_claim": None,
            "parallel_efficiency_claim": None,
            "gpu_speedup_claim": None,
            "timing_load_gate_failed": True,
        },
        "remaining_physical_gates": [
            "Expanded-model mixed-CI response and orbital stationarity on already computed balanced families",
            "Core-double/higher-rank and radial/virtual-space convergence",
            "Same-state complete QED and isotope/operator coverage",
            "Independent unfitted experiment and same-state observable validation",
            "Controlled end-to-end and parallel/GPU performance",
        ],
    }
    target = HERE / "results_stage002.json"
    if target.exists():
        raise FileExistsError("Immutable stage result already exists")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    registry_path = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    registry = json.loads(registry_path.read_text())
    registry["updated_at"] = now
    registry["yb_mcdhf_result"] = paths["current_mcdhf"]
    registry["physics_budget_result"] = target.relative_to(REPO).as_posix()
    tasks = registry["tasks"]
    tasks[0].update(status="68步闭合原驻值门；仅五参考数值基线合格",
                    detail="原初始化/metric不变，24径向轨道共同优化，残差5.143e−8 <1e−7，独立完整能量/梯度/原metric重放通过。混合CI响应与更大模型不能继承此认证。")
    tasks[1].update(status="同005轨道平衡294→468→531-CSF RCI已完成",
                    detail=f"最终D3/D5/P1/P3 gap {final_gaps} cm⁻¹；完整4f补回基态通道，P改善而D变差，四条仍未过1%。下一步混合CI响应轨道优化与核双激发量级估算；未认证扩展MCDHF或完整CV。")
    tasks[2].update(status="静态RCI及独立同003轨道修正账已实算",
                    detail=f"动态−静态Breit{min(dynamic):.2f}–{max(dynamic):.2f} cm⁻¹；现SE模型差{min(se_difference):.0f}–{max(se_difference):.0f} cm⁻¹。非局域/屏蔽多体QED与有限核覆盖待补；不拼到不同轨道/态、不按NIST拟合模型。")
    tasks[3].update(status="五参考驻值已过；扩展相关态性质仍待计算")
    registry["spectra"][0].update(
        comparison="Yb⁺原生五态MCDHF原驻值门通过；主表明确列五参考静态RCI/NIST基线。平衡294→468→531固定轨道RCI另列实际gap；强度/寿命未在扩展模型完成。",
        status="数值驻值、有限关联与实验精度分开；历史SD保留。005五参考通过不代表531-CSF轨道已优化或最高水平。")
    registry["physics_next_setups"] = [
        "Native common-orbital MCDHF/RCI: retain original stationarity metric and independent constrained objective gates; extend CI-response-safe comparison before mixed-target production certification.",
        "Balanced 294/468/531 fixed-orbital RCI is computed with matched lower/upper states. Next compare mixed-CI orbital relaxation and missing core-double/higher-rank differential contributions; no dominant mechanism is established by these finite increments.",
        "Complete same-state RCI Breit/QED/recoil coverage; SE model differences and unsupported nonlocal/screened terms remain explicit, never fitted to NIST.",
        "Native same-wavefunction E1/E2/M1 and response after matching/expanded-model stationarity; retain Ne/Eu successful baselines and historical lifetime failures.",
        "NIST/reference-aware atlas and independent FAC/GRASP/AMBiT comparisons; external programs remain comparators, not Atomx production backends.",
        "Exact shared contractions and immutable identities first; then controlled same-model end-to-end CPU/GPU/parallel benchmarks. No timing claim under failed load gates.",
    ]
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(target.relative_to(REPO).as_posix()),
                      "registry": binding(registry_path.relative_to(REPO).as_posix())}))


if __name__ == "__main__":
    main()
