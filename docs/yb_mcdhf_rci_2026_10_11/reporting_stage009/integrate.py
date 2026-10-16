"""Publish current evidence to the hourly registry without evaluating physics."""

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
    previous = TOP / "results_stage008b.json"
    result = read(previous)
    for row in result["inputs"].values():
        assert binding(ROOT / row["path"])["sha256"] == row["sha256"]
    vdir = TOP / "virtual_only_discriminator"
    virtual = read(vdir / "result.json")
    virtualproof = read(vdir / "saved_array_audit/verification.json")
    radial = read(vdir / "radial_support/result.json")
    assert radial["all_radial_norm_checks_passed"]
    assert radial["new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves"] == 0
    for path, digest in radial["bound_inputs_sha256"].items():
        assert binding(ROOT / path)["sha256"] == digest
    assert virtual["numeric_identity_gates_passed"]
    assert virtualproof["all_independent_saved_array_checks_passed"]
    assert not virtual["original_stationarity_passed"]
    assert not virtual["expanded_stationarity_evaluated"]
    assert virtual["calls"]["completed_complete_CI"] == 2
    assert virtual["calls"]["completed_explicit_H"] == 2
    pdir = TOP / "retained_hamiltonian_performance"
    performance = read(pdir / "verification.json")
    assert performance["all_passed"] and performance["check_count"] == 10041
    assert performance["new_performance_measurements"] == 0
    for directory in [vdir, pdir]:
        execution = read(directory / "root_execution.json")
        assert execution["exit_code"] == 0 and not execution["outer_wall_cap_stopped"]
    publication = read(TOP / "atlas_update003/root_publication.json")
    assert publication["deployment_status"] == "succeeded"
    assert publication["audience_unchanged"]
    assert publication["models"] == 14 and publication["rows"] == 56
    result["inputs"]["previous_stage008b"] = binding(previous)
    for role, directory, names in [
        ("virtual", vdir, ["contract.json", "execution_binding.json", "result.json",
                           "virtual_construction_controls.json", "root_execution.json",
                           "saved_array_audit/verification.json", "decision.json",
                           "radial_support/contract.json", "radial_support/result.json"]),
        ("performance_audit", pdir, ["verification.json", "verification_contract.json",
                                     "actual_handoff.json", "RESULTS.md"]),
        ("website_v4", TOP / "atlas_update003", ["handoff.json", "root_publication.json"]),
    ]:
        for name in names:
            result["inputs"][role + "/" + name] = binding(directory / name)
    result["fixed160_balanced_virtual_response"] = {
        "live_numeric_checks": len(virtual["checks"]),
        "independent_checks": virtualproof["checks_count"],
        "stages": [s for s in virtual["stages"] if "rows" in s],
        "actual_calls": virtual["calls"], "scope": virtual["scope"],
        "old24_wave_bytes_preserved": virtual["old24_wave_bytes_preserved"],
        "expanded_stationarity_evaluated": False,
        "original_failed_gvx_record_preserved": True,
        "missing_physics_or_radial_convergence_bound": None,
    }
    result["retained_H_chain_independent_verification"] = {
        "checks": performance["check_count"],
        "complete_archives": performance["full_archives_verified"],
        "signed_clusters": performance["signed_clusters_verified"],
        "median_ratio_implied_elapsed_reduction": performance["median_ratio_implied_elapsed_reduction"],
        "new_physical_or_timing_calls": 0,
    }
    result["saved_virtual_radial_support"] = {
        "radial_comparisons": radial["radial_comparisons"],
        "retained_CI_diagnostics": radial["retained_CI_diagnostics"],
        "limits": radial["limits"], "new_science_calls": 0,
    }
    result["historical_website_v3"] = result["website_publication"]
    result["website_publication"] = publication
    result["historical_stage008b_card_summary"] = result["card_summary"]
    result["selection_reason"] = (
        "Fixed160 complete5p doubles worsen total gaps. The same160 first balanced nuclear "
        "virtual layer changes lines by only a few cm-1, far below the remaining ~2000cm-1. "
        "Saved radial moments place the added nuclear virtuals at mean radii 0.287–0.348a0 "
        "versus old valence 3.008–4.690a0, with new-CSF diagonal denominators around 50Ha. "
        "Next distinguish screened closed68-Fock virtuals inside the SAME original no-pair "
        "space, keeping old24/accepted160 and the same complete1046 target sectors fixed. "
        "The finite response and basis-dependent EN estimate prove neither completeness nor "
        "radial convergence, and different space increments must not be added.")
    result["next_execution_status"] = (
        "Both finite rank/virtual controls are complete and independently checked. "
        "Next native MCDHF work is chosen by differential physical scale; production "
        "highest accuracy, original stationarity and experiment gates remain unmet.")
    result["card_summary"] = (
        "原生完整531 MCDHF累计320步，原残差1.94348e−6仍FAIL，实验千cm⁻¹gap仍在。"
        "固定160点完整5p双空穴使RMS gap2022.830→2225.974；"
        "同点平衡核场虚轨道层四线只变−4.782/−4.796/+2.485/+2.511cm⁻¹，"
        "RMS2022.830→2019.201。均按原态/原门实算，不据此推断物理完备或径向收敛。"
        "保存数组显示新增虚轨道均径0.287–0.348a₀，价轨道3.008–4.690a₀；"
        "下轮检验原no-pair空间中的闭核屏蔽Fock虚轨道。"
        "完整H复用经1491运行/3217资格核验，五组暖态受检归档链中位1.24259倍，"
        "链用时约降19.52%，另10041项独立数组/统计核验通过；非全优化器/GPU/并行加速。"
        "远程周期表版本4，14模型56行，320点DC与固定160点双激发分列，历史与失败保留。")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage009.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    progress_path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(progress_path.read_bytes())
    progress = read(progress_path)
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["updated_at"] = result["created_at"]
    progress["tasks"][1]["status"] = "完整双空穴与平衡虚层分别实算并独立核验；主实验gap仍在"
    progress["tasks"][1]["detail"] = (
        "固定accepted160：5p双空穴令RMS2022.830→2225.974；一层平衡核场虚轨道"
        "令RMS→2019.201，四线只变−4.782/−4.796/+2.485/+2.511cm⁻¹。"
        "保留原24轨道、态身份和门限；扩展模型未优化，无径向/关联收敛界。"
        "新增核场虚轨道均径0.287–0.348a₀，价轨道3.008–4.690a₀；"
        "下一步用闭核屏蔽Fock虚轨道判别，原no-pair空间/旧24轨道/160点/1046目标块不变。")
    progress["tasks"][4]["status"] = "远程周期表版本4已发布；14模型56行，Ne/Eu及历史保留"
    progress["tasks"][4]["detail"] = (
        "320点完整531 DC与另一个固定160点完整5p双激发RCI分别列谱、来源和原驻值失败；"
        "未拼接静态Breit、动态/QED或其他点，原始NIST与历史48行不变。")
    progress["tasks"][5]["detail"] = (
        "默认关闭；完整531 dense DC，五组交替配对受检归档链中位1.24259倍"
        "（1.19672–1.29885），对应约19.52%链用时降低；实际21H/14CI/14梯度。"
        "另10041项独立数组/统计核验通过；不认证整个优化器/GPU/并行效率。")
    progress["physics_next_setups"][1] = result["selection_reason"]
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "progress": binding(progress_path), "new_science_calls": 0}))


if __name__ == "__main__":
    main()
