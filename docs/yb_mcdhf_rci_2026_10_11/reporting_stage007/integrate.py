"""Append completed saved trajectory and bounded retention controls; no science."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]


def binding(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    previous = TOP / "results_stage006b.json"
    result = json.loads(previous.read_text())
    for row in result["inputs"].values():
        assert binding(ROOT / row["path"])["sha256"] == row["sha256"]
    completion = TOP / "public_mixed_resume_run003/completion"
    implementation = TOP / "checkpoint_hamiltonian_reuse_implementation"
    trajectory = json.loads((completion / "verification.json").read_text())
    handoff = json.loads((implementation / "handoff.json").read_text())
    controls = json.loads((implementation / "verification.json").read_text())
    assert binding(completion / "verification.json")["sha256"] == "9b49a7f68652d3681fb109da776274103f03f27a4f7ab581dc8976a2f430d9ce"
    assert binding(implementation / "handoff.json")["sha256"] == "7e4523e955d2856a91b2eb6b58790b6322565d5f1a21c028890ff2ecdc1e0575"
    assert trajectory["all_passed"] and trajectory["checks_count"] == 4963
    assert all(v == 0 for v in trajectory["new_physical_calls"].values())
    assert controls["all_passed"] and controls["check_count"] == 767
    assert handoff["actual_unit_tests_passed"] == 85
    for relative, expected in handoff["artifacts_sha256"].items():
        assert binding(implementation / relative)["sha256"] == expected
    result["inputs"]["previous_stage006b"] = binding(previous)
    for relative in ("completion.json", "verification.json", "trajectory_report.json", "handoff.json"):
        result["inputs"]["run003_completion/" + relative] = binding(completion / relative)
    for relative in ("handoff.json", "verification.json", "unit_contract003.json", "unit_execution003.json", "source_identity002.json"):
        result["inputs"]["retained_H_small_controls/" + relative] = binding(implementation / relative)
    result["saved_run003_trajectory"] = {
        "independent_checks": 4963, "accepted_points": 161,
        "new_physical_calls": 0, "original_stationarity_passed": False,
        "last_energy_decrease_ULPs": 1,
        "scope": "saved arrays and original metric; not proof of roundoff causation or new optimization"}
    result["retained_H_implementation"] = {
        "source_map_sha256": handoff["source_map_sha256"],
        "focused_tests": 85, "readonly_checks": 767,
        "actual_small_control_H_builds": {"trial": 1, "accepted_gradient": 0, "getter": 0, "independent_fresh": 1},
        "default_enabled": False, "scope": "complete dense real B-spline Dirac-Coulomb",
        "full531_qualified": False, "controlled_speedup_measured": False,
        "full531_next_point": "accepted320", "physical_discriminator_fixed_point": "accepted160"}
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = TOP / "results_stage007.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    progress_path = ROOT / "docs/atomx_hourly_progress_2026_10_10/progress.json"
    with (HERE / "before_progress.json").open("xb") as stream:
        stream.write(progress_path.read_bytes())
    progress = json.loads(progress_path.read_text())
    progress["physics_budget_result"] = str(output.relative_to(ROOT))
    progress["updated_at"] = result["created_at"]
    progress["tasks"][0]["detail"] += " 全161接受点另经4963项保存数组复核，原驻值FAIL不变。"
    progress["tasks"][5]["status"] = "公共同点H复用已实现并通过85项小模型数值测试；完整531验证与受控计时待执行"
    progress["tasks"][5]["detail"] = (
        "默认关闭，仅完整dense B-spline DC；实测小模型trial构H1次，接受梯度/读取额外0次。"
        "candidate354源冻结，完整531同点H/CI/signed RDM/g/pg资格验证将独立执行，"
        "随后至少5暖态交替配对并单列冷启动；当前未认证CPU/GPU或并行加速。"
        "完整531与261五目标等价性1795项证明仍保留，仅限声明EOL范围。")
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "progress": binding(progress_path),
                      "new_physical_calls": 0}))


if __name__ == "__main__":
    main()
