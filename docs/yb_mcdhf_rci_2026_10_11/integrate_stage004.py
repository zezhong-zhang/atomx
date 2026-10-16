"""Append public mixed531 evidence without replacing any frozen spectrum."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    run = HERE / "public_mixed_resume_run002"
    endpoint = load(run / "independent_endpoint_verification.json")
    assert endpoint["all_identity_replay_checks_passed"]
    launch = load(run / "root_execution.json")
    replay = load(run / "root_endpoint_execution.json")
    assert launch["execution_contract_unchanged"] and launch["driver_unchanged"]
    assert replay["exit_code"] == 0 and replay["execution_contract_unchanged"]
    baseline = load(run / "new_source_baseline_controls.json")
    assert baseline["bare_same_recovered_point_gradient_replay_max"] == 0
    assert baseline["bare_same_recovered_point_preconditioned_gradient_replay_max"] == 0
    contract = load(run / "contract.json")
    for relative, expected in contract["inputs_sha256"].items():
        assert binding(ROOT / relative)["sha256"] == expected, relative
    pointer = load(run / "latest_checkpoint.json")
    checkpoint_path = run / pointer["path"]
    checkpoint = load(checkpoint_path)
    assert binding(checkpoint_path)["sha256"] == pointer["sha256"]
    assert endpoint["accepted_checkpoint_sha256"] == pointer["sha256"]
    assert checkpoint["accepted_complete_point"]
    result = load(HERE / "results_stage003.json")
    for value in result["inputs"].values():
        assert binding(ROOT / value["path"])["sha256"] == value["sha256"]
    for name in ("handoff.json", "contract.json", "source_identity.json", "execution_contract.json",
                 "new_source_baseline_controls.json", "root_execution.json", "root_endpoint_execution.json",
                 "independent_endpoint_verification.json"):
        result["inputs"]["public531/" + name] = binding(run / name)
    result["inputs"].update(
        previous_stage003=binding(HERE / "results_stage003.json"),
        actual_public531_checkpoint=binding(checkpoint_path),
        actual_public531_arrays=binding(checkpoint_path.with_suffix(".npz")),
        matched_point_reference=binding(HERE / "matched_recovered_point_reference/result.json"),
        matched_point_verification=binding(HERE / "matched_recovered_point_reference/verification.json"),
        physical_frontier004=binding(HERE / "physical_frontier004/decision_and_prospective_experiment.json"),
        physical_frontier004_verification=binding(HERE / "physical_frontier004/verification.json"),
    )
    residual = endpoint["actual_original_metric_orbital_residual"]
    indicator = endpoint["actual_original_metric_energy_indicator_Ha"]
    passed = endpoint["original_stationarity_gates_passed"]
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["public_mixed531_optimization"] = {
        "actual_complete_public_optimizer_call": True,
        "accepted_iteration": checkpoint["iteration"],
        "original_metric_sha256": checkpoint["original_metric_sha256"],
        "whole_same_point_bare_g_and_pg_max_error": 0,
        "original_residual": residual,
        "original_energy_indicator_Ha": indicator,
        "original_stationarity_gates_passed": passed,
        "independent_identity_checks": endpoint["checks_count"],
        "DC_lower_upper_rows": endpoint["DC_lower_upper_rows"],
        "operator": "Dirac-Coulomb equal-five-state EOL only",
        "static_Breit_spectrum_changed": False,
        "full_core_valence_or_radial_convergence_certified": False,
    }
    states = "通过" if passed else "未通过"
    lines = [r for r in endpoint["DC_lower_upper_rows"] if r["label"] != "6s_1/2"]
    gaps = "/".join(f"{r['nominal_gap_cm1']:+.1f}" for r in lines)
    result["card_summary"] += (
        f"\n实际公共531-CSF MCDHF完成{checkpoint['iteration']}个接受步；"
        "同点旧/新源码完整g/pg逐元素差0，原metric保持。"
        f"独立终点重建{endpoint['checks_count']}项身份/CI/梯度检查通过，"
        f"残差{residual:.6e}、指标{indicator:.6e} Ha，原驻值门{states}。"
        f"同态DC四线NIST gap {gaps} cm⁻¹；不能加到静态531或003修正账。"
        "物理下一步区分完整5p双核空穴与平衡虚轨道层；"
        "五个完整目标对称块先验等价性与端到端性能分别核验。"
    )
    result["selection_reason"] = (
        "Use the independently verified actual public531 point and unchanged original stationarity gates. "
        "Resolve incomplete numerical stationarity separately from same-point missing-rank and virtual-space "
        "differential tests; exact complete target-sector equivalence precedes performance use."
    )
    result["next_execution_status"] = (
        "真实公共531求解及独立终点重放已完成；后继物理与性能判别依原门和完整模型推进。"
    )
    result.update(production_certified=False, experimental_gate_passed=False, highest_level_reached=False)
    path = HERE / "results_stage004.json"
    assert not path.exists(), "Never overwrite immutable stage evidence"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(path), "original_stationarity_passed": passed,
                      "accepted_steps": checkpoint["iteration"], "DC_gaps_cm1": gaps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
