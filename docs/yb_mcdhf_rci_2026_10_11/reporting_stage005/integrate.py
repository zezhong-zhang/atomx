"""Present the verified complete531 same-point RCI without changing old evidence."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
ROOT = TOPIC.parents[1]


def load(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    directory = TOPIC / "public531_endpoint_static_rci"
    static = load(directory / "result.json")
    proof = load(directory / "saved_array_verification.json")
    launch = load(directory / "root_execution.json")
    assert proof["all_passed"] and all(c["passed"] for c in proof["checks"])
    assert proof["checks_count"] == len(proof["checks"])
    assert proof["result_sha256"] == binding(directory / "result.json")["sha256"]
    assert proof["arrays_sha256"] == binding(directory / "arrays.npz")["sha256"]
    assert launch["exit_code"] == 0 and launch["driver_unchanged"]
    assert launch["execution_contract_unchanged"]
    assert static["static_RCI_numeric_integrity_passed"]
    assert static["DC_original_stationarity_passed"] is False
    assert not any(static[k] for k in ("stationarity_certified", "production_certified",
                                      "experimental_gate_passed", "highest_level_reached"))
    result = load(TOPIC / "results_stage004.json")
    for old in result["inputs"].values():
        assert binding(ROOT / old["path"])["sha256"] == old["sha256"]
    result["inputs"]["previous_stage004"] = binding(TOPIC / "results_stage004.json")
    names = ("contract.json", "execution_contract.json", "run.py", "verify_arrays.py",
             "result.json", "saved_array_verification.json", "arrays.npz", "root_execution.json")
    inputs = {name: binding(directory / name) for name in names}
    result["inputs"].update({"samepoint_static/" + k: v for k, v in inputs.items()})
    labels = ("6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    by_label = {r["label"]: r for r in static["static_rows"]}
    assert set(by_label) == set(labels)
    rows = [by_label[label] for label in labels]
    stationarity = static["inherited_DC_stationarity_evidence"]
    residual, indicator = stationarity["orbital_residual"], stationarity["estimated_energy_error_Ha"]
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["public531_samepoint_static_RCI"] = {
        "CSFs": 531, "blocks": 16, "orbitals": 24, "accepted_parent_iteration": 160,
        "stationarity_passed": False, "static_rows": rows, "multiplets": static["multiplets"],
        "independent_saved_array_checks": proof["checks_count"],
        "calls": static["calls"], "timings_uncontrolled_s": static["timings_uncontrolled_s"],
        "performance_comparison_valid": False,
        "operator": "Same-point full static Breit Gaunt and zero-frequency retardation; no QED/recoil",
    }
    shifts = "；".join(f"{r['label']} {r['static_minus_DC_transition_cm1']:+.2f}" for r in rows[1:])
    gaps = "；".join(f"{r['label']} {r['nominal_gap_cm1']:+.2f}" for r in rows[1:])
    result["card_summary"] = (
        "主线为 Atomx 原生 MCDHF→RCI。五参考005原驻值基线保留；"
        "完整531-CSF公共优化已接受160步，同点旧/新完整g/pg差0，原P/逆度量不变。"
        f"独立终点988项通过，原残差 {residual:.6e} / 门1e−7仍未过，"
        f"指标 {indicator:.6e} Ha / 门1e−8通过，不能认证驻值。\n"
        f"同一accepted160点的完整531静态Breit RCI实算并独立核验{proof['checks_count']}项；"
        f"四线静态−DC位移(cm⁻¹)：{shifts}。NIST gap：{gaps}，均未过1%。\n"
        "这些差能不能闭合约1700–2300 cm⁻¹ gap。优先判别完整5p双核空穴与平衡虚轨道层，"
        "并继续原度量驻值；未将其它点的动态Breit/QED账拼入。"
        f"本次非受控资源观察{static['timings_uncontrolled_s']['total']:.2f}s，"
        f"其中Hamiltonian构造{static['timings_uncontrolled_s']['static_H']:.2f}s、"
        f"完整分块CI {static['timings_uncontrolled_s']['complete_blocked_CI']:.4f}s；"
        "以Hamiltonian收缩为性能重点，完整目标块等价性和并行效率另验。"
    )
    result.update(production_certified=False, experimental_gate_passed=False, highest_level_reached=False)
    output = TOPIC / "results_stage005.json"
    card = {
        "method": "MCDHF -> RCI", "computed_run": "public531 accepted160 → same-point static Breit RCI",
        "producer": "Atomx native public orbital optimizer and complete native static CSF Hamiltonian",
        "CSFs": 531, "radial_orbitals": 24, "electrons": 69,
        "stationarity": {"converged": False, "orbital_residual": residual, "orbital_tolerance": 1e-7,
                         "estimated_energy_error": indicator, "energy_tolerance": 1e-8,
                         "residual_metric_sha256": result["public_mixed531_optimization"]["original_metric_sha256"],
                         "termination_reason": "parent maximum_iterations", "iterations": 160},
        "rows": [{"label": r["label"], "twoJ": r["twoJ"], "parity": r["parity"],
                  "energy_Ha": r["energy_Ha"], "computed_cm1": r["excitation_cm1"],
                  "experimental_cm1": r["NIST_nominal_cm1"], "error_cm1": r["nominal_gap_cm1"],
                  "relative_error": r["nominal_relative_gap"], "gate_pass": r["nominal_1percent_pass"],
                  "uncertainty_cm1": r["NIST_uncertainty_cm1"]} for r in rows],
        "production_certified": False, "experimental_gate_passed": False,
        "highest_level_reached": False, "fitted_corrections": False, "inputs": inputs,
        "card_summary": "本表为完整531同点静态RCI诊断。原DC驻值失败保留，不能认证完整核价/虚空间收敛。",
    }
    assert not output.exists() and not (HERE / "current_card_evidence.json").exists()
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "current_card_evidence.json").open("x") as stream:
        stream.write(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "card": binding(HERE / "current_card_evidence.json")}))


if __name__ == "__main__":
    main()
