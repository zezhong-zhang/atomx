"""Report the actual same-objective320 endpoint and separately point160 static RCI."""

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
    previous = TOPIC / "results_stage005b.json"
    result = load(previous)
    for old in result["inputs"].values():
        assert binding(ROOT / old["path"])["sha256"] == old["sha256"]
    directory = TOPIC / "public_mixed_resume_run003"
    proof = load(directory / "independent_endpoint_verification.json")
    assert binding(directory / "independent_endpoint_verification.json")["sha256"] == "082d24ab57cada5c83101e7ac2a6930732838b1617724530d7ce5611303d65b4"
    assert proof["all_identity_replay_checks_passed"] and all(c["passed"] for c in proof["checks"])
    assert proof["checks_count"] == len(proof["checks"]) == 1011
    assert proof["original_stationarity_gates_passed"] is False
    for name in ("root_execution.json", "root_endpoint_execution.json"):
        execution = load(directory / name)
        assert execution["exit_code"] == 0 and execution["driver_unchanged"]
        assert execution["execution_contract_unchanged"]
    checkpoint = directory / "checkpoints/accepted_0160.json"
    arrays = checkpoint.with_suffix(".npz")
    assert binding(checkpoint)["sha256"] == proof["accepted_checkpoint_sha256"]
    assert binding(arrays)["sha256"] == proof["accepted_arrays_sha256"]
    run = load(directory / "result_unverified.json")
    assert run["actual_full_public_optimizer_call"] and run["same_objective_as_parent"]
    assert run["parent_accepted_iteration"] == 160
    diagnostic = run["diagnostics"]
    assert diagnostic["iterations"] == 160 and diagnostic["converged"] is False
    residual = proof["actual_original_metric_orbital_residual"]
    indicator = proof["actual_original_metric_energy_indicator_Ha"]
    assert residual == diagnostic["orbital_residual"] > 1e-7
    assert indicator == diagnostic["estimated_energy_error"] < 1e-8
    labels = ("6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    by_label = {r["label"]: r for r in proof["DC_lower_upper_rows"]}
    assert set(by_label) == set(labels)
    rows = [by_label[label] for label in labels]
    names = ("contract.json", "execution_contract.json", "run.py", "verify_endpoint.py",
             "result_unverified.json", "independent_endpoint_verification.json", "source_identity.json",
             "root_execution.json", "root_endpoint_execution.json")
    inputs = {name: binding(directory / name) for name in names}
    inputs.update(accepted_checkpoint=binding(checkpoint), accepted_arrays=binding(arrays))
    result["inputs"].update({"public320/" + k: v for k, v in inputs.items()})
    result["inputs"]["previous_stage005b"] = binding(previous)
    result["inputs"]["historical_point160_current_mcdhf"] = result["inputs"]["current_mcdhf"]
    stationarity = {"converged": False, "orbital_residual": residual, "orbital_tolerance": 1e-7,
                    "estimated_energy_error": indicator, "energy_tolerance": 1e-8,
                    "residual_metric_sha256": diagnostic["residual_metric_sha256"],
                    "termination_reason": diagnostic["termination_reason"], "iterations": 320}
    card = {"method": "MCDHF -> RCI", "computed_run": "public531 DC total320 accepted steps",
            "producer": "Atomx native public orbital optimizer; RCI at point160 reported separately",
            "CSFs": 531, "radial_orbitals": 24, "electrons": 69, "stationarity": stationarity,
            "rows": [{"label": r["label"], "twoJ": int(r["label"].split("_")[1].split("/")[0]),
                      "parity": -1 if r["label"].startswith("6p") else 1,
                      "energy_Ha": r["energy_Ha"], "computed_cm1": r["excitation_cm1"],
                      "experimental_cm1": r["NIST_nominal_cm1"], "error_cm1": r["nominal_gap_cm1"],
                      "relative_error": r["nominal_gap_cm1"] / r["NIST_nominal_cm1"] if r["NIST_nominal_cm1"] else None,
                      "gate_pass": r["nominal_1percent_pass"], "uncertainty_cm1": None} for r in rows],
            "production_certified": False, "experimental_gate_passed": False,
            "highest_level_reached": False, "fitted_corrections": False, "inputs": inputs,
            "card_summary": "本表为最新完整531 DC MCDHF累计320步诊断，原驻值仍失败。静态Breit的已核验表属于另一个accepted160点，不能拼接为当前320点RCI。"}
    card_path = HERE / "current_card_evidence.json"
    with card_path.open("x") as stream:
        stream.write(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    result["inputs"]["current_mcdhf"] = binding(card_path)
    result["historical_stationarity_point160"] = result["stationarity"]
    result["stationarity"] = stationarity
    result["computed_run"] = card["computed_run"]
    result["public_mixed531_continuation"] = {"total_accepted_steps": 320,
        "additional_steps": 160, "independent_checks": 1011, "stationarity": stationarity,
        "rows": rows, "same_objective_as_parent": True, "fresh_secant_history_declared": True,
        "full_gradient_calls": run["total_run_full_gradient_calls"],
        "energy_trial_calls": run["energy_trial_calls"], "line_search_fallbacks": diagnostic["line_search_fallbacks"],
        "current_static_Breit_computed": False, "performance_comparison_valid": False}
    delta = "；".join(f"{r['label']} {r['same_lower_upper_transition_delta_from_parent160_cm1']:+.3f}" for r in rows[1:])
    result["card_summary"] = (
        f"原生完整531 MCDHF同目标累计320接受步；第二轮160步独立同点重验1011项通过。"
        f"原P/逆度量残差4.71705e−5→{residual:.6e}，门1e−7仍失败；指标{indicator:.6e}Ha通过1e−8。"
        f"四线第二轮差(cm⁻¹)：{delta}，不能闭合千cm⁻¹实验gap。\n"
        "另一个accepted160点的完整531静态Breit RCI已独立3710项核验，四线static−DC仅+2.384/−28.751/−20.697/−44.262 cm⁻¹(D3/D5/P1/P3)；不能拼入320点。\n"
        "优先在已选accepted160点判别完整5p双核空穴和共同workspace的平衡虚轨道层。"
        "完整531与261目标块谱/EOL/24轨道梯度/RDM/原metric差均0，独立1795项通过，仅限五目标EOL。"
        "性能先复用带物理身份的完整H，随后测同模型端到端和CPU/GPU并行效率。"
    )
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result.update(production_certified=False, experimental_gate_passed=False, highest_level_reached=False)
    output = TOPIC / "results_stage006.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": binding(output), "card": binding(card_path)}))


if __name__ == "__main__":
    main()
