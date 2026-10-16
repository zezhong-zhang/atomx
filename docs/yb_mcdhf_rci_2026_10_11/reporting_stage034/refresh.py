"""Integrate the closed native31 trajectory using saved JSON only."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
HOURLY = REPO / "docs/atomx_hourly_progress_2026_10_10"


def sha(path):
    if Path(path).suffix == ".npz":
        raise ValueError("No numerical arrays in reporting integration")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    return dict(path=str(path.relative_to(REPO)), sha256=sha(path))


def once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-binding", type=Path, required=True)
    args = parser.parse_args()
    contract, seal = read(HERE / "contract.json"), read(args.root_binding)
    assert seal["authorized"] is True
    assert seal["driver_sha256"] == contract["driver_sha256"] == sha(__file__)
    assert seal["contract_sha256"] == sha(HERE / "contract.json")
    registry = HOURLY / "progress.json"
    assert sha(registry) == seal["registry_before_sha256"] == contract["registry_before_sha256"]
    assert all(sha(REPO / name) == value for name, value in contract["inputs"].items())
    assert not any((REPO / name).exists() for name in contract["outputs"])
    roles = {name: REPO / path for name, path in contract["roles"].items()}
    receipt = read(roles["native_receipt"])
    assert receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0
    assert not receipt["timed_out"] and all(receipt[k] for k in
        ("complete_shared_source_unchanged", "frozen_source_unchanged", "input_unchanged"))
    actual, closure = read(roles["native_result"]), read(roles["native_closure"])
    assert actual["status"] == "bounded_f043_public_optimizer_returned" and actual["error"] is None
    assert actual["all_completed_controls_passed"] and len(actual["checks"]) == 1582
    assert all(row["passed"] for row in actual["checks"])
    assert closure["all_scalar_completion_controls_passed"] and len(closure["checks"]) == 77
    assert all(row["passed"] for row in closure["checks"])
    assert closure["scientific_calls"] == closure["NPZ_reads_or_rehashes"] == 0
    assert closure["actual_new_qualified_steps"] == closure["finite_stored_H_acceptance_count"] == 16
    assert actual["last_qualified_checkpoint"] == str(roles["native_terminal"].relative_to(REPO))
    terminal = read(roles["native_terminal"])
    assert terminal["iteration"] == 16 and terminal["qualified"] is True
    assert actual["trajectory"][-1]["manifest_sha256"] == sha(roles["native_terminal"])
    assert not actual["original_stationarity_passed"] and not terminal["original_stationarity_passed"]
    assert closure["source_map_sha256"] == actual["source_map_sha256"]
    assert not closure["current_source_adopted"] and not closure["experiment_or_physical_completeness_certified"]
    old_panel, old_result, record = read(roles["base_panel"]), read(roles["base_result"]), read(registry)
    panel, result = copy.deepcopy(old_panel), copy.deepcopy(old_result)
    panel["historical_stage033_panel"] = bound(roles["base_panel"])
    panel["computed_run"] = "31轨道共同MCDHF有限16步末态；未按实验gap选点"
    panel["CSFs"], panel["radial_orbitals"] = 1595, 31
    panel["stationarity"] = copy.deepcopy(actual["optimizer_diagnostics"])
    panel["stationarity"]["scope"] = "Frozen f043 source, original3689 metric, original five .2 physical UIDs; nonstationary terminal16."
    panel["rows"] = []
    old_rows = {row["label"]: row for row in old_panel["rows"]}
    for row in terminal["rows"]:
        uid = row["UID"]
        original = old_rows[row["label"]]
        assert (original["twoJ"], original["parity"]) == tuple(uid[:2])
        assert original["experimental_cm1"] == row["NIST_nominal_cm1"]
        relative = row["nominal_gap_cm1"] / row["NIST_nominal_cm1"] if row["NIST_nominal_cm1"] else None
        panel["rows"].append(dict(label=row["label"], twoJ=uid[0], parity=uid[1],
            energy_Ha=row["energy_Ha"], computed_cm1=row["excitation_cm1"],
            experimental_cm1=row["NIST_nominal_cm1"], error_cm1=row["nominal_gap_cm1"],
            relative_error=relative, gate_pass=relative is None or abs(relative) <= .01,
            uncertainty_cm1=original["uncertainty_cm1"]))
    assert not all(row["gate_pass"] for row in panel["rows"])
    note = ("31轨道原生弛豫16步、17点、1582实时控制及77保存标量闭合通过；"
        "四线RMS1915.98→1283.77cm⁻¹，原驻值及实验1%门仍失败。12→13步切向梯度与残差突增，"
        "先核验局部导数/态跟踪，再决定续算；g·pg/2仅是指标，不能作为谱线误差界。"
        "共享横向owner同源基准完整H/g逐位相等，历史混合CI42通过/2失败仍保留；"
        "同点Breit量级和公平CPU/GPU效率仍待实际闭合。网页v9仍为23模型92行，尚未包含本16步轨迹。"
        "04HKT自动服务完成四项既有探针，Yb MCDHF自动作业为0；本31弛豫由root独立合同执行。")
    panel["card_summary"] = result["card_summary"] = note
    panel["native31_relaxation"] = dict(result=bound(roles["native_result"]),
        receipt=bound(roles["native_receipt"]), closure=bound(roles["native_closure"]),
        terminal=bound(roles["native_terminal"]), no_gap_selection=True,
        original_stationarity_passed=False, current_source_adopted=False)
    for role in ("native_result", "native_receipt", "native_closure", "native_terminal", "matched_owner", "matched_owner_receipt"):
        panel["inputs"][str(roles[role].relative_to(REPO))] = bound(roles[role])
    preserved = set(old_panel) - {"computed_run", "CSFs", "radial_orbitals", "stationarity", "rows", "card_summary", "inputs"}
    assert all(panel[key] == old_panel[key] for key in preserved)
    result["historical_stage033_result"] = bound(roles["base_result"])
    result["native31_relaxation"] = copy.deepcopy(panel["native31_relaxation"])
    result["selection_reason"] = "先判别原生31有限弛豫12→13局部响应、同点算符贡献及受控效率；不依据gap选择中间末态或增加截断。"
    result["next_execution_status"] = "SOURCE-bound fixed-point discrimination pending; original stationarity and experiment FAIL; no automatic optimizer continuation."
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    once(HERE / "registry_before.json", record)
    once(HERE / "current_card_evidence.json", panel)
    result["current_native_card_evidence"] = bound(HERE / "current_card_evidence.json")
    result["inputs"]["current_mcdhf"] = result["current_native_card_evidence"]
    once(TOPIC / "results_stage034.json", result)
    record.update(physics_budget_result=str((TOPIC / "results_stage034.json").relative_to(REPO)),
        yb_mcdhf_result=str((HERE / "current_card_evidence.json").relative_to(REPO)), updated_at=result["created_at"])
    record["tasks"][0].update(status="原生31 MCDHF有限16步已闭合；驻值/实验仍FAIL", detail="完整1582门通过；末态RMS1283.77cm⁻¹，先判别12→13局部响应，不自动追加步数。")
    record["spectra"][0].update(comparison="原生31/1595 MCDHF有限16步：21464.30/22852.53/25725.28/29779.42cm⁻¹；NIST22960.80/24332.69/27061.82/30392.23。RMS1283.77cm⁻¹。", status="未拟合、未驻值；完整轨迹保留，原实验1%门失败；旧29与固定31为历史独立模型。")
    sys.path.insert(0, str(HOURLY))
    import sync_card
    card, citations = sync_card.render(record, datetime.now(timezone.utc))
    content = "\n".join(element.get("content", "") for element in card["body"]["elements"])
    assert card["schema"] == "2.0" and all(x in content for x in ("21464.30", "29779.42", "1283.77", "12→13", "实验1%门仍失败"))
    once(HERE / "preview_card.json", card)
    once(HERE / "preview_citations.json", citations)
    assert sha(registry) == contract["registry_before_sha256"]
    temp = HOURLY / "progress.stage034.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temp.replace(registry)
    spec = importlib.util.spec_from_file_location("stage034_existing_planner", HOURLY / "parallel_planner.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    decision = dict(status="implementation_frontier", jobs=[], frontier=[], evidence=[])
    planner.annotate_physics_priority(REPO, HOURLY, decision)
    assert decision["direction"] == note
    assert not any(n == "numpy" or n.startswith("numpy.") or n == "atomx" or n.startswith("atomx.") for n in sys.modules)
    once(HERE / "integration_proof.json", dict(all_passed=True, result=bound(TOPIC / "results_stage034.json"),
        panel=bound(HERE / "current_card_evidence.json"), registry=bound(registry),
        previous_stage033_preserved=True, literal_terminal_not_gap_selected=True,
        elements=len(card["body"]["elements"]), citations=len(citations), new_science_NPZ_API_card_calls=0, hourly_timer_unchanged=True))
    print(json.dumps(dict(status="metadata_stage034_integrated", elements=len(card["body"]["elements"]), citations=len(citations))))


if __name__ == "__main__":
    main()
