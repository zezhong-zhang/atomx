"""Refresh current reporting labels without altering scientific evidence."""
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
T = B.parent
R = T.parents[1]
H = R / "docs/atomx_hourly_progress_2026_10_10"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bound(p):
    return {"path": str(p.relative_to(R)), "sha256": sha(p)}


def once(p, value):
    with p.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    contract = read(B / "contract.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        assert passed, name

    check("driver", sha(Path(__file__)) == contract["driver_sha256"])
    registry = H / "progress.json"
    check("registry_before", sha(registry) == contract["registry_before_sha256"])
    for p, digest in contract["inputs"].items():
        check("input/" + p, sha(R / p) == digest)
    original = read(T / "results_stage029.json")
    result = copy.deepcopy(original)
    original_panel = read(T / "reporting_stage029/current_card_evidence.json")
    panel = copy.deepcopy(original_panel)
    panel["card_summary"] = "原能量指标是 g·pg/2 的诊断量，不能作为能级误差界；新增31轨道未优化，驻值量仍属于原29轨道208点。\n" + panel["card_summary"]
    check("all_scientific_fields_preserved", all(panel[k] == v for k, v in original_panel.items() if k != "card_summary"))
    once(B / "current_card_evidence.json", panel)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["inputs"]["historical_current_mcdhf_stage029"] = original["inputs"]["current_mcdhf"]
    result["inputs"]["current_mcdhf"] = bound(B / "current_card_evidence.json")
    result["current_native_card_evidence"] = bound(B / "current_card_evidence.json")
    once(T / "results_stage030.json", result)
    record = read(registry)
    once(B / "registry_before.json", record)
    record.update(physics_budget_result=str((T / "results_stage030.json").relative_to(R)), yb_mcdhf_result=str((B / "current_card_evidence.json").relative_to(R)), updated_at=result["created_at"])
    record["tasks"][4].update(status="远程v8已发布21模型84行；v9准备中", detail="保留全部NIST与277轨迹记录。两组固定31 RCI和原208梯度资格将另列；31未优化，原驻值及实验门失败。")
    record["spectra"][0].update(comparison="当前原生29/1046 MCDHF共同轨道到208步，驻值失败，四线RMS2054.32cm⁻¹。屏蔽f固定31/1595 RCI四线25690.48/26569.19/25981.58/29360.74，对NIST22960.80/24332.69/27061.82/30392.23；RMS1915.98cm⁻¹，941门通过，但31轨道尚未优化。", status="原208一切向两尺度导数7053门通过，驻值与实验门仍失败；新增种子响应不是物理完备性证明。531/320步属于历史模型，算符修正不可跨点拼接。")
    sys.path.insert(0, str(H))
    import sync_card
    card, citations = sync_card.render(record, datetime.now(timezone.utc))
    check("card_schema", card["schema"] == "2.0")
    content = "\n".join(v.get("content", "") for v in card["body"]["elements"])
    check("current_metadata_visible", "21模型84行" in content and "RMS1915.98" in content and "不能作为能级误差界" in content and "最新Yb⁺原生完整531" not in content)
    once(B / "preview_card.json", card)
    once(B / "preview_citations.json", citations)
    check("registry_atomic_before", sha(registry) == contract["registry_before_sha256"])
    tmp = registry.with_suffix(".stage030.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(registry)
    spec = importlib.util.spec_from_file_location("stage030_planner", H / "parallel_planner.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    decision = {"status": "implementation_frontier", "jobs": [], "frontier": [], "evidence": []}
    planner.annotate_physics_priority(R, H, decision)
    check("actual_live_planner", decision["direction"] == result["card_summary"])
    once(B / "integration_proof.json", {"all_passed": True, "checks": checks, "check_count": len(checks), "result": bound(T / "results_stage030.json"), "panel": bound(B / "current_card_evidence.json"), "registry": bound(registry), "new_science_arrays_API_cards": 0, "historical_manifest_preserved_without_array_rehash": True})
    print(json.dumps({"PASS": len(checks), "result": bound(T / "results_stage030.json"), "registry": bound(registry), "elements": len(card["body"]["elements"]), "citations": len(citations)}))


if __name__ == "__main__":
    main()
