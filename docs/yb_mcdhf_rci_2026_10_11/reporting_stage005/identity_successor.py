"""Bind current reporting identity without rewriting delivered stage005 evidence."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
ROOT = TOPIC.parents[1]


def binding(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    old_path = TOPIC / "results_stage005.json"
    card_path = HERE / "current_card_evidence.json"
    assert binding(old_path)["sha256"] == "12eca73e6c89d31296603769ea0cd6f0c8fe118b2c9670a06fbd3a4358d62a71"
    assert binding(card_path)["sha256"] == "bb7247af6b5ca5038f0c518b67eeb8afd576529e7123f58ec4133760e9e79033"
    result = json.loads(old_path.read_text())
    card = json.loads(card_path.read_text())
    for item in result["inputs"].values():
        assert binding(ROOT / item["path"])["sha256"] == item["sha256"]
    result["inputs"]["historical_current_mcdhf_stage005"] = result["inputs"]["current_mcdhf"]
    result["inputs"]["current_mcdhf"] = binding(card_path)
    result["inputs"]["previous_delivered_stage005"] = binding(old_path)
    result["historical_five_reference_stationarity"] = result["stationarity"]
    result["stationarity"] = card["stationarity"]
    result["computed_run"] = card["computed_run"]
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["card_summary"] = result["card_summary"].replace(
        "这些差能不能闭合", "这些已计算的差能不足以闭合")
    result["reporting_identity_successor"] = {
        "cause": "Stage005 card used current accepted160/static evidence, but planner current_mcdhf still named the historical five-reference005 card.",
        "delivered_stage005_preserved": True,
        "scientific_results_changed": False,
        "science_calls": 0,
        "fitted_corrections": False,
    }
    output = TOPIC / "results_stage005b.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(binding(output)))


if __name__ == "__main__":
    main()
