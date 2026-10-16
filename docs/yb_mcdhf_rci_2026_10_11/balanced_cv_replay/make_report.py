"""Source-bound presentation from the completed scientific arrays and result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / "result.json").read_text())
    verification = json.loads((HERE / "verification.json").read_text())
    if not verification["all_passed"] or sha(HERE / "result.json") != verification["artifacts_sha256"]["result.json"]:
        raise ValueError("scientific replay must pass for these exact inputs")
    levels, increments = [], []
    for stage in result["stages"]:
        for model, rows in stage["tracked_levels"].items():
            for row in rows:
                levels.append({"stage": stage["name"], "CSFs": stage["CSFs"], "operator": model,
                    "state": row["label"], "column": row["column"], "local_root": row["local_root"],
                    "excitation_cm1": row["excitation_from_tracked_6s_cm1"],
                    "NIST_nominal_cm1": row["NIST_nominal_cm1"], "gap_cm1": row["gap_cm1"],
                    "relative_error": row["relative_error"], "bare_character": row["bare_character_weight"],
                    "state_binding_unambiguous": row["binding_unambiguous"]})
                increments.append({"stage": stage["name"], "operator": model, "state": row["label"],
                    "absolute_delta_from_previous_Ha": row["energy_delta_from_previous_Ha"],
                    "absolute_delta_from_bare_Ha": row["energy_delta_from_bare_Ha"],
                    "transition_delta_from_previous_cm1": row["transition_delta_from_previous_cm1"],
                    "transition_delta_from_bare_cm1": row["transition_delta_from_bare_cm1"]})
    for name, records in (("spectral_summary.csv", levels), ("energy_increments.csv", increments)):
        with (HERE / name).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    final = result["stages"][-1]
    rows = final["tracked_levels"]["static_Breit"][1:]
    summary = (
        "固定005已驻值MCDHF的同24径向轨道，原生RCI完成4f→4f+5p→4f+5p+5s完整允许pair族（294/468/531 CSF），"
        f"全部DC/static Breit根与有符号态绑定、{verification['checks_count']}项独立重放均通过。"
        "真实基态仍6s；完整4f使ground发生815 cm⁻¹量级降能，揭示旧68族缺少ground通道。"
        "5p对跃迁差能改变304–808 cm⁻¹，5s为15.8–34.2 cm⁻¹。"
        "最终D3/D5/P1/P3实验gap依次"
        + "/".join(f"{row['gap_cm1']:+.1f}" for row in rows)
        + " cm⁻¹，仍未过1%。这是固定轨道的balanced-family RCI判别，尚未做扩展空间MCDHF驻值，不能称完整CV或生产谱。"
    )
    presentation = {"scientific_result_sha256": sha(HERE / "result.json"),
                    "verification_sha256": sha(HERE / "verification.json"),
                    "script_sha256": sha(HERE / "make_report.py"), "card_summary_text": summary,
                    "final_static_rows": rows,
                    "scope": result["scope"],
                    "artifacts_sha256": {name: sha(HERE / name) for name in ("spectral_summary.csv", "energy_increments.csv")}}
    (HERE / "presentation.json").write_text(json.dumps(presentation, indent=2, ensure_ascii=False) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
