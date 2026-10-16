"""Verify already archived card/readback and explicitly supplied closed root receipt; zero API/science."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = HERE.parent
MESSAGE = "om_x100b638eadb5b068c1205d85d5d5cb0"
KEY = "392ee019-c98b-53ac-82dd-3016109af6f3"
CHAT = "oc_fb2fc627b3e98dc9555de31580878758"
BOT = "ou_c697e48a528ad9aa6157f62531561b59"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def visible(value):
    return re.sub(r"\s+", "", re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-closed-receipt", help="Explicit root-approved closed run005 receipt; omit while pending")
    args = parser.parse_args()
    card, receipt, status, back = [read(HERE / p) for p in ("card.json", "receipt.json", "send_status.json", "readback.json")]
    readback_execution = read(HERE / "readback_execution.json")
    request, service = read(HERE / "iteration_request.json"), read(HERE / "service_iteration_status.json")
    message = back["data"]["messages"][0]
    assert readback_execution["exit_code"] == 0 and readback_execution["actual_readback_calls"] == 1
    assert back["ok"] and back["identity"] == "bot" and back["data"]["total"] == 1
    assert message["message_id"] == receipt["data"]["message_id"] == status["message_id"] == MESSAGE
    assert message["chat_id"] == receipt["data"]["chat_id"] == CHAT
    assert message["sender"]["open_bot_id"] == BOT and message["sender"]["sender_type"] == "app"
    assert message["sender"]["name"] == "aisi codex" and message["msg_type"] == "interactive"
    assert not message["deleted"] and not message["updated"]
    assert status["idempotency_key"] == KEY and status["passed"] and status["iteration_dispatch_ok"]
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert len(elements) == 15 and all(e["tag"] == "markdown" for e in elements)
    expected = '<card title="' + card["header"]["title"]["content"] + '">\n' + "\n".join(e["content"] for e in elements) + "\n</card>"
    actual = message["content"]
    assert isinstance(actual, str) and visible(expected) == visible(actual)
    differences = [{"operation": op, "payload": expected[a:b], "readback": actual[c:d]}
                   for op, a, b, c, d in difflib.SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes() if op != "equal"]
    assert differences == [*({"operation": "delete", "payload": "\n", "readback": ""} for _ in range(6)),
                           {"operation": "insert", "payload": "", "readback": "["},
                           {"operation": "insert", "payload": "", "readback": "](https://doi.org/10.1103/PhysRevA.110.L030802)"}]
    assert sha(HERE / "registry_stage021.json") == status["registry_sha256"]
    stage = CAMPAIGN / "results_stage021.json"
    assert any(e["path"] == str(stage.relative_to(REPO)) and e["sha256"] == sha(stage) for e in status["evidence"])
    integration = read(CAMPAIGN / "reporting_stage021/integration_proof.json")
    assert integration["all_passed"] and integration["checks_count"] == 158 and integration["new_science_calls"] == 0
    assert request["status"] == service["status"] == "implementation_frontier"
    assert request["key"] == service["key"] == KEY and request["message_id"] == service["message_id"] == MESSAGE
    assert request["decision"]["job"] is None and service["decision"]["job"] is None
    assert not request["decision"]["highest_level_reached"]

    run = CAMPAIGN / "correlated29_optimization_run005_preparation"
    execution, once = read(run / "execution_contract.json"), read(run / "once_marker.json")
    assert once["exactly_one_launch"] and once["execution_contract_sha256"] == sha(run / "execution_contract.json")
    assert execution["execution_authorized_by_root"] and execution["source_map_sha256"] == "f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe"
    assert execution["max_new_accepted_steps"] == 128 and execution["max_optimizer_calls"] == 1
    assert datetime.fromisoformat(once["created_utc"]) > datetime.fromisoformat(status["finished_at"])
    proof = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_readback_verified": True, "final_root_iteration_closure_bound": False,
        "status": "delivery_verified_root_run005_closure_pending", "cycle": status["cycle"],
        "message_id": MESSAGE, "idempotency_key": KEY, "chat_id": CHAT, "bot_open_id": BOT,
        "message_position": message["message_position"], "actual_message_link": message["message_app_link"],
        "source_stage021": {"path": str(stage.relative_to(REPO)), "sha256": sha(stage), "metadata_checks": 158, "new_science_calls": 0},
        "registry_sha256": status["registry_sha256"],
        "actual_card": {"payload_schema": "2.0", "elements": 15, "title": card["header"]["title"]["content"],
                        "receipt_create_time_Asia_Hong_Kong": receipt["data"]["create_time"],
                        "readback_create_time_Asia_Hong_Kong": message["create_time"],
                        "all_visible_content_matches": True, "readback_content_characters": len(actual),
                        "raw_readback_schema_available": False, "readback_representation": "CLI-rendered full card text; sent schema2 DSL archived separately",
                        "rendering_differences": differences},
        "service_send": {"started_at": status["started_at"], "finished_at": status["finished_at"], "passed": True, "iteration_dispatch_ok": True},
        "service_iteration": {"status": "implementation_frontier", "job": None, "automatic_physics_job_started": False,
                              "finished_at": request["finished_at"], "highest_level_reached": False},
        "root_bounded_iteration": {"kind": "separate native same29/1046/original3451metric continuation from closedglobal80", "started_after_successful_card": True,
                                   "immutable_start_marker": {"path": str((run / "once_marker.json").relative_to(REPO)), "sha256": sha(run / "once_marker.json"), "created_utc": once["created_utc"]},
                                   "execution_contract": {"path": str((run / "execution_contract.json").relative_to(REPO)), "sha256": sha(run / "execution_contract.json")},
                                   "max_new_steps": 128, "actual_result_bound": False},
        "actual_readback_calls_by_root": 1, "new_API_calls_by_this_audit": 0,
        "new_scientific_calls_by_this_audit": 0, "cards_sent_by_this_audit": 0,
        "configurations_or_secrets_read": False, "registry_or_stage021_changed": False,
        "files_sha256": {p: sha(HERE / p) for p in ("card.json", "receipt.json", "send_status.json", "service_iteration_status.json", "iteration_request.json",
                                                      "registry_stage021.json", "readback.json", "readback_execution.json", "root_send_stdout.json")},
        "verifier_sha256": sha(__file__),
        "scope": "Actual stage021 send/readback is separate from service jobnull frontier and separate root-run005 continuation. Historical physical/stationarity/experiment FAIL not promoted."
    }
    if args.root_closed_receipt:
        closed_path = (REPO / args.root_closed_receipt).resolve()
        assert closed_path == (run / "root_execution.json").resolve()
        closed = read(closed_path)
        assert closed["status"] == "foreground_closed" and "exit_code" in closed and "finished_utc" in closed
        assert closed["execution_contract_sha256"] == sha(run / "execution_contract.json")
        assert closed["driver_sha256"] == execution["driver_sha256"]
        assert datetime.fromisoformat(closed["started_utc"]) >= datetime.fromisoformat(status["finished_at"])
        proof.update(status="actual_delivery_and_separate_root_iteration_closed", final_root_iteration_closure_bound=True)
        proof["root_bounded_iteration"].update(closed=True, root_closed_receipt={"path": str(closed_path.relative_to(REPO)), "sha256": sha(closed_path)},
                                              exit_code=closed["exit_code"], started_utc=closed["started_utc"], finished_utc=closed["finished_utc"],
                                              elapsed_uncontrolled_seconds=closed["elapsed_uncontrolled_seconds"],
                                              scope="Receipt closure only; no independent H/CI/g or physical-success claim by delivery audit")
        proof["passed"] = True
        write(HERE / "delivery_proof.json", proof)
        output = HERE / "delivery_proof.json"
    else:
        write(HERE / "delivery_prepared.json", proof)
        output = HERE / "delivery_prepared.json"
    print(json.dumps({"passed": True, "output": output.name, "sha256": sha(output), "position": message["message_position"],
                      "root_iteration_closed": proof["final_root_iteration_closure_bound"], "API_calls": 0, "science_calls": 0}))


if __name__ == "__main__":
    main()
