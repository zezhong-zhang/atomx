"""Two already archived card readbacks, with distinct sources/iteration scopes; zero API/science."""
from __future__ import annotations

import difflib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = HERE.parents[2]
UTILITY = CAMPAIGN / "delivery_stage021/verify_saved_delivery.py"
spec = importlib.util.spec_from_file_location("saved_stage021_delivery_utilities", UTILITY)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
read, sha, write, visible = shared.read, shared.sha, shared.write, shared.visible


def verify(directory, stage_number, expected_id, position):
    status, receipt, card, back, executed, request, service = [read(directory / p) for p in (
        "send_status.json", "receipt.json", "card.json", "readback.json", "readback_execution.json",
        "iteration_request.json", "service_iteration_status.json")]
    message = back["data"]["messages"][0]
    assert back["ok"] and back["identity"] == "bot" and back["data"]["total"] == 1
    assert message["message_id"] == receipt["data"]["message_id"] == status["message_id"] == expected_id
    assert message["message_position"] == str(position) and message["chat_id"] == receipt["data"]["chat_id"] == shared.CHAT
    assert message["sender"]["open_bot_id"] == shared.BOT and message["sender"]["sender_type"] == "app" and message["sender"]["name"] == "aisi codex"
    assert message["msg_type"] == "interactive" and not message["deleted"] and not message["updated"]
    assert executed["actual_readback_calls"] == 1 and executed["exit_code"] == 0 and executed["readback_sha256"] == sha(directory / "readback.json")
    assert status["passed"] and status["iteration_dispatch_ok"] and card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert len(elements) == 15 and all(e["tag"] == "markdown" for e in elements)
    expected = '<card title="' + card["header"]["title"]["content"] + '">\n' + "\n".join(e["content"] for e in elements) + "\n</card>"
    actual = message["content"]
    assert isinstance(actual, str) and visible(expected) == visible(actual)
    differences = [{"operation": op, "payload": expected[a:b], "readback": actual[c:d]} for op, a, b, c, d in
                   difflib.SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes() if op != "equal"]
    assert differences == [*({"operation": "delete", "payload": "\n", "readback": ""} for _ in range(6)),
                           {"operation": "insert", "payload": "", "readback": "["},
                           {"operation": "insert", "payload": "", "readback": "](https://doi.org/10.1103/PhysRevA.110.L030802)"}]
    stage = CAMPAIGN / f"results_stage{stage_number:03d}.json"
    assert any(e["path"] == str(stage.relative_to(REPO)) and e["sha256"] == sha(stage) for e in status["evidence"])
    registry = directory / "registry_stage022.json" if stage_number == 22 else CAMPAIGN / "delivery_stage021/registry_stage021.json"
    assert sha(registry) == status["registry_sha256"]
    assert request["status"] == service["status"] == "implementation_frontier"
    assert request["decision"]["job"] is None and service["decision"]["job"] is None
    assert request["decision"]["jobs"] == service["decision"]["jobs"] == []
    assert request["key"] == service["key"] == status["idempotency_key"] and request["message_id"] == service["message_id"] == expected_id
    proof = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "message_id": expected_id,
        "position": position, "actual_message_link": message["message_app_link"], "chat_id": shared.CHAT, "bot_open_id": shared.BOT,
        "cycle": status["cycle"], "idempotency_key": status["idempotency_key"],
        "source_stage": {"number": stage_number, "path": str(stage.relative_to(REPO)), "sha256": sha(stage), "global_accepted_step": 208 if stage_number == 22 else 80},
        "registry_source": {"path": str(registry.relative_to(REPO)), "sha256": sha(registry)},
        "actual_card": {"schema": "2.0", "elements": 15, "title": card["header"]["title"]["content"],
                        "receipt_create_time_Asia_Hong_Kong": receipt["data"]["create_time"], "readback_create_time_Asia_Hong_Kong": message["create_time"],
                        "full_visible_content_matches": True, "readback_characters": len(actual), "readback_is_raw_DSL": False,
                        "readback_representation": "CLI-rendered full card text; sent schema2 card archived separately", "rendering_differences": differences},
        "actual_send": {"started_at": status["started_at"], "finished_at": status["finished_at"], "passed": True},
        "service_iteration": {"status": "implementation_frontier", "job": None, "jobs": [], "automatic_physics_jobs": 0, "finished_at": service["finished_at"]},
        "actual_readback_calls_by_root": 1, "new_API_calls_by_this_audit": 0, "new_scientific_calls_by_this_audit": 0, "cards_sent_by_this_audit": 0,
        "files_sha256": {p.name: sha(p) for p in sorted(directory.glob("*.json")) if p.name != "delivery_proof.json"},
        "verifier_sha256": sha(__file__), "shared_utilities_sha256": sha(UTILITY),
        "scope": "Saved-text/hash/receipt binding only; no producer replay or physical/stationarity/experiment certification upgrade."
    }
    if stage_number == 22:
        run = CAMPAIGN / "correlated29_radial_reuse_qualification003_preparation"
        closed, once, execution, result = [read(run / p) for p in ("root_execution.json", "once_marker.json", "execution_contract.json", "actual/result.json")]
        assert closed["status"] == "foreground_closed" and closed["exit_code"] == 0 and once["exactly_one_launch"]
        assert sha(run / "root_execution.json") == "27fbab86c8be671bb0b390b96cf621e68444f70c5073e45fa61039e6d9649946"
        assert closed["execution_contract_sha256"] == once["execution_contract_sha256"] == sha(run / "execution_contract.json")
        assert closed["driver_sha256"] == execution["driver_sha256"] and result["all_equivalence_gates_passed"]
        assert datetime.fromisoformat(closed["started_utc"]) > datetime.fromisoformat(status["finished_at"])
        proof["separate_root_bounded_iteration"] = {"kind": "Private numerical radial-reuse qualification003 at original corrected29 same point; not new orbital optimization or shared adoption",
            "started_after_this_card": True, "actual_closed_exit_code": 0, "started_utc": closed["started_utc"], "finished_utc": closed["finished_utc"],
            "elapsed_uncontrolled_seconds": closed["elapsed_uncontrolled_seconds"], "actual_declared_calls": {k: result[k] for k in ("H", "CI", "gradient", "optimizer", "GPU", "scipy_eigh", "Gauss_setup_eigvalsh")},
            "all_equivalence_gates_passed": True, "shared_adoption_claim": False, "speedup_certified": False,
            "files_sha256": {p: sha(run / p) for p in ("root_execution.json", "once_marker.json", "execution_contract.json", "actual/result.json", "actual/comparison.json")}}
    else:
        proof["native_iteration_scope"] = {"new_independent_native_launch_for_this_hourly_card": False,
            "continuing_work": "Parallel source/math and qualification preparation continued under existing authorization; no new independent native launch is attributed to this hour24 card.",
            "historical_run005_claimed_as_new_launch": False, "later_stage022_root_qualification003_is_separate": True}
    write(directory / "delivery_proof.json", proof)
    return {"directory": directory.name, "proof_sha256": sha(directory / "delivery_proof.json"), "position": position,
            "source_stage": stage_number, "service_jobs": [], "new_API_calls": 0, "new_scientific_calls": 0}


if __name__ == "__main__":
    assert sha(UTILITY) == "f8bf88f8caae2285620273f449c6f3b3d705364cc8115b2699284dd7c0bba093"
    summaries = [verify(HERE, 22, "om_x100b638f067e5ca0dee5748e0a93e50", 1616),
                 verify(CAMPAIGN / "delivery_hour24", 21, "om_x100b638f66f890a4c022bc622939307", 1615)]
    print(json.dumps(summaries))
