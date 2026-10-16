"""Verify only archived hourly delivery evidence; no API or science calls."""

import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEY = "30cd70c8-315f-5d8c-b970-62c134673601"
MESSAGE = "om_x100b638d288ee13cc4302107d4fb072"
CHAT = "oc_fb2fc627b3e98dc9555de31580878758"
BOT = "ou_c697e48a528ad9aa6157f62531561b59"
STAGE_SHA = "247ba573bf265c23254760f39df16c3517012999cade1c7d4c898b2d9727eccb"
REGISTRY_SHA = "ade386ad9689c1a8ef054d22ec55893f3468830986aa34a580169ac5a5a42bf1"
ROOT_SUCCESSOR_SHA = "395b7dd64dedba5d71b16c56adc75f36394250ec7561527f8acdcf8cd40784c4"


def read(name):
    return json.loads((HERE / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_text(value):
    # The CLI adds a DOI hyperlink wrapper and folds blank lines. Retain every
    # non-whitespace visible character; do not claim raw CardKit DSL equality.
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    return re.sub(r"\s+", "", value)


card = read("card.json")
receipt = read("receipt.json")
status = read("send_status.json")
request = read("iteration_request.json")
preview = read("iteration_preview.json")
decision = read("iteration_decision.json")
readback = read("readback.json")
execution = read("readback_execution.json")
assert execution["exit_code"] == 0 and execution["actual_readback_calls"] == 1
assert readback["ok"] and readback["identity"] == "bot"
assert readback["data"]["total"] == 1
message = readback["data"]["messages"][0]
assert message["message_id"] == receipt["data"]["message_id"] == MESSAGE
assert message["chat_id"] == receipt["data"]["chat_id"] == CHAT
assert message["sender"]["open_bot_id"] == BOT
assert message["sender"]["name"] == "aisi codex"
assert message["sender"]["sender_type"] == "app"
assert message["msg_type"] == "interactive"
assert not message["deleted"] and not message["updated"]
assert message["create_time"] == "2026-10-11 22:00"
assert receipt["data"]["create_time"] == "2026-10-11 22:00:04"
assert status["idempotency_key"] == request["key"] == KEY
assert status["message_id"] == request["message_id"] == MESSAGE
assert status["cycle"] == request["cycle"] == "2026-10-11T14"
assert status["passed"] and status["iteration_dispatch_ok"]
assert status["started_at"] == "2026-10-11T14:00:01.872478+00:00"
assert status["finished_at"] == "2026-10-11T14:00:04.756317+00:00"
assert status["registry_sha256"] == REGISTRY_SHA
assert digest(HERE / "registry_stage016.json") == REGISTRY_SHA
assert digest(REPO / "docs/yb_mcdhf_rci_2026_10_11/reporting_stage017/before_progress.json") == REGISTRY_SHA
assert digest(REPO / "docs/yb_mcdhf_rci_2026_10_11/results_stage017.json") == ROOT_SUCCESSOR_SHA
assert any(
    item["path"] == "docs/yb_mcdhf_rci_2026_10_11/results_stage016.json"
    and item["sha256"] == STAGE_SHA
    for item in status["evidence"]
)
assert digest(REPO / "docs/yb_mcdhf_rci_2026_10_11/results_stage016.json") == STAGE_SHA
assert request["preview_sha256"] == status["iteration_preview_sha256"]
assert digest(HERE / "iteration_preview.json") == request["preview_sha256"]
assert request["decision"] == decision
for item in (request, preview, decision):
    assert item["status"] == "implementation_frontier"
for item in (preview, decision):
    assert item["job"] is None and item["jobs"] == []
assert card["schema"] == "2.0"
elements = card["body"]["elements"]
assert len(elements) == 15 and all(item["tag"] == "markdown" for item in elements)
expected = (
    '<card title="' + card["header"]["title"]["content"] + '">\n'
    + "\n".join(item["content"] for item in elements) + "\n</card>"
)
actual = message["content"]
assert visible_text(expected) == visible_text(actual)
differences = []
for operation, a, b, c, d in difflib.SequenceMatcher(
    None, expected, actual, autojunk=False
).get_opcodes():
    if operation != "equal":
        differences.append({"operation": operation, "payload": expected[a:b], "readback": actual[c:d]})
assert differences == [
    *({"operation": "delete", "payload": "\n", "readback": ""} for _ in range(6)),
    {"operation": "insert", "payload": "", "readback": "["},
    {"operation": "insert", "payload": "", "readback": "](https://doi.org/10.1103/PhysRevA.110.L030802)"},
]
inputs = [
    "card.json", "receipt.json", "send_status.json", "iteration_request.json",
    "iteration_preview.json", "iteration_decision.json", "readback.json",
    "readback.stderr.log", "readback_execution.json", "registry_stage016.json",
    "verify_delivery.py",
]
proof = {
    "proof_created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "passed": True,
    "successful_send_and_readback": True,
    "cycle": status["cycle"],
    "idempotency_key": KEY,
    "message_id": MESSAGE,
    "actual_message_link": message["message_app_link"],
    "message_position": message["message_position"],
    "chat_id": CHAT,
    "bot_open_id": BOT,
    "source_stage016_sha256": STAGE_SHA,
    "registry_sha256": REGISTRY_SHA,
    "registry_copy_provenance": "docs/yb_mcdhf_rci_2026_10_11/reporting_stage017/before_progress.json; immutable prior registry copied after successor update",
    "renderer_sha256": status["renderer_sha256"],
    "actual_card": {
        "payload_schema": card["schema"], "elements": len(elements),
        "title": card["header"]["title"]["content"],
        "receipt_create_time_Asia_Hong_Kong": receipt["data"]["create_time"],
        "readback_create_time_Asia_Hong_Kong": message["create_time"],
        "readback_content_characters": len(actual),
        "all_visible_content_matches": True,
        "raw_readback_schema_available": False,
        "readback_representation": "CLI-rendered card text; raw v2 DSL is in copied send payload only",
        "rendering_differences": differences,
    },
    "service_send": {
        "started_at": status["started_at"], "finished_at": status["finished_at"],
        "passed": status["passed"], "iteration_dispatch_ok": status["iteration_dispatch_ok"],
        "exit_code": 0, "exit_code_evidence": "explicit root delegation of actual service closure",
    },
    "iteration": {
        "request_status": request["status"], "requested_at": request["requested_at"],
        "finished_at": request["finished_at"], "service_job": None, "service_jobs": [],
        "service_automatically_started_science": False,
        "root_bounded_iteration": "Root explicitly continued this authorized hourly-card iteration with the single bounded native normalized-gradient control (1 H / 1 CI / 2 weighted gradients).",
        "root_bounded_iteration_result": "Pending at delegation; root subsequently reported actual closure: 1 H / 1 CI / 2 weighted gradients, 24.946 s, 104 live checks passed. This delivery proof only binds that separate successor report, and does not independently certify its scientific results.",
        "root_bounded_iteration_report": "docs/yb_mcdhf_rci_2026_10_11/results_stage017.json",
        "root_bounded_iteration_report_sha256": ROOT_SUCCESSOR_SHA,
        "root_authorization_source": "Parent /root task message delegating this hour22 delivery closure",
    },
    "actual_readback_calls": 1,
    "new_scientific_calls": 0,
    "copied_configurations_or_secrets": False,
    "files_sha256": {name: digest(HERE / name) for name in inputs},
    "scope": "Actual bot/current-chat delivery and full visible content readback only. Delivered stage016 is preserved; stage017 is a separately bound root continuation. Preserve implementation_frontier/no service job, original numerical and physical failures, and optimizer HOLD; this proof independently executes no science.",
}
(HERE / "delivery_proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "passed": True, "message_id": MESSAGE, "link": proof["actual_message_link"],
    "proof_sha256": digest(HERE / "delivery_proof.json"),
    "iteration_status": request["status"], "service_job": None,
    "actual_readback_calls": 1, "new_scientific_calls": 0,
}, ensure_ascii=False))
