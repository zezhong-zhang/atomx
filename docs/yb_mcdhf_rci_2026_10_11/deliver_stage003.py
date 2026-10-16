"""Send the authorized display card, read it back, and archive its iteration request."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHAT = "oc_fb2fc627b3e98dc9555de31580878758"
BOT = "ou_c697e48a528ad9aa6157f62531561b59"


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    out = HERE / "delivery_stage003"
    out.mkdir(exist_ok=True)
    sender = REPO / "docs/atomx_hourly_progress_2026_10_10/sync_card.py"
    command = ["python3", str(sender), "--config",
               "/home/zzhang/.local/state/atomx-lark-hourly-aisi-codex/config.json"]
    preview = subprocess.run(command + ["--render-only"], cwd=REPO, text=True,
                             capture_output=True, check=False)
    (out / "preview_stdout.json").write_text(preview.stdout)
    (out / "preview_stderr.log").write_text(preview.stderr)
    if preview.returncode:
        raise RuntimeError("Source-bound card preview failed; consult saved diagnostic")
    sent = subprocess.run(command + ["--cycle", "native-mcdhf-rci-stationary-20261011-stage003"],
                          cwd=REPO, text=True, capture_output=True, check=False)
    (out / "sender_stdout.json").write_text(sent.stdout)
    (out / "sender_stderr.log").write_text(sent.stderr)
    if sent.returncode:
        raise RuntimeError("Authorized card send failed; consult saved bridge diagnostic")
    record = json.loads(sent.stdout)
    assert record["passed"] and record["iteration_dispatch_ok"]
    state = Path(record["iteration_preview"]).parent.parent
    for name in ("latest_card.json", "last_success.json", "iteration_status.json"):
        if (state / name).exists():
            shutil.copyfile(state / name, out / name)
    shutil.copyfile(record["iteration_preview"], out / "iteration_preview.json")
    shutil.copyfile(record["iteration_request"], out / "iteration_request_initial.json")
    readback = subprocess.run(["lark-cli", "im", "+messages-mget", "--message-ids",
                               record["message_id"], "--no-reactions"], cwd=REPO,
                              text=True, capture_output=True, check=False)
    (out / "readback.json").write_text(readback.stdout)
    (out / "readback_stderr.log").write_text(readback.stderr)
    if readback.returncode:
        raise RuntimeError("Card was sent; readback failed. Preserve its real receipt.")
    message = json.loads(readback.stdout)["data"]["messages"][0]
    text = message["content"]
    checks = {
        "actual_message_id": message["message_id"] == record["message_id"],
        "correct_chat": message["chat_id"] == CHAT,
        "own_bot": message["sender"]["open_bot_id"] == BOT,
        "interactive": message["msg_type"] == "interactive",
        "primary_MCDHF_visible": "五参考驻值基线" in text and "5.143e-08" in text,
        "mixed_stationarity_diagnostic_visible": "0.03676" in text and "0.02499" in text,
        "historical_SD_separate": "历史 SD 对照" in text,
        "iteration_dispatched": record["iteration_dispatch_ok"],
    }
    assert all(checks.values()), checks
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message_id": record["message_id"], "chat_id": CHAT,
        "message_app_link": message["message_app_link"],
        "checks": checks, "passed": True, "highest_level_reached": False,
        "source_identity": record["evidence"],
        "registry_sha256": record["registry_sha256"],
        "renderer_sha256": record["renderer_sha256"],
        "sent_card_sha256": hashlib.sha256((out / "latest_card.json").read_bytes()).hexdigest(),
        "readback_sha256": hashlib.sha256((out / "readback.json").read_bytes()).hexdigest(),
        "iteration_scope": "Receipt-backed decision/dispatch of registered bounded scientific jobs; no unregistered expanded-MCDHF executor is claimed.",
    }
    save(out / "receipt.json", receipt)
    print(json.dumps({"passed": True, "message_id": record["message_id"],
                      "message_app_link": message["message_app_link"],
                      "iteration_dispatched": True}))


if __name__ == "__main__":
    main()
