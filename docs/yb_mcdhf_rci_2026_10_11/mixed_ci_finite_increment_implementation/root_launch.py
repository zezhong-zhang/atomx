"""One reviewed finite qualification, foreground with whole-group wall cap."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    binding = json.loads((HERE / "execution_contract.json").read_text())
    assert binding["finite_qualification_execution_authorized"] is True
    for name, expected in binding["root_reviewed_files_sha256"].items():
        assert sha(HERE.parents[2] / name) == expected, name
    assert not (HERE / "root_execution.json").exists()
    before = sha(HERE / "execution_contract.json")
    command = [sys.executable, "-u", str(HERE / "run.py")]
    started = time.monotonic()
    stopped = False
    with (HERE / "root_execution.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=315)
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "foreground_command": command, "execution_contract_sha256": before,
              "launcher_sha256": sha(Path(__file__)), "outer_wall_cap_s": 315,
              "exit_code": code, "outer_wall_cap_stopped": stopped,
              "elapsed_s": time.monotonic() - started,
              "log_sha256": sha(HERE / "root_execution.log"),
              "execution_contract_unchanged": before == sha(HERE / "execution_contract.json"),
              "controlled_performance_claim": False}
    with (HERE / "root_execution.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code or int(stopped))


if __name__ == "__main__":
    main()
