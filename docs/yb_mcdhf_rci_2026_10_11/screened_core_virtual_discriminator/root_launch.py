"""Keep the reviewed screened-virtual invocation foreground and bounded."""

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
    binding_path = HERE / "execution_binding.json"
    binding = json.loads(binding_path.read_text())
    driver = HERE / "run.py"
    assert binding["execution_authorized_by_root"]
    assert binding["driver_sha256"] == sha(driver)
    assert binding["preparation_contract_sha256"] == sha(HERE / "contract.json")
    assert not (HERE / "root_execution.json").exists()
    command = [sys.executable, "-u", str(driver), "--execute", "--binding", str(binding_path)]
    before = {"driver": sha(driver), "binding": sha(binding_path)}
    started = time.monotonic()
    stopped = False
    with (HERE / "root_execution.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=2055)
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "foreground_command": command, "driver_sha256": before["driver"],
              "execution_contract_sha256": before["binding"],
              "launcher_sha256": sha(Path(__file__)), "outer_wall_cap_s": 2055,
              "exit_code": code, "outer_wall_cap_stopped": stopped,
              "elapsed_s": time.monotonic() - started,
              "log_sha256": sha(HERE / "root_execution.log"),
              "driver_unchanged": before["driver"] == sha(driver),
              "execution_contract_unchanged": before["binding"] == sha(binding_path),
              "controlled_performance_claim": False}
    with (HERE / "root_execution.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if code or stopped:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
