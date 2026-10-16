"""Execute one exact reviewed native29 original-rotation Rayleigh qualification with a foreground process cap."""

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
    seal = HERE / "execution_contract.json"
    binding = json.loads(seal.read_text())
    assert binding["execution_authorized_by_root"] is True
    for name, expected in binding["root_reviewed_files_sha256"].items():
        assert sha(HERE.parents[2] / name) == expected, name
    assert not (HERE / "actual").exists()
    assert not (HERE / "root_execution.json").exists()
    before = sha(seal)
    command = [sys.executable, "-u", str(HERE / "run.py"),
               "--execution-contract", str(seal)]
    started = time.monotonic()
    stopped = False
    with (HERE / "root_execution.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=135)
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  foreground_command=command, execution_contract_sha256=before,
                  launcher_sha256=sha(Path(__file__)), outer_wall_cap_s=135,
                  exit_code=code, outer_wall_cap_stopped=stopped,
                  elapsed_s=time.monotonic() - started,
                  log_sha256=sha(HERE / "root_execution.log"),
                  execution_contract_unchanged=before == sha(seal),
                  controlled_performance_claim=False, automatic_retry=False)
    with (HERE / "root_execution.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code or int(stopped))


if __name__ == "__main__":
    main()
