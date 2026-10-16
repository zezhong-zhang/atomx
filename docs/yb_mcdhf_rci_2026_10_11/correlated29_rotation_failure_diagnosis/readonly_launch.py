"""One foreground read-only launch with independent outer hard timeout."""
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract_path = HERE / "outer_execution_contract.json"
    authority_path = HERE / "root_authorization.json"
    contract = json.loads(contract_path.read_text())
    if (sha(__file__) != contract["launcher_sha256"]
            or sha(authority_path) != contract["authorization_sha256"]
            or sha(HERE / "contract.json") != contract["diagnostic_contract_sha256"]
            or sha(HERE / "diagnose_saved.py") != contract["driver_sha256"]):
        raise ValueError("Changed exact read-only execution input")
    if (HERE / "root_execution.json").exists() or (HERE / "actual").exists():
        raise ValueError("One invocation only; no implicit retry")
    command = [str(REPO / ".venv/bin/python"), "-u", str(HERE / "diagnose_saved.py"),
               "--authorization", str(authority_path)]
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "1"
    start = time.monotonic()
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "foreground_command": command,
              "outer_execution_contract_sha256": sha(contract_path), "outer_wall_cap_s": 105,
              "automatic_retry": False, "controlled_performance_claim": False,
              "original_failed_gate_unchanged": True}
    stopped = False
    with (HERE / "root_execution.log").open("w") as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=105)
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    record.update(exit_code=code, outer_wall_cap_stopped=stopped, elapsed_s=time.monotonic() - start,
                  log_sha256=sha(HERE / "root_execution.log"),
                  execution_contract_unchanged=sha(contract_path) == record["outer_execution_contract_sha256"])
    (HERE / "root_execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    raise SystemExit(code if code >= 0 else 1)


if __name__ == "__main__":
    main()
