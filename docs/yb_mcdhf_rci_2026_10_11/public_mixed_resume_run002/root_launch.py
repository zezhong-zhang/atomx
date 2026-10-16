"""Root-owned foreground launch; preserve exact child exit and streamed output."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mode = sys.argv[1]
    assert mode in ("solve", "verify")
    output = HERE / ("root_execution.json" if mode == "solve" else "root_endpoint_execution.json")
    assert not output.exists(), "Never overwrite a completed scientific invocation record"
    contract = HERE / "execution_contract.json"
    approval = json.loads(contract.read_text())
    driver = HERE / ("run.py" if mode == "solve" else "verify_endpoint.py")
    expected = approval["driver_sha256" if mode == "solve" else "verifier_sha256"]
    assert sha(driver) == expected
    assert sha(contract) == "9b219c1a2481e70010dabf497ebacf7ea1ab879514a69fbd78e1862becbd3aa2"
    command = [str(ROOT / ".venv/bin/python"), "-u", str(driver)]
    if mode == "solve":
        command.append("solve")
    command += ["--execution-contract", str(contract)]
    admission = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "foreground_command": command,
        "driver_sha256": sha(driver),
        "execution_contract_sha256": sha(contract),
        "launcher_sha256": sha(Path(__file__)),
    }
    admission_path = HERE / ("root_launch_admission.json" if mode == "solve" else "root_endpoint_admission.json")
    assert not admission_path.exists()
    admission_path.write_text(json.dumps(admission, indent=2) + "\n")
    started = time.monotonic()
    log = HERE / ("run.log" if mode == "solve" else "verify_endpoint.log")
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    record = {**admission, "exit_code": code, "elapsed_s": time.monotonic() - started,
              "log_sha256": sha(log), "execution_contract_unchanged": sha(contract) == admission["execution_contract_sha256"],
              "driver_unchanged": sha(driver) == admission["driver_sha256"],
              "controlled_performance_claim": False}
    output.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
