"""Record a single reviewed foreground scientific invocation without replay."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--contract-name", default="execution_contract.json")
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--driver", default="run.py")
    parser.add_argument("--hash-key", default="driver_sha256")
    parser.add_argument("--record", default="root_execution")
    parser.add_argument("--outer-wall-cap", type=int, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    contract = directory / args.contract_name
    driver = directory / args.driver
    assert directory.is_relative_to(ROOT)
    assert sha(contract) == args.contract_sha256
    approval = json.loads(contract.read_text())
    assert approval["execution_authorized_by_root"] is True
    assert sha(driver) == approval[args.hash_key]
    record = directory / (args.record + ".json")
    admission_path = directory / (args.record + "_admission.json")
    log = directory / (args.record + ".log")
    assert not any(p.exists() for p in (record, admission_path, log))
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [str(ROOT / ".venv/bin/python"), "-u", str(driver), *arguments]
    admission = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "foreground_command": command,
        "driver_sha256": sha(driver),
        "execution_contract_sha256": sha(contract),
        "launcher_sha256": sha(Path(__file__)),
        "outer_wall_cap_s": args.outer_wall_cap,
    }
    with admission_path.open("x") as stream:
        stream.write(json.dumps(admission, indent=2) + "\n")
    started = time.monotonic()
    process = None
    stopped = False

    def timeout(*_):
        nonlocal stopped
        stopped = True
        if process is not None:
            os.killpg(process.pid, signal.SIGKILL)

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.outer_wall_cap)
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
    finally:
        signal.alarm(0)
    result = {**admission, "exit_code": code, "outer_wall_cap_stopped": stopped,
              "elapsed_s": time.monotonic() - started, "log_sha256": sha(log),
              "execution_contract_unchanged": sha(contract) == args.contract_sha256,
              "driver_unchanged": sha(driver) == admission["driver_sha256"],
              "controlled_performance_claim": False}
    with record.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
