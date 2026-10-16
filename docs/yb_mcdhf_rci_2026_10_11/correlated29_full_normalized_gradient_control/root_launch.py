"""Exactly one root-authorized foreground process, with a 315s outer cap."""
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    authority = json.loads((HERE / "execution_contract.json").read_text())
    originals = {"run.py": authority["driver_sha256"],
                 "contract.json": authority["contract_sha256"],
                 "preparation_verification.json": authority["preparation_sha256"],
                 "handoff.json": authority["handoff_sha256"],
                 "root_authorization_message.json": authority["root_authorization_message_sha256"]}
    if not all(sha(HERE / name) == value for name, value in originals.items()):
        raise ValueError("Changed exact root authorization/input")
    if (HERE / "root_launch_started.json").exists() or (HERE / "actual").exists():
        raise ValueError("Single authorized launch already consumed")
    environment = os.environ.copy()
    environment.update(authority["thread_values"])
    record = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "execution_contract_sha256": sha(HERE / "execution_contract.json"),
              "launcher_sha256": sha(__file__), "outer_timeout_s": 315,
              "foreground_single_launch": True, "retry_count": 0,
              "timing_controlled": False, "thread_values": authority["thread_values"],
              "host_affinity_before": sorted(os.sched_getaffinity(0)),
              "host_load_before": list(os.getloadavg()), "frozen_files_before": originals}
    write(HERE / "root_launch_started.json", record)
    start = time.monotonic()
    with (HERE / "root_execution.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "run.py"), "--execution-contract",
                                    str(HERE / "execution_contract.json")], cwd=HERE.parents[2],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(json.dumps({"foreground_child_pid": process.pid, "outer_timeout_s": 315}), flush=True)
        timed_out = False
        try:
            returncode = process.wait(timeout=315)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    record.update(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  child_pid=process.pid, returncode=returncode, outer_timeout_triggered=timed_out,
                  elapsed_uncontrolled_s=time.monotonic() - start,
                  host_load_after=list(os.getloadavg()),
                  frozen_files_after={name: sha(HERE / name) for name in originals},
                  frozen_files_unchanged=all(sha(HERE / n) == h for n, h in originals.items()),
                  log_sha256=sha(HERE / "root_execution.log"),
                  result_sha256=(sha(HERE / "actual/result.json") if (HERE / "actual/result.json").exists() else None),
                  failure_sha256=(sha(HERE / "actual/failure.json") if (HERE / "actual/failure.json").exists() else None))
    write(HERE / "root_execution.json", record)
    print(json.dumps({"returncode":returncode, "elapsed_uncontrolled_s":record["elapsed_uncontrolled_s"],
                      "result_sha256":record["result_sha256"], "failure_sha256":record["failure_sha256"],
                      "frozen_files_unchanged":record["frozen_files_unchanged"]}), flush=True)
    sys.exit(returncode if returncode >= 0 else 128 - returncode)


if __name__ == "__main__":
    main()
