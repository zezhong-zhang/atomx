"""One root-authorized foreground launcher; stdlib receipt only."""
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    seal = json.loads((B / "execution_contract.json").read_text())
    for field, name in (("driver_sha256", "run.py"), ("contract_sha256", "contract.json"),
                        ("preparation_sha256", "preparation_verification.json")):
        if seal[field] != sha(B / name):
            raise ValueError("Root seal changed: " + field)
    host = {"load_average": list(os.getloadavg()), "affinity": sorted(os.sched_getaffinity(0))}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            host["mem_available_KiB"] = int(line.split()[1])
    if 31 not in host["affinity"] or host["mem_available_KiB"] < int(3.95 * 2**20):
        raise ValueError("Resource admission failed")
    if host["load_average"][0] > seal["maximum_initial_load"]:
        raise ValueError("Initial load exceeds root budget guard")
    with (B / "once_marker.json").open("x") as handle:
        json.dump({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "execution_contract_sha256": sha(B / "execution_contract.json"),
                   "exactly_one_launch": True}, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    command = ["timeout", "--signal=TERM", "--kill-after=5s", "315s",
               "/data/code/atomx/.venv/bin/python", str(B / "run.py"),
               "--execution", str(B / "execution_contract.json")]
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = "1"
    receipt = {"host_admission": host, "status": "foreground_started", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": command, "cwd": str(R), "launcher_sha256": sha(__file__),
               "execution_contract_sha256": sha(B / "execution_contract.json"),
               "driver_sha256": sha(B / "run.py"), "contract_sha256": sha(B / "contract.json"),
               "preparation_sha256": sha(B / "preparation_verification.json"),
               "thread_variables": {name: environment[name] for name in
               ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")},
               "timing_scope": "uncontrolled cold fixed-CI operator discriminator, no speedup certificate"}
    write(B / "root_execution.json", receipt)
    begin = time.monotonic()
    with (B / "root_execution.log").open("wb") as log:
        result = subprocess.run(command, cwd=R, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt.update(status="foreground_closed", exit_code=result.returncode,
                   elapsed_uncontrolled_seconds=time.monotonic() - begin,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   log_sha256=sha(B / "root_execution.log"))
    if (B / "actual").exists():
        receipt["actual_files_sha256"] = {str(p.relative_to(B)): sha(p) for p in
                sorted((B / "actual").rglob("*.json"))}
    write(B / "root_execution.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "exit_code", "elapsed_uncontrolled_seconds", "log_sha256")}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
