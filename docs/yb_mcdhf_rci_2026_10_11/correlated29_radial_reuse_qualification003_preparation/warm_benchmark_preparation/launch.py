"""Exactly-once outer receipt, including hard timeout; stdlib only."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run import THREADS, admit, sha, write

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    args = parser.parse_args()
    contract, execution = admit(args.execution, require_affinity=True)
    if execution.get("launcher_sha256") != sha(__file__):
        raise ValueError("Exact launcher seal missing")
    if (HERE / "actual").exists():
        raise ValueError("Actual directory already exists")
    with (HERE / "launch_once_marker.json").open("x") as handle:
        handle.write(json.dumps({"execution_contract_sha256": sha(args.execution),
                                "launcher_sha256": sha(__file__), "sole_launch": True}))
        handle.flush()
        os.fsync(handle.fileno())
    environment = dict(os.environ)
    if any(environment.get(name) != "1" for name in THREADS):
        raise ValueError("One-thread environment required")
    command = ["timeout", "--signal=TERM", "--kill-after=5s", "315s",
               "/data/code/atomx/.venv/bin/python", str(HERE / "run.py"),
               "--execution", str(Path(args.execution).absolute())]
    start = time.monotonic()
    with (HERE / "execution.log").open("x") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                cwd=contract["working_directory"])
    write(HERE / "execution_receipt.json", {"status": "foreground_closed", "exit_code": result.returncode,
          "elapsed_wall_seconds": time.monotonic() - start, "command": command,
          "execution_contract_sha256": sha(args.execution), "launcher_sha256": sha(__file__),
          "driver_sha256": sha(HERE / "run.py"), "helper_sha256": sha(HERE / "helper.py"),
          "log_sha256": sha(HERE / "execution.log"), "outer_seconds": 315, "kill_after_seconds": 5,
          "partial_actual_outputs_retained": True, "no_retry": True,
          "GPU_optimizer_parallel_whole_run_speed_certificate": False})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
