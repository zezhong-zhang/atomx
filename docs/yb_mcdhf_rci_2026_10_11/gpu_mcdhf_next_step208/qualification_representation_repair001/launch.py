"""Stdlib foreground guardian; no child may launch without the exact root seal."""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def process_group_RSS(group):
    total, pids = 0, []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            text = (directory / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            if int(fields[2]) != group:
                continue
            status = (directory / "status").read_text().splitlines()
            row = next((r for r in status if r.startswith("VmRSS:")), "VmRSS: 0 kB")
            total += int(row.split()[1]) * 1024
            pids.append(int(directory.name))
        except (OSError, ValueError):
            continue
    return total, pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute is None:
        print("SOURCE_ONLY: no root binding supplied; no child/NumPy/Atomx/CUDA execution")
        return
    spec = importlib.util.spec_from_file_location("native31_private_qualification_guarded_driver", HERE / "run.py")
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    # Exact root admission checks only source/JSON, then array hashes, no array loads.
    authority, contract, _, _ = driver.authorize(args.execute)
    if authority.get("launcher_sha256") != sha(__file__):
        raise ValueError("root seal must bind this exact guardian")
    controller = HERE / "controller"
    controller.mkdir(exist_ok=False)
    write(controller / "ONE_ATTEMPT.json", dict(root_seal_sha256=sha(args.execute), driver_sha256=sha(HERE / "run.py"), launcher_sha256=sha(__file__)))
    caps = contract["resources"]
    if not set(caps["CPU4"]) <= os.sched_getaffinity(0):
        raise RuntimeError("actual affinity does not admit CPU30 and all CPU4 workers")
    environment = os.environ.copy()
    for name in caps["single_thread_env"]:
        environment[name] = "1"
    environment["NUMBA_NUM_THREADS"] = "4"
    runtime = ROOT / contract["runtime"]["isolated_runtime"]
    environment["CUDA_PATH"] = str(runtime / "nvidia/cu13")
    environment["LD_LIBRARY_PATH"] = str(runtime / "nvidia/cu13/lib")
    cmd = [caps["interpreter_literal"], "-I", str(HERE / "run.py"), "--execute", str(args.execute.absolute())]
    write(controller / "command.json", dict(argv=cmd, cwd=str(ROOT), environment={k: environment[k] for k in (*caps["single_thread_env"], "NUMBA_NUM_THREADS", "CUDA_PATH", "LD_LIBRARY_PATH")},
        virtual_AS_bytes=caps["virtual_AS_bytes"], RSS_hard_watch_bytes=caps["maximum_RSS_bytes"],
        sample_seconds=.05, timing_controlled=False))
    start, peak, reason = time.monotonic(), 0, None
    with (controller / "execution.log").open("w") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                rss, pids = process_group_RSS(child.pid)
                peak = max(peak, rss)
                if rss > caps["maximum_RSS_bytes"]:
                    reason = "RSS_hard_watch_stop"
                elif time.monotonic() - start > caps["outer_seconds"]:
                    reason = "outer_wall_limit"
                if reason:
                    write(controller / "resource_stop.json", dict(reason=reason, actual_RSS_bytes=rss,
                          PIDs=pids, elapsed_s=time.monotonic() - start))
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=caps["kill_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.05)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            exit_code = child.wait()
    receipt = dict(status="foreground_closed", exit_code=exit_code, resource_stop=reason,
        peak_process_group_RSS_bytes=peak, elapsed_uncontrolled_s=time.monotonic() - start,
        root_seal_sha256=sha(args.execute), contract_sha256=sha(HERE / "contract.json"),
        driver_sha256=sha(HERE / "run.py"), launcher_sha256=sha(__file__),
        execution_log_sha256=sha(controller / "execution.log"), controlled_speedup_qualified=False,
        RSS_monitor_note="50ms sampled process-group RSS; threshold breach terminates group, transient unsampled peak cannot be certified")
    write(controller / "execution_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(exit_code if exit_code else (1 if reason else 0))


if __name__ == "__main__":
    main()
