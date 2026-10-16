"""Root-sealed once-only foreground local browser QA; no scientific imports."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STOP = None
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "GOTO_NUM_THREADS",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_signal(number, _frame):
    global STOP
    STOP = "signal_" + str(number)


def process_table():
    """Linux process identity includes start time; never signal a reused PID."""
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            resident = int((entry / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            table[int(entry.name)] = {
                "ppid": int(fields[1]), "start": int(fields[19]),
                "state": fields[0], "rss": resident,
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return table


def descendants(table, root):
    found = {root}
    while True:
        children = {pid for pid, value in table.items() if value["ppid"] in found}
        updated = found | children
        if updated == found:
            return found - {root}
        found = updated


def signal_owned(pid, identity, number):
    try:
        fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
        if int(fields[19]) == identity:
            os.kill(pid, number)
    except (FileNotFoundError, ProcessLookupError):
        pass


def cleanup(processes, known, deadline):
    """Subreaper captures Chrome children even after their launcher detaches/exits."""
    events = []
    for stage, number in (("TERM", signal.SIGTERM), ("KILL", signal.SIGKILL)):
        stage_deadline = min(deadline, time.monotonic() + (0.4 if stage == "TERM" else 1.8))
        while True:
            table = process_table()
            live = descendants(table, os.getpid())
            for pid in live:
                known[pid] = table[pid]["start"]
            targets = [pid for pid, identity in known.items()
                       if pid in table and table[pid]["start"] == identity
                       and table[pid]["state"] != "Z"]
            for pid in targets:
                signal_owned(pid, known[pid], number)
            for process in processes:
                process.poll()
            # Popen has reaped its own direct children before adopted-child reaping.
            if all(process.returncode is not None for process in processes):
                while True:
                    try:
                        child, _status = os.waitpid(-1, os.WNOHANG)
                        if not child:
                            break
                    except ChildProcessError:
                        break
            if not targets or time.monotonic() >= stage_deadline:
                events.append({"signal": stage, "last_live_owned_pids": targets})
                break
            time.sleep(0.05)
    table = process_table()
    remaining = [pid for pid in descendants(table, os.getpid()) if table[pid]["state"] != "Z"]
    return {"all_owned_descendants_stopped": not remaining, "remaining_pids": remaining, "events": events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    receipt = {"status": "admission_started", "started_utc": utc(),
               "science_calls": 0, "publication_calls": 0, "credentials_read": False}
    processes, handles, known, samples = [], [], {}, []
    peak, browser_exit, exit_code = 0, None, 1
    browser = None
    deadline = started + 90.0
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    attempted = False
    try:
        execution_path = Path(args.execution_contract).resolve()
        if sha(execution_path) != args.execution_sha256:
            raise ValueError("Root execution seal hash mismatch")
        seal = json.loads(execution_path.read_text())
        contract_path = HERE / "browser_launcher_contract.json"
        contract = json.loads(contract_path.read_text())
        if (seal["scope"] != "atlas_update007_local_browser_once"
                or seal["execution_authorized_by_root"] is not True or seal["maximum_attempts"] != 1
                or seal["launcher_sha256"] != sha(__file__)
                or seal["launcher_contract_sha256"] != sha(contract_path)
                or seal["inputs_sha256"] != contract["inputs_sha256"]
                or seal["resources"] != contract["resources"]):
            raise ValueError("Root authorization/launcher/input/resource binding mismatch")
        resources = contract["resources"]
        if resources["CPU"] not in os.sched_getaffinity(0):
            raise ValueError("Declared CPU30 is unavailable")
        receipt.update(execution_contract_sha256=sha(execution_path), launcher_sha256=sha(__file__),
                       launcher_contract_sha256=sha(contract_path), resources=resources,
                       available_accounting_cpus=sorted(os.sched_getaffinity(0)))
        os.sched_setaffinity(0, {resources["CPU"]})
        receipt["actual_affinity"] = sorted(os.sched_getaffinity(0))
        for key in THREAD_KEYS:
            os.environ[key] = "1"
        receipt["thread_environment"] = {key: os.environ[key] for key in THREAD_KEYS}
        for name, expected in contract["inputs_sha256"].items():
            if sha(REPO / name) != expected:
                raise ValueError("Changed prepared input: " + name)
        site_contract = json.loads((HERE / "contract.json").read_text())
        site = Path(site_contract["site"])
        manifest = json.loads((HERE / "site_manifest.json").read_text())
        for name, value in manifest["files"].items():
            if sha(site / name) != value["sha256"]:
                raise ValueError("Changed local site asset: " + name)
        if not json.loads((HERE / "data_verification.json").read_text())["all_passed"]:
            raise ValueError("Independent local data verification failed")
        if time.monotonic() >= deadline - 3 or STOP:
            raise RuntimeError("Stopped during admission")
        # Never reuse or terminate an existing listener.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 39319))
        if any((HERE / name).exists() for name in contract["exclusive_outputs"]):
            raise ValueError("Previous browser artifacts exist; no automatic retry")
        with (HERE / "browser_once_marker.json").open("x") as marker:
            json.dump({"started_utc": utc(), "execution_contract_sha256": args.execution_sha256}, marker)
        attempted = True
        # No RLIMIT_AS/cgroup/systemd mutation: Chromium reserves large virtual spans.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER, Linux.
            raise OSError(ctypes.get_errno(), "Cannot establish owned descendant reaping")
        receipt["subreaper_enabled"] = True
        env = os.environ.copy()

        def launch(command, label):
            out = (HERE / (label + ".stdout.log")).open("xb")
            err = (HERE / (label + ".stderr.log")).open("xb")
            handles.extend((out, err))
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err,
                                       start_new_session=True)
            processes.append(process)
            table = process_table()
            if process.pid in table:
                known[process.pid] = table[process.pid]["start"]
            receipt[label] = {"command": command, "pid": process.pid,
                              "pid_start": known.get(process.pid)}
            return process

        server = launch([sys.executable, "-u", "-m", "http.server", "39319", "--bind", "127.0.0.1",
                         "--directory", str(site / "dist")], "browser_server")
        ready = False
        last_sample = None
        max_gap = 0.0
        while time.monotonic() < deadline - 3:
            sample_started = time.monotonic()
            table = process_table()
            owned = descendants(table, os.getpid())
            for pid in owned:
                known[pid] = table[pid]["start"]
            # Conservative RSS sum includes shared resident pages and the supervisor itself.
            aggregate = sum(table[pid]["rss"] for pid in owned | {os.getpid()} if pid in table)
            peak = max(peak, aggregate)
            if last_sample is not None:
                max_gap = max(max_gap, sample_started - last_sample)
                if sample_started - last_sample > resources["maximum_observed_sample_gap_seconds"]:
                    raise RuntimeError("RSS sampling gap exceeded declared 0.25s; stopping owned processes")
            last_sample = sample_started
            samples.append({"seconds": sample_started - started, "aggregate_rss_bytes": aggregate,
                            "owned_processes": len(owned)})
            if aggregate > resources["aggregate_rss_stop_bytes"]:
                raise MemoryError("Observed aggregate RSS exceeded 2GiB; possible sampling overshoot")
            if STOP:
                raise RuntimeError(STOP)
            if server.poll() is not None:
                raise RuntimeError("Owned loopback server exited before QA completion")
            if not ready:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:39319/index.html", timeout=0.03) as response:
                        ready = response.status == 200
                except (urllib.error.URLError, TimeoutError):
                    pass
                if ready:
                    browser = launch([contract["node_executable"], str(HERE / "browser_test.js")], "browser")
            if browser is not None and browser.poll() is not None:
                browser_exit = browser.returncode
                if browser_exit:
                    raise RuntimeError("Browser QA exited " + str(browser_exit))
                result = json.loads((HERE / "browser_result.json").read_text())
                if result["all_passed"] is not True:
                    raise ValueError("Browser result did not qualify")
                exit_code = 0
                receipt["browser_result_sha256"] = sha(HERE / "browser_result.json")
                break
            time.sleep(max(0.0, sample_started + resources["rss_sample_interval_seconds"] - time.monotonic()))
        else:
            raise TimeoutError("90s foreground budget reached cleanup reserve; no retry")
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(json.dumps(receipt["failure"]), file=sys.stderr, flush=True)
    finally:
        if attempted:
            stopped = cleanup(processes, known, deadline)
            receipt["cleanup"] = stopped
            if not stopped["all_owned_descendants_stopped"]:
                exit_code = 1
            for handle in handles:
                handle.close()
            for name in ("browser_server", "browser"):
                if name in receipt:
                    index = 0 if name == "browser_server" else 1
                    receipt[name]["exit_code_after_cleanup"] = processes[index].poll()
                    receipt[name]["stdout_sha256"] = sha(HERE / (name + ".stdout.log"))
                    receipt[name]["stderr_sha256"] = sha(HERE / (name + ".stderr.log"))
            artifact_names = ("browser_result.json", "browser_failure.json", "downloaded_update.json",
                              "desktop.png", "mobile.png")
            receipt["artifacts_sha256"] = {name: sha(HERE / name) for name in artifact_names if (HERE / name).exists()}
            write(HERE / "browser_rss_samples.json", {"sampling_interval_seconds": 0.2, "samples": samples})
            receipt["rss_samples_sha256"] = sha(HERE / "browser_rss_samples.json")
        receipt.update(status="foreground_closed" if attempted else "prelaunch_refused", exit_code=exit_code,
                       browser_exit_code=browser_exit, finished_utc=utc(),
                       elapsed_uncontrolled_seconds=time.monotonic() - started,
                       peak_observed_aggregate_rss_bytes=peak, memory_kernel_hard_cap=False,
                       memory_sampling_may_overshoot=True, execution_attempts=int(attempted),
                       browser_attempts=int(browser is not None),
                       maximum_observed_sample_gap_seconds=locals().get("max_gap", 0.0))
        # A prior closure is immutable; rejected re-invocations cannot overwrite it.
        output = HERE / ("browser_execution.json" if attempted else "browser_prelaunch_refusal.json")
        with output.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "exit_code": exit_code,
                          "elapsed_seconds": receipt["elapsed_uncontrolled_seconds"], "peak_RSS_bytes": peak}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
