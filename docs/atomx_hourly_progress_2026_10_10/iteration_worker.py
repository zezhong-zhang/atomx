"""Receipt-backed bounded parallel execution with serial durable claiming."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from parallel_planner import plan_batch as plan  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def resource_preflight(state, caps):
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    free_disk = shutil.disk_usage(state).free
    result = dict(
        available_bytes=available,
        free_disk_bytes=free_disk,
        load=os.getloadavg(),
        threads=caps["threads"],
        passed=available >= caps["minimum_available_bytes"]
        and free_disk >= caps["minimum_free_disk_bytes"],
        timing_claim=False,
    )
    parallel = caps.get("parallel")
    if parallel:
        cpus = len(os.sched_getaffinity(0))
        reservations = caps["reserved_bytes"]
        threads = caps["total_threads"]
        result.update(
            cpus=cpus,
            reserved_bytes=reservations,
            host_headroom_bytes=parallel["host_headroom_bytes"],
            service_memory_bytes=parallel["service_memory_bytes"],
            total_threads=threads,
            passed=result["passed"]
            and available >= reservations + parallel["host_headroom_bytes"]
            and reservations <= parallel["maximum_reserved_bytes"]
            and threads <= min(cpus, parallel["maximum_threads"])
            and result["load"][0] + threads <= cpus * parallel["maximum_load_per_cpu"],
        )
    return result


def archive_decision_sources(state, key):
    """Retain the exact controller source, independently of scientific inputs."""
    files = sorted(HERE.glob("*.py")) + [
        HERE / name
        for name in (
            "iteration_policy.json",
            "iteration_queue.json",
            "progress.json",
            "parallel_registry.json",
        )
        if (HERE / name).exists()
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = state / "iteration_decisions" / (key + "_sources.tar.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=p.name)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in files):
        raise RuntimeError("controller changed while archiving selected decision")
    return dict(
        files=hashes,
        archive=str(path),
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def run_process(command, *, cwd, env, log, timeout, on_start=None):
    """A timeout terminates the entire lane, including a spawned backend."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        if on_start:
            on_start(process.pid)
        return process.wait(timeout=timeout)
    except BaseException:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
        # A descendant may ignore SIGTERM even if its parent exited.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def execute_job(record, state, environment):
    record = dict(record)
    job = record["job"]
    job_path = state / "iteration_inputs" / (job["id"] + ".json")
    directory = REPO / record["output"]
    runner = (
        "run_registered_step.py"
        if job.get("kind") == "registered_probe"
        else "run_gap_step.py"
        if job.get("kind") in ("bloch_partition", "whole_transverse_control")
        else "run_iteration_step.py"
    )
    log_dir = state / "iteration_logs"
    log_dir.mkdir(exist_ok=True)
    try:
        with (log_dir / (job["id"] + ".log")).open("w") as log:
            record["process_started_at"] = now()
            exit_code = run_process(
                [
                    str(REPO / ".venv/bin/python"),
                    str(HERE / runner),
                    "--job",
                    str(job_path),
                    "--output",
                    str(directory),
                ],
                cwd=REPO,
                env=environment,
                log=log,
                timeout=record["caps"]["timeout_s"],
                on_start=lambda pid: record.update(
                    process_pid=pid, process_spawned_at=now()
                ),
            )
            record["process_finished_at"] = now()
        if exit_code:
            raise RuntimeError(
                f"scientific process exit {exit_code}; retained log and failure evidence"
            )
        result_path = directory / "result.json"
        data = json.loads(result_path.read_text())
        if job.get("kind") == "registered_probe" and (
            data["job"] != job
            or not data["execution_gate_pass"]
            or not data["sources_and_inputs_unchanged"]
            or data["highest_level_reached"]
        ):
            raise ValueError("registered result failed its execution identity contract")
        record.update(
            status="completed",
            result=data,
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        )
    except Exception as error:
        record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        record["process_finished_at"] = now()
    record["finished_at"] = now()
    # Distinct lanes write only their own ledger; batch status has one owner.
    save(state / "iteration_ledger" / (job["id"] + ".json"), record)
    return record


def recover_request(request, ledger):
    job_ids = request.get("job_ids") or [request["job_id"]]
    records = []
    for identifier in job_ids:
        path = ledger / (identifier + ".json")
        record = json.loads(path.read_text())
        if record["status"] == "running":
            record.update(
                status="interrupted",
                finished_at=now(),
                error="Previous worker stopped; durable claim retained and not replayed.",
            )
            save(path, record)
        records.append(record)
    request.update(
        status="completed"
        if all(r["status"] == "completed" for r in records)
        else "interrupted",
        finished_at=now(),
        jobs=records,
        reason="Recovered each durable lane independently; completed siblings retained, no silent replay.",
    )


def drain(config):
    state = Path(config["state_directory"])
    requests = state / "iteration_requests"
    ledger = state / "iteration_ledger"
    ledger.mkdir(exist_ok=True)
    jobs_raw = (HERE / "iteration_queue.json").read_bytes()
    jobs = json.loads(jobs_raw)["jobs"]
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    lib = "/home/zzhang/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = (
        lib
        + ":"
        + lib
        + "/openblas-serial"
        + (
            ":" + environment["LD_LIBRARY_PATH"]
            if environment.get("LD_LIBRARY_PATH")
            else ""
        )
    )
    for request_path in sorted(requests.glob("*.json")):
        request = json.loads(request_path.read_text())
        marker = request_path.with_suffix(".ready")
        if request["status"] == "running":
            recover_request(request, ledger)
            save(request_path, request)
            save(state / "iteration_status.json", request)
        if request["status"] != "queued":
            marker.unlink(missing_ok=True)
            continue
        decision = None
        if (HERE / "iteration_policy.json").exists():
            try:
                decision = plan(REPO, state, here=HERE)
                decision["execution_sources"] = archive_decision_sources(
                    state, request["key"]
                )
            except Exception as error:
                decision = dict(
                    status="waiting_for_evidence_repair",
                    job=None,
                    reason=type(error).__name__ + ": " + str(error),
                    highest_level_reached=False,
                )
            save(state / "iteration_decisions" / (request["key"] + ".json"), decision)
            selected_jobs = decision.get(
                "jobs", [decision["job"]] if decision["job"] else []
            )
        else:
            done_ids = {
                json.loads(p.read_text())["job_id"] for p in ledger.glob("*.json")
            }
            job = next((j for j in jobs if j["id"] not in done_ids), None)
            selected_jobs = [job] if job else []
        if not selected_jobs:
            request.update(
                status=decision["status"]
                if decision
                else "waiting_for_queue_expansion",
                finished_at=now(),
                reason=decision["reason"]
                if decision
                else "Declared scientific matrix exhausted; no automatic replay or highest-level claim.",
                decision=decision,
            )
            save(request_path, request)
            save(state / "iteration_status.json", request)
            marker.unlink(missing_ok=True)
            continue
        if len({job["id"] for job in selected_jobs}) != len(selected_jobs):
            raise ValueError("duplicate lane in scientific batch")
        for job in selected_jobs:
            if not re.fullmatch(r"[A-Za-z0-9_]+", job["id"]):
                raise ValueError("invalid scientific job identifier")
            if (ledger / (job["id"] + ".json")).exists():
                raise ValueError(
                    "selector attempted to overwrite a durable scientific claim"
                )
        execution_caps = dict(decision["caps"]) if decision else None
        if decision and "parallel_caps" in decision:
            execution_caps.update(
                parallel=decision["parallel_caps"],
                reserved_bytes=sum(
                    j.get("resources", {}).get("reserved_bytes", 4 * 1024**3)
                    for j in selected_jobs
                ),
                total_threads=len(selected_jobs),
            )
        preflight = resource_preflight(state, execution_caps) if decision else None
        if preflight and not preflight["passed"]:
            request.update(
                status="deferred_resources",
                finished_at=now(),
                reason="Resource preflight failed; no physics truncation or claimed execution.",
                decision=decision,
                preflight=preflight,
            )
            save(request_path, request)
            save(state / "iteration_status.json", request)
            marker.unlink(missing_ok=True)
            continue
        records = []
        for job in selected_jobs:
            directory = (
                REPO
                / "docs/atomic_pci_sd_native_2026_10_09/hourly_iterations"
                / job["id"]
            )
            job_path = state / "iteration_inputs" / (job["id"] + ".json")
            save(job_path, job)
            record = dict(
                job_id=job["id"],
                job=job,
                cycle=request["cycle"],
                request_key=request["key"],
                message_id=request["message_id"],
                status="running",
                started_at=now(),
                queue_sha256=hashlib.sha256(jobs_raw).hexdigest(),
                output=str(directory.relative_to(REPO)),
                caps=dict(
                    job.get(
                        "resources",
                        dict(
                            timeout_s=600,
                            reserved_bytes=4 * 1024**3,
                            address_bytes=4 * 1024**3,
                            threads=1,
                        ),
                    )
                ),
                highest_level_reached=False,
                decision=decision,
                preflight=preflight,
                preview_sha256=request.get("preview_sha256"),
            )
            save(ledger / (job["id"] + ".json"), record)
            records.append(record)
        # Claim the complete admitted batch before any child process starts.
        request.update(
            status="running",
            job_ids=[r["job_id"] for r in records],
            started_at=now(),
            decision=decision,
            preflight=preflight,
            jobs=records,
            highest_level_reached=False,
        )
        if len(records) == 1:
            request["job_id"] = records[0]["job_id"]
        save(request_path, request)
        save(state / "iteration_status.json", request)
        marker.unlink(missing_ok=True)
        with ThreadPoolExecutor(max_workers=len(records)) as pool:
            pending = {
                pool.submit(execute_job, r, state, environment): r["job_id"]
                for r in records
            }
            for future in as_completed(pending):
                completed = future.result()
                records = [
                    completed if r["job_id"] == completed["job_id"] else r
                    for r in records
                ]
                request.update(jobs=records)
                save(request_path, request)
                save(state / "iteration_status.json", request)
        statuses = [r["status"] for r in records]
        request.update(
            status="completed"
            if all(s == "completed" for s in statuses)
            else "failed"
            if all(s == "failed" for s in statuses)
            else "partially_completed",
            finished_at=now(),
        )
        if len(records) == 1:
            request.update(
                {k: records[0][k] for k in ("result", "error") if k in records[0]}
            )
        save(request_path, request)
        save(state / "iteration_status.json", request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    state = Path(config["state_directory"])
    with (state / "iteration.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        drain(config)


if __name__ == "__main__":
    main()
