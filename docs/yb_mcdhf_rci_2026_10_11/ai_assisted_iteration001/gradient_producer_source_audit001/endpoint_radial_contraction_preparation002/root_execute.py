"""Run the once-only, root-reviewed complete endpoint producer discriminator."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    seal = read(HERE / "root_review_seal.json")
    frozen = seal["preparation_files_sha256"]
    assert seal["complete_source_and_contract_reviewed"] is True
    assert seal["guardian_sha256"] == sha(Path(__file__))
    assert all(sha(HERE / name) == value for name, value in frozen.items())
    c = read(HERE / "contract.json")
    prep = read(HERE / "preparation.json")
    assert prep["all_SOURCE_checks_passed"]
    assert all(row["passed"] for row in prep["checks"])
    caps = c["resources"]
    binding_path = HERE / "execution_binding.json"
    assert not binding_path.exists() and not (HERE / "actual").exists()
    binding = dict(
        execution_authorized_by_root=True,
        root_review=seal["scope"],
        root_review_seal_sha256=sha(HERE / "root_review_seal.json"),
        driver_sha256=frozen["run.py"],
        contract_sha256=frozen["contract.json"],
        preparation_sha256=frozen["preparation.json"],
        handoff_sha256=frozen["handoff.json"],
        guardian_sha256=sha(Path(__file__)),
        input_roles=c["input_roles"],
        resources=caps,
        once_only=True,
    )
    write(binding_path, binding)
    guards = {HERE / name: value for name, value in frozen.items()}
    guards.update({HERE / "root_review_seal.json": sha(HERE / "root_review_seal.json"),
                   Path(__file__): sha(Path(__file__)), binding_path: sha(binding_path)})
    guards.update({ROOT / row["path"]: row["sha256"] for row in c["input_roles"].values()})
    identity = read(ROOT / c["input_roles"]["source_identity"]["path"])
    source = ROOT / c["source_archive"]
    guards.update({source / name.removeprefix("src/"): value
                   for name, value in identity["all_atomx_python_sha256"].items()})
    assert len(identity["all_atomx_python_sha256"]) == 356
    assert all(sha(path) == value for path, value in guards.items())
    env = os.environ.copy()
    for name in c["thread_environment"]:
        env[name] = "1"
    command = [caps["interpreter_literal"], "-I", str(HERE / "run.py"),
               "--execute", str(binding_path)]
    started = time.monotonic()
    wall_stop = False
    log = HERE / "root_execution.log"
    with log.open("xb") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 stdout=stream, stderr=subprocess.STDOUT)
        try:
            exit_code = child.wait(timeout=caps["outer_seconds"])
        except subprocess.TimeoutExpired:
            wall_stop = True
            os.killpg(child.pid, signal.SIGKILL)
            exit_code = child.wait()
    unchanged = all(sha(path) == value for path, value in guards.items())
    result, failure = HERE / "actual/result.json", HERE / "actual/failure.json"
    receipt = dict(
        status="foreground_closed", exit_code=exit_code, outer_wall_stop=wall_stop,
        elapsed_uncontrolled_s=time.monotonic() - started,
        source_and_inputs_unchanged=unchanged, frozen_guard_files=len(guards),
        command=command, guardian_sha256=sha(Path(__file__)),
        root_review_seal_sha256=sha(HERE / "root_review_seal.json"),
        execution_binding_sha256=sha(binding_path), execution_log_sha256=sha(log),
        result_sha256=sha(result) if result.exists() else None,
        failure_sha256=sha(failure) if failure.exists() else None,
        controlled_timing=False, ordinary_FD_failures_preserved=True,
        physical_stationarity_and_experimental_gates_upgraded=False,
    )
    write(HERE / "root_execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    if exit_code or not unchanged:
        raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
