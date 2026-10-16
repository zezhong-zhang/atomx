"""Root guardian: one reviewed, complete saved-array contraction replay."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
FROZEN = {
    "run.py": "a0449492bf286a9444b0d38019669e81929a3ca5a6b5576632fac5fff21bea16",
    "contract.json": "47441b29c2197a3b48cf6268d50e55769f394ea5a5b5554e929f6786403616a6",
    "preparation.json": "1caaab7bb488c0a9d134dcd0efa8bb9c9cd5617eb3f5146a986e91ce8ffa1743",
    "handoff.json": "1106c4f16a249aef6c9168897d4f26e52e455a09593b4df12cbbcf29e57c0b42",
    "verify_preparation.py": "bc18a2035d14da5a2e5d99440e8aec9fb7ee48f12f2446b10dcc1c29023a70bc",
    "README.md": "75879bf33eb8b0925a7fe665e3264eec7079e82994e168db567bb9c21eca0357",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert all(sha(HERE / n) == s for n, s in FROZEN.items())
    contract = json.loads((HERE / "contract.json").read_text())
    prep = json.loads((HERE / "preparation.json").read_text())
    handoff = json.loads((HERE / "handoff.json").read_text())
    assert prep["all_SOURCE_checks_passed"] and all(c["passed"] for c in prep["checks"])
    assert handoff["resources"] == contract["resources"]
    guards = {HERE / n: s for n, s in FROZEN.items()}
    guards.update(
        {ROOT / r["path"]: r["sha256"] for r in contract["input_roles"].values()}
    )
    identity = json.loads(
        (ROOT / contract["input_roles"]["source_identity"]["path"]).read_text()
    )
    source = ROOT / contract["source_archive"]
    guards.update(
        {
            source / n.removeprefix("src/"): s
            for n, s in identity["all_atomx_python_sha256"].items()
        }
    )
    assert len(identity["all_atomx_python_sha256"]) == 356
    assert all(sha(p) == s for p, s in guards.items())
    binding_path = HERE / "execution_binding.json"
    assert not binding_path.exists() and not (HERE / "actual").exists()
    binding = dict(
        execution_authorized_by_root=True,
        root_review="Complete source/contract read; full1595 arrays, all5 UID phases and actual norms, original residual/Gram gates and wide signed response replay; no new physical calls or forced zero hypothesis.",
        driver_sha256=FROZEN["run.py"],
        contract_sha256=FROZEN["contract.json"],
        preparation_sha256=FROZEN["preparation.json"],
        handoff_sha256=FROZEN["handoff.json"],
        input_roles=contract["input_roles"],
        resources=contract["resources"],
        guardian_sha256=sha(__file__),
        once_only=True,
    )
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    guards[binding_path] = sha(binding_path)
    guards[Path(__file__)] = sha(__file__)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    command = [
        contract["resources"]["interpreter_literal"],
        "-I",
        str(HERE / "run.py"),
        "--execute",
        str(binding_path),
    ]
    log = HERE / "root_execution.log"
    started = time.monotonic()
    outer_stop = False
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            start_new_session=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            exit_code = process.wait(timeout=contract["resources"]["outer_seconds"])
        except subprocess.TimeoutExpired:
            outer_stop = True
            os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait()
    unchanged = all(sha(p) == s for p, s in guards.items())
    result, failure = HERE / "actual/result.json", HERE / "actual/failure.json"
    receipt = dict(
        status="foreground_closed",
        exit_code=exit_code,
        outer_wall_stop=outer_stop,
        elapsed_uncontrolled_s=time.monotonic() - started,
        source_and_inputs_unchanged=unchanged,
        frozen_guard_files=len(guards),
        command=command,
        execution_binding_sha256=sha(binding_path),
        guardian_sha256=sha(__file__),
        execution_log_sha256=sha(log),
        result_sha256=sha(result) if result.exists() else None,
        failure_sha256=sha(failure) if failure.exists() else None,
        controlled_timing=False,
        ordinary_FD_failures_preserved=True,
        new_physics_calls=0,
        full31_MCDHF_qualified=False,
    )
    (HERE / "root_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if exit_code or not unchanged:
        raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
