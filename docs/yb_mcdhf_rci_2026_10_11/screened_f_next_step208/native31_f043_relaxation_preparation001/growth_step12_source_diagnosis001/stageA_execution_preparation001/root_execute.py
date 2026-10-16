"""One bounded foreground execution of the frozen offline secant replay."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    frozen = {
        "run.py": "80540c01c0709b1a4cdfe6cd9292e6ccad3f7373e3e2fd29fdd0560833663967",
        "contract.json": "4a40f8ad3037a1ff5663ff43723b93cde4af5d5c9871e4d9cde538a655ff9de6",
        "preparation.json": "9904da0597e6cea9da18aa8cf64e3a251bf6836f77163eb3feb2d1d2a6895613",
        "handoff.json": "7b2df118ae647ca57591dd94927c839809fd5299205832a0366d75ebb3d1da17",
    }
    contract = json.loads((HERE / "contract.json").read_text())
    inputs = {REPO / r["path"]: r["sha256"] for r in contract["metadata_inputs"].values()}
    for name, digest in frozen.items():
        inputs[HERE / name] = digest
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError("Frozen source or metadata changed before execution")
    if (HERE / "actual").exists():
        raise ValueError("An actual attempt already exists; no automatic retry")
    binding = dict(execution_authorized_by_root=True,
                   driver_sha256=frozen["run.py"], contract_sha256=frozen["contract.json"],
                   preparation_sha256=frozen["preparation.json"], CPU=31, threads=1,
                   AS_GiB=1, inner_seconds=30, outer_seconds=45,
                   max_small_Gram_eigh=7, max_NPZ_loads=14,
                   max_H=0, max_CI=0, max_gradient=0, max_Dirac=0, max_optimizer=0,
                   guardian_sha256=sha(__file__), scope="offline chronological history and endpoint only",
                   expected_controls="unchanged native metric, gradients, slopes and literal point13",
                   no_cards=True, no_stageB=True)
    seal = HERE / "execution_binding.json"
    write(seal, binding)
    env = os.environ.copy()
    for name in contract["resources"]["thread_environment"]:
        env[name] = "1"
    started = time.monotonic()
    stop = None
    with (HERE / "root_execution.log").open("xb") as output:
        child = subprocess.Popen([
            "/data/code/atomx/.venv/bin/python", "-I", str(HERE / "run.py"),
            "--execution-binding", str(seal)], cwd=REPO, env=env,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            stop = "outer_45s_wall_cap"
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=2)
    unchanged = all(sha(path) == digest for path, digest in inputs.items())
    result = HERE / "actual" / "result.json"
    receipt = dict(status="foreground_closed", exit_code=code, resource_stop=stop,
                   elapsed_uncontrolled_s=time.monotonic() - started,
                   source_and_metadata_unchanged=unchanged,
                   execution_binding_sha256=sha(seal), guardian_sha256=sha(__file__),
                   execution_log_sha256=sha(HERE / "root_execution.log"),
                   result_sha256=sha(result) if result.exists() else None,
                   physical_or_stationarity_certificate=False)
    write(HERE / "root_execution.json", receipt)
    print(json.dumps(receipt))
    if code != 0 or stop is not None or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
