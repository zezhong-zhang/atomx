"""Root guardian for one immutable same-point DC/static/zero RCI run."""
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    frozen = {
        "run.py": "f915725fce0b4fbeeb179256270265c382619c2d2d5075497124c527a1e0efc8",
        "contract.json": "5c668e3f38a97a60d77d42a33a5cbdacac410919506534aa8030116b570412dc",
        "preparation.json": "d429c97c29594a8647289ccbdc914cec94efec975bbb269505631e079d3b84ee",
        "handoff.json": "0b27933295f71993975deed3551d52c4e15fd45cf7dc0acbff1c1ed56ce45f65",
    }
    for name, expected in frozen.items():
        if sha(HERE / name) != expected:
            raise ValueError("Prepared executor changed: " + name)
    contract = json.loads((HERE / "contract.json").read_text())
    inputs = {REPO / r["path"]: r["sha256"] for r in contract["input_roles"].values()}
    for name, digest in frozen.items():
        inputs[HERE / name] = digest
    identity = json.loads((REPO / contract["source_identity"]["path"]).read_text())
    for name, digest in identity["all_source_sha256"].items():
        inputs[REPO / contract["source_archive"] / name.removeprefix("src/")] = digest
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError("Frozen input/source changed")
    if (HERE / "actual").exists():
        raise ValueError("Existing actual attempt; no automatic retry")
    seal = HERE / "execution_binding.json"
    binding = dict(execution_authorized_by_root=True,
                   contract_sha256=frozen["contract.json"], driver_sha256=frozen["run.py"],
                   preparation_sha256=frozen["preparation.json"],
                   source_map_sha256=contract["source_map_sha256"],
                   input_roles=contract["input_roles"], resources=contract["resources"],
                   guardian_sha256=sha(__file__), no_cards=True,
                   fixed_RCI_only=True, no_MCDHF_continuation=True)
    write(seal, binding)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    started, resource_stop = time.monotonic(), None
    with (HERE / "root_execution.log").open("xb") as output:
        child = subprocess.Popen(["/data/code/atomx/.venv/bin/python", "-I", str(HERE / "run.py"),
                                  "--execute", str(seal)], cwd=REPO, env=env,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=contract["resources"]["outer_seconds"])
        except subprocess.TimeoutExpired:
            resource_stop = "outer_1215s_wall_cap"
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=5)
    unchanged = all(sha(path) == digest for path, digest in inputs.items())
    result, failure = HERE / "actual" / "result.json", HERE / "actual" / "failure.json"
    receipt = dict(status="foreground_closed", exit_code=code, resource_stop=resource_stop,
                   elapsed_uncontrolled_s=time.monotonic() - started,
                   source_and_inputs_unchanged=unchanged,
                   execution_binding_sha256=sha(seal), guardian_sha256=sha(__file__),
                   execution_log_sha256=sha(HERE / "root_execution.log"),
                   result_sha256=sha(result) if result.exists() else None,
                   failure_sha256=sha(failure) if failure.exists() else None,
                   controlled_timing=False, current_MCDHF_optimizer_qualified=False)
    write(HERE / "root_execution.json", receipt)
    print(json.dumps(receipt))
    if code != 0 or resource_stop is not None or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
