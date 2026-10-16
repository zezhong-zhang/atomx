"""Root foreground guardian for one complete current31 core-correlation arm."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
MANIFEST_SHA = "d5b55ed4eb90a0865b140d903ebc8322de1fdd4a976cdf0edd1bb24e35d3da1b"
DRIVER_SHA = "7cce05dfe0e1347337f62639638ae1d2dc8ca808ce1b2f2525bf97e3f738ff41"
CONTRACT_SHA = "aebbe1c3526152623b9669393362946af8e0c16d94d3956fb088e0723a4cced2"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    manifest_path = HERE / "preparation_manifest.json"
    assert sha(manifest_path) == MANIFEST_SHA
    assert sha(HERE / "run.py") == DRIVER_SHA
    assert sha(HERE / "contract.json") == CONTRACT_SHA
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads((HERE / "contract.json").read_text())
    validation = json.loads((HERE / "source_validation.json").read_text())
    assert validation["driver_AST_compiles"]
    assert validation["ruff_full"] == "all_checks_passed"
    assert validation["formatting_successor"]["physical_code_and_gates_unchanged"]
    guards = {HERE / n: s for n, s in manifest["files_sha256"].items()}
    guards[manifest_path] = MANIFEST_SHA
    for name, row in contract["input_roles"].items():
        path = ROOT / row["path"]
        if row["deferred_numerical"]:
            assert (
                path.stat().st_size
                <= contract["array_access"][name]["compressed_bytes_max"]
            )
        guards[path] = row["sha256"]
    identity = json.loads(
        (ROOT / contract["input_roles"]["source_identity"]["path"]).read_text()
    )
    source = ROOT / contract["source_archive"]
    assert identity["source_map_sha256"] == contract["source_map_sha256"]
    assert len(identity["all_atomx_python_sha256"]) == 356
    guards.update(
        {
            source / n.removeprefix("src/"): s
            for n, s in identity["all_atomx_python_sha256"].items()
        }
    )
    parent = HERE.parent / "execution_preparation001"
    parent_manifest = parent / "preparation_manifest.json"
    assert not (parent / "actual").exists()
    guards[parent_manifest] = manifest["formatting_parent_preparation_manifest_sha256"]
    for n, s in json.loads(parent_manifest.read_text())["files_sha256"].items():
        guards[parent / n] = s
    assert all(sha(p) == s for p, s in guards.items())
    binding_path = HERE / "execution_binding.json"
    assert not binding_path.exists() and not (HERE / "actual").exists()
    binding = json.loads((HERE / "root_binding_template.json").read_text())
    assert (
        binding["driver_sha256"] == DRIVER_SHA
        and binding["contract_sha256"] == CONTRACT_SHA
    )
    assert binding["resources"] == contract["resources"]
    assert binding["input_roles"] == contract["input_roles"]
    binding.update(
        execution_authorized_by_root=True,
        pending_exact_root_execution_contract=False,
        preparation_manifest_sha256=MANIFEST_SHA,
        guardian_sha256=sha(__file__),
        root_review="Complete source/contract/resource and full-model review; SOURCE/default and full AST equivalence verified; one new retained native H/CI, exact embedded old1595 H before diagonalization, all complete residual/Gram/signed tracker gates; no optimizer or source adoption.",
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
        binding["literal_interpreter"],
        "-I",
        str(HERE / "run.py"),
        "--execution-binding",
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
            code = process.wait(timeout=contract["resources"]["outer_seconds"])
        except subprocess.TimeoutExpired:
            outer_stop = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    unchanged = all(sha(p) == s for p, s in guards.items())
    result, failure = HERE / "actual/result.json", HERE / "actual/failure.json"
    receipt = dict(
        status="foreground_closed",
        exit_code=code,
        outer_wall_stop=outer_stop,
        elapsed_uncontrolled_s=time.monotonic() - started,
        source_and_inputs_unchanged=unchanged,
        frozen_guard_files=len(guards),
        command=command,
        binding_sha256=sha(binding_path),
        guardian_sha256=sha(__file__),
        log_sha256=sha(log),
        result_sha256=sha(result) if result.exists() else None,
        failure_sha256=sha(failure) if failure.exists() else None,
        controlled_timing=False,
        baseline_H_rebuilt=False,
        stationarity_and_original_ordinary_FD_failures_preserved=True,
        model_pruned=False,
        automatic_retry=False,
    )
    (HERE / "root_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if code or not unchanged:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
