"""Once-only root foreground guardian for the unmodified saved-H diagnostic."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent
MANIFEST_SHA = "c5eaea9793c2f2e1c1e6faffe66d54b543b71668eca45d4f3fb1f46a0969d192"
DRIVER_SHA = "8b83e68ea74eab275b3ab3050d146c44804272dccc4e9771ff8f6142ff61cf37"
CONTRACT_SHA = "5cab2c6eebe9e030aaf76f9f62679a33f9f9188704217cac6dcfdce008e97c6e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest_path = HERE / "preparation_manifest.json"
    assert sha(manifest_path) == MANIFEST_SHA
    assert sha(HERE / "run.py") == DRIVER_SHA
    assert sha(HERE / "contract.json") == CONTRACT_SHA
    manifest, contract = read(manifest_path), read(HERE / "contract.json")
    validation = read(HERE / "source_validation.json")
    assert validation["full_Ruff"] == "PASS" and validation["whole_driver_AST_compiles"]
    assert validation["no_saved_H_mutation_targets"]
    guards = {HERE / n: s for n, s in manifest["files_sha256"].items()}
    guards[manifest_path] = MANIFEST_SHA
    for role, row in contract["input_roles"].items():
        path = ROOT / row["path"]
        if row["deferred_numerical"]:
            assert (
                path.stat().st_size
                <= contract["array_access"][role]["compressed_bytes_max"]
            )
        guards[path] = row["sha256"]
    identity = read(ROOT / contract["input_roles"]["source_identity"]["path"])
    source = ROOT / contract["source_archive"]
    assert identity["source_map_sha256"] == contract["source_map_sha256"]
    assert len(identity["all_atomx_python_sha256"]) == 356
    guards.update(
        {
            source / n.removeprefix("src/"): s
            for n, s in identity["all_atomx_python_sha256"].items()
        }
    )
    parent = ROOT / contract["parent_directory"]
    parent_manifest = read(parent / "preparation_manifest.json")
    guards.update({parent / n: s for n, s in parent_manifest["files_sha256"].items()})
    binding_path = HERE / "execution_binding.json"
    assert not binding_path.exists() and not (HERE / "actual").exists()
    binding = read(HERE / "root_binding_template.json")
    assert (
        binding["driver_sha256"] == DRIVER_SHA
        and binding["contract_sha256"] == CONTRACT_SHA
    )
    assert (
        binding["resources"] == contract["resources"]
        and binding["input_roles"] == contract["input_roles"]
    )
    assert (
        binding["original_byte_gate_remains_failed"] and not binding["new_H_authorized"]
    )
    assert not binding["retained_owner_fabrication"]
    # Root reviewed complete frozen driver/contract, original failure, signed
    # map/metric/reference identities, exact-ratio theorem and full block gates.
    binding.update(
        execution_authorized_by_root=True,
        pending_exact_root_execution_contract=False,
        preparation_manifest_sha256=MANIFEST_SHA,
        guardian_sha256=sha(__file__),
        root_review="Complete frozen source/contract/handoff review; unmodified saved2574 H, exactly five complete evd blocks, full E/C residual/Gram and original531 signed UID tracker; exact operand Fraction row sums with ordered-root-only theorem; zero new physics producers and no parent gate upgrade.",
    )
    assert all(sha(p) == expected for p, expected in guards.items())
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
    unchanged = all(sha(p) == expected for p, expected in guards.items())
    result, failure = HERE / "actual/result.json", HERE / "actual/failure.json"
    receipt = dict(
        status="foreground_closed",
        diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
        exit_code=code,
        outer_wall_stop=outer_stop,
        elapsed_uncontrolled_s=time.monotonic() - started,
        source_and_inputs_unchanged=unchanged,
        frozen_guard_files=len(guards),
        command=command,
        execution_binding_sha256=sha(binding_path),
        guardian_sha256=sha(__file__),
        log_sha256=sha(log),
        result_sha256=sha(result) if result.exists() else None,
        failure_sha256=sha(failure) if failure.exists() else None,
        controlled_timing=False,
        original_parent_byte_gate_remains_failed=True,
        source_or_model_qualified=False,
        new_physics_producers=0,
        automatic_retry=False,
    )
    (HERE / "root_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if code or not unchanged:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
