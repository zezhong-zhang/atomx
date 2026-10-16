"""One bounded foreground metadata build/test receipt, without scientific imports."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_bytes())
    for name, expected in contract["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Frozen metadata consumer changed")
    output = HERE / "actual"
    output.mkdir(exist_ok=True)
    with (output / "once.json").open("x") as stream:
        json.dump({"contract_sha256": sha(args.contract), "scientific_execution": False}, stream)
    started = time.monotonic()
    os.sched_setaffinity(0, {31})
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    python = "/data/code/atomx/.venv/bin/python"
    commands = [
        ("manifest", [python, str(HERE / "make_manifest.py")]),
        ("tests", [python, "-m", "unittest", "-v", "test_dataset_builder"]),
        ("builder", [python, str(HERE / "dataset_builder.py"), "--repo", str(REPO),
                     "--manifest", str(HERE / "manifest.json"), "--output", str(HERE / "dataset.json")]),
    ]
    records = []
    for phase, command in commands:
        remaining = contract["whole_wall_cap_s"] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Metadata whole wall budget exceeded")
        phase_start = time.monotonic()
        with (output / (phase + ".stdout")).open("wb") as stdout, (output / (phase + ".stderr")).open("wb") as stderr:
            completed = subprocess.run(command, cwd=HERE, env=environment,
                stdout=stdout, stderr=stderr, timeout=remaining, check=False)
        records.append({"phase": phase, "command": command, "exit_code": completed.returncode,
                        "elapsed_s": time.monotonic() - phase_start,
                        "stdout_sha256": sha(output / (phase + ".stdout")),
                        "stderr_sha256": sha(output / (phase + ".stderr"))})
        (output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        if completed.returncode:
            (output / "failure.json").write_text(json.dumps({"phase": phase, "records": records,
                "scientific_calls": 0, "no_retry": True}, indent=2) + "\n")
            raise RuntimeError("Metadata phase failed; evidence preserved")
    dataset = json.loads((HERE / "dataset.json").read_bytes())
    for relative, expected in dataset["inputs_sha256"].items():
        if Path(relative).suffix != ".json" or sha(REPO / relative) != expected:
            raise ValueError("Closed JSON input changed during dataset build")
    if any(sha(HERE / name) != expected for name, expected in contract["source_sha256"].items()):
        raise ValueError("Frozen source changed during metadata run")
    result = {"status": "metadata_build_and_tests_foreground_closed", "exit_code": 0,
        "all_metadata_commands_passed": True, "records": records, "summary": dataset["summary"],
        "metadata_IO": dataset["metadata_IO"], "inputs_sha256": dataset["inputs_sha256"],
        "source_sha256": contract["source_sha256"], "contract_sha256": sha(args.contract),
        "manifest_sha256": sha(HERE / "manifest.json"), "dataset_sha256": sha(HERE / "dataset.json"),
        "actual_affinity": sorted(os.sched_getaffinity(0)), "address_space_cap_bytes": 1024**3,
        "elapsed_uncontrolled_s": time.monotonic() - started,
        "scientific_calls_or_NPZ_reads_hashes": 0, "learned_prediction_or_performance_claim": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result_sha256": sha(output / "result.json"), "summary": dataset["summary"]}))


if __name__ == "__main__":
    main()
