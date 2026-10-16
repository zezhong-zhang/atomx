"""Run a reviewed source-bound probe in its own output and retained contract."""

import argparse
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from parallel_planner import bound_file, registered_job, sha, validate_registry

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_bytes())
    registry = json.loads((HERE / "parallel_registry.json").read_bytes())
    validate_registry(registry)
    task = next(
        t for t in registry["tasks"] if t["name"] == job["lane"] and t["enabled"]
    )
    if job != registered_job(REPO, HERE, task):
        raise ValueError("registered source/input/job changed after selection")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {**job["sources"], **job["inputs"]}
    contract = dict(
        job=job,
        hypothesis=task["hypothesis"],
        fixed=task["fixed"],
        distinguishing_outcomes=task["distinguishing_outcomes"],
        scientific_gates=task["scientific_gates"],
        stopping_condition="One bounded reviewed action, preserve failures and original physical spaces/gates.",
        resources=job["resources"],
        source_and_input_sha256=files,
        load=os.getloadavg(),
        highest_level_reached=False,
    )
    (args.output / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2)
    )
    with tarfile.open(args.output / "sources_and_inputs.tar.gz", "w:gz") as archive:
        for relative in files:
            archive.add(bound_file(REPO, relative), arcname=relative)
        archive.add(args.job, arcname="selected_job.json")
        archive.add(Path(__file__), arcname="run_registered_step.py")
    if any(sha(bound_file(REPO, p)) != value for p, value in files.items()):
        raise ValueError("registered source/input changed while archiving")
    memory = job["resources"]["address_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    started = time.monotonic()
    probe_output = args.output / "probe"
    command = [
        sys.executable,
        str(bound_file(REPO, job["entry_point"])),
        *job["arguments"],
        "--output",
        str(probe_output),
    ]
    # This process and its probe share the worker's isolated process group.
    with (args.output / "probe.log").open("w") as log:
        process = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode:
        raise RuntimeError(
            f"registered probe exit {process.returncode}; retained raw output"
        )
    result_path = probe_output / "result.json"
    probe = json.loads(result_path.read_bytes())
    if probe["highest_level_reached"] or not isinstance(probe["gates"], dict):
        raise ValueError("probe result lacks explicit scientific gates/scope")
    unchanged = all(sha(bound_file(REPO, p)) == value for p, value in files.items())
    result = dict(
        job=job,
        execution_gate_pass=unchanged,
        sources_and_inputs_unchanged=unchanged,
        probe_result_sha256=sha(result_path),
        probe_outputs_sha256={
            str(p.relative_to(probe_output)): sha(p)
            for p in sorted(probe_output.rglob("*"))
            if p.is_file()
        },
        scientific_result=probe,
        observed_wall_seconds=time.monotonic() - started,
        timing_scope="Reviewed probe subprocess after controller archive; not end-to-end/GPU performance certification",
        highest_level_reached=False,
        card_summary=probe["card_summary"],
    )
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    if not unchanged:
        raise RuntimeError(
            "registered source/input changed during execution; no valid completion"
        )


if __name__ == "__main__":
    main()
