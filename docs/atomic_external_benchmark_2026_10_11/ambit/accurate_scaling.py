"""Retained measurement refinement: fixed native BeII, monotonic child wall."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import shutil

import numpy as np

import benchmark as original

HERE = Path(__file__).resolve().parent


def main():
    previous = json.loads((HERE / "result.json").read_text())
    amendment = dict(
        fixed_scientific_work="Original full 4000-point BeII HF/basis/MBPT/CI; same executable and exact inputs/outputs",
        measurement_change="C++ monotonic fork/exec/wait4 timer excludes Python timeout wait polling latency",
        retained_previous="result.json and every be_pair* v002 execution/output remain unchanged",
        gates="Original contract precision/load/600s/8GiB gates unchanged",
        source_sha256=original.digest(HERE / "child_timer.cpp"),
        created_at=original.stamp(),
    )
    (HERE / "timer_amendment.json").write_text(json.dumps(amendment, indent=2) + "\n")
    with original.LOCK.open("a") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX)
        build = original.execute(
            ["/usr/bin/c++", "-O2", "-std=c++17", str(HERE / "child_timer.cpp"),
             "-o", str(HERE / "child_timer")], HERE, "child_timer_build",
        )
    results = {}
    input_file = original.ROOT / "docs/ci_fscc_reference_upgrade_2026_10_09/ambit_be_smoke_refined/BeII.input"
    golden = previous["ambit_native_scaling"]["results"]["2"]["pairs"][0]["complete_output_sha256"]["1"]
    for workers in (2, 4, 8):
        pairs = []
        for round_id in range(5):
            order = (1, workers) if round_id % 2 == 0 else (workers, 1)
            records, timings, outputs = {}, {}, {}
            with original.LOCK.open("a") as locked:
                fcntl.flock(locked, fcntl.LOCK_EX)
                for worker in order:
                    directory = HERE / f"be_v003_pair{workers}_round{round_id:02d}_threads{worker}"
                    directory.mkdir()
                    shutil.copy2(input_file, directory / "BeII.input")
                    records[worker] = original.execute(
                        [str(HERE / "child_timer"), str(directory / "child_time.json"),
                         str(original.BUILD / "ambit"), "BeII.input"],
                        directory, "ambit", worker,
                    )
                    timings[worker] = json.loads((directory / "child_time.json").read_text())
                    outputs[worker] = {path.name: original.digest(path) for path in
                                       sorted(directory.glob("BeII_native_reference_0.*"))}
                    text = (directory / "ambit.stdout").read_text()
                    energies = [float(value) for value in re.findall(r"([-0-9.e+]+) /cm", text)]
                    np.savez_compressed(directory / "complete_levels_cm.npz", energies_cm=energies)
                    if len(energies) != 6 or "Finished" not in text:
                        raise RuntimeError("incomplete original native panel")
            gate = outputs[1] == golden and outputs[workers] == golden
            pairs.append(dict(
                round=round_id, order=order, executions=records, child_timings=timings,
                complete_output_sha256=outputs, scientific_gate=gate,
                controlled_timing=all(record["controlled_timing"] for record in records.values()),
                serial_seconds=timings[1]["seconds"], parallel_seconds=timings[workers]["seconds"],
            ))
            print("accurate_native", workers, round_id, pairs[-1]["serial_seconds"],
                  pairs[-1]["parallel_seconds"], gate, flush=True)
        stats = original.pair_statistics(pairs, "serial_seconds", "parallel_seconds")
        stats["parallel_efficiency"] = stats["speedup"] / workers if stats["speedup"] is not None else None
        results[str(workers)] = dict(pairs=pairs, statistics=stats)
    report = dict(
        created_at=original.stamp(), source_identity_same_as_v002=True,
        timer=dict(source_sha256=original.digest(HERE / "child_timer.cpp"),
                   binary_sha256=original.digest(HERE / "child_timer"),
                   script_sha256=original.digest(Path(__file__)), build=build),
        input_sha256=original.digest(input_file), reference_golden_complete_output_sha256=golden,
        scopes="Full native AMBiT child wall from fork/exec to wait4, including CLI I/O and startup; fresh scientific workspace each invocation; OS/shared-library caches warm. No Python wait polling; no cross-software physics ranking.",
        results=results,
    )
    (HERE / "accurate_scaling_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["statistics"] for key, value in results.items()}), flush=True)


if __name__ == "__main__":
    main()
