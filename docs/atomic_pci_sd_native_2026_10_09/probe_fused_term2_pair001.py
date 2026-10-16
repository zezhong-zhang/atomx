"""Controlled complete frozen 6s term2 GPU pairs, including all transfers.

The original CPU arrays independently anchor every measured output. This is
one complete stage, not a fresh COMMON, full SD map, solve or spectrum timing.
External compute contexts or the original load guard invalidate timing.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from probe_batched_projection001 import FrozenFused, FrozenNative

from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_guard():
    report = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pids = sorted({int(line.strip()) for line in report.splitlines() if line.strip()})
    return dict(
        load=list(os.getloadavg()),
        other_compute_pids=[pid for pid in pids if pid != os.getpid()],
        this_pid=os.getpid(),
    )


def guard_passes(guard):
    return max(guard["load"]) <= 27 and not guard["other_compute_pids"]


def main(run_name):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    parent = OWNER / "valence_term2_cuda009"
    plan_path = parent / "cuda_full_plan_6_-1.json"
    field_path = parent / "cuda_full_fields_6_-1.npz"
    library = parent / "backend/native_core_sd.so"
    reference = parent / "term2_6_-1.npz"
    physical = OWNER / "yb_full_run008/yb_full_inputs.npz"
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((parent / "backend/manifest.json").read_text())
    dims = manifest["dimensions"]
    shape = (dims["NXX"], dims["NXX"], manifest["valence_sd"]["nvh"])
    host_bound = int(
        8 * (12 * np.prod(shape) + 4 * dims["NHF"] * dims["NX"] * dims["NK"])
    )
    original_resource = json.loads((parent / "resource.json").read_text())
    fused_proof = json.loads((OWNER / "batched_projection003/result.json").read_text())
    if (
        digest(library) != plan["library_sha256"]
        or digest(physical) != original_resource["input_identity"]["input_sha256"]
        or plan["state"] != [6, -1]
        or fused_proof["status"] != "pass"
        or fused_proof["shape"] != list(shape)
        or fused_proof["angular_rows"] != plan["plan_rows"]
    ):
        raise ValueError("complete original physical/library/validated shape differs")
    fused_resource = json.loads(
        (OWNER / "batched_projection003/batched/fused_angular.json").read_text()
    )
    combined_bound = (
        plan["device_bound_bytes"] + fused_resource["full_device_payload_bound_bytes"]
    )
    contract = dict(
        hypothesis="exact reach plus disjoint angular fusion reduces warm complete GPU term2 time at unchanged full signed value/response/transfer gates",
        fixed="complete archived68core414virtual/ns123/Fermi/l<=3/5Ha/6s/22538 native angular rows, same complete numerator/prime/energy and FP64",
        gate_Ha=1e-11,
        host_cap_bytes=32 * 1024**3,
        host_payload_bound_bytes=host_bound,
        device_cap_bytes=2 * 1024**3,
        combined_device_payload_bound_bytes=combined_bound,
        load_guard=27,
        device_guard="GPU0 no compute context except this process before and after every pair invocation",
        timing="complete warm term2 including CPU validation and all host/device input/output copies; signed comparison outside timer; both full-response and value-only policies",
        pair_order=["original,fused", "fused,original", "original,fused"],
        stop="first signed/finite/source/cap failure rejects; failed load/device guard retains correctness and records timing rejection without claiming a ratio",
        scope="frozen complete stage; no new COMMON/full-map/solve/physical-spectrum/end-to-end/public adoption claim",
    )
    save("contract.json", contract)
    if (
        host_bound > contract["host_cap_bytes"]
        or combined_bound > contract["device_cap_bytes"]
    ):
        raise MemoryError("complete paired payload exceeds original caps")
    paths = [
        Path(__file__),
        *(
            OWNER / n
            for n in (
                "probe_batched_projection001.py",
                "cuda_fused_angular_control.py",
                "cuda_batched_projection_control.py",
                "cuda_full_term2_control.py",
                "cuda_valence_bridge.py",
                "cuda_density_control.py",
            )
        ),
        ROOT / "src/atomx/numerics/bspline/coulomb_features.py",
    ]
    hashes = {}
    for path in paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    save("source_sha256.json", hashes)
    proof_sources = json.loads(
        (OWNER / "batched_projection003/source_sha256.json").read_text()
    )
    for name in (
        "probe_batched_projection001.py",
        "cuda_density_control.py",
        "cuda_fused_angular_control.py",
        "cuda_batched_projection_control.py",
        "cuda_full_term2_control.py",
    ):
        if (
            hashes[str((OWNER / name).relative_to(ROOT))]
            != proof_sources[str((OWNER / name).relative_to(ROOT))]
        ):
            raise ValueError("validated original/fused source differs")
    proof_inputs = json.loads(
        (OWNER / "batched_projection003/input_sha256.json").read_text()
    )
    for path in (library, field_path, reference, physical, plan_path):
        if digest(path) != proof_inputs[str(path.relative_to(ROOT))]:
            raise ValueError("complete signed-proof physical input differs")
    save(
        "input_binding.json",
        {
            str(f.relative_to(ROOT)): digest(f)
            for f in (
                library,
                field_path,
                reference,
                physical,
                plan_path,
                OWNER / "batched_projection003/result.json",
                OWNER / "batched_projection003/batched/fused_angular.json",
            )
        },
    )
    with np.load(physical, allow_pickle=False) as data:
        workspace = {
            n.removeprefix("workspace_"): data[n].copy()
            for n in data.files
            if n.startswith("workspace_")
        }
    workspace = {n: v.item() if v.ndim == 0 else v for n, v in workspace.items()}
    feature = PolynomialCellCoulombFeatures(workspace)
    with np.load(reference, allow_pickle=False) as data:
        numerator, prime, expected, expected_prime = (
            np.array(data[n], dtype=np.float64, order="F", copy=True)
            for n in ("numerator", "prime", "original", "original_prime")
        )
    if any(a.shape != shape for a in (numerator, prime, expected, expected_prime)):
        raise ValueError("complete original padded arrays differ")
    energy = fused_proof["energy_Ha"]
    controls = {}
    for label, cls in (("original", FrozenNative), ("fused", FrozenFused)):
        directory = out / label
        directory.mkdir()
        control = cls(
            feature,
            dims,
            directory,
            SimpleNamespace(_name=str(library)),
            memory_bytes=contract["device_cap_bytes"],
            **({"projection_policy": "reachable"} if label == "fused" else {}),
        )
        control.field_path, control.field_plan = field_path, plan
        controls[label] = control
    cp = controls["fused"].cp
    records, warm_outputs = [], {}

    def invoke(label, response, phase, pair):
        value, derivative = (np.zeros(shape, order="F") for _ in range(2))
        before = device_guard()
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        controls[label].compute(
            energy, numerator, prime, value, derivative, response=response
        )
        cp.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter() - start
        after = device_guard()
        errors = [float(np.max(abs(value - expected)))]
        if response:
            errors.append(float(np.max(abs(derivative - expected_prime))))
        if max(errors) > contract["gate_Ha"] or not all(
            np.all(np.isfinite(a)) for a in (value, derivative)
        ):
            np.savez_compressed(
                out / "failed_outputs.npz", value=value, derivative=derivative
            )
            raise AssertionError("complete originalCPU signed paired output failed")
        row = dict(
            variant=label,
            response=response,
            phase=phase,
            pair=pair,
            seconds=seconds,
            signed_errors_Ha=errors,
            before=before,
            after=after,
            timing_guard_passed=guard_passes(before) and guard_passes(after),
        )
        records.append(row)
        save("observations.json", records)
        if phase == "warmup":
            warm_outputs[f"{label}_{response}_value"] = value
            warm_outputs[f"{label}_{response}_prime"] = derivative
        return row

    for response in (True, False):
        for label in controls:
            invoke(label, response, "warmup", None)
    np.savez_compressed(out / "warm_signed_outputs.npz", **warm_outputs)
    warm_outputs.clear()
    if all(r["timing_guard_passed"] for r in records):
        for response in (True, False):
            for pair, order in enumerate(
                (("original", "fused"), ("fused", "original"), ("original", "fused")), 1
            ):
                for label in order:
                    invoke(label, response, "pair", pair)
    accepted = len(records) == 16 and all(r["timing_guard_passed"] for r in records)
    ratios = {}
    if accepted:
        for response in (True, False):
            values = []
            for pair in (1, 2, 3):
                match = {
                    r["variant"]: r["seconds"]
                    for r in records
                    if r["phase"] == "pair"
                    and r["response"] == response
                    and r["pair"] == pair
                }
                values.append(match["original"] / match["fused"])
            ratios["full_response" if response else "value_only"] = dict(
                original_over_fused=values, median=float(np.median(values))
            )
    save(
        "result.json",
        dict(
            status="complete signed outputs passed; controlled pairs passed"
            if accepted
            else "complete signed outputs passed; controlled timing rejected by declared guards",
            controlled_stage_speedup=ratios if accepted else None,
            measurements=records,
            scope=contract["scope"],
            max_signed_error_Ha=max(max(r["signed_errors_Ha"]) for r in records),
            public_adopted=False,
            highest_level_achieved=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
    except BaseException as error:
        (OWNER / args.run_name / "failure.json").write_text(
            json.dumps(dict(type=type(error).__name__, error=str(error)), indent=2)
            + "\n"
        )
        raise
