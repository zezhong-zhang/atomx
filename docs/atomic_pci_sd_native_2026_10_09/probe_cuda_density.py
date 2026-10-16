"""Live complete GPU weighted-density block and FP64 GEMM control."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]
sys.path.append(str(OWNER / "cuda_python_runtime"))
from cuda_density_control import CUDADensityControl  # noqa: E402

from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run_name):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    contract = dict(
        hypothesis="Resident batched float64 cell Coulomb actions and GEMM can use available GPU without dropping tagged nodes, incident edges or signed interference",
        fixed="Actual fullYb008 V68/ns123/Fermi/radial workspace and all59s virtuals from original414 pool; full3910 tagged coordinates, all3481 directed s-pair densities and ranks0..7",
        scope="one complete term2 radial subblock/control, not full414all-partial term2/SD or spectra; no FP32/TF32/quad GPU claim",
        kernel="shared immutable CPU feature geometry/weights, exact ordered per-node incident edge sums, original two orientations including diagonal; explicit --fmad=false",
        raw_signed_gate_Ha=1e-11,
        device_cap_bytes=512 * 1024**2,
        host_cap_bytes=2 * 1024**3,
        timing="three alternating warm complete CPU/GPU density pairs at rank0 after full0..7correctness; both GPU-resident synchronized kernel and host-to-device/action/device-to-host timings retained; includes complete C copies, no end-to-end SD claim",
        load_guard=24,
        stop="Any actual driver/precision/model/count/cap/finite/raw gate failure rejects; failed load guards invalidate timing only; no automatic public adoption",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    paths = [
        Path(__file__),
        OWNER / "cuda_density_control.py",
        ROOT / "src/atomx/numerics/bspline/coulomb_features.py",
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    import cupy as cp

    cuda_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )
    device = cp.cuda.Device(0)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    runtime = dict(
        cupy_version=cp.__version__,
        numpy_version=np.__version__,
        numpy_path=np.__file__,
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        device_name=properties["name"].decode(),
        device_capability=device.compute_capability,
        actual_device_total_bytes=int(properties["totalGlobalMem"]),
        device_report=cuda_info,
        packages={},
    )
    for name in (
        "cupy-cuda13x",
        "cuda-toolkit",
        "nvidia-cublas",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-nvjitlink",
    ):
        runtime["packages"][name] = importlib.metadata.version(name)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    # Limit only this experiment's CuPy pool; do not free another user's data.
    cp.get_default_memory_pool().set_limit(size=contract["device_cap_bytes"])
    parent = OWNER / "yb_full_run008"
    plan = json.loads((parent / "yb_full_plan.json").read_text())
    keys = tuple(tuple(key) for key in plan["virtual_keys"] if key[1] == -1)
    if len(keys) != 59 or len(plan["virtual_keys"]) != 414:
        raise AssertionError("original complete radial block/pool identity differs")
    inputs = parent / "yb_full_inputs.npz"
    with np.load(inputs, allow_pickle=False) as data:
        workspace = {
            name.removeprefix("workspace_"): data[name].copy()
            for name in data.files
            if name.startswith("workspace_")
        }
        workspace = {
            name: value.item() if value.ndim == 0 else value
            for name, value in workspace.items()
        }
        waves = [
            (data[f"P_{n}_{k}"].copy(), data[f"Q_{n}_{k}"].copy()) for n, k in keys
        ]
    feature = PolynomialCellCoulombFeatures(workspace)
    control = CUDADensityControl(feature, memory_bytes=contract["device_cap_bytes"])
    batch, nodes = len(keys) ** 2, len(feature.radii)
    if nodes != 3910:
        raise AssertionError("original full tagged radial coordinates differ")
    bound = control.preflight(batch)
    host_bound = 8 * (5 * nodes * batch + 6 * nodes * len(keys))
    if host_bound > contract["host_cap_bytes"]:
        raise MemoryError("complete host control block exceeds cap")
    samples = np.column_stack([feature.sample(p, q) for p, q in waves])
    densities = np.ascontiguousarray(
        (samples[:, :, None] * samples[:, None, :]).reshape(nodes, batch)
    )
    if not np.all(np.isfinite(densities)):
        raise AssertionError("nonfinite actual complete density input")
    np.savez_compressed(
        out / "density_inputs.npz", samples=samples, densities=densities
    )
    (out / "resource_identity.json").write_text(
        json.dumps(
            dict(
                device_bound_bytes=bound,
                host_bound_bytes=host_bound,
                nodes=nodes,
                batch=batch,
                radial_keys=keys,
                parent_input_sha256=digest(inputs),
                feature_identity=dict(feature.identity),
            ),
            indent=2,
        )
        + "\n"
    )
    device_input = cp.asarray(densities)

    def cpu_action(rank):
        output = np.empty_like(densities)
        for j in range(batch):
            output[:, j] = feature.apply_density(
                np.ascontiguousarray(densities[:, j]), rank
            )
        return output

    records = []
    for rank in range(8):
        before = time.perf_counter()
        reference = cpu_action(rank)
        cpu_seconds = time.perf_counter() - before
        device.synchronize()
        before = time.perf_counter()
        device_output = control.apply(device_input, rank)
        device.synchronize()
        gpu_seconds = time.perf_counter() - before
        actual = cp.asnumpy(device_output)
        error = float(np.max(abs(reference - actual)))
        if not np.all(np.isfinite(actual)) or error > 1e-11:
            np.savez_compressed(
                out / f"rank_{rank}_failure.npz", reference=reference, actual=actual
            )
            raise AssertionError("complete signed GPU weighted density gate failed")
        np.savez_compressed(
            out / f"rank_{rank}.npz", reference=reference, actual=actual
        )
        record = dict(
            rank=rank,
            max_signed_error_Ha=error,
            cpu_diagnostic_seconds=cpu_seconds,
            gpu_diagnostic_seconds=gpu_seconds,
            bitwise_equal=bool(np.array_equal(reference, actual)),
        )
        records.append(record)
        (out / "partial_result.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        del actual, reference, device_output
    # Actual signed block matrices, float64 throughout. The cuBLAS control is
    # a radial contraction only, not a complete angular term2 result.
    rng = np.random.default_rng(731049)
    rho = rng.normal(size=(len(keys), len(keys))) * 1e-4
    phi_gpu, rho_gpu = cp.asarray(samples), cp.asarray(rho)
    reference = samples @ rho.T
    projected_cpu = reference.T @ samples
    device.synchronize()
    before = time.perf_counter()
    actual_gpu = phi_gpu @ rho_gpu.T
    projected_gpu = actual_gpu.T @ phi_gpu
    device.synchronize()
    gemm_seconds = time.perf_counter() - before
    actual, projected = cp.asnumpy(actual_gpu), cp.asnumpy(projected_gpu)
    errors = dict(
        contraction=float(np.max(abs(actual - reference))),
        projection=float(np.max(abs(projected - projected_cpu))),
    )
    np.savez_compressed(
        out / "gemm.npz",
        rho=rho,
        reference=reference,
        actual=actual,
        projected_cpu=projected_cpu,
        projected_gpu=projected,
    )
    if not np.all(np.isfinite(projected)) or max(errors.values()) > 1e-11:
        raise AssertionError("complete signed FP64 GEMM control failed")
    del (
        actual_gpu,
        projected_gpu,
        phi_gpu,
        rho_gpu,
        actual,
        projected,
        reference,
        projected_cpu,
    )
    # Prewarm then alternate. Copies are measured separately and retained.
    cpu_action(0)
    warm = control.apply(device_input, 0)
    device.synchronize()
    del warm
    pairs = []
    for pair in range(3):
        before_load = os.getloadavg()
        before = time.perf_counter()
        cpu = cpu_action(0)
        cpu_seconds = time.perf_counter() - before
        device.synchronize()
        before = time.perf_counter()
        gpu = control.apply(device_input, 0)
        device.synchronize()
        resident_seconds = time.perf_counter() - before
        actual = cp.asnumpy(gpu)
        resident_error = float(np.max(abs(cpu - actual)))
        del gpu, actual
        device.synchronize()
        before = time.perf_counter()
        copied = cp.asarray(densities)
        gpu = control.apply(copied, 0)
        roundtrip = cp.asnumpy(gpu)
        device.synchronize()
        copied_seconds = time.perf_counter() - before
        copied_error = float(np.max(abs(cpu - roundtrip)))
        if max(resident_error, copied_error) > 1e-11:
            raise AssertionError("paired complete GPU density numerical gate failed")
        pairs.append(
            dict(
                pair=pair,
                cpu_seconds=cpu_seconds,
                gpu_resident_seconds=resident_seconds,
                gpu_roundtrip_seconds=copied_seconds,
                before_load=before_load,
                after_load=os.getloadavg(),
                resident_error_Ha=resident_error,
                roundtrip_error_Ha=copied_error,
            )
        )
        del copied, gpu, roundtrip, cpu
    load_passed = all(
        max(*p["before_load"], *p["after_load"]) <= contract["load_guard"]
        for p in pairs
    )
    result = dict(
        status="complete GPU density block and FP64 radial GEMM gates passed",
        records=records,
        gemm_errors_Ha=errors,
        gemm_diagnostic_seconds=gemm_seconds,
        pairs=pairs,
        load_guard_passed=load_passed,
        resident_median_speedup=float(
            np.median([p["cpu_seconds"] / p["gpu_resident_seconds"] for p in pairs])
        )
        if load_passed
        else None,
        roundtrip_median_speedup=float(
            np.median([p["cpu_seconds"] / p["gpu_roundtrip_seconds"] for p in pairs])
        )
        if load_passed
        else None,
        scope="one complete radial subblock; no full term2/SD end-to-end or spectroscopy claim",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
    except BaseException as error:
        out = OWNER / args.run_name
        if out.exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
