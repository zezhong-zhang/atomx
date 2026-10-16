"""Source-bound full radial-subblock CPU/CUDA value and direction control.

This is not a full SD map, solver response or spectroscopic calculation.
The independent oracle is the existing public ordered edge action, not the
new shared CPU/CUDA scalar functions. Failed device guards reject timing only.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
site.addsitedir(str(OWNER / "numba_cuda_runtime"))
sys.path[:0] = [str(ROOT / "src"), str(OWNER)]

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
from numba_density_control import NumbaDensityControl  # noqa: E402

from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def guard():
    report = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return dict(
        load=list(os.getloadavg()),
        this_pid=os.getpid(),
        other_compute_pids=sorted(
            {
                int(v.strip())
                for v in report.splitlines()
                if v.strip() and int(v.strip()) != os.getpid()
            }
        ),
    )


def main(run_name):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    contract = dict(
        hypothesis="same complete ordered weighted cell arithmetic can compile to Numba CPU parallel and CUDA kernels at original signed FP64 gates",
        fixed="Yb full008 68core/414radial-virtual/ns123/Fermi workspace; all59s keys,3481directed pair densities,3910tagged coordinates,all ranks0..7; same original polynomial A.T K_native A weights",
        direction="fixed-geometry analytic density direction phi*dphi+dphi*phi, seed731050/native ns123 coefficient directions; not an SD energy/state derivative",
        gate=1e-11,
        dtype="real float64 C only; complex/other layout rejects",
        rounding="CPU fastmath=False; CUDA explicit dadd_rn/dmul_rn; compiled PTX must contain no fma or atom instructions",
        threads=[1, 8],
        host_cap_bytes=2 * 1024**3,
        device_cap_bytes=512 * 1024**2,
        load_guard=27,
        timing="compile/first-use separate; three alternating warm CPU1/CPU8 full rank0 pairs. GPU resident and transfer-inclusive pairs only if load<=27 AND no other GPU compute PID before/after; otherwise no GPU ratio",
        stop="failure of complete identity, cap, dtype, finite, signed or restoration gates rejects arithmetic. Failed load/device guards reject timing, not correctness. No public adoption or automatic backend fallback",
        scope="one complete s-density radial subblock only, not all414partial waves, term2/SD/end-to-end/experimental accuracy or highest-level certification",
    )
    save("contract.json", contract)
    initial_guard = guard()
    save("initial_guard.json", initial_guard)
    if max(initial_guard["load"]) > contract["load_guard"]:
        raise RuntimeError("original host-load preflight rejects complete run")
    # Bound the complete retained arrays, both controls, comparisons and original
    # feature construction before loading/allocating the density payload.
    nodes, batch, nv = 3910, 59**2, 115
    host_bound = 8 * (14 * nodes * batch + 12 * nv * batch) + 128 * 1024**2
    disk_bound = 8 * nodes * batch * (2 + 6 * 8) + 256 * 1024**2
    if host_bound > contract["host_cap_bytes"]:
        raise MemoryError("full host payload/scratch exceeds original2GiB cap")
    if shutil.disk_usage(out).free < disk_bound:
        raise MemoryError("insufficient complete evidence disk before allocation")
    save(
        "preflight.json",
        dict(
            host_bound_bytes=host_bound,
            archive_bound_bytes=disk_bound,
            disk_free_bytes=shutil.disk_usage(out).free,
        ),
    )
    source_paths = [Path(__file__), OWNER / "numba_density_control.py"]
    source_paths += sorted((ROOT / "src/atomx/numerics/bspline").glob("*.py"))
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    from numba import cuda

    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "numba",
            "llvmlite",
            "numba-cuda",
            "cuda-bindings",
            "cuda-core",
            "cuda-toolkit",
            "nvidia-nvvm",
            "nvidia-nvjitlink",
            "nvidia-cuda-runtime",
            "nvidia-cuda-nvrtc",
        )
    }
    save(
        "runtime.json",
        dict(
            packages=packages,
            numpy_path=np.__file__,
            numba_path=nb.__file__,
            cuda_path=cuda.__file__,
            cuda_source_sha256=digest(cuda.__file__),
            CUDA_PATH=os.environ.get("CUDA_PATH"),
            LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH"),
            device=str(cuda.get_current_device()),
            driver_report=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.free",
                    "--format=csv",
                ],
                text=True,
            ),
        ),
    )
    physical = OWNER / "yb_full_run008/yb_full_inputs.npz"
    plan_path = OWNER / "yb_full_run008/yb_full_plan.json"
    physical_hash, plan_hash = digest(physical), digest(plan_path)
    plan = json.loads(plan_path.read_text())
    keys = [tuple(key) for key in plan["virtual_keys"] if key[1] == -1]
    if len(keys) != 59 or len(plan["virtual_keys"]) != 414:
        raise AssertionError("full original radial band identity differs")
    with np.load(physical, allow_pickle=False) as data:
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
    if len(feature.radii) != nodes or feature.outer_indices.shape[1] != nv:
        raise AssertionError("complete tagged geometry differs from plan")
    cpu = NumbaDensityControl(
        feature,
        backend="cpu",
        batch=batch,
        memory_bytes=contract["device_cap_bytes"],
        threads=8,
    )
    gpu = NumbaDensityControl(
        feature,
        backend="cuda",
        batch=batch,
        memory_bytes=contract["device_cap_bytes"],
        threads=8,
    )
    save(
        "resource_identity.json",
        dict(
            host_bound_bytes=host_bound,
            control_payload_bound_bytes=gpu.bound,
            nodes=nodes,
            batch=batch,
            nv=nv,
            radial_keys=keys,
            physical_sha256=physical_hash,
            plan_sha256=plan_hash,
            feature_identity=dict(feature.identity),
        ),
    )
    samples = np.column_stack([feature.sample(p, q) for p, q in waves])
    rng = np.random.default_rng(731050)
    directions = np.column_stack(
        [
            feature.sample(
                rng.normal(size=len(p)) * 1e-3, rng.normal(size=len(q)) * 1e-3
            )
            for p, q in waves
        ]
    )
    density = np.ascontiguousarray(
        (samples[:, :, None] * samples[:, None, :]).reshape(nodes, batch)
    )
    prime = np.ascontiguousarray(
        (
            directions[:, :, None] * samples[:, None, :]
            + samples[:, :, None] * directions[:, None, :]
        ).reshape(nodes, batch)
    )
    np.save(out / "density.npy", density)
    np.save(out / "direction.npy", prime)
    np.savez(out / "sample_directions.npz", samples=samples, directions=directions)

    def original(x, rank):
        result = np.empty_like(x)
        for j in range(batch):
            result[:, j] = feature.apply_density(np.ascontiguousarray(x[:, j]), rank)
        return result

    def check(reference, actual):
        if not np.all(np.isfinite(actual)):
            raise AssertionError("nonfinite full output")
        error = float(np.max(np.abs(reference - actual)))
        if error > contract["gate"]:
            raise AssertionError(f"original full signed gate failed: {error}")
        return dict(
            max_signed_error=error,
            bitwise_equal=bool(np.array_equal(reference, actual)),
        )

    def timed(action, *args):
        before = time.perf_counter()
        result = action(*args)
        return result, time.perf_counter() - before

    previous_threads = nb.get_num_threads()
    records = []
    for rank in range(8):
        record = dict(rank=rank, diagnostic_seconds={}, checks={})
        references = []
        for label, x in (("value", density), ("direction", prime)):
            ref, elapsed = timed(original, x, rank)
            record["diagnostic_seconds"][f"original_{label}"] = elapsed
            references.append(ref)
            np.save(out / f"rank{rank}_{label}_reference.npy", ref)
            for backend, control in (("cpu8", cpu), ("cuda", gpu)):
                y = np.empty_like(x)
                _, elapsed = timed(control.apply_into, x, rank, y)
                record["diagnostic_seconds"][f"{backend}_{label}"] = elapsed
                record["checks"][f"{backend}_{label}"] = check(ref, y)
                np.save(out / f"rank{rank}_{label}_{backend}.npy", y)
            if label == "value":
                cpu.threads = 1
                _, elapsed = timed(cpu.apply_into, x, rank, y)
                record["diagnostic_seconds"]["cpu1_value"] = elapsed
                record["checks"]["cpu1_value"] = check(ref, y)
                cpu.threads = 8
            del y
        # Real value->direction->value restoration; original public arrays anchor it.
        y = np.empty_like(density)
        gpu.apply_into(density, rank, y)
        record["checks"]["cuda_value_restored"] = check(references[0], y)
        if not record["checks"]["cuda_value_restored"]["bitwise_equal"]:
            raise AssertionError("complete CUDA borrowed output restoration differs")
        del references, y
        if nb.get_num_threads() != previous_threads:
            raise AssertionError("CPU caller thread mask not restored")
        records.append(record)
        save("partial_result.json", records)
        print(json.dumps(record), flush=True)
    # Compile product, not just decorator options, certifies no contraction/race.
    ptx_audit = []
    for i, kernel in enumerate(gpu.kernels):
        for ordinal, (signature, assembly) in enumerate(kernel.inspect_asm().items()):
            path = out / f"kernel{i}_{ordinal}.ptx"
            path.write_text(assembly)
            forbidden = re.findall(r"\b(?:fma|atom|red)\.[^\s;]+", assembly)
            if forbidden:
                raise AssertionError(
                    f"forbidden CUDA fused/atomic instructions: {forbidden}"
                )
            ptx_audit.append(
                dict(
                    kernel=i,
                    signature=str(signature),
                    sha256=digest(path),
                    forbidden=forbidden,
                )
            )
    save("ptx_audit.json", ptx_audit)
    # Actual resource/layout rejection before execution; no cast or small-space fallback.
    rejects = {}
    for name, action in (
        (
            "cap",
            lambda: NumbaDensityControl(
                feature, backend="cpu", batch=batch, memory_bytes=1
            ),
        ),
        (
            "complex",
            lambda: cpu.apply_into(density.astype(complex), 0, np.empty_like(density)),
        ),
        ("layout", lambda: cpu.apply_into(density.T, 0, np.empty_like(density))),
        (
            "backend",
            lambda: NumbaDensityControl(
                feature,
                backend="auto",
                batch=batch,
                memory_bytes=contract["device_cap_bytes"],
            ),
        ),
    ):
        try:
            action()
        except (ValueError, MemoryError) as error:
            rejects[name] = str(error)
        else:
            raise AssertionError(f"required rejection absent: {name}")
    save("rejections.json", rejects)
    # Prewarm complete rank0. CPU timings retain both masks and the original oracle.
    ref = original(density, 0)
    y = np.empty_like(density)
    cpu.apply_into(density, 0, y)
    cpu_pairs = []
    for pair in range(3):
        record = dict(
            pair=pair, before_load=list(os.getloadavg()), seconds={}, checks={}
        )
        for workers in (1, 8) if pair % 2 == 0 else (8, 1):
            cpu.threads = workers
            _, elapsed = timed(cpu.apply_into, density, 0, y)
            record["seconds"][str(workers)] = elapsed
            record["checks"][str(workers)] = check(ref, y)
        record["after_load"] = list(os.getloadavg())
        record["valid"] = max(record["before_load"] + record["after_load"]) <= 27
        cpu_pairs.append(record)
    save("cpu_pairs.json", cpu_pairs)
    gpu_pairs = []
    gpu_guard = guard()
    gpu_timing_allowed = (
        max(gpu_guard["load"]) <= 27 and not gpu_guard["other_compute_pids"]
    )
    if gpu_timing_allowed:
        gpu.set_input(density)
        gpu.apply_resident(0)
        gpu.synchronize()
        cpu.threads = 8
        for pair in range(3):
            record = dict(pair=pair, before=guard(), seconds={})
            order = ("cpu8", "cuda_roundtrip", "cuda_resident")
            if pair % 2:
                order = tuple(reversed(order))
            for mode in order:
                if mode == "cpu8":
                    _, elapsed = timed(cpu.apply_into, density, 0, y)
                elif mode == "cuda_roundtrip":
                    _, elapsed = timed(gpu.apply_into, density, 0, y)
                else:

                    def action():
                        gpu.apply_resident(0)
                        gpu.synchronize()

                    _, elapsed = timed(action)
                record["seconds"][mode] = elapsed
                if mode == "cuda_resident":
                    gpu.y.copy_to_host(y, stream=gpu.stream)
                    gpu.synchronize()
                check(ref, y)
            record["after"] = guard()
            record["valid"] = all(
                max(g["load"]) <= 27 and not g["other_compute_pids"]
                for g in (record["before"], record["after"])
            )
            gpu_pairs.append(record)
            if not record["valid"]:
                break
    save(
        "gpu_pairs.json",
        dict(initial=gpu_guard, allowed=gpu_timing_allowed, pairs=gpu_pairs),
    )
    unchanged = all(digest(ROOT / name) == sha for name, sha in hashes.items())
    unchanged &= digest(physical) == physical_hash and digest(plan_path) == plan_hash
    if not unchanged:
        raise AssertionError("actual source or full original inputs changed during run")
    cpu_valid = all(p["valid"] for p in cpu_pairs)
    cpu_ratio = (
        float(
            np.median([p["seconds"]["1"] for p in cpu_pairs])
            / np.median([p["seconds"]["8"] for p in cpu_pairs])
        )
        if cpu_valid
        else None
    )
    gpu_ratios = None
    if len(gpu_pairs) == 3 and all(p["valid"] for p in gpu_pairs):
        medians = {
            key: float(np.median([p["seconds"][key] for p in gpu_pairs]))
            for key in gpu_pairs[0]["seconds"]
        }
        gpu_ratios = {
            key: medians["cpu8"] / medians[key]
            for key in ("cuda_roundtrip", "cuda_resident")
        }
    result = dict(
        status="pass",
        records=records,
        cpu_pairs=cpu_pairs,
        controlled_cpu1_over_cpu8_ratio=cpu_ratio,
        controlled_cpu8_over_gpu_ratios=gpu_ratios,
        gpu_timing_guard=gpu_guard,
        ptx_audit=ptx_audit,
        source_input_unchanged=bool(unchanged),
        caller_thread_mask_restored=nb.get_num_threads() == previous_threads,
        scope=contract["scope"],
        highest_level=False,
    )
    save("result.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("records", "cpu_pairs", "ptx_audit")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
    except Exception as error:
        out = OWNER / args.run_name
        if out.is_dir():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
