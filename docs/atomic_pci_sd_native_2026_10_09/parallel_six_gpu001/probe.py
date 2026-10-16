#!/usr/bin/env python3
"""Full signed CPU/CUDA contraction correctness before bounded timing."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import resource
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path


def measured(function):
    started = time.perf_counter()
    result = function()
    return time.perf_counter() - started, result


def guard():
    report = dict(
        load=list(os.getloadavg()), this_pid=os.getpid(), host_load_ceiling=27
    )
    try:
        report["device"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        report["other_compute_pids"] = [
            int(v.strip())
            for v in pids.splitlines()
            if v.strip() and int(v.strip()) != os.getpid()
        ]
    except Exception as error:
        report["device_query_error"] = f"{type(error).__name__}: {error}"
        report["other_compute_pids"] = None
    report["timing_guard_passed"] = (
        max(report["load"]) <= 27 and report["other_compute_pids"] == []
    )
    return report


def address_observation():
    lines = Path("/proc/self/status").read_text().splitlines()
    return {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in lines
        if line.startswith(("VmSize:", "VmPeak:", "VmRSS:", "VmHWM:"))
    }


def save(path, result):
    from probe import clean

    path.write_text(json.dumps(clean(result), indent=2, default=repr) + "\n")


def run(output, manual_gpu):
    import numba as nb
    import numpy as np
    import payload
    import runtime_guard
    from numba_density_control import NumbaDensityControl
    from pair_ladder_numba import NumbaPairLadderControl
    from probe import active_sources, archive_sources, sha

    from atomx.methods.sd_pair_diagrams import CorePairLadders
    from atomx.methods.sd_transverse import build_j_coupled_sd_interaction

    owner = Path(__file__).resolve().parent
    root = owner.parents[2]
    inputs = owner.parent / "parallel_six_sd_diagrams001/inputs.npz"
    shutil.copy2(inputs, output / "inputs.npz")
    data = np.load(inputs, allow_pickle=False)
    feature = payload.FrozenDensityFeature(data)
    density = np.ascontiguousarray(data["density_full"])
    x = np.ascontiguousarray(
        np.column_stack(
            (
                data["amplitudes_basis"],
                data["amplitudes_signed"].real,
                data["amplitudes_signed"].imag,
            )
        )
    )
    sources = active_sources(
        root,
        (
            Path(__file__).resolve(),
            owner / "contract.json",
            owner / "pair_ladder_numba.py",
            owner / "runtime_dependencies_sha256.json",
            root / "tests/test_numba_pair_ladder_control.py",
            owner.parent / "parallel_six_sd_diagrams001/probe.py",
        ),
    )
    before = archive_sources(root, output, sources)
    original_workers = nb.get_num_threads()
    arrays, cpu_cases, cpu_controls, production = {}, {}, {}, {}
    cap = 32 * 1024**2
    started = time.perf_counter()
    cpu_density = NumbaDensityControl(
        feature, backend="cpu", batch=density.shape[1], memory_bytes=cap, threads=1
    )
    density_construct = time.perf_counter() - started
    cpu_density_outputs, density_gates = {}, {}
    for rank in range(4):
        y = np.empty_like(density)
        seconds, _ = measured(
            lambda rank=rank, y=y: cpu_density.apply_into(density, rank, y)
        )
        error = float(np.max(abs(y - data[f"density_rank{rank}_public"])))
        if error > 1e-11:
            raise AssertionError("complete density original public FP64 gate failed")
        arrays[f"density_cpu_rank{rank}"] = y.copy()
        cpu_density_outputs[rank] = y.copy()
        density_gates[f"rank{rank}_public_full_signed_max"] = error
        cpu_cases[f"density_rank{rank}"] = dict(
            first_action_seconds=seconds, shape=list(y.shape)
        )
    for mode in ("dc", "static", "dynamic"):
        ladders = CorePairLadders(
            build_j_coupled_sd_interaction(payload.load_hamiltonian(data, mode))
        )
        production[mode] = ladders
        construction, control = measured(
            lambda ladders=ladders: NumbaPairLadderControl(
                ladders, backend="cpu", batch=x.shape[1], memory_bytes=cap, threads=1
            )
        )
        cpu_controls[mode] = control
        p, h = np.empty_like(x), np.empty_like(x)
        compilation, _ = measured(
            lambda control=control, p=p, h=h: control.kernel.compile(
                tuple(nb.typeof(a) for a in (*control.geometry, x, p, h))
            )
        )
        cold, _ = measured(
            lambda control=control, p=p, h=h: control.apply_into(x, p, h)
        )
        expected_p = np.asarray(
            [ladders.evaluate(t).particle_particle_Ha for t in x.T]
        ).T.real
        expected_h = np.asarray([ladders.evaluate(t).hole_hole_Ha for t in x.T]).T.real
        gates = dict(
            pp_complete_max_Ha=float(np.max(abs(p - expected_p))),
            hh_complete_max_Ha=float(np.max(abs(h - expected_h))),
        )
        if max(gates.values()) > 1e-11:
            raise AssertionError("complete real signed CPU ladder original gate failed")
        arrays[f"{mode}_cpu_pp"], arrays[f"{mode}_cpu_hh"] = p.copy(), h.copy()
        cpu_cases[mode] = dict(
            gates=gates,
            construction_seconds=construction,
            first_action_including_jit_seconds=cold,
            cold_compilation_seconds=compilation,
            first_action_scope="complete host input validation/workers/PP/HH action; compilation separated",
            shape=list(x.shape),
            whole_owned_payload_bound_bytes=control.bound,
        )
    worker_gate = nb.get_num_threads() == original_workers
    if not worker_gate:
        raise AssertionError("caller worker mask changed")
    result = dict(
        status="passed",
        gates=dict(
            cpu_density=density_gates,
            cpu_ladders={k: v["gates"] for k, v in cpu_cases.items() if "gates" in v},
            caller_workers_restored=worker_gate,
            cpu_complete_signed_correctness=True,
            gpu_admitted=False,
            gpu_complete_signed_correctness=False,
            gpu_controlled_timing=False,
        ),
        cpu_cases=cpu_cases,
        density_construction_seconds=density_construct,
        gpu_status="admission_pending",
        highest_level_reached=False,
        native_scalable_sd_transverse_complete=False,
        gpu_speedup_claim=None,
        source_sha256=before,
        source_files=list(before),
        input_files=[str(inputs.relative_to(root))],
        input_sha256=sha(inputs),
        cap_kind="manual explicit GPU VAS contract"
        if manual_gpu
        else "automatic hard 4GiB VAS contract",
        address_before_cuda=address_observation(),
        card_summary="Numba CPU 全 signed DC/static/dynamic source-PP-HH contractions 和公共 native density action 已通过原 FP64 门；CUDA 先准入，未过门不报告加速，native_scalable_sd_transverse_complete=false。",
    )
    save(
        output / "result.json", result
    )  # preserve complete CPU gates before CUDA admission
    result["initial_guard"] = guard()
    gpu_controls = []
    try:
        lock = (owner.parent / "parallel_six_gpu001/device0.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        from numba import cuda

        result["cuda_runtime"] = dict(
            module_path=cuda.__file__,
            module_sha256=sha(cuda.__file__),
            numba=nb.__version__,
            numpy=np.__version__,
            numba_cuda=importlib.metadata.version("numba-cuda"),
        )
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable under declared address cap; no target fallback"
            )
        context_seconds, context = measured(cuda.current_context)
        result["cuda_context_seconds"] = context_seconds
        result["address_after_context"] = address_observation()
        result["device_free_total_bytes"] = list(map(int, context.get_memory_info()))
        result["gates"]["gpu_admitted"] = True
        setup_seconds, gpu_density = measured(
            lambda: NumbaDensityControl(
                feature,
                backend="cuda",
                batch=density.shape[1],
                memory_bytes=cap,
                threads=1,
            )
        )
        gpu_controls.append(gpu_density)
        gpu_cases, density_cuda_gates = {}, {}
        for rank in range(4):
            y = np.empty_like(density)
            cold, _ = measured(
                lambda rank=rank, y=y: gpu_density.apply_into(density, rank, y)
            )
            error = float(np.max(abs(y - cpu_density_outputs[rank])))
            public = float(np.max(abs(y - data[f"density_rank{rank}_public"])))
            exact = np.array_equal(y, cpu_density_outputs[rank])
            if max(error, public) > 1e-11 or not exact:
                raise AssertionError(
                    "complete CUDA density original gate/IEEE identity failed"
                )
            arrays[f"density_cuda_rank{rank}"] = y.copy()
            density_cuda_gates[f"rank{rank}"] = dict(
                cpu_full_signed_max=error,
                public_full_signed_max=public,
                bitwise_cpu_equal=exact,
            )
            gpu_cases[f"density_rank{rank}"] = dict(
                first_action_with_jit_and_transfer_seconds=cold
            )
        pair_gpu = {}
        for mode in ("dc", "static", "dynamic"):
            construction, control = measured(
                lambda mode=mode: NumbaPairLadderControl(
                    production[mode],
                    backend="cuda",
                    batch=x.shape[1],
                    memory_bytes=cap,
                    threads=1,
                )
            )
            pair_gpu[mode] = control
            gpu_controls.append(control)
            p, h = np.empty_like(x), np.empty_like(x)
            compilation, _ = measured(
                lambda control=control: control.kernel.compile(
                    tuple(
                        nb.typeof(a)
                        for a in (*control.geometry, control.x, control.p, control.h)
                    )
                )
            )
            cold, _ = measured(
                lambda control=control, p=p, h=h: control.apply_into(x, p, h)
            )
            gates = dict(
                pp_cpu_complete_max_Ha=float(np.max(abs(p - arrays[f"{mode}_cpu_pp"]))),
                hh_cpu_complete_max_Ha=float(np.max(abs(h - arrays[f"{mode}_cpu_hh"]))),
                pp_bitwise_cpu_equal=bool(np.array_equal(p, arrays[f"{mode}_cpu_pp"])),
                hh_bitwise_cpu_equal=bool(np.array_equal(h, arrays[f"{mode}_cpu_hh"])),
            )
            if (
                any(
                    gates[k] > 1e-11
                    for k in ("pp_cpu_complete_max_Ha", "hh_cpu_complete_max_Ha")
                )
                or not gates["pp_bitwise_cpu_equal"]
                or not gates["hh_bitwise_cpu_equal"]
            ):
                raise AssertionError(
                    "complete CUDA ladder original FP64/IEEE identity gate failed"
                )
            arrays[f"{mode}_cuda_pp"], arrays[f"{mode}_cuda_hh"] = p.copy(), h.copy()
            gpu_cases[mode] = dict(
                gates=gates,
                construction_allocation_upload_seconds=construction,
                setup_layers=control.setup_layers,
                cold_compilation_seconds=compilation,
                first_action_with_jit_and_transfer_seconds=cold,
                first_action_scope="full amplitude upload/PP/HH/output download after explicit separated compilation; no source-diagram GPU compute",
                whole_owned_payload_bound_bytes=control.bound,
            )
        # Audit actual compiled code, including existing reused density code.
        ptx_reports = []
        for name, kernels in (
            ("density", gpu_density.kernels),
            ("ladder", (pair_gpu["dynamic"].kernel,)),
        ):
            for index, kernel in enumerate(kernels):
                for signature, ptx in kernel.inspect_asm().items():
                    forbidden = re.findall(r"\b(?:fma|atom|red)\.[^\n]*", ptx)
                    path = output / f"{name}_{index}.ptx"
                    path.write_text(ptx)
                    ptx_reports.append(
                        dict(
                            name=name,
                            kernel=index,
                            signature=str(signature),
                            sha256=sha(path),
                            forbidden=forbidden,
                        )
                    )
                    if forbidden:
                        raise AssertionError(
                            "actual CUDA PTX violates no-FMA/no-atomic contract"
                        )
        realized = 0
        for control in gpu_controls:
            device_arrays = list(control.geometry)
            if hasattr(control, "work"):
                device_arrays += (
                    list(control.work) + [control.x, control.y] + list(control.weights)
                )
            else:
                device_arrays += [control.x, control.p, control.h]
            realized += sum(value.nbytes for value in device_arrays)
        if realized > cap:
            raise MemoryError(
                "sum of all live owned GPU arrays exceeds declared device cap"
            )
        result.update(
            gpu_status="correctness_passed",
            gpu_cases=gpu_cases,
            density_cuda_gates=density_cuda_gates,
            gpu_density_setup_allocation_upload_seconds=setup_seconds,
            realized_owned_device_bytes=realized,
            ptx_audit=ptx_reports,
        )
        result["gates"]["gpu_complete_signed_correctness"] = True
        result["gates"]["ptx_no_fma_or_atomic"] = True
        # Separate scopes, alternate pair order. Fixed tiny physical workload;
        # no extrapolation to full SD, large native dimensions or end-to-end.
        cpu, gpu = cpu_controls["dynamic"], pair_gpu["dynamic"]
        p, h = np.empty_like(x), np.empty_like(x)
        pairs, repeats = [], 20
        gpu.set_input(x)
        gpu.synchronize()
        for sample in range(5):
            timings = {}

            def cpu_call():
                for _ in range(repeats):
                    cpu.apply_into(x, p, h)

            def gpu_host_call():
                for _ in range(repeats):
                    gpu.apply_into(x, p, h)

            order = (
                ("cpu_warm_host_seconds", cpu_call),
                ("gpu_warm_transfer_inclusive_seconds", gpu_host_call),
            )
            for name, function in order if sample % 2 == 0 else reversed(order):
                timings[name] = measured(function)[0] / repeats
            from numba_density_control import _workers

            resident_cpu_start = time.perf_counter()
            with _workers(1):
                for _ in range(repeats):
                    cpu.kernel(*cpu.geometry, x, p, h)
            timings["cpu_warm_kernel_and_scoped_workers_seconds"] = (
                time.perf_counter() - resident_cpu_start
            ) / repeats

            def upload():
                gpu.set_input(x)
                gpu.synchronize()

            timings["gpu_complete_amplitude_upload_seconds"] = measured(upload)[0]
            gpu.set_input(x)
            gpu.synchronize()
            event_start, event_end = cuda.event(), cuda.event()
            event_start.record(gpu.stream)
            resident_start = time.perf_counter()
            for _ in range(repeats):
                gpu.apply_resident()
            event_end.record(gpu.stream)
            gpu.synchronize()
            timings["gpu_warm_resident_host_seconds"] = (
                time.perf_counter() - resident_start
            ) / repeats
            timings["gpu_warm_device_event_seconds"] = float(
                cuda.event_elapsed_time(event_start, event_end)
            ) / (1000 * repeats)

            def download():
                gpu.p.copy_to_host(p, stream=gpu.stream)
                gpu.h.copy_to_host(h, stream=gpu.stream)
                gpu.synchronize()

            timings["gpu_complete_separated_output_download_seconds"] = measured(
                download
            )[0]
            pairs.append(timings)
        result["warm_paired_layers"] = dict(
            model="dynamic signed complete PP/HH 10x12; excludes source owner setup and entire SD solve",
            pairs=pairs,
            repeats_per_pair=repeats,
            event_scope="CUDA events bracket all 20 kernels and intervening host-launch gaps, not a single-kernel device duration",
            comparison_scope="CPU warm host and GPU transfer-inclusive use same full apply_into validation; resident observations have separate explicit scopes",
        )
        result["final_guard"] = guard()
        timing_ok = (
            result["initial_guard"]["timing_guard_passed"]
            and result["final_guard"]["timing_guard_passed"]
        )
        result["gates"]["gpu_controlled_timing"] = timing_ok
        if timing_ok:
            result["controlled_scoped_cpu_over_gpu_host_ratio"] = float(
                np.median(
                    [
                        p["cpu_warm_host_seconds"]
                        / p["gpu_warm_transfer_inclusive_seconds"]
                        for p in pairs
                    ]
                )
            )
        result["card_summary"] = (
            "共享 Numba CPU/CUDA ordered FP64 完整 signed DC/static/dynamic PP/HH 和 2688×25 全 native density、rank0..3逐元素及 bitwise 对照通过；真实 PTX 无 FMA/atomic，显存 payload 通过 32 MiB cap。只报告有界 kernel 分层计时；负载门失败则 speedup=null，完整 SD/GPU规模化仍未完成。"
        )
    except Exception as error:
        result["gpu_error"] = f"{type(error).__name__}: {error}"
        result["address_after_cuda_failure"] = address_observation()
        result["gpu_status"] = (
            "admission_deferred"
            if not result["gates"]["gpu_admitted"]
            else "gate_failed"
        )
        if result["gates"]["gpu_admitted"]:
            result["status"] = "failed"
        result["final_guard"] = guard()
    finally:
        np.savez_compressed(output / "arrays.npz", **arrays)
        after = {str(path.relative_to(root)): sha(path) for path in sources}
        result["source_unchanged"] = before == after
        result["final_address_observation"] = address_observation()
        mapped = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            candidate = line.split()[-1]
            if candidate.startswith("/") and any(
                name in candidate
                for name in (
                    "libcuda.so",
                    "libcudart.so",
                    "libnvvm.so",
                    "libnvJitLink.so",
                    "libnvrtc.so",
                )
            ):
                mapped.add(Path(candidate).resolve())
        result["actual_loaded_cuda_libraries_sha256"] = {
            str(path): sha(path) for path in sorted(mapped) if path.is_file()
        }
        result["arrays_sha256"] = sha(output / "arrays.npz")
        result["gates"]["source_and_input_unchanged"] = before == after and sha(
            inputs
        ) == sha(output / "inputs.npz")
        result["gates"]["caller_workers_restored_after_cuda"] = (
            nb.get_num_threads() == original_workers
        )
        result["runtime_after"] = runtime_guard.verify(
            owner / "runtime_dependencies_sha256.json"
        )
        result["gates"]["complete_runtime_dependencies_unchanged"] = result[
            "runtime_after"
        ]["matched"]
        if (
            not result["gates"]["source_and_input_unchanged"]
            or not result["gates"]["caller_workers_restored_after_cuda"]
        ):
            result["status"] = "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manual-gpu",
        action="store_true",
        help="explicit separate prospective128GiB VAS/32MiB device contract; never used by automatic four-GiB runner",
    )
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    runtime_root = owner.parent / "numba_cuda_runtime/nvidia/cu13"
    if os.environ.get("ATOMX_PAIR_CONTROL_CUDA_READY") != str(runtime_root):
        # The ELF loader reads LD_LIBRARY_PATH at exec startup. Same PID and
        # inherited hard resource limits; this sets the existing scoped
        # optional runtime, never changes bridge credentials or profiles.
        environment = dict(os.environ)
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment.update(
            CUDA_PATH=str(runtime_root),
            LD_LIBRARY_PATH=str(runtime_root / "lib")
            + (":" + previous if previous else ""),
            ATOMX_PAIR_CONTROL_CUDA_READY=str(runtime_root),
        )
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise SystemExit("refuse to overwrite evidence")
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = "1"
    address = 128 * 1024**3 if args.manual_gpu else 4 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (address, address))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    root = owner.parents[2]
    # Reuse the existing installed optional numba-cuda CUDA implementation.
    sys.path[:0] = [
        str(owner.parent / "parallel_six_sd_diagrams001"),
        str(owner),
        str(owner.parent),
        str(root / "src"),
        str(root / "tests"),
    ]
    started = time.monotonic()
    result = dict(
        status="failed",
        gates={},
        highest_level_reached=False,
        card_summary="CPU/GPU 探针未通过；不报告加速。",
    )
    try:
        import runtime_guard

        result["runtime_before"] = runtime_guard.verify(
            owner / "runtime_dependencies_sha256.json"
        )
        site.addsitedir(str(owner.parent / "numba_cuda_runtime"))
        result.update(run(output, args.manual_gpu))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result.update(
            caps=dict(
                address_bytes=address,
                cpu_seconds=120,
                threads=1,
                owned_device_payload_bytes=32 * 1024**2,
            ),
            observed_wall_seconds=time.monotonic() - started,
            process_peak_rss_bytes=1024
            * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            highest_level_reached=False,
            gpu_speedup_claim=None,
            native_scalable_sd_transverse_complete=False,
        )
        save(output / "result.json", result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "gpu_status",
                        "observed_wall_seconds",
                        "gpu_error",
                        "gpu_speedup_claim",
                    )
                },
                default=repr,
            )
        )


if __name__ == "__main__":
    main()
