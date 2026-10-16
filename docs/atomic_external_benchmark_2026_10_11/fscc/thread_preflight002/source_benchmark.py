"""Fixed-integral finite CI/EXP-T agreement and scoped strong-scaling audit.

This private benchmark does not change Atomx production or reference sources.
The Atomx FSCC interface calls EXP-T; the wrapper comparison is overhead, not
an independent solver comparison. The complete finite Fermi CI is an oracle.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import resource
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OWNER = Path(__file__).resolve().parent
OLD = REPO / "docs/atomic_pci_sd_native_2026_10_09"
CANDIDATE = (
    REPO
    / "docs/atomx_hourly_progress_2026_10_10/parallel001/public_backend_source-build/expt.x"
)
DEFAULT = OLD / "expt_compact_public_source001-build/expt.x"
CASES = {
    "two_core": OLD / "transverse_precision007/tight",
    "four_core": OLD / "parallel_six_fscc_scope001/frozen001/fscc",
}
CAP = 8 * 1024**3
TENSORS = ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine():
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return {
        "load": list(os.getloadavg()),
        "affinity": sorted(os.sched_getaffinity(0)),
        "available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
        "free_disk_bytes": shutil.disk_usage(OWNER).free,
    }


def guard(before, after, workers):
    return {
        "load_before_plus_workers": max(before["load"]) + workers
        <= len(before["affinity"]),
        "load_after_plus_workers": max(after["load"]) + workers
        <= len(after["affinity"]),
        "load_change_at_most_four": max(
            abs(a - b) for a, b in zip(before["load"], after["load"], strict=True)
        )
        <= 4.0,
        "memory_headroom": min(before["available_bytes"], after["available_bytes"])
        >= CAP + 8 * 1024**3,
        "disk_headroom": min(before["free_disk_bytes"], after["free_disk_bytes"])
        >= 8 * 1024**3,
    }


@contextlib.contextmanager
def timing_lock():
    with (OWNER.parent / "timing.lock").open("a") as handle:
        print("waiting shared timing lock", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment(workers):
    value = os.environ.copy()
    libs = (
        REPO.parent
        / "atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu"
    )
    value["LD_LIBRARY_PATH"] = ":".join(
        (str(libs), str(libs / "openblas-serial"), value.get("LD_LIBRARY_PATH", ""))
    )
    value["OMP_NUM_THREADS"] = str(workers)
    value["OMP_DYNAMIC"] = "FALSE"
    value["OMP_THREAD_LIMIT"] = str(workers)
    value["OMP_MAX_ACTIVE_LEVELS"] = "1"
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        value[variable] = "1"
    return value


class BackendSampler:
    """Same 10-ms observer for bare and wrapper runs; observer time is included."""

    def __init__(self, workers, directory):
        self.stop = threading.Event()
        self.workers = workers
        self.total_thread_cap = 8
        self.directory = directory
        self.thread_cap_violation = False
        self.thread_CPU_ticks = {}
        self.max_observed_active_threads = 0
        self.max_threads = self.peak_RSS_bytes = self.samples = 0
        self.libraries = set()
        self.pids = set()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                children = (
                    Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
                    .read_text()
                    .split()
                )
            except OSError:
                children = ()
            for pid in children:
                try:
                    exe = Path(f"/proc/{pid}/exe").resolve(strict=True)
                    if exe.name != "expt.x":
                        continue
                    fields = dict(
                        line.split(":", 1)
                        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                        if ":" in line
                    )
                    self.max_threads = max(self.max_threads, int(fields["Threads"]))
                    active = 0
                    for task in Path(f"/proc/{pid}/task").iterdir():
                        stat = (task / "stat").read_text().rsplit(")", 1)[1].split()
                        ticks = int(stat[11]) + int(stat[12])
                        key = (int(pid), int(task.name))
                        if ticks > self.thread_CPU_ticks.get(key, ticks):
                            active += 1
                        self.thread_CPU_ticks[key] = ticks
                    self.max_observed_active_threads = max(
                        self.max_observed_active_threads, active
                    )
                    if int(fields["Threads"]) > self.total_thread_cap:
                        self.thread_cap_violation = True
                        # A direct child with the verified executable name; stop
                        # only this owned scientific process on cap excess.
                        process_command = Path(f"/proc/{pid}/cmdline").read_bytes()
                        if (
                            b"--no-clean" in process_command
                            and b"cc.inp" in process_command
                        ):
                            import signal

                            os.kill(int(pid), signal.SIGTERM)
                    self.peak_RSS_bytes = max(
                        self.peak_RSS_bytes, int(fields["VmHWM"].split()[0]) * 1024
                    )
                    self.pids.add(int(pid))
                    self.samples += 1
                    for row in Path(f"/proc/{pid}/maps").read_text().splitlines():
                        token = row.split()[-1]
                        if token.startswith("/") and (
                            ".so" in token or "ld-linux" in token
                        ):
                            self.libraries.add(str(Path(token).resolve()))
                except (OSError, KeyError, ValueError):
                    pass
            self.stop.wait(0.01)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if self.thread_cap_violation:
            dump(self.directory / "runtime_thread_cap_rejection.json", self.data())

    def data(self):
        return {
            "actual_backend_pids": sorted(self.pids),
            "observed_backend_max_threads": self.max_threads,
            "requested_OpenMP_team_size": self.workers,
            "total_process_thread_cap": self.total_thread_cap,
            "thread_cap_violation": self.thread_cap_violation,
            "maximum_observed_CPU_active_threads": self.max_observed_active_threads,
            "observed_thread_CPU_seconds": {
                f"{pid}/{tid}": ticks / os.sysconf("SC_CLK_TCK")
                for (pid, tid), ticks in self.thread_CPU_ticks.items()
            },
            "sampled_backend_peak_RSS_bytes": self.peak_RSS_bytes,
            "samples": self.samples,
            "mapped_runtime_library_sha256": {
                p: sha(p) for p in sorted(self.libraries)
            },
            "scope": "10 ms /proc observer; peak RSS/threads are sampled lower bounds, not exact resource counters",
        }


def parse_backend(directory, integrals):
    from atomx.methods.fscc import read_heff

    log = (directory / "cc.log").read_text()
    statuses = {
        s: [int(ok), int(it), *(float(v) for v in (a, b, c))]
        for s, ok, it, a, b, c in re.findall(
            r"ATOMX_CC_STATUS (\dh\dp) (\d+) (\d+) (\S+) (\S+) (\S+)", log
        )
    }
    if set(statuses) != {"0h0p", "0h1p"} or any(v[0] != 1 for v in statuses.values()):
        raise RuntimeError(f"complete backend solve failed: {directory}")
    blocks = read_heff(directory / "scratch/HEFF")
    matrix = blocks["0h1p", 1]
    core = float(blocks["0h0p", 1][0, 0].real)
    active = np.arange(integrals.noccupied, len(integrals.labels), dtype=np.int32)
    if matrix.shape != (len(active), len(active)):
        raise RuntimeError("incomplete eight-state model matrix")
    np.savez_compressed(
        directory / "model_space_matrix.npz",
        matrix_Ha=matrix,
        global_active_indices=active,
    )
    times = {
        name.strip(): float(value)
        for name, value in re.findall(r"^  (.+?)\s{2,}(\d+\.\d+)\s*$", log, re.M)
    }
    update = max(max(v[2:]) for v in statuses.values())
    bound = 6 * float(np.max(abs(np.diag(integrals.fock_Ha)))) * update
    return {
        "core_energy_Ha": core,
        "statuses": statuses,
        "raw_equation_bound_Ha": bound,
        "backend_reported_stage_seconds": times,
        "reported_max_owned_memory_bytes": int(
            re.search(r"max memory usage = (\d+) bytes", log)[1]
        ),
        "core_and_all_eight_attachment_energies_Ha": [
            core,
            *np.sort(np.linalg.eigvals(matrix).real + core).tolist(),
        ],
    }


def backend(integrals, template, binary, directory, workers, scope):
    from atomx.methods.fscc import compute

    before = machine()
    if (
        before["available_bytes"] < CAP + 8 * 1024**3
        or before["free_disk_bytes"] < 8 * 1024**3
    ):
        raise MemoryError(
            "resource admission failed without shrinking the declared fixture"
        )
    tolerance = 1e-14 if integrals.noccupied == 2 else 1e-13
    if scope == "bare":
        directory.mkdir(exist_ok=False)
        for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json"):
            shutil.copy2(template / name, directory / name)
        text = (template / "cc.inp").read_text()
        text = re.sub(r"nthreads \d+", f"nthreads {workers}", text)
        (directory / "cc.inp").write_text(text)
    with BackendSampler(workers, directory) as sampler:
        start = time.perf_counter()
        if scope == "wrapper":
            compute(
                integrals,
                executable=binary,
                directory=directory,
                model="CCSDT",
                sector="0h1p",
                active_indices=tuple(range(integrals.noccupied, len(integrals.labels))),
                tolerance=tolerance,
                residual_bound_tolerance_Ha=1e-11,
                memory_bytes=512 * 1024**2,
                tile_size=4,
                threads=workers,
                timeout_seconds=120,
                environment=environment(workers),
                tensor_symmetry="C1",
            )
        else:
            with (directory / "cc.log").open("w") as log:
                process = subprocess.run(
                    [str(binary), "--no-clean", "cc.inp"],
                    cwd=directory,
                    env=environment(workers),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            if process.returncode:
                raise RuntimeError(f"nonzero EXP-T return code {process.returncode}")
        elapsed = time.perf_counter() - start
    after = machine()
    parsed = parse_backend(directory, integrals)
    gate = guard(before, after, workers)
    gate["actual_thread_count_within_parent_cap"] = (
        sampler.max_threads <= 8 and not sampler.thread_cap_violation
    )
    result = dict(
        parsed,
        scope=scope,
        workers=workers,
        wall_seconds=elapsed,
        load_before=before,
        load_after=after,
        timing_environment_gate=gate,
        controlled_timing=all(gate.values()),
        executable_sha256=sha(binary),
        actual_runtime=sampler.data(),
        directory=str(directory.relative_to(OWNER)),
        inputs={
            name: sha(directory / name)
            for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json", "cc.inp")
        },
    )
    dump(directory / "benchmark_sample.json", result)
    return result


def complete_arrays(reader, directory, n):
    with np.load(directory / "model_space_matrix.npz") as archive:
        result = {"model_space_matrix_Ha": archive["matrix_Ha"].copy()}
    for name in TENSORS:
        result[name] = reader.decode_diagram(directory / "scratch" / (name + ".dg"), n)[
            0
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
    audit = load(
        OLD / "parallel_six_fscc_scope001/audit.py", "external_benchmark_audit"
    )
    reader, oracle = audit.helpers()
    from atomx.methods.fscc import load_integrals, resource_estimate

    files = [
        Path(__file__),
        OWNER.parent / "contract.json",
        *REPO.joinpath("src/atomx").rglob("*.py"),
        OLD / "parallel_six_fscc_scope001/audit.py",
        OLD / "parallel_fscc012/probe.py",
        OLD / "probe_bloch_oracle.py",
        REPO / "tests/test_transverse_spinor_integrals.py",
        REPO / "tests/test_spinor_integrals_fscc.py",
    ]
    for binary in (CANDIDATE, DEFAULT):
        files.extend(
            [
                binary,
                binary.parent / "atomx_backend.json",
                binary.parent / "CMakeCache.txt",
                binary.parent / "CMakeFiles/expt.x.dir/link.txt",
                binary.parent / "CMakeFiles/expt.x.dir/flags.make",
                binary.parent / "openblas/config.h",
                binary.parent / "openblas/lib/libopenblas.a",
            ]
        )
        source = binary.parent.with_name(binary.parent.name.removesuffix("-build"))
        files.extend(source.joinpath("src/rcc").rglob("*.c"))
        files.extend(source.joinpath("src/rcc").rglob("*.h"))
    files = sorted(set(p.resolve() for p in files if p.is_file()))
    sources = {str(p.relative_to(REPO)): sha(p) for p in files}
    with tarfile.open(output / "source_inputs.tar.gz", "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(REPO)))
    inputs = {}
    integrals = {}
    setup = {}
    for case, original in CASES.items():
        destination = output / "input" / case
        destination.mkdir(parents=True)
        for name in ("integrals.json", "ATOMX_ONE", "ATOMX_TWO"):
            shutil.copy2(original / name, destination / name)
        t = time.perf_counter()
        integrals[case] = load_integrals(destination)
        setup[case] = {
            "input_load_seconds": time.perf_counter() - t,
            "estimated_resources": resource_estimate(
                integrals[case], model="CCSDT", sector="0h1p", active_count=8
            ),
            "input_origin": str(original.relative_to(REPO)),
            "labels": integrals[case].labels,
            "noccupied": integrals[case].noccupied,
            "complete_model_indices": list(
                range(integrals[case].noccupied, len(integrals[case].labels))
            ),
        }
        inputs[case] = {
            name: sha(destination / name)
            for name in ("integrals.json", "ATOMX_ONE", "ATOMX_TWO")
        }
    dump(
        output / "contract.json",
        {
            "parent_contract_sha256": sha(OWNER.parent / "contract.json"),
            "source_sha256": sources,
            "inputs": inputs,
            "setup": setup,
            "machine": machine(),
            "address_cap_bytes": CAP,
            "hypothesis": "Full signed finite equations distinguish old backend Fov-T2 defect from physical CCSDT truncation; identical full input permits backend strong scaling and wrapper overhead measurements.",
            "fixed": "Exact archived complete dynamic transverse signed integrals; C1 tensor blocks; tilesize 4; all eight active spinors; no warm-amplitude restart; no cutoff or H change.",
            "gates": {
                "complete_projected_equations_Ha": 1e-11,
                "heff_projection_Ha": 1e-10,
                "same_target_full_array_Ha": 1e-11,
            },
            "repeats": 5,
            "workers": [1, 2, 4, 8],
            "stopping": "600 s under timing lock; no gate relaxation, no source changes; preserve any failure.",
            "scope_limits": "Finite Fermi CI oracle, not production CI performance; wrapper inherits EXP-T. Prebuilt source build and radial/orbital production not timed. CPU affinity unchanged.",
            "runtime_thread_policy": "OMP_THREAD_LIMIT=workers and OMP_MAX_ACTIVE_LEVELS=1; observer stops owned child if observed Threads exceeds requested workers. Parent cap remains 8. Original requested-only failure remains run001/stopped_thread_budget.json.",
        },
    )
    records, accuracy, ci_records = [], {}, []
    templates, baselines = {}, {}
    with timing_lock():
        timed_start = time.perf_counter()
        for case, a in integrals.items():
            accuracy[case] = {}
            for label, binary in (("default", DEFAULT), ("corrected", CANDIDATE)):
                directory = output / f"accuracy_{case}_{label}"
                record = backend(a, None, binary, directory, 1, "wrapper")
                metrics, arrays = audit.audit_case(directory)
                np.savez_compressed(directory / "full_signed_audit.npz", **arrays)
                gates = {
                    "core_projected": metrics["core_projected_rank123_max_Ha"] <= 1e-11,
                    "attachment_projected": metrics[
                        "attachment_projected_rank012_max_Ha"
                    ]
                    <= 1e-11,
                    "heff_projection": metrics["heff_signed_projection_max_Ha"]
                    <= 1e-10,
                    "intermediate_normalization": metrics[
                        "intermediate_normalization_max"
                    ]
                    <= 1e-12,
                    "positive_metric": metrics["physical_metric_min_eigenvalue"] > 0,
                }
                accuracy[case][label] = dict(
                    metrics,
                    gates=gates,
                    sample=record,
                    full_H_FCI_error_is_required_zero=(case == "two_core"),
                    exact_two_core_gate=(
                        metrics["full_attachment_bloch_max_Ha"] <= 1e-11
                        and metrics["attachment_fci_method_error_max_Ha"] <= 1e-11
                    )
                    if case == "two_core"
                    else None,
                )
                dump(directory / "accuracy.json", accuracy[case][label])
                if label == "corrected":
                    if not all(gates.values()) or (
                        case == "two_core"
                        and not accuracy[case][label]["exact_two_core_gate"]
                    ):
                        raise RuntimeError(
                            "corrected scientific gate failed before timing"
                        )
                    templates[case] = directory
                    baselines[case] = complete_arrays(reader, directory, len(a.labels))
            print(
                "accuracy",
                case,
                {
                    k: v["attachment_projected_rank012_max_Ha"]
                    for k, v in accuracy[case].items()
                },
                flush=True,
            )
        # Five paired full wrapper / bare executions in alternating orders.
        for round_number in range(5):
            order = (
                ("bare", "wrapper") if round_number % 2 == 0 else ("wrapper", "bare")
            )
            for scope in order:
                directory = output / f"wrapper_pair_r{round_number}_{scope}"
                sample = backend(
                    integrals["two_core"],
                    templates["two_core"],
                    CANDIDATE,
                    directory,
                    1,
                    scope,
                )
                observed = complete_arrays(reader, directory, 10)
                defect = max(
                    float(np.max(abs(observed[k] - baselines["two_core"][k])))
                    for k in observed
                )
                sample.update(
                    repeat=round_number,
                    fixed_full_array_defect_Ha=defect,
                    correctness_pass=defect <= 1e-11,
                )
                records.append(sample)
        # Five interleaved worker sweeps; every run has complete signed output.
        for round_number in range(5):
            order = (1, 2, 4, 8) if round_number % 2 == 0 else (8, 4, 2, 1)
            for workers in order:
                if time.perf_counter() - timed_start > 570:
                    raise TimeoutError("predeclared 600-second lane budget")
                directory = output / f"scaling_r{round_number}_p{workers}"
                sample = backend(
                    integrals["four_core"],
                    templates["four_core"],
                    CANDIDATE,
                    directory,
                    workers,
                    "bare",
                )
                observed = complete_arrays(reader, directory, 12)
                defects = {
                    k: float(np.max(abs(observed[k] - baselines["four_core"][k])))
                    for k in observed
                }
                sample.update(
                    repeat=round_number,
                    fixed_full_array_defects_Ha=defects,
                    correctness_pass=max(defects.values()) <= 1e-11,
                )
                if round_number == 0:
                    metrics, arrays = audit.audit_case(directory)
                    np.savez_compressed(directory / "full_signed_audit.npz", **arrays)
                    sample["independent_projected_audit"] = metrics
                    sample["correctness_pass"] &= (
                        metrics["core_projected_rank123_max_Ha"] <= 1e-11
                        and metrics["attachment_projected_rank012_max_Ha"] <= 1e-11
                        and metrics["heff_signed_projection_max_Ha"] <= 1e-10
                    )
                records.append(sample)
                print(
                    "scaling",
                    round_number,
                    workers,
                    sample["wall_seconds"],
                    sample["correctness_pass"],
                    flush=True,
                )
        # Complete matrix construction + full eigensystem and eigensystem-only
        # are separate scopes. Neither includes native orbital/integral setup.
        for round_number in range(5):
            order = (
                ("two_core", "four_core")
                if round_number % 2 == 0
                else ("four_core", "two_core")
            )
            for case in order:
                a = integrals[case]
                before = machine()
                start = time.perf_counter()
                mc, hc = oracle.determinant_matrix(a, a.noccupied)
                ma, ha = oracle.determinant_matrix(a, a.noccupied + 1)
                assembly_seconds = time.perf_counter() - start
                start = time.perf_counter()
                ec, vc = np.linalg.eigh(hc)
                ea, va = np.linalg.eigh(ha)
                solve_seconds = time.perf_counter() - start
                after = machine()
                residual = max(
                    float(np.max(abs(hc @ vc - vc * ec))),
                    float(np.max(abs(ha @ va - va * ea))),
                )
                metric = max(
                    float(np.max(abs(vc.conj().T @ vc - np.eye(len(mc))))),
                    float(np.max(abs(va.conj().T @ va - np.eye(len(ma))))),
                )
                gate = guard(before, after, 1)
                np.savez_compressed(
                    output / f"finite_ci_{case}_r{round_number}.npz",
                    core_masks=mc,
                    attachment_masks=ma,
                    core_H_Ha=hc,
                    attachment_H_Ha=ha,
                    core_all_roots_Ha=ec,
                    attachment_all_roots_Ha=ea,
                    core_all_vectors=vc,
                    attachment_all_vectors=va,
                )
                ci_records.append(
                    dict(
                        case=case,
                        repeat=round_number,
                        assembly_seconds=assembly_seconds,
                        eigensystem_seconds=solve_seconds,
                        full_matrix_plus_full_eigensystem_seconds=assembly_seconds
                        + solve_seconds,
                        full_H_residual_Ha=residual,
                        orthogonality_max=metric,
                        complete_core_dimension=len(mc),
                        complete_attachment_dimension=len(ma),
                        all_selected_attachment_roots_Ha=ea[:8].tolist(),
                        correctness_pass=residual <= 1e-11 and metric <= 1e-12,
                        controlled_timing=all(gate.values()),
                        timing_environment_gate=gate,
                        load_before=before,
                        load_after=after,
                    )
                )
        dump(output / "raw_samples.json", {"backend": records, "finite_ci": ci_records})
    source_unchanged = all(sha(REPO / p) == digest for p, digest in sources.items())

    def summary(samples, field="wall_seconds"):
        numbers = [s[field] for s in samples]
        return {
            "repeats": len(numbers),
            "observed_seconds": numbers,
            "median_seconds": float(np.median(numbers)),
            "minimum_seconds": min(numbers),
            "maximum_seconds": max(numbers),
            "controlled_timing": all(
                s["controlled_timing"] and s["correctness_pass"] for s in samples
            ),
        }

    scaling = {
        str(p): summary(
            [
                r
                for r in records
                if r["directory"].split("/")[-1].startswith("scaling_")
                and r["workers"] == p
            ]
        )
        for p in (1, 2, 4, 8)
    }
    for p in (1, 2, 4, 8):
        valid = (
            scaling["1"]["controlled_timing"]
            and scaling[str(p)]["controlled_timing"]
            and source_unchanged
        )
        scaling[str(p)]["speedup"] = (
            scaling["1"]["median_seconds"] / scaling[str(p)]["median_seconds"]
            if valid
            else None
        )
        scaling[str(p)]["parallel_efficiency"] = (
            scaling[str(p)]["speedup"] / p if valid else None
        )
    wrapper = {
        scope: summary(
            [
                r
                for r in records
                if Path(r["directory"]).name.startswith("wrapper_pair_")
                and r["scope"] == scope
            ]
        )
        for scope in ("bare", "wrapper")
    }
    wrapper["median_wrapper_extra_seconds"] = (
        wrapper["wrapper"]["median_seconds"] - wrapper["bare"]["median_seconds"]
        if source_unchanged
        and all(wrapper[k]["controlled_timing"] for k in ("bare", "wrapper"))
        else None
    )
    result = {
        "status": "finite_comparison_completed",
        "accuracy": accuracy,
        "backend_scaling": scaling,
        "wrapper_same_backend_overhead": wrapper,
        "finite_CI_oracle_cost": {
            case: {
                field: summary([r for r in ci_records if r["case"] == case], field)
                for field in (
                    "assembly_seconds",
                    "eigensystem_seconds",
                    "full_matrix_plus_full_eigensystem_seconds",
                )
            }
            for case in CASES
        },
        "gates": {
            "source_unchanged": source_unchanged,
            "all_repeats_full_signed_correct": all(
                r["correctness_pass"] for r in records
            ),
            "all_finite_CI_residual_metric_pass": all(
                r["correctness_pass"] for r in ci_records
            ),
        },
        "source_sha256": sources,
        "inputs": inputs,
        "source_archive_sha256": sha(output / "source_inputs.tar.gz"),
        "cold_source_build_seconds": None,
        "cold_atomic_orbital_integral_production_seconds": None,
        "total_wall_seconds": time.perf_counter() - started,
        "parent_process_peak_RSS_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024,
        "performance_ranking": None,
        "highest_level_reached": False,
        "limits": [
            "Atomx FSCC calls EXP-T; wrapper and backend are not independent solvers.",
            "Two-core/ten-spinor CCSDT can exhaust the finite three-electron excitation ranks, not experimental physics.",
            "Four-core/twelve-spinor retains CCSDT but genuine T4/S4 method errors remain; compare retained equations separately.",
            "Bare stage includes executable startup, solve, scratch I/O and exit; pre-export setup and parsing excluded.",
            "Wrapper stage includes full integral export, executable validation, solve, I/O and result parsing; Python import excluded.",
            "Complete finite CI oracle cost includes all fixed-N determinant rows and eigensystems, not a production CI software benchmark.",
            "Tiny fixtures cannot predict large-scale software speed or parallel efficiency.",
        ],
    }
    dump(output / "result.json", result)
    manifests = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "output_sha256.json"
    }
    dump(output / "output_sha256.json", manifests)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "scaling": scaling,
                "wrapper": wrapper,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
