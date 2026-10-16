"""Root-sealed controlled warm trial/gradient benchmark; stdlib-only on import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def admit(execution_path, *, require_affinity=False):
    execution = json.loads(Path(execution_path).read_text())
    contract = json.loads((HERE / "contract.json").read_text())
    preparation = json.loads((HERE / "preparation_verification.json").read_text())
    required = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "helper_sha256": sha(HERE / "helper.py"), "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "CPU": 30, "controller_CPU": 0, "max_H": 14, "max_CI": 14,
                "max_gradient": 14, "max_scipy_eigh": 70, "wall_seconds": 300,
                "combined_RSS_GiB": 3.95, "root_reserves_no_other_H_CI_gradient": True,
                "both_history32_pilots_foreground_closed": True,
                "purpose": "controlled_warm_fresh_complete29_trial_gradient_scoped"}
    if any(execution.get(k) != v for k, v in required.items()):
        raise ValueError("Exact root execution seal missing or changed")
    if not preparation.get("all_passed") or preparation.get("driver_sha256") != sha(__file__):
        raise ValueError("Prepared source mismatch")
    if preparation.get("helper_sha256") != sha(HERE / "helper.py") or preparation.get("contract_sha256") != sha(HERE / "contract.json"):
        raise ValueError("Prepared helper/contract mismatch")
    pilots = execution.get("history32_pilot_receipts_sha256", {})
    if len(pilots) != 2:
        raise ValueError("Two closed pilot receipt identities required")
    for name, expected in pilots.items():
        if sha(REPO / name) != expected:
            raise ValueError("Pilot receipt mismatch")
        if json.loads((REPO / name).read_text()).get("status") != "foreground_closed":
            raise ValueError("Prior pilot is not closed")
    for row in contract["inputs"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Input changed: " + row["path"])
    for variant in ("ordinary", "candidate"):
        row = contract[variant]
        identity = json.loads((REPO / row["identity"]).read_text())
        if identity["source_map_sha256"] != row["source_map_sha256"]:
            raise ValueError("Source map changed")
        for name, expected in identity["all_atomx_python_sha256"].items():
            if sha(REPO / row["archive"] / name.removeprefix("src/")) != expected:
                raise ValueError("Frozen source changed: " + name)
    qualified = json.loads((REPO / contract["inputs"]["qualification003_result"]["path"]).read_text())
    shared = json.loads((REPO / contract["inputs"]["shared_adoption_completion"]["path"]).read_text())
    if not qualified.get("all_equivalence_gates_passed") or shared.get("test_counts") != {"tests": 123, "errors": 0, "failures": 0, "skipped": 0}:
        raise ValueError("Closed qualification/adoption prerequisite failed")
    if any(os.environ.get(k) != "1" for k in THREADS) or os.environ.get("NUMBA_DISABLE_JIT", "0") != "0":
        raise ValueError("All numerical threads must be explicitly one")
    if not {0, 30} <= set(range(os.cpu_count() or 0)):
        raise ValueError("Declared CPUs unavailable")
    if require_affinity and not {0, 30} <= os.sched_getaffinity(0):
        raise ValueError("CPU0 and CPU30 must both be permitted before pinning")
    return contract, execution


def proc_rss(pid):
    try:
        for row in Path(f"/proc/{pid}/status").read_text().splitlines():
            if row.startswith("VmRSS:"):
                return int(row.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def host():
    row = next(s for s in Path("/proc/stat").read_text().splitlines() if s.startswith("cpu30 "))
    ticks = list(map(int, row.split()[1:]))
    freq = Path("/sys/devices/system/cpu/cpu30/cpufreq/scaling_cur_freq")
    return {"load": list(os.getloadavg()), "online_cpus": os.cpu_count(), "cpu30_ticks": ticks,
            "cpu30_frequency_kHz": int(freq.read_text()) if freq.exists() else None,
            "monotonic": time.monotonic()}


def host_gate(before, after, idle=False):
    delta = [b - a for a, b in zip(before["cpu30_ticks"], after["cpu30_ticks"], strict=True)]
    total = sum(delta[:8])
    busy = 1. - (delta[3] + delta[4]) / total if total else 0.
    interference = sum(delta[i] for i in (2, 4, 5, 6, 7)) / total if total else 0.
    load_ok = max(before["load"][0], after["load"][0]) <= .5 * (after["online_cpus"] or 1)
    return {"before": before, "after": after, "busy_fraction": busy,
            "system_iowait_irq_softirq_steal_fraction": interference,
            "passed": load_ok and (busy <= .05 if idle else interference <= .05)}


class Calls:
    def __init__(self, caps):
        self.caps, self.started, self.completed = caps, {}, {}
        self.raw_hs, self.centered = [], []

    def wrap(self, name, function, capture=None):
        def call(*args, **kwargs):
            self.started[name] = self.started.get(name, 0) + 1
            if name in self.caps and self.started[name] > self.caps[name]:
                raise RuntimeError("Call cap exceeded: " + name)
            answer = function(*args, **kwargs)
            self.completed[name] = self.completed.get(name, 0) + 1
            if capture is not None:
                capture(args, answer)
            return answer
        return call

    def install(self):
        import numpy as np
        from scipy import linalg

        import atomx.methods.ci as ci_module
        import atomx.methods.csf.variational as variational
        import atomx.numerics.bspline.convol as convol
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.tracking as tracking

        def capture_hs(args, answer):
            self.raw_hs.append((int(args[1]), answer[0].copy(), answer[1].copy()))

        original = dirac.build_dirac_matrices
        variational.build_dirac_matrices = self.wrap("initial_hS", original, capture_hs)
        matrix.build_dirac_matrices = self.wrap("CI_internal_hS", original, capture_hs)
        dirac.solve_dirac = self.wrap("Dirac_eigensolve", dirac.solve_dirac)
        matrix.build_ci_matrix = self.wrap("H", matrix.build_ci_matrix)
        variational.ci_solve = self.wrap("CI", variational.ci_solve)
        variational.CSFOrbitalFunctional.gradient_from_weights = self.wrap("gradient", variational.CSFOrbitalFunctional.gradient_from_weights)
        variational.CSFOrbitalFunctional._pair_potential = self.wrap("gradient_source_pair", variational.CSFOrbitalFunctional._pair_potential)
        tracking.CIBasisTracking.resolve = self.wrap("tracking_resolution", tracking.CIBasisTracking.resolve)
        tracking.ci_subspace_overlaps = self.wrap("tracking_overlap", tracking.ci_subspace_overlaps)
        np.linalg.svd = self.wrap("tracking_SVD", np.linalg.svd)
        np.linalg.eigvalsh = self.wrap("eigvalsh", np.linalg.eigvalsh)
        np.linalg.eigh = self.wrap("numpy_eigh", np.linalg.eigh)
        np.load = self.wrap("npz_load", np.load)
        linalg.eigh = self.wrap("scipy_eigh", linalg.eigh)

        def capture_centered(args, answer):
            self.centered.append({"size": len(args[0]), "origin_Ha": float(answer[2]),
                                  "first_real_diagonal_Ha": float(args[0][0, 0].real)})

        ci_module._centered_dense_eigh = self.wrap("centered_dense_helper", ci_module._centered_dense_eigh, capture_centered)
        density, dense, semi, summation = (convol._density_sym, convol._convol_case1_t,
                                          convol._convol_case1_semiseparable_t, convol._sum_amb_banded)
        for module in (convol, matrix):
            module._density_sym = self.wrap("density_band", density)
            module._convol_case1_t = self.wrap("direct_full_bank", dense)
            module._convol_case1_semiseparable_t = self.wrap("direct_semiseparable", semi)
            module._sum_amb_banded = self.wrap("band_scalar", summation)
        matrix._bspline_pair_potential = self.wrap("CI_source_pair", matrix._bspline_pair_potential)
        convol.ConvolCache.get_rkb_t = self.wrap("bank_get", convol.ConvolCache.get_rkb_t)
        convol.ConvolCache.get_coulomb_factors = self.wrap("factor_get", convol.ConvolCache.get_coulomb_factors)
        for name in ("_density_sym", "_convol_case1_t", "_convol_case1_semiseparable_t"):
            setattr(variational, name, getattr(convol, name))
        variational._banded_to_full_add = self.wrap("gradient_band_expand", variational._banded_to_full_add)


def worker(variant, execution_path, deadline):
    worker_start = time.monotonic()
    contract, _execution = admit(execution_path)
    admission_seconds = time.monotonic() - worker_start
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {30})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Shared300s cap")))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(MemoryError("Root/controller resource stop")))
    signal.alarm(max(1, int(deadline - time.monotonic())))
    output = HERE / "actual" / variant
    output.mkdir(exist_ok=False)
    protocol = sys.stdout
    log = (output / "worker.log").open("w", buffering=1)
    sys.stdout = log

    def reply(value):
        protocol.write(json.dumps(plain(value), allow_nan=False) + "\n")
        protocol.flush()

    calls, checks = Calls(contract["worker_caps"][variant]), []
    def flush(status):
        write(output / "progress.json", {"status": status, "started": calls.started,
              "completed": calls.completed, "checks": checks})

    def gate(name, condition, error=None, limit=None):
        passed = bool(condition)
        row = {"name": name, "passed": passed}
        if error is not None:
            row.update(error=float(error), limit=float(limit))
        checks.append(row)
        if not passed:
            raise ValueError("Scientific gate failed: " + name)

    try:
        source = REPO / contract[variant]["archive"]
        if any(n == "atomx" or n.startswith("atomx.") for n in sys.modules):
            raise ValueError("Atomx imported before source admission")
        sys.path.insert(0, str(source))
        import_start = time.monotonic()
        import helper
        import numba
        import numpy as np
        import scipy
        from threadpoolctl import threadpool_info

        def close(name, a, b, atol=0., rtol=0.):
            delta = abs(np.asarray(a) - np.asarray(b))
            bounds = atol + rtol * abs(np.asarray(b))
            gate(name, np.all(np.isfinite(delta)) and np.all(delta <= bounds),
                 np.max(delta, initial=0.), np.max(bounds, initial=0.))

        def check(name, error, limit):
            gate(name, np.isfinite(error) and error <= limit, error, limit)

        calls.install()
        pools = threadpool_info()
        gate("actual_BLAS_threads_one", bool(pools) and all(p["num_threads"] == 1 for p in pools))
        gate("actual_Numba_threads_one", numba.get_num_threads() == 1)
        import_install_seconds = time.monotonic() - import_start
        start = time.monotonic()
        point = helper.prepare_point(contract, calls.raw_hs, gate, close, plain)
        setup_seconds = time.monotonic() - start
        gate("worker_affinity", os.sched_getaffinity(0) == {30})
        for name, module in tuple(sys.modules.items()):
            if name.startswith("atomx") and getattr(module, "__file__", None):
                gate("frozen_module_" + name, Path(module.__file__).resolve().is_relative_to(source.resolve()))
        flush("ready")
        reply({"status": "ready", "variant": variant, "setup_seconds": setup_seconds,
               "source_input_admission_seconds": admission_seconds,
               "imports_instrumentation_thread_admission_seconds": import_install_seconds,
               "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
               "threadpools": pools, "Numba_threads": numba.get_num_threads(),
               "pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0)),
               "started": calls.started, "completed": calls.completed})
        cold, previous_ci, completed_chains = None, None, 0
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "stop":
                for name, expected in contract["worker_exact_calls"].items():
                    gate("final_started_" + name, calls.started.get(name, 0) == expected)
                    gate("final_completed_" + name, calls.completed.get(name, 0) == expected)
                gate("complete7chains", completed_chains == 7)
                for name, expected in contract["variant_exact_calls"][variant].items():
                    gate("variant_final_started_" + name, calls.started.get(name, 0) == expected)
                    gate("variant_final_completed_" + name, calls.completed.get(name, 0) == expected)
                admit(execution_path)
                flush("closed")
                reply({"status": "closed", "started": calls.started, "completed": calls.completed,
                       "checks": checks, "completed_chains": completed_chains})
                return
            if request["command"] == "compare":
                start = time.monotonic()
                clusters = helper.compare_saved_chains(contract, request["ordinary"], request["candidate"], check)
                flush("pair_compared")
                reply({"status": "compared", "cluster_count": len(clusters),
                       "comparison_seconds": time.monotonic() - start})
                continue
            if request["command"] != "chain" or completed_chains >= 7:
                raise ValueError("Unexpected/extra workload")
            chain_output = output / request["label"]
            chain_output.mkdir(exist_ok=False)
            checks_before = len(checks)
            calls_before = dict(calls.completed)
            calls.centered.clear()
            flush("before_energy_" + request["label"])
            start = time.monotonic()
            energy, ci = point.functional.evaluate(point.waves, gradient=False, ci_result=None)
            energy_seconds = time.monotonic() - start
            flush("energy_completed_" + request["label"])
            start = time.monotonic()
            gate("fresh_CI", ci is not previous_ci)
            gate("fresh_H_owner", previous_ci is None or ci.hamiltonian_snapshot is not previous_ci.hamiltonian_snapshot)
            trial = helper.validate_trial(point, energy, ci, calls.centered, gate, close)
            trial_validation_seconds = time.monotonic() - start
            flush("before_gradient_" + request["label"])
            start = time.monotonic()
            gradient_result = point.functional.evaluate(point.waves, ci_result=trial.ci, gradient=True)
            gradient_seconds = time.monotonic() - start
            flush("gradient_completed_" + request["label"])
            paths = helper.finish_checked_chain(point, trial, gradient_result, variant, contract,
                                                chain_output, write, gate, close)
            repeat_start = time.monotonic()
            if cold is not None:
                helper.compare_saved_chains(contract, cold, chain_output, check, same_worker=True)
            repeat_seconds = time.monotonic() - repeat_start
            if cold is None:
                cold = chain_output
            previous_ci = ci
            completed_chains += 1
            for name, expected in contract["per_chain_exact_calls"].items():
                gate("chain_calls_" + name, calls.completed.get(name, 0) - calls_before.get(name, 0) == expected)
            gate("per_chain_geometry_hS_after_cold", completed_chains == 1 or
                 all(calls.completed.get(n, 0) == calls_before.get(n, 0) for n in ("initial_hS", "CI_internal_hS")))
            timings = {"energy_trial_seconds": energy_seconds, "accepted_gradient_seconds": gradient_seconds,
                       "primary_seconds": energy_seconds + gradient_seconds,
                       "trial_validation_seconds": trial_validation_seconds,
                       "sameworker_full_repeat_validation_seconds": repeat_seconds,
                       **{k: paths[k] for k in ("validation_metadata_seconds", "checked_compressed_archive_seconds")}}
            write(chain_output / "source_header.json", {"source_map_sha256": contract[variant]["source_map_sha256"],
                  "driver_sha256": sha(__file__), "helper_sha256": sha(HERE / "helper.py"),
                  "input_files_sha256": contract["inputs"], "H_bytes_sha256": hashlib.sha256(trial.h.tobytes()).hexdigest(),
                  "actual_new_H_number": calls.completed["H"], "fresh_owner": True})
            result = {"status": "chain_passed", "label": request["label"], "variant": variant,
                      "directory": str(chain_output), "timings": timings,
                      "started": dict(calls.started), "completed": dict(calls.completed),
                      "checks": checks[checks_before:]}
            write(chain_output / "result.json", result)
            flush("completed_" + request["label"])
            reply(result)
        raise RuntimeError("Controller ended before closed ledger")
    except BaseException as error:
        flush("failed")
        failure = {"status": "failed", "variant": variant, "error": repr(error),
                   "started": calls.started, "completed": calls.completed, "checks": checks}
        write(output / "failure.json", failure)
        reply(failure)
        raise


def controller(execution_path):
    admission_start = time.monotonic()
    permitted_cpus_before_pinning = sorted(os.sched_getaffinity(0))
    contract, execution = admit(execution_path, require_affinity=True)
    admission_seconds = time.monotonic() - admission_start
    os.sched_setaffinity(0, {0})
    actual = HERE / "actual"
    actual.mkdir(exist_ok=False)
    with (actual / "once_marker.json").open("x") as handle:
        handle.write(json.dumps({"execution_contract_sha256": sha(execution_path), "driver_sha256": sha(__file__)}))
        handle.flush()
        os.fsync(handle.fileno())
    start, children, rows, pairs = time.monotonic(), {}, [], []
    deadline = start + 300
    child_lock = threading.Lock()
    monitor_stop, resource_failed = threading.Event(), threading.Event()
    memory = {"peak_combined_worker_RSS_bytes": 0, "sampling_seconds": .25}

    def monitor():
        while not monitor_stop.wait(.25):
            with child_lock:
                current = tuple(children.values())
            rss = sum(proc_rss(child.pid) for child in current)
            memory["peak_combined_worker_RSS_bytes"] = max(rss, memory["peak_combined_worker_RSS_bytes"])
            if rss > int(3.95 * 2**30) or time.monotonic() >= deadline:
                resource_failed.set()
                for child in current:
                    if child.poll() is None:
                        child.terminate()
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()

    def receive(variant):
        line = children[variant].stdout.readline()
        if not line:
            raise RuntimeError("Worker closed without response: " + variant)
        result = json.loads(line)
        if result["status"] == "failed":
            raise RuntimeError("Worker scientific/resource failure: " + json.dumps(result))
        if resource_failed.is_set():
            raise MemoryError("Combined memory/shared wall stop")
        return result

    def ask(variant, command):
        child = children[variant]
        child.stdin.write(json.dumps(command) + "\n")
        child.stdin.flush()
        return receive(variant)

    try:
        for variant in ("ordinary", "candidate"):
            env = dict(os.environ)
            cache = actual / (variant + "_numba_cache")
            cache.mkdir()
            env["NUMBA_CACHE_DIR"] = str(cache)
            stderr = (actual / (variant + "_stderr.log")).open("w")
            child = subprocess.Popen(
                ["/data/code/atomx/.venv/bin/python", str(HERE / "run.py"), "--worker", variant,
                 "--execution", str(Path(execution_path).absolute()), "--deadline", str(deadline)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True,
                env=env, bufsize=1)
            with child_lock:
                children[variant] = child
            ready = receive(variant)
            rows.append(ready)
            write(actual / "progress.json", {"status": "setup", "rows": rows})
        controlled = True
        for pair_index, label in enumerate(("cold", "warmup", "pair1", "pair2", "pair3", "pair4", "pair5")):
            idle0 = host()
            time.sleep(1.)
            idle = host_gate(idle0, host(), idle=True)
            before = host()
            order = ("candidate", "ordinary") if label in ("pair2", "pair4") else ("ordinary", "candidate")
            outcomes = {}
            for variant in order:
                outcomes[variant] = ask(variant, {"command": "chain", "label": label})
                rows.append(outcomes[variant])
                write(actual / "progress.json", {"status": label, "rows": rows, "pairs": pairs})
            running = host_gate(before, host())
            comparison = ask("ordinary", {"command": "compare", "ordinary": outcomes["ordinary"]["directory"],
                                           "candidate": outcomes["candidate"]["directory"]})
            passed = idle["passed"] and running["passed"]
            controlled = controlled and passed
            pair = {"index": pair_index, "label": label, "order": order, "outcomes": outcomes,
                    "host_idle_guard": idle, "host_running_guard": running, "host_passed": passed,
                    "complete_scientific_comparison": comparison}
            pairs.append(pair)
            write(actual / "progress.json", {"status": "pair_completed", "rows": rows, "pairs": pairs})
        final = {variant: ask(variant, {"command": "stop"}) for variant in ("ordinary", "candidate")}
        for child in children.values():
            if child.wait(timeout=5) != 0:
                raise RuntimeError("Worker nonzero final exit")
        ratios = [p["outcomes"]["ordinary"]["timings"]["primary_seconds"] /
                  p["outcomes"]["candidate"]["timings"]["primary_seconds"] for p in pairs if p["label"].startswith("pair")]
        median = statistics.median(ratios)
        result = {"all_science_gates_passed": True, "all_host_gates_passed": controlled,
                  "scoped_warm_improvement_certified": controlled and median >= 1.05,
                  "median_ordinary_over_candidate_primary_ratio": median, "warm_pair_ratios": ratios,
                  "pairs": pairs, "setup": rows[:2], "final_worker_ledgers": final, "memory": memory,
                  "total_H_CI_gradient_calls": 14, "total_scipy_eigh": 70, "total_hS": 28,
                  "wall_seconds": time.monotonic() - start,
                  "controller_initial_source_input_admission_seconds": admission_seconds,
                  "permitted_cpus_before_pinning": permitted_cpus_before_pinning,
                  "execution_contract_sha256": sha(execution_path), "driver_sha256": sha(__file__),
                  "helper_sha256": sha(HERE / "helper.py"), "contract_sha256": sha(HERE / "contract.json"),
                  "optimizer_GPU_parallel_experiment_stationarity_certified": False}
        write(actual / "result.json", result)
    except BaseException as error:
        write(actual / "failure.json", {"all_science_gates_passed": False, "error": repr(error),
              "rows": rows, "pairs": pairs, "memory": memory, "wall_seconds": time.monotonic() - start,
              "partial_inflight_counts": "use any worker failure/progress; a hard-killed call is incomplete, never inferred completed",
              "controlled_speedup_certified": False})
        raise
    finally:
        monitor_stop.set()
        for child in children.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    parser.add_argument("--worker", choices=("ordinary", "candidate"))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.worker:
        if args.deadline is None:
            raise ValueError("Shared absolute deadline required")
        worker(args.worker, args.execution, args.deadline)
    else:
        controller(args.execution)


if __name__ == "__main__":
    main()
