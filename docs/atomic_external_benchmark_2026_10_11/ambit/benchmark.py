"""Locked, fixed-work AMBiT comparison and its independent native scaling."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import tarfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT.parent / "atomx-references/2026-10-09-ci-fscc"
BUILD = BASE / "AMBiT-build/src"
CONTRACT = json.loads((HERE.parent / "contract.json").read_text())
LOCK = Path(CONTRACT["timing_contract"]["lock_file"])
CAP = CONTRACT["timing_contract"]["max_per_process_address_bytes"]


def digest(path):
    with Path(path).open("rb") as source:
        result = hashlib.file_digest(source, "sha256") if hasattr(hashlib, "file_digest") else None
        if result is None:
            result = hashlib.sha256()
            for block in iter(lambda: source.read(2**20), b""):
                result.update(block)
    return result.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def snapshot():
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(load=os.getloadavg(), affinity=sorted(os.sched_getaffinity(0)),
                available_bytes=int(meminfo["MemAvailable"].split()[0]) * 1024,
                disk_bytes=shutil.disk_usage(HERE).free, utc=stamp())


def preflight(before, workers):
    return dict(cpu_load=max(before["load"]) + workers <= len(before["affinity"]),
                memory=before["available_bytes"] >= CONTRACT["timing_contract"]["max_reserved_total_bytes"]
                + CONTRACT["timing_contract"]["minimum_host_headroom_bytes"],
                disk=before["disk_bytes"] >= CONTRACT["timing_contract"]["minimum_free_disk_bytes"])


def environment(workers):
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    env["OMP_NUM_THREADS"] = str(workers)
    env["OMP_DYNAMIC"] = "FALSE"
    env["PYTHONPATH"] = str(ROOT / "src") + ":" + str(ROOT)
    lib = BASE / "toolchain/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(lib) + ":" + str(lib / "openblas-serial") + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def limit():
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))


def execute(command, directory, tag, workers=1, output_directory=None):
    destination = directory if output_directory is None else output_directory
    before = snapshot()
    checks = preflight(before, workers)
    # Load failure permits observational execution; memory/disk excess reject allocation.
    if not checks["memory"] or not checks["disk"]:
        raise RuntimeError("resource excess before allocation")
    record = dict(command=command, cwd=str(directory), workers=workers, before=before,
                  admission=checks, started_utc=stamp(), address_cap_bytes=CAP)
    started = time.perf_counter()
    with (destination / (tag + ".stdout")).open("w") as out, (destination / (tag + ".stderr")).open("w") as err:
        timed = ["/usr/bin/time", "-f", "%M", "-o", str(destination / (tag + ".maxrss")), *command]
        try:
            child = subprocess.run(timed, cwd=directory, env=environment(workers),
                                   stdout=out, stderr=err, timeout=600, preexec_fn=limit)
            record["returncode"] = child.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = "timeout"
    record["cli_wall_seconds"] = time.perf_counter() - started
    record["finished_utc"] = stamp()
    record["after"] = snapshot()
    record["guard"] = dict(checks, load_change=max(abs(a - b) for a, b in zip(
        before["load"], record["after"]["load"])) <= 4.)
    record["controlled_timing"] = all(record["guard"].values())
    rss_lines = (destination / (tag + ".maxrss")).read_text().splitlines()
    record["peak_rss_bytes"] = int(rss_lines[-1]) * 1024 if rss_lines[-1].isdigit() else None
    (destination / (tag + ".execution.json")).write_text(json.dumps(record, indent=2) + "\n")
    if record["returncode"] != 0:
        raise RuntimeError(f"preserved execution failure: {directory}/{tag}")
    return record


def compile_probe():
    options = []
    for line in (BUILD / "CMakeFiles/ambit.dir/flags.make").read_text().splitlines():
        if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
            options += shlex.split(line.split("=", 1)[1])
    obj, exe = HERE / "timed_sigma.o", HERE / "timed_sigma"
    compile_command = ["/usr/bin/c++", *options, "-c", str(HERE / "timed_sigma.cpp"), "-o", str(obj)]
    link = shlex.split((BUILD / "CMakeFiles/ambit.dir/link.txt").read_text())
    link[link.index("CMakeFiles/ambit.dir/ambit.cpp.o")] = str(obj)
    link[link.index("-o") + 1] = str(exe)
    # Avoid modifying the upstream compiler dependency file.
    link = [arg for arg in link if not arg.startswith("-Wl,--dependency-file=")]
    records = []
    for tag, command in (("compile", compile_command), ("link", link)):
        records.append(execute(command, BUILD, tag, output_directory=HERE))
    return records


def prepare_inputs():
    target = HERE / "fresh_inputs001"
    target.mkdir()
    for name in ("run_matched_sigma.py", "ambit_sigma_probe.cpp"):
        shutil.copy2(ROOT / "docs/ci_mbpt_matched_qspace_2026_10_09" / name, target / name)
    source = (target / "run_matched_sigma.py").read_text().replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[3]")
    source = source.replace("    data=core._bsp_data", """    meta=dict(converged=core.converged,iterations=core.iterations,residual=core.residual,
              total_energy=core.total_energy,orbital_energies=core.orbital_energies,
              subshells=[dict(n=sh.n,kappa=sh.kappa,occupation=sh.occupation) for sh in core.subshells])
    (HERE/'core_metadata.json').write_text(json.dumps(meta,indent=2)+'\\n')
    data=core._bsp_data""")
    # The original link driver would overwrite a generated dependency file in BUILD.
    source = source.replace("    link[start]=str(obj)", "    link[start]=str(obj)\n    link=[arg for arg in link if not arg.startswith('-Wl,--dependency-file=')]")
    (target / "run_matched_sigma.py").write_text(source)
    return target, execute([str(ROOT / ".venv/bin/python"), str(target / "run_matched_sigma.py")], target, "prepare")


def pair_statistics(pairs, left_key, right_key):
    accepted = len(pairs) >= 3 and all(pair["controlled_timing"] and pair["scientific_gate"] for pair in pairs)
    ratios = [pair[left_key] / pair[right_key] for pair in pairs]
    return dict(pairs=len(pairs), controlled=accepted,
                left_median_seconds=statistics.median(pair[left_key] for pair in pairs),
                right_median_seconds=statistics.median(pair[right_key] for pair in pairs),
                observational_ratios=ratios,
                speedup=statistics.median(ratios) if accepted else None)


def matched(inputs):
    results = []
    for round_id in range(5):
        directory = HERE / f"matched_v002_round{round_id:02d}"
        directory.mkdir()
        records = {}
        order = ("atomx", "ambit") if round_id % 2 == 0 else ("ambit", "atomx")
        with LOCK.open("a") as locked:
            fcntl.flock(locked, fcntl.LOCK_EX)
            for software in order:
                if software == "atomx":
                    command = [str(ROOT / ".venv/bin/python"), str(HERE / "atomx_action.py"),
                               str(inputs), str(directory / "atomx_timings.json")]
                else:
                    command = [str(HERE / "timed_sigma"), *[str(inputs / name) for name in
                               ("orbitals.txt", "integrals.txt", "requests.txt")],
                               str(directory / "ambit_timings.json")]
                records[software] = execute(command, directory, software)
        atomx = np.load(directory / "atomx_timings.npz")["output"]
        ambit = np.loadtxt(directory / "ambit.stdout")
        np.savez_compressed(directory / "complete_comparison.npz", atomx=atomx, ambit=ambit, error=ambit - atomx)
        gate = (atomx.shape == (27, 6) and np.array_equal(atomx[:, :3], ambit[:, :3])
                and np.allclose(atomx[:, 3:], ambit[:, 3:], atol=1e-11, rtol=1e-10))
        if not gate:
            raise RuntimeError("same-Hamiltonian complete signed matrix gate failed")
        timings = {name: json.loads((directory / (name + "_timings.json")).read_text())
                   for name in ("atomx", "ambit")}
        results.append(dict(round=round_id, order=order, executions=records, scientific_gate=bool(gate),
                            controlled_timing=all(record["controlled_timing"] for record in records.values()),
                            atomx_seconds=timings["atomx"]["action_seconds"],
                            ambit_seconds=timings["ambit"]["action_seconds"],
                            max_error_Ha=float(np.max(np.abs(ambit[:, 3:] - atomx[:, 3:])))))
        print("matched", round_id, results[-1]["atomx_seconds"], results[-1]["ambit_seconds"], flush=True)
    return dict(orbitals=19, integrals=37350, signed_matrix_entries=27, components=3,
                reference_native=True, shared_radial_integrals=True, experimental_accuracy_comparison=False,
                timing_scope="Warm complete same fixed-energy matrix output: Atomx additionally constructs reusable pole/residue snapshots and captures the radial physical binding; AMBiT directly sums each fixed-energy pp/hh/total request. CLI parse/setup/output excluded; no end-to-end ranking.",
                pairs=results, statistics=pair_statistics(results, "atomx_seconds", "ambit_seconds"))


def native_scaling():
    results = {}
    input_file = ROOT / "docs/ci_fscc_reference_upgrade_2026_10_09/ambit_be_smoke_refined/BeII.input"
    golden = None
    for workers in (2, 4, 8):
        pairs = []
        for round_id in range(5):
            order = (1, workers) if round_id % 2 == 0 else (workers, 1)
            records, outputs = {}, {}
            with LOCK.open("a") as locked:
                fcntl.flock(locked, fcntl.LOCK_EX)
                for worker in order:
                    directory = HERE / f"be_pair{workers}_round{round_id:02d}_threads{worker}"
                    directory.mkdir()
                    shutil.copy2(input_file, directory / "BeII.input")
                    records[worker] = execute([str(BUILD / "ambit"), "BeII.input"], directory, "ambit", worker)
                    output_files = sorted(directory.glob("BeII_native_reference_0.*"))
                    outputs[worker] = {path.name: digest(path) for path in output_files}
                    if not {".basis", ".one.int", ".two.int", ".levels"}.issubset(
                        {"." + path.name.split(".", 1)[1] for path in output_files}
                    ):
                        raise RuntimeError("incomplete full AMBiT run artifacts")
                    text = (directory / "ambit.stdout").read_text()
                    energies = [float(value) for value in re.findall(r"([-0-9.e+]+) /cm", text)]
                    np.savez_compressed(directory / "complete_levels_cm.npz", energies_cm=energies)
                    if len(energies) != 6 or "Finished" not in text:
                        raise RuntimeError("incomplete native level panel")
                    # Exact complete signed output bytes are stronger than the fixed numerical gate.
                    if golden is None:
                        golden = outputs[worker]
            gate = outputs[1] == golden and outputs[workers] == golden
            pairs.append(dict(round=round_id, order=order, executions=records, complete_output_sha256=outputs,
                              scientific_gate=gate,
                              controlled_timing=all(record["controlled_timing"] for record in records.values()),
                              serial_seconds=records[1]["cli_wall_seconds"],
                              parallel_seconds=records[workers]["cli_wall_seconds"]))
            print("native", workers, round_id, pairs[-1]["serial_seconds"], pairs[-1]["parallel_seconds"], gate, flush=True)
        stats = pair_statistics(pairs, "serial_seconds", "parallel_seconds")
        stats["parallel_efficiency"] = stats["speedup"] / workers if stats["speedup"] is not None else None
        results[str(workers)] = dict(pairs=pairs, statistics=stats)
    return dict(input_sha256=digest(input_file), radial_points=4000, software="native upstream AMBiT",
                atomx_matched=False, scopes="Cold scientific workspace each invocation (no old basis/integral/levels cache); OS/shared-library caches warm; full CLI HF+basis+MBPT+CI and I/O; fixed BeII panel six levels and all persisted wave/integral arrays.",
                results=results)


def identity():
    paths = list((BASE / "AMBiT/src").rglob("*")) + list((ROOT / "src/atomx").rglob("*.py"))
    paths += list(BUILD.rglob("*.a")) + list(BUILD.rglob("*.o"))
    paths += [BUILD / "ambit", HERE / "timed_sigma", HERE / "timed_sigma.o", HERE / "timed_sigma.cpp",
              HERE / "atomx_action.py", HERE / "benchmark.py", HERE.parent / "contract.json",
              BUILD / "CMakeFiles/ambit.dir/flags.make", BUILD / "CMakeFiles/ambit.dir/link.txt"]
    files = sorted({path.resolve() for path in paths if path.is_file() and "__pycache__" not in str(path)})
    manifests = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in files}
    runtime_paths = set()
    for maps in HERE.rglob("*.maps"):
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/") and Path(fields[5]).is_file():
                runtime_paths.add(Path(fields[5]).resolve())
    runtime = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in sorted(runtime_paths)}
    with tarfile.open(HERE / "actual_sources_objects.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path).lstrip("/"), recursive=False)
    report = dict(actual_sources_objects=manifests, actual_mapped_runtime=runtime,
                  native_source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE / "AMBiT", text=True).strip(),
                  native_source_status=subprocess.check_output(["git", "status", "--short"], cwd=BASE / "AMBiT", text=True),
                  source_archive_sha256=digest(HERE / "actual_sources_objects.tar.gz"),
                  runtime_copied=False)
    (HERE / "execution_identity.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(source_count=len(files), runtime_count=len(runtime), archive_sha256=report["source_archive_sha256"])


def main():
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
    with LOCK.open("a") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX)
        inputs = HERE / "fresh_inputs001"
        preparation = json.loads((inputs / "prepare.execution.json").read_text())
        builds = [json.loads((HERE / (name + ".execution.json")).read_text()) for name in ("compile", "link")]
    input_files = sorted(path for path in inputs.iterdir() if path.suffix in (".txt", ".npz", ".json")
                         and "execution" not in path.name and "results" not in path.name)
    before = {path.name: digest(path) for path in input_files}
    comparison = matched(inputs)
    scaling = native_scaling()
    after = {path.name: digest(path) for path in input_files}
    if before != after:
        raise RuntimeError("scientific input changed during timing")
    report = dict(created_at=stamp(), preparation=preparation, build=builds, matched=comparison,
                  ambit_native_scaling=scaling, input_sha256=before, input_postcheck=True,
                  execution_identity=identity(), highest_level_reached=False,
                  assumptions="AMBiT Sigma probe invokes unmodified pinned upstream diagrams. Shared Atomx radial arrays do not independently verify radial physics. AMBiT BeII alone is not a cross-software accuracy ranking.")
    (HERE / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(matched=comparison["statistics"], scaling={key: value["statistics"]
                     for key, value in scaling["results"].items()})), flush=True)


if __name__ == "__main__":
    main()
