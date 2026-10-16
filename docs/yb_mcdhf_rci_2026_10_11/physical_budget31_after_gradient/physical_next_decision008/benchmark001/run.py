"""Once-only ABBA comparison of exact singleton-bucket compression.

Default and --prepare are stdlib-only. Scientific workers require the root seal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
THREADS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), separators=(",", ":")).encode()).hexdigest()


def function_node(path, name="_connected_pairs"):
    return next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_check(c):
    for relative, expected in c["sources"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("source changed: " + relative)
    for name, item in c["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("input changed: " + name)
    if sha(HERE / "run.py") != c["driver_sha256"]:
        raise RuntimeError("driver changed")


def prepare():
    base = read(HERE.parent.parent / "physical_next_decision006/execution001/contract.json")
    names = ("models", "basis_maps")
    old = HERE.parent / "angular_before.py"
    archived = read(HERE.parent / "source_before.json")
    if sha(old) != archived["sha256"]:
        raise RuntimeError("before archive differs from recorded actual source")
    old_ast = ast.parse(old.read_text())
    new_ast = ast.parse((ROOT / "src/atomx/angular/csf_hamiltonian.py").read_text())
    for tree in (old_ast, new_ast):
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != "_connected_pairs"]
    if ast.dump(old_ast) != ast.dump(new_ast):
        raise RuntimeError("source changed outside _connected_pairs")
    old_relative = str(old.relative_to(ROOT))
    inputs = {name: base["inputs"][name] for name in names}
    inputs["old_angular"] = {"path": old_relative, "sha256": sha(old)}
    c = {
        "status": "SOURCE_ONLY_READY_NO_SCIENTIFIC_EXECUTION",
        "hypothesis": "Compressing singleton spectator buckets to one integer retains exactly the same ordered Slater-Condon candidates and full signed angular tables while reducing bucket object residency.",
        "fixed": {"electrons": 69, "A_NR_CSF": [91, 1595], "B_NR_CSF": [100, 2574],
                  "A_parent_CSF": 3789, "B_parent_CSF": 5741,
                  "sectors": [[1, 1], [1, -1], [3, 1], [3, -1], [5, 1]],
                  "radial_symmetry": False, "max_determinants": 2659,
                  "all_other_source": "Same current dirty imported Atomx source; old replaces literal archived _connected_pairs only; typed CSR _coefficient_table remains identical. No radial, H assembly, CI or eigensolver calls."},
        "resources": {"CPU": 27, "threads": 1, "worker_AS_bytes": 6 * 1024**3,
                      "worker_alarm_seconds": 300, "worker_outer_seconds": 310,
                      "host_load1_max": 16, "host_MemAvailable_GiB_min": 10},
        "order": ["old", "new", "new", "old"],
        "inputs": inputs,
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/atomx").rglob("*.py"))},
        "driver_sha256": sha(HERE / "run.py"),
        "old_function_AST_sha256": {name: digest(ast.dump(function_node(old, name)))
                                    for name in ("_connected_pairs",)},
        "all_other_angular_AST_identical": True,
        "gates": {"identical": ["full CSF signatures and dimensions", "raw determinants, spinor labels, slices and patterns", "all connected candidate pairs in order", "radial-key order", "retained pair coordinates", "CSR data/indices/indptr and shapes", "full signed sparse projection arrays", "signed synthetic integral actions and bilinears", "cached repeat object identity"],
                  "signed_bilinear_action_max_abs": 1e-10,
                  "signed_absolute_control_min_difference": 1e-8},
        "timing_scope": "Fresh workers, cold in-process Atomx caches, identical A then B and five-sector order. Table and projection construction timed separately. Connection iterator pull time is accumulated in fixed1024-coordinate batches; hashing and downstream coefficient work excluded. Coefficient remainder=table time minus measured connection pull time, approximate due to timer/batch overhead. No radial/H/CI timing; no end-to-end claim.",
        "work_memory": "Ten complete existing angular tables per worker, <=2659 raw determinants/block, <=4million table nnz/block and40million total; <=600MB table CSR retained payload, plus projections/determinants/Python build temporaries. No dense determinant/CSF Hamiltonian or reference NPZ load. Existing prior complete workloads fit~2GiB;6GiB hardstop is admission cap, not predicted peak.",
        "stop": "First source/input/resource/correctness/timeout failure stops run; preserve partial output, no retry, changed gate, configuration exclusion or backend substitution. Failed load guard prevents controlled timing claim; correctness workers continue under hard resource caps.",
    }
    write(HERE / "contract.json", c)
    source_check(c)
    print(json.dumps({"status": c["status"], "source_files": len(c["sources"]),
                      "driver_sha256": c["driver_sha256"], "contract_sha256": sha(HERE / "contract.json")}))


def host():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        mem[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {"load1": os.getloadavg()[0], "MemAvailable_bytes": mem["MemAvailable"]}


def admission(c):
    sample = host()
    sample["timing_load_guard_pass"] = (
        sample["load1"] <= c["resources"]["host_load1_max"]
        and sample["MemAvailable_bytes"] >= c["resources"]["host_MemAvailable_GiB_min"] * 1024**3)
    return sample


def validate_seal(c, path):
    seal = read(path)
    expected = {"execution_authorized_by_root": True, "once_only": True,
                "driver_sha256": sha(HERE / "run.py"),
                "contract_sha256": sha(HERE / "contract.json"),
                "resources": c["resources"]}
    if seal != expected:
        raise RuntimeError("root seal differs from exact driver/contract/resource identity")


def worker(c, variant, ordinal, seal_path):
    validate_seal(c, seal_path)
    source_check(c)
    cap = c["resources"]
    if set(os.sched_getaffinity(0)) != {cap["CPU"]}:
        raise RuntimeError("worker CPU affinity mismatch")
    if any(os.environ.get(k) != "1" for k in THREADS):
        raise RuntimeError("worker thread environment mismatch")
    resource.setrlimit(resource.RLIMIT_AS, (cap["worker_AS_bytes"], cap["worker_AS_bytes"]))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("worker300s hard stop")))
    signal.alarm(cap["worker_alarm_seconds"])
    out = HERE / "actual" / f"worker_{ordinal:02d}_{variant}.json"
    report = {"variant": variant, "ordinal": ordinal, "host_before": admission(c),
              "blocks": [], "success": False}
    write(out, report)
    start = time.perf_counter()
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import scipy

        from atomx.angular import csf_hamiltonian as angular
        from atomx.config.csf import enumerate_jj_csfs

        def array_hash(a):
            a = np.ascontiguousarray(a)
            return digest((a.shape, a.dtype.str, hashlib.sha256(a.tobytes()).hexdigest()))

        def csr_hash(a):
            if a.format != "csr":
                raise RuntimeError("expected CSR")
            return digest((a.shape, array_hash(a.data), array_hash(a.indices), array_hash(a.indptr)))

        before_caches = {name: str(value.cache_info()) for name, value in vars(angular).items()
                         if hasattr(value, "cache_info")}
        if any(value.cache_info().currsize for value in vars(angular).values() if hasattr(value, "cache_info")):
            raise RuntimeError("angular process caches not cold at import")
        if variant == "old":
            nodes = [function_node(ROOT / c["inputs"]["old_angular"]["path"], name)
                     for name in ("_connected_pairs",)]
            module = ast.Module(body=nodes, type_ignores=[])
            exec(compile(module, c["inputs"]["old_angular"]["path"], "exec"), vars(angular))
        selected = angular._connected_pairs
        report.update(numpy=np.__version__, scipy=scipy.__version__, cold_angular_caches=before_caches,
                      selected_function_AST_sha256={name: digest(ast.dump(function_node(
                          ROOT / c["inputs"]["old_angular"]["path"] if variant == "old" else Path(angular.__file__), name)))
                          for name in ("_connected_pairs",)})
        measure = {}

        def connected(dets, *, triangular=False):
            t = time.perf_counter()
            iterator = iter(selected(dets, triangular=triangular))
            elapsed = time.perf_counter() - t
            count, h = 0, hashlib.sha256()
            while True:
                t = time.perf_counter()
                batch = list(itertools.islice(iterator, 1024))
                elapsed += time.perf_counter() - t
                if not batch:
                    break
                count += len(batch)
                h.update(np.asarray(batch, dtype="<i8").tobytes())
                yield from batch
            measure.update(connected_seconds=elapsed, connected_candidates=count,
                           connected_pair_sha256=h.hexdigest())

        angular._connected_pairs = connected
        specs = read(ROOT / c["inputs"]["models"]["path"])
        maps = read(ROOT / c["inputs"]["basis_maps"]["path"])
        sectors = set(map(tuple, specs["selected_sectors"]))
        total_nnz = total_bytes = 0
        for arm, model, parent_n, csf_n in (("A", "baseline", 3789, 1595), ("B", "expanded", 5741, 2574)):
            full = [z for row in specs[model + "_configs"] for z in enumerate_jj_csfs(row["config"])]
            csfs = [z for z in full if (z.total_J2, z.parity) in sectors]
            signatures = [angular.csf_signature(z) for z in csfs]
            actual_signatures = [json.dumps(plain((s.subshells, s.coupling, z.total_J2, z.parity)), separators=(",", ":"))
                                 for s, z in zip(signatures, csfs, strict=True)]
            if len(full) != parent_n or len(csfs) != csf_n or actual_signatures != maps[model]["selected_signatures"]:
                raise RuntimeError("complete CSF/source identity mismatch: " + arm)
            if not all(z.n_electrons == 69 for z in full):
                raise RuntimeError("electron count changed")
            for j2, parity in c["fixed"]["sectors"]:
                grouped = {}
                for sig, csf in zip(signatures, csfs, strict=True):
                    if (csf.total_J2, csf.parity) == (j2, parity):
                        grouped.setdefault(sig.pattern, []).append(sig)
                t = time.perf_counter()
                structure = angular.block_terms(tuple(grouped), j2, max_determinants=2659, radial_symmetry=False)
                table_seconds = time.perf_counter() - t
                sig_groups = tuple(tuple(grouped[p]) for p in structure.patterns)
                t = time.perf_counter()
                projection = angular.csf_block_matrix(structure.patterns, j2, sig_groups, sparse=True)
                projection_seconds = time.perf_counter() - t
                if angular.block_terms(tuple(grouped), j2, max_determinants=2659, radial_symmetry=False) is not structure:
                    raise RuntimeError("structure cache repeat identity failed")
                if angular.csf_block_matrix(structure.patterns, j2, sig_groups, sparse=True) is not projection:
                    raise RuntimeError("projection cache repeat identity failed")
                total_nnz += structure.table.nnz
                table_bytes = sum(v.nbytes for v in (structure.table.data, structure.table.indices, structure.table.indptr))
                total_bytes += table_bytes
                if structure.table.nnz > 4_000_000 or total_nnz > 40_000_000 or total_bytes > 600_000_000:
                    raise RuntimeError("table nnz/payload cap exceeded")
                rng = np.random.default_rng(716 + j2 + 10 * parity)
                left, right = rng.normal(size=(2, projection.shape[1]))
                left /= np.linalg.norm(left)
                right /= np.linalg.norm(right)
                radial = rng.normal(size=len(structure.keys))
                values = structure.table @ radial
                a, b = projection @ left, projection @ right
                action_det = np.zeros(len(structure.dets))
                rows, cols = structure.pairs.T
                np.add.at(action_det, rows, values * b[cols])
                action = projection.T @ action_det
                bilinear = float(left @ action)
                direct = float(np.dot(values, a[rows] * b[cols]))
                absolute = float(np.dot(values, abs(a[rows]) * abs(b[cols])))
                if abs(bilinear - direct) > 1e-10 or abs(bilinear - absolute) < 1e-8:
                    raise RuntimeError("signed bilinear/action gate failed")
                record = {"arm": arm, "sector": [j2, parity], "raw_determinants": len(structure.dets),
                          "CSFs": projection.shape[1], "radial_keys": len(structure.keys),
                          "table_nnz": structure.table.nnz, "table_bytes": table_bytes,
                          "projection_nnz": projection.nnz, "table_seconds": table_seconds,
                          "projection_seconds": projection_seconds, **measure,
                          "coefficient_remainder_seconds_approx": table_seconds - measure["connected_seconds"],
                          "RSS_highwater_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "identity": {"raw_determinants": digest(structure.dets),
                                       "spinor_labels": digest(structure.spinor_labels),
                                       "slices_patterns": digest((structure.slices, structure.patterns)),
                                       "radial_keys": digest(structure.keys), "retained_pairs": array_hash(structure.pairs),
                                       "table_CSR": csr_hash(structure.table), "projection_CSR": csr_hash(projection),
                                       "signed_action": array_hash(action), "signed_bilinear": bilinear,
                                       "connected_pairs": measure["connected_pair_sha256"],
                                       "signed_direct": direct, "abs_control": absolute},
                          "signed_bilinear_action_error": abs(bilinear - direct),
                          "signed_absolute_control_difference": abs(bilinear - absolute), "cache_identity": True}
                report["blocks"].append(record)
                write(out, report)
        imported = {}
        for name, module in list(sys.modules.items()):
            if name != "atomx" and not name.startswith("atomx."):
                continue
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            path = Path(path).resolve()
            if not path.is_relative_to(ROOT / "src/atomx"):
                raise RuntimeError("wrong imported Atomx origin: " + str(path))
            relative = str(path.relative_to(ROOT))
            if sha(path) != c["sources"].get(relative):
                raise RuntimeError("import hash mismatch: " + relative)
            imported[relative] = sha(path)
        source_check(c)
        report.update(imported_sources=imported, success=True)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - start,
                      RSS_highwater_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      host_after=admission(c))
        report["controlled_timing_admissible"] = report["host_before"]["timing_load_guard_pass"] and report["host_after"]["timing_load_guard_pass"]
        write(out, report)
        signal.alarm(0)
    print(json.dumps({"worker": ordinal, "variant": variant, "success": report["success"],
                      "seconds": report["elapsed_seconds"], "RSS_KiB": report["RSS_highwater_KiB"]}), flush=True)


def execute(c, seal_path):
    validate_seal(c, seal_path)
    source_check(c)
    actual = HERE / "actual"
    actual.mkdir()
    results = []
    checks, baseline = [], None
    report = {"success": False, "order": c["order"], "workers": results}
    write(actual / "result.json", report)
    try:
        for ordinal, variant in enumerate(c["order"]):
            source_check(c)
            env = dict(os.environ, **{name: "1" for name in THREADS})
            cmd = ["taskset", "-c", str(c["resources"]["CPU"]), str(ROOT / ".venv/bin/python"),
                   "-I", str(HERE / "run.py"), "--worker", variant, "--ordinal", str(ordinal),
                   "--seal", str(Path(seal_path).resolve())]
            completed = subprocess.run(cmd, env=env, timeout=c["resources"]["worker_outer_seconds"], check=False)
            path = actual / f"worker_{ordinal:02d}_{variant}.json"
            item = read(path) if path.exists() else {"success": False, "missing_worker_receipt": True}
            results.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path) if path.exists() else None,
                            "returncode": completed.returncode})
            write(actual / "result.json", report)
            if completed.returncode != 0 or not item["success"]:
                raise RuntimeError("worker failed; no subsequent run")
            if len(item["blocks"]) != 10:
                raise RuntimeError("missing complete symmetry blocks")
            identities = [record["identity"] for record in item["blocks"]]
            if baseline is None:
                baseline = identities
            for block, identity in enumerate(identities):
                checks.append({"worker": ordinal, "block": block,
                               "same_identity": identity == baseline[block]})
                if not checks[-1]["same_identity"]:
                    raise RuntimeError("old/new bitwise angular/action identity failed; stop before next worker")
        workers = [read(ROOT / item["path"]) for item in results]
        timings = [{"variant": w["variant"], "table_seconds": sum(b["table_seconds"] for b in w["blocks"]),
                    "connection_seconds": sum(b["connected_seconds"] for b in w["blocks"]),
                    "projection_seconds": sum(b["projection_seconds"] for b in w["blocks"]),
                    "RSS_highwater_KiB": w["RSS_highwater_KiB"]} for w in workers]
        report.update(success=True, identical_block_checks=checks, timings=timings,
                      controlled_timing_admissible=all(w["controlled_timing_admissible"] for w in workers),
                      scope="Angular construction only; no radial/H/CI, large4f12 or spectroscopy precision claim.")
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["identical_block_checks"] = checks
        write(actual / "result.json", report)
    print(json.dumps({"success": report["success"], "identity_checks": len(checks),
                      "controlled_timing_admissible": report["controlled_timing_admissible"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute")
    p.add_argument("--worker", choices=("old", "new"))
    p.add_argument("--ordinal", type=int)
    p.add_argument("--seal")
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(read(HERE / "contract.json"), args.execute)
    elif args.worker:
        worker(read(HERE / "contract.json"), args.worker, args.ordinal, args.seal)
    else:
        print(json.dumps({"status": "SOURCE_ONLY_NOT_EXECUTED", "driver_sha256": sha(HERE / "run.py")}))


if __name__ == "__main__":
    main()
