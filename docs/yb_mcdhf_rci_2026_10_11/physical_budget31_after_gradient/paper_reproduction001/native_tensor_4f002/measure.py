"""Matched fixed-physics core-residual and EA-action timing, one source epoch."""

import argparse
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

import numpy as np

from atomx.methods.tensor_coupled_cluster import CCSDTTensorOperator, TensorCoreResult
from atomx.physics.csf.spinor_integrals import SpinorIntegrals

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=["A1", "B1", "B2", "A2"])
    args = parser.parse_args()
    contract = json.loads((HERE / "performance_contract.json").read_text())
    storage = "dense" if args.label.startswith("A") else "canonical"
    baseline = storage == "dense"
    source = json.loads(
        (HERE / ("before_source.json" if baseline else "after_source.json")).read_text()
    )
    tree = HERE / "before_tree" if baseline else ROOT
    assert all(sha(tree / p) == h for p, h in source["files"].items())
    import atomx.methods.tensor_coupled_cluster as module

    assert (
        Path(module.__file__).resolve()
        == (tree / "src/atomx/methods/tensor_coupled_cluster.py").resolve()
    )
    assert all(sha(HERE / name) == h for name, h in contract["inputs"].items())
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    def alarm(*_):
        raise TimeoutError("matched performance cell300second limit")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    assert all(
        os.environ.get(x) == "1"
        for x in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    )
    before_load = os.getloadavg()[0]
    assert before_load <= 8
    ledger = json.loads((HERE / "operator.json").read_text())
    with np.load(HERE / "integrals.npz") as z:
        obj = SpinorIntegrals(
            tuple(map(tuple, ledger["labels"])),
            18,
            z["h"],
            z["fock"],
            z["g"],
            ledger["shifted_frozen_energy_Ha"],
            0.0,
            ledger["identity"],
        )
    op = (
        CCSDTTensorOperator(obj, memory_MiB=4096)
        if baseline
        else CCSDTTensorOperator(obj, memory_MiB=4096, projection_storage="canonical")
    )
    with np.load(HERE / "fixed_inputs.npz") as z:
        vector, right = z["core"], z["right"]
    probe = json.loads((HERE / "probe_result.json").read_text())
    core = TensorCoreResult(
        vector,
        probe["core_energy_Ha"],
        probe["core_energy_Ha"],
        probe["new4f_residual_by_rank"]["2"],
        0,
        op.operator_sha256,
    )
    rows = []
    last = {}
    for sample in range(3):
        load_start = os.getloadavg()[0]
        assert load_start <= 8
        start = time.perf_counter()
        residual = op.core_residual(vector)
        core_seconds = time.perf_counter() - start
        start = time.perf_counter()
        product = op.attachment_action(core, right)
        ea_seconds = time.perf_counter() - start
        load_end = os.getloadavg()[0]
        assert load_end <= 8
        with np.load(HERE / "before_values.npz") as z:
            core_error = float(np.max(abs(residual - z["core_residual"])))
            ea_error = float(np.max(abs(product - z["EA_action"])))
        assert core_error <= 1e-10 and ea_error <= 1e-10
        rows.append(
            dict(
                sample=sample,
                warmup=sample == 0,
                core_seconds=core_seconds,
                EA_seconds=ea_seconds,
                core_error_Ha=core_error,
                EA_error_Ha=ea_error,
                load_start=load_start,
                load_end=load_end,
            )
        )
        last = dict(core_residual=residual, EA_action=product)
    signal.alarm(0)
    np.savez(HERE / (args.label + "_values.npz"), **last)
    result = dict(
        status="PASS_MATCHED_CELL",
        label=args.label,
        source_map_sha256=source["source_map_sha256"],
        projection_storage=storage,
        operator_sha256=op.operator_sha256,
        rows=rows,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        scope="warm complete core-residual plus EA-action at the fixed embedded off-shell full4f point; no full-solver or spectral speedup claim",
    )
    (HERE / (args.label + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
