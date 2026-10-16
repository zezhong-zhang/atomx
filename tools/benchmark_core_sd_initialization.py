#!/usr/bin/env python3
"""Replay a full native band and check original vs BLAS SD intermediates."""

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from build_pci_sd_backend import build

from atomx.angular.coefficients import reduced_C_jj
from atomx.methods.core_sd import CoreSD
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.potentials.nuclear import make_nuclear_potential


def channels(problem, kind):
    d = problem.plan["dimensions"]
    shape = (d["NN"], d["NN"], d["NK"], d["NK"], d["KK"] + 1)
    nlookup = int(np.prod(shape))
    nvalues = d["NXX"] ** 2 * d["NXK"]
    buffer = (ctypes.c_char * (4 * nlookup + 8 * nvalues + 4)).in_dll(
        problem._backend, f"xcore{kind}_"
    )
    lookup = np.frombuffer(buffer, dtype=np.int32, count=nlookup).reshape(
        shape, order="F"
    )
    value = np.frombuffer(
        buffer, dtype=np.float64, count=nvalues, offset=4 * nlookup
    ).reshape((d["NXX"], d["NXX"], d["NXK"]), order="F")
    return lookup, value


def main(args):
    args.directory.mkdir(parents=True, exist_ok=False)

    def save(name, obj):
        (args.directory / name).write_text(json.dumps(obj, indent=2) + "\n")

    plan_name = (
        "yb_full_plan.json"
        if (args.input / "yb_full_plan.json").exists()
        else "plan.json"
    )
    plan = json.loads((args.input / plan_name).read_text())
    occupied = tuple(map(tuple, plan["occupied_keys"]))
    virtual = tuple(map(tuple, plan["virtual_keys"]))
    inputs_name = (
        "yb_full_inputs.npz"
        if (args.input / "yb_full_inputs.npz").exists()
        else "native_inputs.npz"
    )
    cap = args.memory_gib * 1024**3
    blas_extra = 24 * plan["dimensions"]["NHF"] * plan["dimensions"]["NXX"]
    if plan["complete_payload_bound_bytes"] + blas_extra > cap:
        raise MemoryError("complete SD plus BLAS initialization scratch exceeds cap")
    save(
        "contract.json",
        dict(
            source_run=str(args.input.resolve()),
            Z=args.Z,
            fixed="all saved core and virtual orbitals; exact same native metric, nucleus and DC quadrature",
            hypothesis="BLAS constructs identical signed complete X1/X2 matrices with less repeated grid work",
            gate_Ha=1e-11,
            seed=31917,
            sampled_elements_per_kind=50,
            baseline=str(args.baseline) if args.baseline else None,
            memory_bytes=cap,
            initialization_extra_bound_bytes=blas_extra,
            repetitions=args.repetitions,
            load=os.getloadavg(),
            load_guard_max=24,
            thread_environment={
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
            timing_scope="construction and repeated complete SD initialization only; no equation-solve or experiment claim",
            stopping="reject failed signed radial/control matrices or whole-model resource cap; no pruning",
        ),
    )
    with np.load(args.input / inputs_name, allow_pickle=False) as data:
        workspace = {
            k.removeprefix("workspace_"): data[k]
            for k in data
            if k.startswith("workspace_")
        }
        for k, a in tuple(workspace.items()):
            if a.ndim == 0:
                workspace[k] = a.item()
        waves = {
            a: (data[f"P_{a[0]}_{a[1]}"], data[f"Q_{a[0]}_{a[1]}"])
            for a in occupied + virtual
        }
    nucleus = None
    if (args.input / "nuclear_model.json").exists():
        profile = json.loads((args.input / "nuclear_model.json").read_text())
        nucleus = make_nuclear_potential(
            args.Z, profile["model"], profile["requested_mass_number"]
        )
    build(
        args.source,
        args.directory / "backend",
        plan["dimensions"],
        args.compiler,
        runtime_directory=args.runtime_directory,
        blas_library=args.blas_library,
    )

    def prepare(directory):
        start = time.perf_counter()
        problem = CoreSD(
            waves,
            workspace,
            args.Z,
            workspace["ns"],
            occupied_keys=occupied,
            virtual_keys=virtual,
            backend_directory=directory,
            nuclear_potential=nucleus,
            memory_bytes=cap,
        )
        return problem, time.perf_counter() - start

    problem, total_prepare = prepare(args.directory / "backend")
    print(
        json.dumps(dict(stage="complete BLAS preparation", seconds=total_prepare)),
        flush=True,
    )
    old = None
    baseline_prepare = None
    if args.baseline:
        old, baseline_prepare = prepare(args.baseline)
    errors = []
    if old is not None:
        for kind in (1, 2):
            lookup, value = channels(problem, kind)
            old_lookup, old_value = channels(old, kind)
            if not np.array_equal(lookup, old_lookup):
                raise AssertionError("whole declared angular-channel identity changed")
            # Keep complete matrices, including all padding; bounded comparison scratch.
            a, b = value.ravel(order="F"), old_value.ravel(order="F")
            errors.append(
                max(
                    float(
                        np.max(abs(a[start : start + 65536] - b[start : start + 65536]))
                    )
                    for start in range(0, len(a), 65536)
                )
            )
        if max(errors) > 1e-11:
            raise AssertionError("complete original vs BLAS X1/X2 matrix gate failed")
    cache = ConvolCache(workspace)
    rng = np.random.default_rng(31917)
    controls = []
    groups = problem.plan["groups"]
    for kind in (1, 2):
        lookup, values = channels(problem, kind)
        allowed = np.argwhere(lookup > 0)
        for index in rng.integers(len(allowed), size=50):
            i, j, im, jn, rank = map(int, allowed[index])
            m, n = rng.integers(len(groups[im])), rng.integers(len(groups[jn]))
            a, b, ci, cj = groups[im][m], groups[jn][n], occupied[i], occupied[j]
            if kind == 1:
                quartet, angular = (a, ci, b, cj), ((a[1], ci[1]), (b[1], cj[1]))
            else:
                quartet, angular = (a, b, ci, cj), ((a[1], b[1]), (ci[1], cj[1]))
            c = (-1) ** rank
            for ka, kb in angular:
                c *= (-1) ** (abs(ka) + abs(kb) + rank - 1) * reduced_C_jj(ka, kb, rank)
            expected = c * compute_slater_rk_cell(
                *(x for a in quartet for x in waves[a]), workspace, rank, cache
            )
            actual = values[m, n, lookup[i, j, im, jn, rank] - 1]
            controls.append(
                dict(
                    kind=kind,
                    indices=[i, j, im, jn, rank, int(m), int(n)],
                    keys=quartet,
                    expected_Ha=float(expected),
                    actual_Ha=float(actual),
                    error_Ha=float(abs(expected - actual)),
                )
            )
    if max(a["error_Ha"] for a in controls) > 1e-11:
        raise AssertionError("signed independent public radial X1/X2 gate failed")
    save("radial_controls.json", controls)
    measurements = []
    for repeat in range(args.repetitions):
        row = {}
        names = ("old", "blas") if repeat % 2 == 0 else ("blas", "old")
        for name in names:
            p = old if name == "old" else problem
            if p is not None:
                start = time.perf_counter()
                p._prepare(waves)
                row[name + "_seconds"] = time.perf_counter() - start
        measurements.append(row)
    save(
        "result.json",
        dict(
            prepare_seconds=total_prepare,
            baseline_prepare_seconds=baseline_prepare,
            full_matrix_max_differences_Ha=errors,
            radial_max_difference_Ha=max(a["error_Ha"] for a in controls),
            paired_initialization=measurements,
            load=os.getloadavg(),
            performance_load_guard_passed=max(os.getloadavg()) < 24,
            resources=dict(problem.plan, dimensions=dict(problem.plan["dimensions"])),
            provenance=problem.provenance,
            scope="full signed X1/X2 identity, shared radial gate and initialization stage; no SD solve",
        ),
    )
    save(
        "source_sha256.json",
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("tools/build_pci_sd_backend.py"),
                Path("src/atomx/methods/_pci_core_blas.f90"),
            )
        },
    )
    print(
        json.dumps(
            dict(
                stage="passed initialization controls",
                errors=errors,
                radial_max=max(a["error_Ha"] for a in controls),
                measurements=measurements,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--Z", type=int, default=10)
    parser.add_argument("--memory-gib", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path)
    parser.add_argument("--blas-library", type=Path, required=True)
    main(parser.parse_args())
