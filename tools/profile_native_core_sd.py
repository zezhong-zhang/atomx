#!/usr/bin/env python3
"""Replay complete native core SD and profile every original diagram stage."""

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from build_pci_sd_backend import build

from atomx.methods.core_sd import CoreSD
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures
from atomx.numerics.bspline.pair_contraction import CellPairContraction


def main(args):
    args.directory.mkdir(parents=True, exist_ok=False)

    def save(name, obj):
        (args.directory / name).write_text(json.dumps(obj, indent=2) + "\n")

    plan = json.loads((args.input / "plan.json").read_text())
    old = json.loads((args.input / "result.json").read_text())
    save(
        "contract.json",
        dict(
            source_run=str(args.input.resolve()),
            load=os.getloadavg(),
            fixed="replay all saved native orbitals and every raw source amplitude",
            gate_Ha=1e-11,
            repetitions=3,
            edge_batch=1024,
            hypothesis="original diagram stage costs identify repeated radial work; exact pair contraction improves that stage without any cutoff",
            stopping="failed complete action/residual or pair-matrix agreement rejects timing claim",
            timing_scope="warm same-vector whole action, diagram sub-stages, and paired full pair-matrix kernel; no end-to-end solver speedup claim",
        ),
    )
    occupied = tuple(map(tuple, plan["occupied_keys"]))
    virtual = tuple(map(tuple, plan["virtual_keys"]))
    with np.load(args.input / "native_inputs.npz", allow_pickle=False) as data:
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
    with np.load(args.input / "solution.npz") as data:
        amplitudes = data["amplitudes"].copy()
    build(
        args.source,
        args.directory / "backend",
        plan["dimensions"],
        args.compiler,
        runtime_directory=args.runtime_directory,
    )
    p = CoreSD(
        waves,
        workspace,
        args.Z,
        workspace["ns"],
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=args.directory / "backend",
    )
    baseline = p.action(amplitudes)
    baseline_error = float(np.max(abs(baseline - p.source)))
    if baseline_error > 1e-11:
        raise AssertionError("saved complete native equation does not replay")
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    function = p._backend.atomx_sd_profile_vector_action
    function.argtypes = [ctypes.c_int64, pointer, pointer, pointer]
    function.restype = None
    stages = (
        "unpack",
        "outin",
        "terma1",
        "terma2",
        "terma3",
        "terma4",
        "term1",
        "term21",
        "term31",
        "term32",
        "term41",
        "term42",
        "term5_1",
        "term5_2",
        "noncanonical_fock",
        "pack_result",
    )
    timings = []
    for _ in range(3):
        result, elapsed = np.empty_like(amplitudes), np.zeros(16)
        began = time.perf_counter()
        function(len(amplitudes), amplitudes, result, elapsed)
        error = float(np.max(abs(baseline - result)))
        if error > 1e-11:
            raise AssertionError("profiled whole action changed signed equations")
        timings.append(
            dict(
                zip(stages, elapsed.tolist(), strict=True),
                wall_seconds=time.perf_counter() - began,
                action_max_difference_Ha=error,
            )
        )
    save(
        "stages.json",
        dict(
            timings=timings,
            baseline_residual_max_Ha=baseline_error,
            original_residual_max_Ha=old["residual_max_Ha"],
            provenance=p.provenance,
        ),
    )

    f = CellCoulombFeatures(workspace)
    target_keys = [a for a in virtual if a[1] == -5]
    source_keys = [a for a in virtual if a[1] == -4]
    if not target_keys or not source_keys:
        raise ValueError(
            "this exact complete g/f profile requires both retained partial waves"
        )
    u, r = [
        np.column_stack([f.sample(*waves[a]) for a in group])
        for group in (target_keys, source_keys)
    ]
    # Select an actual solved complete f/f rho_3 channel for the first core pair.
    offset = sum(sum(a[1] == b[1] for a in virtual) for b in occupied)
    density = None
    for i, j, im, inn, rank in plan["channels"]:

        def kap(ind):
            ind += 1
            return -(ind + 1) // 2 if ind % 2 else ind // 2

        nr = sum(a[1] == kap(im) for a in virtual)
        ns = sum(a[1] == kap(inn) for a in virtual)
        if (i, j, kap(im), kap(inn), rank) == (0, 0, -4, -4, 3):
            density = (
                amplitudes[offset : offset + nr * ns]
                .reshape((nr, ns), order="F")
                .copy()
            )
        offset += nr * ns
    if density is None:
        raise ValueError("complete native source channel absent")
    begin = time.perf_counter()
    pair = CellPairContraction(f, u, r, r, u, 1)
    pair_prepare = time.perf_counter() - begin
    required = len(f.radii) * r.shape[1] * u.shape[1] * 8
    if required > 128 * 1024**2:
        raise MemoryError("whole baseline pair-potential array exceeds profile cap")
    begin = time.perf_counter()
    potential = np.empty((len(f.radii), r.shape[1], u.shape[1]))
    for m in range(u.shape[1]):
        for ir in range(r.shape[1]):
            potential[:, ir, m] = f.apply_density(u[:, m] * r[:, ir], 1)
    baseline_prepare = time.perf_counter() - begin

    def old_action():
        v = r @ density.T
        uv = np.einsum("irm,ir->im", potential, v)
        return uv.T @ u

    errors = []
    measured = []
    for repeat in range(3):
        row = {}
        for name in ("old", "pair") if repeat % 2 == 0 else ("pair", "old"):
            start = time.perf_counter()
            value = old_action() if name == "old" else pair(density)
            row[name + "_seconds"] = time.perf_counter() - start
            row[name] = value
        error = float(np.max(abs(row["old"] - row["pair"])))
        if error > 1e-11:
            raise AssertionError(
                "complete native pair contraction changed signed action"
            )
        errors.append(error)
        measured.append({k: v for k, v in row.items() if k.endswith("seconds")})
    np.savez_compressed(
        args.directory / "pair_controls.npz",
        density=density,
        original=old_action(),
        pair=pair(density),
    )
    save(
        "pair.json",
        dict(
            identity=dict(pair.identity),
            measurements=measured,
            errors_Ha=errors,
            baseline_prepare_seconds=baseline_prepare,
            pair_prepare_seconds=pair_prepare,
            baseline_potential_bytes=required,
            scope="same full native target/source partial waves and actual solved rho; pair-kernel timing only",
        ),
    )
    save(
        "source_sha256.json",
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("src/atomx/numerics/bspline/pair_contraction.py"),
            )
        },
    )
    print(
        json.dumps(dict(stages=timings, pair=measured, errors_Ha=errors), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--Z", type=int, default=10)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path)
    main(parser.parse_args())
