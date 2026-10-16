#!/usr/bin/env python3
"""Same-space signed full-action replay and paired native term21 timing."""

import argparse
import hashlib
import inspect
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from build_pci_sd_backend import build

from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.numerics.bspline.coulomb_features import (
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)


def main(args):
    args.directory.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.directory / name).write_text(json.dumps(value, indent=2) + "\n")

    load = os.getloadavg()
    save(
        "contract.json",
        dict(
            input=str(args.input.resolve()),
            Z=args.Z,
            seed=314159,
            fixed="all saved native DC orbitals, core/virtual channels, native cell rules, complete signed raw amplitudes",
            hypothesis="disjoint core-pair work and component-tagged scratch improve full action without changed equations",
            action_gate_Ha=1e-11,
            residual_gate_Ha=1e-10,
            stop="any complete signed action or root mismatch rejects optimization; no physical pruning",
            requested_threads=args.threads,
            term21_mode=args.term21_mode,
            core_singles_blas=args.core_singles_blas,
            radial_representation=args.radial_representation,
            load=load,
            load_guard=args.load_guard,
            timing_scope="alternating warm full actions; initialization and root solve separately",
            memory_bytes=args.memory_gib * 1024**3,
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(build.__code__.co_filename),
                    Path(inspect.getfile(CoreSD)),
                    Path(inspect.getfile(CoreSD)).with_name("_pci_term21_compact.f90"),
                    Path(inspect.getfile(CoreSD)).with_name("_pci_term21_streamed.f90"),
                    Path(inspect.getfile(CoreSD)).with_name("_pci_core_blas.f90"),
                    Path(inspect.getfile(CoreSD)).with_name(
                        "_pci_core_singles_blas.f90"
                    ),
                )
                if p.exists()
            },
        ),
    )
    saved = json.loads((args.input / "plan.json").read_text())
    if "occupied_keys" not in saved:
        # The independently anchored finite probes store the full resource
        # identity in result.json, while plan.json holds only the ABI.
        result = json.loads((args.input / "result.json").read_text())
        finite_plan = result["resources"]
        if finite_plan["dimensions"] != result["dimensions"]:
            raise ValueError("finite probe result/resource ABI disagreement")
        saved = finite_plan
    occupied = tuple(map(tuple, saved["occupied_keys"]))
    virtual = tuple(map(tuple, saved["virtual_keys"]))
    with np.load(args.input / "native_inputs.npz", allow_pickle=False) as data:
        workspace = {
            k.removeprefix("workspace_"): data[k].copy()
            for k in data
            if k.startswith("workspace_")
        }
        for k, a in tuple(workspace.items()):
            if a.ndim == 0:
                workspace[k] = a.item()
        waves = {
            a: (data[f"P_{a[0]}_{a[1]}"].copy(), data[f"Q_{a[0]}_{a[1]}"].copy())
            for a in occupied + virtual
        }
        # This replay explicitly supports saved point-nucleus controls.
        if not np.allclose(
            data["nuclear_on_outer_gauss"],
            -args.Z / workspace["gr"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("saved nucleus is not the requested point nucleus")
    solution = args.input / "solution.npz"
    if not solution.exists():
        solution = args.input / "solved.npz"
    with np.load(solution, allow_pickle=False) as data:
        root = data["amplitudes" if "amplitudes" in data else "raw"].copy()
        source, denominators = data["source"].copy(), data["denominators"].copy()
    native_features = CellCoulombFeatures(workspace)
    candidate_features = (
        native_features
        if args.radial_representation == "native"
        else PolynomialCellCoulombFeatures(workspace)
    )
    cap = args.memory_gib * 1024**3
    rng = np.random.default_rng(314159)
    random = rng.normal(size=len(root)) * 1e-3
    outputs = {}
    problems = {}
    preparations = {}
    for threads in (0, *args.threads):
        singles_blas = args.core_singles_blas and threads > 0
        features = candidate_features if threads else native_features
        plan = plan_core_sd(
            occupied,
            virtual,
            features.identity,
            term21_threads=threads,
            term21_mode=args.term21_mode,
            core_singles_blas=singles_blas,
        )
        if (
            plan["complete_payload_bound_bytes"] + 40 * plan["krylov_vector_bytes"]
            > cap
        ):
            raise MemoryError("complete workload plus Krylov exceeds declared cap")
        save(f"plan_{threads}.json", plan)
        directory = args.directory / f"backend_{threads}"
        build(
            args.source,
            directory,
            plan["dimensions"],
            args.compiler,
            runtime_directory=args.runtime_directory,
            blas_library=args.blas_library,
            term21_threads=threads,
            term21_mode=args.term21_mode,
            core_singles_blas=singles_blas,
        )
        began = time.perf_counter()
        p = CoreSD(
            waves,
            workspace,
            args.Z,
            workspace["ns"],
            occupied_keys=occupied,
            virtual_keys=virtual,
            backend_directory=directory,
            memory_bytes=cap,
            term21_threads=threads,
            term21_mode=args.term21_mode,
            core_singles_blas=singles_blas,
            radial_representation=args.radial_representation if threads else "native",
        )
        preparations[threads] = time.perf_counter() - began
        problems[threads] = p
        if (
            np.max(abs(p.source - source)) > 1e-11
            or np.max(abs(p.denominators - denominators)) > 1e-11
        ):
            raise AssertionError("saved complete source/denominators failed replay")
        outputs[f"root_{threads}"] = p.action(root)
        outputs[f"random_{threads}"] = p.action(random)
        print(
            json.dumps(
                dict(
                    stage="signed actions",
                    threads=threads,
                    prepare_seconds=preparations[threads],
                )
            ),
            flush=True,
        )
    np.savez_compressed(
        args.directory / "signed_actions.npz", root=root, random=random, **outputs
    )
    errors = {
        str(n): {
            k: float(np.max(abs(outputs[f"{k}_{n}"] - outputs[f"{k}_0"])))
            for k in ("root", "random")
        }
        for n in args.threads
    }
    if max(v for e in errors.values() for v in e.values()) > 1e-11:
        save("failed_actions.json", errors)
        raise AssertionError("complete signed action mismatch")
    timings = []
    for n in args.threads:
        for repetition in range(3):
            row = dict(threads=n, repetition=repetition, load=os.getloadavg())
            for selected in (0, n) if repetition % 2 == 0 else (n, 0):
                began = time.perf_counter()
                actual = problems[selected].action(root)
                if np.max(abs(actual - outputs[f"root_{selected}"])) > 1e-11:
                    raise AssertionError("repeated signed action mismatch")
                row[str(selected)] = time.perf_counter() - began
            timings.append(row)
            print(json.dumps(row), flush=True)
    roots = {}
    if args.solve:
        for n in args.threads:
            result = problems[n].solve(residual_tolerance_Ha=1e-10)
            error = float(np.max(abs(result.amplitudes - root)))
            roots[str(n)] = dict(
                correlation_energy_Ha=result.correlation_energy_Ha,
                residual_max_Ha=result.residual_max_Ha,
                signed_root_error=error,
                solve_seconds=result.wall_seconds,
                matrix_vector_products=result.matrix_vector_products,
            )
            np.savez_compressed(
                args.directory / f"solution_{n}.npz", amplitudes=result.amplitudes
            )
            if error > 1e-9:
                raise AssertionError("same-space signed root mismatch")
    save(
        "result.json",
        dict(
            errors_Ha=errors,
            timings=timings,
            preparations=preparations,
            roots=roots,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            timing_guard_pass=all(max(t["load"]) <= args.load_guard for t in timings),
            space_converged=False,
            experimental_gate_pass=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--Z", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path)
    parser.add_argument("--blas-library", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--term21-mode", choices=("compact", "streamed"), default="compact"
    )
    parser.add_argument("--memory-gib", type=int, default=4)
    parser.add_argument("--load-guard", type=float, default=24)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--core-singles-blas", action="store_true")
    parser.add_argument(
        "--radial-representation", choices=("native", "polynomial"), default="native"
    )
    main(parser.parse_args())
