#!/usr/bin/env python3
"""Run a complete native virtual band through direct reduced core SD.

Outputs the full space, resource plan, native inputs, equation residual and
timing scope. No experimental accuracy or space convergence is inferred.
"""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from build_pci_sd_backend import build

from atomx.methods.atomic_space import build_atomic_space
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.numerics.bspline.coulomb_features import (
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)


def main(args):
    if args.directory.exists():
        raise FileExistsError("fresh owned scientific run directory required")
    args.directory.mkdir(parents=True)
    requested = vars(args).copy()
    requested = {k: str(v) if isinstance(v, Path) else v for k, v in requested.items()}
    requested.update(
        load=os.getloadavg(),
        threads={
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        equation_residual_gate_Ha=1e-10,
        fixed="native DC closed core; complete requested VN energy band and partial waves",
        stopping="reject oversized complete model or failed raw residual; no virtual pruning",
        phase="larger-space/performance control; no experimental convergence claim",
    )
    (args.directory / "contract.json").write_text(
        json.dumps(requested, indent=2) + "\n"
    )
    root = Path(__file__).resolve().parents[1]
    (args.directory / "source_sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "src/atomx").rglob("*.py"))
                + sorted((root / "src/atomx").rglob("*.f90"))
                + [Path(__file__).resolve(), root / "tools/build_pci_sd_backend.py"]
            },
            indent=2,
        )
        + "\n"
    )
    began = time.perf_counter()
    subshells = [Subshell(1, -1, 2)]
    if args.case in ("be4", "ne10"):
        subshells.append(Subshell(2, -1, 2))
    if args.case == "ne10":
        subshells.extend([Subshell(2, 1, 2), Subshell(2, -2, 4)])
    Z = 10 if args.case == "ne10" else 4
    core = DFSolver(Z=Z, subshells=subshells, ns=args.ns, ks=args.ks).solve()
    scf_seconds = time.perf_counter() - began
    space = build_atomic_space(
        core, virtual_lmax=args.lmax, virtual_energy_max_Ha=args.emax
    )
    features_type = (
        CellCoulombFeatures
        if args.radial_representation == "native"
        else PolynomialCellCoulombFeatures
    )
    features = features_type(core._bsp_data)
    plan = plan_core_sd(
        space.occupied_keys,
        space.virtual_keys,
        features.identity,
        radial_pair_cache_size=args.radial_pair_cache_size,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
    )
    memory = args.memory_gib * 1024**3
    krylov = (
        8 * plan["raw_amplitudes"] * (min(args.restart, plan["raw_amplitudes"]) + 10)
    )
    if plan["complete_payload_bound_bytes"] + krylov > memory:
        raise MemoryError(
            "complete declared band plus Krylov storage exceeds explicit cap"
        )
    (args.directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                stage="complete preflight",
                raw_amplitudes=plan["raw_amplitudes"],
                payload_bytes=plan["complete_payload_bound_bytes"],
                krylov_bytes=krylov,
                seconds=time.perf_counter() - began,
            )
        ),
        flush=True,
    )
    # Retain original native orbital coefficients and quadrature operands.
    arrays = {f"P_{n}_{k}": value[0] for (n, k), value in space.orbitals.items()}
    arrays.update({f"Q_{n}_{k}": value[1] for (n, k), value in space.orbitals.items()})
    arrays.update(
        {
            f"workspace_{k}": v
            for k, v in core._bsp_data.items()
            if isinstance(v, (np.ndarray, int, float, bool))
        }
    )
    potential = core._nuclear_potential
    arrays["nuclear_on_outer_gauss"] = (
        np.vectorize(potential, otypes=[float])(core._bsp_data["gr"])
        if callable(potential)
        else -Z / core._bsp_data["gr"]
        if potential is None
        else np.asarray(potential)
    )
    np.savez_compressed(args.directory / "native_inputs.npz", **arrays)
    build(
        args.source,
        args.directory / "backend",
        plan["dimensions"],
        args.compiler,
        runtime_directory=args.runtime_directory,
        blas_library=args.blas_library,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
    )
    initialized = time.perf_counter()
    problem = CoreSD(
        space.orbitals,
        core._bsp_data,
        Z,
        args.ns,
        occupied_keys=space.occupied_keys,
        virtual_keys=space.virtual_keys,
        backend_directory=args.directory / "backend",
        nuclear_potential=core._nuclear_potential,
        memory_bytes=memory,
        radial_pair_cache_size=args.radial_pair_cache_size,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        radial_representation=args.radial_representation,
    )
    prepare_seconds = time.perf_counter() - initialized
    problem.set_diagram_progress(args.diagram_progress)
    print(
        json.dumps(dict(stage="prepared", prepare_seconds=prepare_seconds)), flush=True
    )
    # Profile repeated action on an identical full vector after preparation.
    t0 = problem.source / problem.denominators
    times = []
    for _ in range(3):
        start = time.perf_counter()
        problem.action(t0)
        times.append(time.perf_counter() - start)
    result = problem.solve(restart=args.restart, maxiter=args.maxiter)
    np.savez_compressed(
        args.directory / "solution.npz",
        amplitudes=result.amplitudes,
        source=problem.source,
        denominators=problem.denominators,
    )
    evidence = dict(
        reference_energy_Ha=result.reference_energy_Ha,
        correlation_energy_Ha=result.correlation_energy_Ha,
        total_energy_Ha=result.total_energy_Ha,
        residual_max_Ha=result.residual_max_Ha,
        residual_l2_Ha=result.residual_l2_Ha,
        solve_matvecs=result.matrix_vector_products,
        solve_wall_seconds=result.wall_seconds,
        prepare_seconds=prepare_seconds,
        scf_seconds=scf_seconds,
        full_wall_seconds=time.perf_counter() - began,
        action_seconds=times,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        space=dict(space.identity),
        provenance=dict(result.provenance),
        experiment_gate="not evaluated",
        space_convergence_gate="not evaluated",
    )
    (args.directory / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in evidence.items() if k not in ("space", "provenance")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=("be2", "be4", "ne10"), default="ne10")
    parser.add_argument("--ns", type=int, default=30)
    parser.add_argument("--ks", type=int, default=7)
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--emax", type=float, required=True)
    parser.add_argument("--memory-gib", type=int, default=2)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path)
    parser.add_argument("--blas-library", type=Path)
    parser.add_argument("--term21-threads", type=int, default=0)
    parser.add_argument(
        "--term21-mode", choices=("compact", "streamed"), default="compact"
    )
    parser.add_argument("--radial-pair-cache-size", type=int, default=16384)
    parser.add_argument("--diagram-progress", action="store_true")
    parser.add_argument("--core-singles-blas", action="store_true")
    parser.add_argument(
        "--radial-representation", choices=("native", "polynomial"), default="native"
    )
    main(parser.parse_args())
