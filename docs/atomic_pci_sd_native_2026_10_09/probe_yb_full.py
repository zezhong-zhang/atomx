"""Full retained Yb V68 / l<=3 / <5 Ha native reduced core-SD workload."""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
REFERENCE = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs/yb_diagnosis_repair_2026_10_08"))
from build_pci_sd_backend import build  # noqa: E402
from common import load_reference  # noqa: E402

from atomx.methods.atomic_space import build_atomic_space  # noqa: E402
from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)


def main(args):
    HERE = OWNER / args.run

    def save(name, value):
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")

    HERE.mkdir(exist_ok=False)
    began = time.perf_counter()
    cap = 32 * 1024**3
    contract = dict(
        reference="original native V68, Fermi nucleus, ns123",
        occupied="all 68 core electrons, no frozen subshell",
        virtual_lmax=3,
        virtual_energy_max_Ha=5,
        memory_bytes=cap,
        residual_tolerance_Ha=1e-10,
        gmres_restart=30,
        maxiter=100,
        load=os.getloadavg(),
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        diagram_progress=args.diagram_progress,
        radial_representation=args.radial_representation,
        reference_first_action=str(args.reference_first_action)
        if args.reference_first_action
        else None,
        reference_wait_seconds=args.reference_wait_seconds,
        signed_action_gate_Ha=1e-11,
        blas_threads=1,
        radial_pair_cache_size=args.radial_pair_cache_size,
        blas_library=str(args.blas_library),
        hypothesis="exact polynomial density coordinates and batched original reduced pCI diagrams preserve the complete native signed Yb action without magnetic quartic storage",
        stop="reject complete overbudget request or failed equation residual; no space reduction or fitted shifts",
        performance_scope="stage times and peak process RSS; no speedup or experimental accuracy claim",
    )
    save("yb_full_contract.json", contract)
    save(
        "yb_full_source_sha256.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                sorted((ROOT / "src/atomx").rglob("*.py"))
                + sorted((ROOT / "src/atomx").rglob("*.f90"))
                + [Path(__file__).resolve(), ROOT / "tools/build_pci_sd_backend.py"]
            )
        },
    )
    core, *_, reference_identity = load_reference()
    space = build_atomic_space(core, virtual_lmax=3, virtual_energy_max_Ha=5)
    features_type = (
        CellCoulombFeatures
        if args.radial_representation == "native"
        else PolynomialCellCoulombFeatures
    )
    feature = features_type(core._bsp_data)
    plan = plan_core_sd(
        space.occupied_keys,
        space.virtual_keys,
        feature.identity,
        radial_pair_cache_size=args.radial_pair_cache_size,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
    )
    assert len(space.occupied_keys) == 19 and len(space.virtual_keys) == 414
    assert plan["independent_scalar_amplitudes"] == 13165103
    save("yb_full_plan.json", plan)
    print(
        json.dumps(
            dict(
                stage="complete space and plan",
                coordinates=plan["raw_amplitudes"],
                payload=plan["complete_payload_bound_bytes"],
                seconds=time.perf_counter() - began,
            )
        ),
        flush=True,
    )
    if plan["complete_payload_bound_bytes"] + 40 * plan["krylov_vector_bytes"] > cap:
        raise MemoryError("complete Yb core SD plus Krylov exceeds explicit 32 GiB cap")
    build(
        REFERENCE / "pCI",
        HERE / "yb_full_backend",
        plan["dimensions"],
        REFERENCE / "gfortran",
        runtime_directory=REFERENCE / "toolchain/usr/lib/x86_64-linux-gnu",
        blas_library=args.blas_library,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
    )
    arrays = {f"P_{n}_{k}": v[0] for (n, k), v in space.orbitals.items()}
    arrays.update({f"Q_{n}_{k}": v[1] for (n, k), v in space.orbitals.items()})
    arrays.update(
        {
            f"workspace_{k}": v
            for k, v in core._bsp_data.items()
            if isinstance(v, (np.ndarray, int, float, bool))
        }
    )
    from atomx.potentials.nuclear import nuclear_profile

    arrays["nuclear_on_outer_gauss"] = np.vectorize(
        core._nuclear_potential, otypes=[float]
    )(core._bsp_data["gr"])
    arrays["nuclear_on_auxiliary_nodes"] = np.vectorize(
        core._nuclear_potential, otypes=[float]
    )(feature.radii)
    save("nuclear_model.json", nuclear_profile(70, "fermi"))
    np.savez_compressed(HERE / "yb_full_inputs.npz", **arrays)
    print(
        json.dumps(
            dict(
                stage="prepare full native operator and original diagrams",
                seconds=time.perf_counter() - began,
            )
        ),
        flush=True,
    )
    preparing = time.perf_counter()
    problem = CoreSD(
        space.orbitals,
        core._bsp_data,
        core.Z,
        core._bsp_data["ns"],
        occupied_keys=space.occupied_keys,
        virtual_keys=space.virtual_keys,
        nuclear_potential=core._nuclear_potential,
        backend_directory=HERE / "yb_full_backend",
        memory_bytes=cap,
        radial_pair_cache_size=args.radial_pair_cache_size,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        radial_representation=args.radial_representation,
    )
    prepare_seconds = time.perf_counter() - preparing
    problem.set_diagram_progress(args.diagram_progress)
    print(
        json.dumps(
            dict(
                stage="prepared",
                prepare_seconds=prepare_seconds,
                seconds=time.perf_counter() - began,
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
        ),
        flush=True,
    )
    initial = problem.source / problem.denominators
    start = time.perf_counter()
    residual = problem.action(initial) - problem.source
    action_seconds = time.perf_counter() - start
    np.savez_compressed(
        HERE / "yb_full_first_action.npz",
        amplitudes=initial,
        residual=residual,
        source=problem.source,
        denominators=problem.denominators,
    )
    save(
        "yb_full_first_action.json",
        dict(
            prepare_seconds=prepare_seconds,
            action_seconds=action_seconds,
            initial_residual_max_Ha=float(np.max(abs(residual))),
            provenance=problem.provenance,
            reference_archive={
                k: v for k, v in reference_identity.items() if k != "source_sha256"
            },
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )
    print(
        json.dumps(
            dict(
                stage="first full action",
                action_seconds=action_seconds,
                initial_residual_max_Ha=float(np.max(abs(residual))),
            )
        ),
        flush=True,
    )
    del initial, residual
    if args.reference_first_action:
        reference = args.reference_first_action.resolve()
        reference_json = reference.with_suffix(".json")
        waiting = time.perf_counter()
        print(
            json.dumps(
                dict(
                    stage="await same-space signed full action",
                    reference=str(reference),
                )
            ),
            flush=True,
        )
        while not reference_json.exists():
            if time.perf_counter() - waiting > args.reference_wait_seconds:
                raise TimeoutError(
                    "reference full action not complete; production solve held"
                )
            time.sleep(1)
        previous = json.loads(reference_json.read_text())
        if previous["provenance"]["feature_identity"]["geometry_sha256"] != (
            problem.features.identity.get(
                "native_geometry_sha256", problem.features.identity["geometry_sha256"]
            )
        ):
            raise ValueError("reference native radial geometry differs")
        old_plan = json.loads((reference.parent / "yb_full_plan.json").read_text())
        for field in (
            "occupied_keys",
            "virtual_keys",
            "raw_amplitudes",
            "independent_scalar_amplitudes",
        ):
            if json.loads(json.dumps(plan[field])) != old_plan[field]:
                raise ValueError("reference complete physical space differs")
        # Zip metadata/timestamps are not a scientific input identity. Compare
        # every native array; only auxiliary sampling follows the coordinate
        # transform and therefore has an intentionally different node list.
        with (
            np.load(reference.parent / "yb_full_inputs.npz", allow_pickle=False) as old,
            np.load(HERE / "yb_full_inputs.npz", allow_pickle=False) as new,
        ):
            if set(old.files) != set(new.files):
                raise ValueError("reference native input coverage differs")
            for field in old.files:
                if field != "nuclear_on_auxiliary_nodes" and not np.array_equal(
                    old[field], new[field]
                ):
                    raise ValueError("reference native input array differs: " + field)
        with np.load(reference, allow_pickle=False) as data:
            t = data["amplitudes"].copy()
            target = data["residual"] + data["source"]
            source_error = float(np.max(abs(data["source"] - problem.source)))
            denominator_error = float(
                np.max(abs(data["denominators"] - problem.denominators))
            )
        checking = time.perf_counter()
        actual = problem.action(t)
        error = float(np.max(abs(actual - target)))
        np.savez_compressed(
            HERE / "yb_full_reference_action.npz", amplitudes=t, action=actual
        )
        comparison = dict(
            reference=str(reference),
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            action_max_error_Ha=error,
            source_max_error_Ha=source_error,
            denominator_max_error_Ha=denominator_error,
            seconds=time.perf_counter() - checking,
            gate_Ha=1e-11,
            scope="identical full signed input vector, native orbitals, Hamiltonian and complete space",
        )
        save("yb_full_reference_action.json", comparison)
        print(
            json.dumps(dict(stage="same-space signed full-action gate", **comparison)),
            flush=True,
        )
        if (
            not np.all(np.isfinite([error, source_error, denominator_error]))
            or max(error, source_error, denominator_error) > 1e-11
        ):
            raise AssertionError(
                "complete Yb signed equation gate failed; production solve held"
            )
        del actual, target, t
    result = problem.solve(restart=30, maxiter=100)
    np.savez_compressed(
        HERE / "yb_full_solution.npz",
        amplitudes=result.amplitudes,
        source=problem.source,
        denominators=problem.denominators,
    )
    outcome = dict(
        reference_energy_Ha=result.reference_energy_Ha,
        correlation_energy_Ha=result.correlation_energy_Ha,
        total_energy_Ha=result.total_energy_Ha,
        residual_max_Ha=result.residual_max_Ha,
        residual_l2_Ha=result.residual_l2_Ha,
        matvecs=result.matrix_vector_products,
        solve_seconds=result.wall_seconds,
        prepare_seconds=prepare_seconds,
        first_action_seconds=action_seconds,
        full_seconds=time.perf_counter() - began,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        provenance=dict(result.provenance),
        optical_spectrum="not calculated: valence effective Hamiltonian remains separate",
        convergence="raw equations only; radial/virtual/partial-wave and experimental gates not passed",
    )
    save("yb_full_result.json", outcome)
    print(
        json.dumps({k: v for k, v in outcome.items() if k != "provenance"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="yb_full_run004")
    parser.add_argument("--radial-pair-cache-size", type=int, default=16384)
    parser.add_argument("--term21-threads", type=int, default=0)
    parser.add_argument("--diagram-progress", action="store_true")
    parser.add_argument("--core-singles-blas", action="store_true")
    parser.add_argument("--reference-first-action", type=Path)
    parser.add_argument("--reference-wait-seconds", type=int, default=3600)
    parser.add_argument(
        "--radial-representation", choices=("native", "polynomial"), default="native"
    )
    parser.add_argument(
        "--term21-mode", choices=("compact", "streamed"), default="compact"
    )
    parser.add_argument(
        "--blas-library",
        type=Path,
        default=REFERENCE
        / "toolchain/usr/lib/x86_64-linux-gnu/openblas-serial/libopenblas.so.0",
    )
    main(parser.parse_args())
