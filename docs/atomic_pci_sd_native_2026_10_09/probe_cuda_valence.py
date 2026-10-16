"""Private complete SD affine/homogeneous/response/solve CPU-CUDA proof."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from build_pci_valence_backend import build_valence  # noqa: E402
from cuda_valence_bridge import BoundCUDAValenceControl, install  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def plain(value):
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.profile_only and not args.full:
        raise ValueError("complete declared full Yb profiling scope required")
    if args.compare_profile_case is not None and not args.profile_only:
        raise ValueError("explicit complete profile comparison scope required")
    if args.compare_map_case is not None and (not args.full or args.profile_only):
        raise ValueError("explicit complete full-map comparison scope required")
    if args.fused_angular and args.projection_policy != "reachable":
        raise ValueError("fused angular control requires exactly reachable fields")
    if args.retained_cpu and (
        not args.full or args.profile_only or args.compare_map_case is None
    ):
        raise ValueError("retained CPU requires complete certified full-map reference")
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2) + "\n")

    contract = dict(
        hypothesis="Whole CUDA term2 is additive in the complete same native affine/homogeneous SD map, retains all supplied analytic responses and fixed-core physical source, and produces fresh complete solved equations",
        fixed="Original Yb68core414virtual/ns123/Fermi/l<=3/5Ha, every native channel and scalar Fock"
        if args.full
        else "Na+ Ne-like complete10core/ns38/point nucleus and declared3s4s3p4p3d virtuals, all magnetic partners",
        gate_Ha=1e-11,
        raw_gate_Ha=1e-10,
        seed=904121,
        host_cap_bytes=32 * 1024**3 if args.full else 2 * 1024**3,
        device_cap_bytes=2 * 1024**3,
        term2_cpu_workers=8,
        blas_inner_threads=1,
        load=os.getloadavg(),
        solve_policy="finite five-column fresh CPU/GPU solves; full case complete maps only, independent production solve follows",
        stop="Reject changed physical identity, incomplete affine/homogeneous/prime/mode or solution gate, nonfinite output or cap overrun; no gate/space alteration",
        scope="Private complete-map and finite solve arithmetic control; no controlled timing, public backend adoption, physical metric or spectrum claim",
        homogeneous_stage_profile_only=args.profile_only,
        private_term41_blas=args.term41_blas,
        private_terma4_blas=args.terma4_blas,
        private_projection_policy=args.projection_policy,
        private_fused_angular=args.fused_angular,
        cpu_reference_policy="retained complete certified same-input CPU arrays; no fresh CPU execution or CPU timing"
        if args.retained_cpu
        else "fresh complete CPU execution",
        complete_profile_reference=args.compare_profile_case,
        complete_map_reference=args.compare_map_case,
    )
    if args.profile_only:
        contract.update(
            hypothesis="Complete current homogeneous map equals terma4+term2+term41+homogeneous Fock; original diagrams, complete fields/coordinates/FP64 and signed1e-11 gate unchanged",
            solve_policy="fresh full current core/source/denominator/energy/raw verification; complete homogeneous five-column decomposition only, no valence root or CPU/GPU map timing claim",
        )
    save("contract.json", contract)
    paths = (
        list((ROOT / "src/atomx").rglob("*.py"))
        + list((ROOT / "src/atomx").rglob("*.f90"))
        + list((ROOT / "tools").glob("build_pci_*.py"))
        + [Path(__file__).resolve()]
        + [
            OWNER / name
            for name in (
                "cuda_valence_bridge.py",
                "cuda_full_term2_control.py",
                "cuda_density_control.py",
            )
        ]
    )
    if args.profile_only:
        paths.append(OWNER / "valence_stage_profile.py")
    if args.projection_policy is not None:
        paths.append(OWNER / "cuda_batched_projection_control.py")
    if args.fused_angular:
        paths.append(OWNER / "cuda_fused_angular_control.py")
    if args.term41_blas:
        paths.append(OWNER / "valence_term41_blas_control.py")
    if args.terma4_blas:
        paths.append(OWNER / "valence_terma4_blas_control.py")
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    if args.full:
        source = OWNER / "yb_full_run008"
        plan_record = json.loads((source / "yb_full_plan.json").read_text())
        occupied, virtual = (
            tuple(map(tuple, plan_record[k])) for k in ("occupied_keys", "virtual_keys")
        )
        with np.load(source / "yb_full_inputs.npz", allow_pickle=False) as data:
            workspace = {
                k.removeprefix("workspace_"): data[k].copy()
                for k in data.files
                if k.startswith("workspace_")
            }
            workspace = {
                k: v.item() if v.ndim == 0 else v for k, v in workspace.items()
            }
            waves = {
                key: (
                    data[f"P_{key[0]}_{key[1]}"].copy(),
                    data[f"Q_{key[0]}_{key[1]}"].copy(),
                )
                for key in occupied + virtual
            }
        z, ns, nuclear = 70, 123, make_nuclear_potential(70, "fermi")
        keys = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
        save(
            "parent_binding.json",
            dict(
                input_sha256=digest(source / "yb_full_inputs.npz"),
                core_solution_sha256=digest(source / "yb_full_solution.npz"),
                declared_all_virtuals=len(virtual),
            ),
        )
    else:
        occupied = ((1, -1), (2, -1), (2, 1), (2, -2))
        virtual = ((3, -1), (4, -1), (3, 1), (4, 1), (3, -2), (4, -2), (3, 2), (3, -3))
        atom = DFSolver(
            Z=11,
            subshells=[Subshell(n, k, 2 * abs(k)) for n, k in occupied],
            ns=38,
            ks=7,
        ).solve()
        waves = {key: atom.orbitals_by_subshell()[key][:2] for key in occupied}
        for kappa in sorted({k[1] for k in virtual}):
            _, states = build_virtual_basis(
                atom, kappa, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
            )
            for key, state in zip(
                [k for k in virtual if k[1] == kappa], states, strict=False
            ):
                waves[key] = state[:2]
        workspace = atom._bsp_data
        z, ns, nuclear = 11, 38, atom._nuclear_potential
        keys = ((3, -1), (3, 1), (3, -2), (3, 2), (3, -3))
    options = dict(
        term21_threads=8 if args.full else 2,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    feature = PolynomialCellCoulombFeatures(workspace)
    core_plan = plan_core_sd(occupied, virtual, feature.identity, **options)
    val_plan = plan_valence_sd(
        core_plan, keys, valence_term2_blas=True, valence_term2_threads=8
    )
    largest = max(val_plan["raw_amplitudes"].values())
    payload = max(
        core_plan["complete_payload_bound_bytes"]
        + 40 * core_plan["krylov_vector_bytes"],
        val_plan["complete_payload_bound_bytes"]
        + 8
        * (50 * largest + 2 * core_plan["dimensions"]["NXX"] ** 2 * val_plan["nvh"]),
    )
    if args.profile_only:
        payload = max(
            payload,
            val_plan["complete_payload_bound_bytes"]
            + 8
            * (
                20 * core_plan["dimensions"]["NXX"] ** 2 * val_plan["nvh"]
                + 12 * largest
            )
            + 3 * core_plan["krylov_vector_bytes"],
        )
    singles_scratch = (
        32 * core_plan["dimensions"]["NHF"] * core_plan["dimensions"]["NX"]
        if args.terma4_blas
        else 0
    )
    payload += singles_scratch
    save(
        "resource.json",
        dict(
            core_plan=core_plan,
            valence_plan=val_plan,
            complete_payload_bound_bytes=payload,
            additional_complete_singles_scratch_bytes=singles_scratch,
        ),
    )
    if payload > contract["host_cap_bytes"]:
        raise MemoryError("complete CPU/CUDA proof payload exceeds cap")
    backend = out / "backend"
    build_valence(
        REF / "pCI",
        backend,
        core_plan["dimensions"],
        REF / "gfortran",
        nxv=val_plan["nxv"],
        nvh=val_plan["nvh"],
        runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
        blas_library=OWNER
        / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
        valence_term2_blas=True,
        valence_term2_threads=8,
        **options,
    )
    install(backend)
    if args.term41_blas:
        from valence_term41_blas_control import install as install_term41

        install_term41(backend)
    if args.terma4_blas:
        from valence_terma4_blas_control import install as install_terma4

        install_terma4(backend)
    core = CoreSD(
        waves,
        workspace,
        z,
        ns,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=nuclear,
        backend_directory=backend,
        memory_bytes=contract["host_cap_bytes"],
        radial_representation="polynomial",
        **options,
    )
    initial = None
    if args.full:
        old = json.loads((source / "yb_full_result.json").read_text())
        with np.load(source / "yb_full_solution.npz", allow_pickle=False) as data:
            source_error = float(np.max(abs(data["source"] - core.source)))
            denominator_equal = np.array_equal(data["denominators"], core.denominators)
            initial = data["amplitudes"].copy()
        if source_error > 1e-11 or not denominator_equal:
            raise ValueError(
                "complete original core source/denominator identity differs"
            )
    result = core.solve(initial_amplitudes=initial)
    if (
        args.full
        and abs(result.correlation_energy_Ha - old["correlation_energy_Ha"]) > 1e-11
    ):
        raise ValueError("complete original solved core energy differs")
    del initial
    save(
        "core.json",
        dict(
            provenance=result.provenance,
            residual_max_Ha=result.residual_max_Ha,
            energy_Ha=result.correlation_energy_Ha,
        ),
    )
    physical = {f"P_{n}_{k}": waves[n, k][0] for n, k in occupied + virtual}
    physical.update({f"Q_{n}_{k}": waves[n, k][1] for n, k in occupied + virtual})
    physical.update(
        {
            f"workspace_{k}": v
            for k, v in workspace.items()
            if isinstance(v, (np.ndarray, int, float))
        }
    )
    np.savez_compressed(out / "physical_inputs.npz", **physical)
    problem = ValenceSD(core, result, valence_keys=keys)
    if args.projection_policy is None:
        control = BoundCUDAValenceControl(
            problem, out / "cuda", memory_bytes=contract["device_cap_bytes"]
        )
    else:
        from cuda_batched_projection_control import BoundBatchedCUDAControl

        control_type = BoundBatchedCUDAControl
        if args.fused_angular:
            from cuda_fused_angular_control import BoundFusedCUDAControl

            control_type = BoundFusedCUDAControl
        control = control_type(
            problem,
            out / "cuda",
            memory_bytes=contract["device_cap_bytes"],
            projection_policy=args.projection_policy,
        )
    parent_profile = None
    parent_map = None
    reference_case = args.compare_profile_case or args.compare_map_case
    if reference_case is not None:
        parent_reference = OWNER / reference_case
        previous = json.loads((parent_reference / "contract.json").read_text())
        if args.retained_cpu:
            certified = json.loads((parent_reference / "result.json").read_text())
            if certified[
                "status"
            ] != "all complete additive map/mode and declared solve gates passed" or [
                tuple(c["key"]) for c in certified["columns"]
            ] != list(keys):
                raise ValueError("complete certified original CPU maps required")
        if any(
            previous[name] != contract[name]
            for name in (
                "fixed",
                "seed",
                "gate_Ha",
                "host_cap_bytes",
                "device_cap_bytes",
            )
        ):
            raise ValueError(
                "complete original profile physical/resource contract differs"
            )
        with np.load(
            parent_reference / "physical_inputs.npz", allow_pickle=False
        ) as data:
            if set(data.files) != set(physical) or any(
                not np.array_equal(data[name], value)
                for name, value in physical.items()
            ):
                raise ValueError("complete original profile physical inputs differ")
        save(
            "complete_reference_binding.json",
            dict(
                parent=reference_case,
                physical_inputs_sha256=digest(parent_reference / "physical_inputs.npz"),
                result_sha256=digest(parent_reference / "result.json"),
            ),
        )
        if args.compare_profile_case is not None:
            parent_profile = parent_reference
        else:
            parent_map = parent_reference
    rng = np.random.default_rng(contract["seed"])
    records = []
    for key in keys:
        partial = next(i for i, group in enumerate(core.plan["groups"]) if key in group)
        anchor = next(k for k in core.plan["groups"][partial] if k in virtual)
        index = core.integrals.labels.index((*anchor, -2 * abs(anchor[1]) + 1))
        energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
        count = problem.plan["raw_amplitudes"][key]
        numerator, prime = (rng.normal(size=count) * 0.0001 for _ in range(2))
        arrays = dict(numerator=numerator, prime=prime)
        row = dict(key=key, energy_Ha=energy, complete_numerators=count, variants={})
        if args.profile_only:
            from valence_stage_profile import profile

            control.select("cuda")
            row = profile(
                problem, key, energy, numerator, out / f"profile_{key[0]}_{key[1]}"
            )
            if parent_profile is not None:
                label = f"profile_{key[0]}_{key[1]}"
                previous = json.loads(
                    (parent_profile / label / "result.json").read_text()
                )
                if previous["energy_Ha"] != energy or previous["key"] != list(key):
                    raise ValueError("complete profile column energy differs")
                with (
                    np.load(
                        parent_profile / label / "outputs.npz", allow_pickle=False
                    ) as old_arrays,
                    np.load(
                        out / label / "outputs.npz", allow_pickle=False
                    ) as new_arrays,
                ):
                    if set(old_arrays.files) != set(
                        new_arrays.files
                    ) or not np.array_equal(old_arrays["vector"], new_arrays["vector"]):
                        raise ValueError(
                            "complete profile input or output coordinates differ"
                        )
                    errors = {
                        name: float(np.max(abs(old_arrays[name] - new_arrays[name])))
                        for name in old_arrays.files
                        if name != "vector"
                    }
                if max(errors.values()) > contract["gate_Ha"]:
                    raise AssertionError(
                        "complete retained original profile signed gate failed"
                    )
                row["retained_original_signed_errors_Ha"] = errors
                row["retained_original_outputs_sha256"] = digest(
                    parent_profile / label / "outputs.npz"
                )
            records.append(row)
            save("partial_result.json", records)
            print(
                json.dumps(
                    dict(
                        stage="complete_homogeneous_profile_passed",
                        key=key,
                        errors=row["complete_signed_errors_Ha"],
                        observations=row["observations"],
                    )
                ),
                flush=True,
            )
            continue
        if args.retained_cpu:
            label = f"{key[0]}_{key[1]}"
            with np.load(
                parent_map / f"column_{label}.npz", allow_pickle=False
            ) as old_arrays:
                if not all(
                    np.array_equal(old_arrays[name], arrays[name])
                    for name in ("numerator", "prime")
                ):
                    raise ValueError("retained complete CPU map inputs differ")
                arrays.update(
                    {
                        name: old_arrays[name].copy()
                        for name in old_arrays.files
                        if name.startswith("cpu_")
                    }
                )
            row["variants"]["cpu"] = dict(
                role="retained certified complete original same-input CPU arrays",
                fresh_executed=False,
                seconds=None,
                outputs_sha256=digest(parent_map / f"column_{label}.npz"),
            )
        for variant in ("cuda",) if args.retained_cpu else ("cpu", "cuda"):
            control.select(variant)
            begin = time.perf_counter()
            with core._lock:
                context = problem._initialize(key, energy)
                affine = problem._evaluate(energy, numerator, prime, context)
                homogeneous = problem._evaluate(
                    energy, numerator, prime, context, linear=True
                )
                value = problem._evaluate_value(energy, numerator, context)
                restored = problem._evaluate(
                    energy, numerator, prime, context, linear=True
                )
            for label, values in (
                ("affine", affine),
                ("homogeneous", homogeneous),
                ("value", value),
                ("restored", restored),
            ):
                arrays.update(
                    {f"{variant}_{label}_{i}": a for i, a in enumerate(values)}
                )
            if not all(
                np.array_equal(a, b) for a, b in zip(homogeneous, restored, strict=True)
            ) or not all(
                np.array_equal(a, b)
                for a, b in zip(homogeneous[:2], value, strict=True)
            ):
                raise AssertionError(
                    "complete homogeneous full/value/full recovery differs"
                )
            row["variants"][variant] = dict(
                seconds=time.perf_counter() - begin, complete_mode_recovery=True
            )
            print(json.dumps(dict(stage="complete_native_maps", **row)), flush=True)
        row["map_signed_errors_Ha"] = {
            label: [
                float(
                    np.max(
                        abs(arrays[f"cpu_{label}_{i}"] - arrays[f"cuda_{label}_{i}"])
                    )
                )
                for i in range(size)
            ]
            for label, size in (("affine", 4), ("homogeneous", 4), ("value", 2))
        }
        if (
            max(v for values in row["map_signed_errors_Ha"].values() for v in values)
            > contract["gate_Ha"]
        ):
            raise AssertionError("complete additive native SD signed gate failed")
        if not args.full:
            solved = {}
            for variant in ("cpu", "cuda"):
                control.select(variant)
                solved[variant] = problem.solve(key, energy)
                column = solved[variant]
                for name in (
                    "numerators",
                    "numerator_derivative",
                    "sigma1_Ha",
                    "sigma1_derivative",
                ):
                    arrays[f"{variant}_{name}"] = getattr(column, name)
                row["variants"][variant].update(
                    raw_root_Ha=column.residual_max_Ha,
                    raw_response_Ha=column.derivative_residual_max_Ha,
                    matvecs=column.matrix_vector_products,
                    solve_seconds=column.wall_seconds,
                )
            row["solve_signed_errors_Ha"] = {
                name: float(
                    np.max(
                        abs(
                            getattr(solved["cpu"], name) - getattr(solved["cuda"], name)
                        )
                    )
                )
                for name in (
                    "numerators",
                    "numerator_derivative",
                    "sigma1_Ha",
                    "sigma1_derivative",
                )
            }
            if max(row["solve_signed_errors_Ha"].values()) > contract["gate_Ha"]:
                raise AssertionError("complete CUDA solved signed gate failed")
        label = f"{key[0]}_{key[1]}"
        if parent_map is not None:
            previous = json.loads((parent_map / f"column_{label}.json").read_text())
            if previous["energy_Ha"] != energy or previous["key"] != list(key):
                raise ValueError("complete original map column energy differs")
            with np.load(
                parent_map / f"column_{label}.npz", allow_pickle=False
            ) as old_arrays:
                if set(old_arrays.files) != set(arrays) or not all(
                    np.array_equal(old_arrays[name], arrays[name])
                    for name in ("numerator", "prime")
                ):
                    raise ValueError(
                        "complete original map input/coordinate identity differs"
                    )
                errors = {
                    name: float(np.max(abs(old_arrays[name] - arrays[name])))
                    for name in old_arrays.files
                    if name not in ("numerator", "prime")
                }
            if max(errors.values()) > contract["gate_Ha"]:
                raise AssertionError(
                    "complete original affine/homogeneous/analytic-response gate failed"
                )
            row["retained_original_signed_errors_Ha"] = errors
            row["retained_original_outputs_sha256"] = digest(
                parent_map / f"column_{label}.npz"
            )
        np.savez_compressed(out / f"column_{label}.npz", **arrays)
        save(f"column_{label}.json", row)
        records.append(row)
        save("partial_result.json", records)
        print(json.dumps(dict(stage="complete_column_passed", **row)), flush=True)
    save(
        "result.json",
        dict(
            status="complete homogeneous stage decomposition gates passed"
            if args.profile_only
            else "all complete additive map/mode and declared solve gates passed",
            columns=records,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            controlled_timing=False,
            public_cuda_adopted=False,
            physical_metric_certified=False,
            experiment_validated=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--term41-blas", action="store_true")
    parser.add_argument("--terma4-blas", action="store_true")
    parser.add_argument("--projection-policy", choices=("all", "reachable"))
    parser.add_argument("--fused-angular", action="store_true")
    parser.add_argument("--retained-cpu", action="store_true")
    parser.add_argument("--compare-profile-case")
    parser.add_argument("--compare-map-case")
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        (OWNER / args.run_name / "failure.json").write_text(
            json.dumps(
                dict(error_type=type(error).__name__, error=str(error)), indent=2
            )
            + "\n"
        )
        raise
