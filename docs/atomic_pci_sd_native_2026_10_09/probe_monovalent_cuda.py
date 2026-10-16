"""Independent finite CPU/CUDA controls for complete standalone SD equations.

Both policies use one actual native COMMON owner and the same declared full
physical space. The GPU specialization retains the existing complete ordered
term2 plan; term31 uses the independently tested FP64 terminal GEMM.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests"), str(OWNER)]

from build_pci_monovalent_backend import build_monovalent  # noqa: E402
from cuda_valence_bridge import BoundCUDAValenceControl  # noqa: E402
from cuda_valence_bridge import install as install_cuda  # noqa: E402
from monovalent_davidson_control import solve  # noqa: E402
from monovalent_term31_blas_control import install, scratch_bytes  # noqa: E402
from valence_term41_blas_control import install as install_term41  # noqa: E402
from valence_terma4_blas_control import install as install_terma4  # noqa: E402
from valence_terma4_blas_control import scratch_bytes as terma4_scratch  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.reduced_monovalent_sd import (  # noqa: E402
    ReducedMonovalentSD,
    plan_reduced_monovalent_sd,
)
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    cap, device_cap = 256 * 1024**2, 2 * 1024**3
    variants = (
        "unchanged",
        "virtual_rotated",
        "occupied_rotated",
        "combined_rotated",
    )
    save(
        "contract.json",
        dict(
            hypothesis="Same complete standalone CPU/CUDA maps and eight-step directed feedback roots agree with the independent full finite primary/Fock oracle for every M",
            fixed="Be+ Z4/point/ns30/ks7; complete declared12magnetic-spinor pool, real DC and native redundant direct-rho coordinates; matched .37virtual/.08occupied-virtual rotations, no amplitude/channel pruning",
            variants=variants,
            signed_and_root_gate_Ha=1e-11,
            numerical_raw_tolerance_Ha=1e-14,
            independent_raw_gate_Ha=1e-10,
            host_cap_bytes=cap,
            device_cap_bytes=device_cap,
            private_scratch_allowance_bytes=8 * 1024**2,
            term2_cpu_workers=8,
            private_term41_blas=args.term41_blas,
            private_terma4_blas=args.terma4_blas,
            core_cpu_workers=2,
            inner_blas_threads=1,
            load=os.getloadavg(),
            timing="single complete root observations with context/JIT overhead; no controlled end-to-end gain",
            stop="Any source/resource/independent/paired signed-map/root/raw/energy/branch failure rejects; no CPU fallback, weakened gates or changed space",
            scope="Private finite standalone CUDA term2 control, original term31 GEMM and complete-image Davidson; no public adoption, full-Yb GPU solve, normalized large states or physical precision certificate",
        ),
    )
    sources = (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx").rglob("*.f90"))
        + sorted((ROOT / "tools").glob("build_pci_*.py"))
        + [
            Path(__file__),
            ROOT / "tests/test_monovalent_sd.py",
            ROOT / "tests/test_reduced_monovalent_sd.py",
        ]
        + [
            OWNER / name
            for name in (
                "monovalent_term31_blas_control.py",
                "monovalent_davidson_control.py",
                "cuda_valence_bridge.py",
                "cuda_full_term2_control.py",
                "cuda_density_control.py",
            )
        ]
    )
    if args.term41_blas:
        sources.append(OWNER / "valence_term41_blas_control.py")
    if args.terma4_blas:
        sources.append(OWNER / "valence_terma4_blas_control.py")
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    helper = runpy.run_path(str(ROOT / "tests/test_reduced_monovalent_sd.py"))
    records, arrays = [], {}
    try:
        for variant in variants:
            current = out / variant
            current.mkdir()
            atom = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
            occupied, virtual = ((1, -1),), ((2, -1), (3, -1), (2, 1), (2, -2))
            waves = {(1, -1): atom.orbitals_by_subshell()[1, -1][:2]}
            for kappa in (-1, 1, -2):
                _, states = build_virtual_basis(
                    atom,
                    kappa,
                    e_max=50,
                    potential_mode="VN",
                    orthogonal_to_occupied=True,
                )
                for i, state in enumerate(states[: 2 if kappa == -1 else 1]):
                    waves[i + 2, kappa] = state[:2]
            rotations = []
            if variant in ("virtual_rotated", "combined_rotated"):
                rotations.append(((2, -1), (3, -1), 0.37))
            if variant in ("occupied_rotated", "combined_rotated"):
                rotations.append(((1, -1), (2, -1), 0.08))
            for left, right, angle in rotations:
                a, b = waves[left], waves[right]
                c, s = np.cos(angle), np.sin(angle)
                waves[left] = tuple(c * x + s * y for x, y in zip(a, b, strict=True))
                waves[right] = tuple(-s * x + c * y for x, y in zip(a, b, strict=True))
            options = dict(
                term21_threads=2,
                term21_mode="streamed",
                core_singles_blas=True,
                core_single_to_doubles_blas=True,
            )
            feature = PolynomialCellCoulombFeatures(atom._bsp_data)
            core_plan = plan_core_sd(occupied, virtual, feature.identity, **options)
            plan = plan_reduced_monovalent_sd(
                core_plan,
                virtual,
                valence_term2_blas=True,
                valence_term2_threads=8,
            )
            private_host = scratch_bytes(core_plan["dimensions"]) + (
                16 * core_plan["dimensions"]["NXX"] ** 2 * plan["nvh"]
            )
            if args.terma4_blas:
                private_host += terma4_scratch(core_plan["dimensions"])
            whole_bound = (
                max(
                    core_plan["complete_payload_bound_bytes"]
                    + 8 * 40 * core_plan["raw_amplitudes"],
                    plan["complete_payload_bound_bytes"]
                    + 8 * (40 * max(plan["amplitudes"].values()) + 8 * 17**2),
                )
                + private_host
            )
            save(
                variant + "_resource.json",
                dict(
                    complete_payload_bound_bytes=whole_bound,
                    private_additional_host_bytes=private_host,
                    host_cap_bytes=cap,
                    device_cap_bytes=device_cap,
                    amplitude_counts={str(k): v for k, v in plan["amplitudes"].items()},
                    status="pass" if whole_bound <= cap else "fail",
                ),
            )
            if whole_bound > cap:
                raise MemoryError("complete private host bound exceeds cap before load")
            directory = current / "backend"
            blas = (
                OWNER
                / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0"
            )
            build_monovalent(
                REF / "pCI",
                directory,
                core_plan["dimensions"],
                REF / "gfortran",
                nxv=plan["nxv"],
                nvh=plan["nvh"],
                runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
                blas_library=blas,
                valence_term2_blas=True,
                valence_term2_threads=8,
                **options,
            )
            manifest = install(directory, memory_bytes=8 * 1024**2)
            if any(
                digest(directory / name) != sha
                for name, sha in manifest["files"].items()
            ):
                raise ValueError("actual pre-CUDA backend artifact changed")
            if args.term41_blas:
                manifest = install_term41(directory)
            if args.terma4_blas:
                manifest = install_terma4(directory)
            if any(
                digest(directory / name) != sha
                for name, sha in manifest["files"].items()
            ):
                raise ValueError("actual pre-CUDA specialized artifact changed")
            install_cuda(directory)
            core = CoreSD(
                waves,
                atom._bsp_data,
                4,
                30,
                occupied_keys=occupied,
                virtual_keys=virtual,
                backend_directory=directory,
                nuclear_potential=atom._nuclear_potential,
                radial_representation="polynomial",
                memory_bytes=cap,
                **options,
            )
            core_result = core.solve(residual_tolerance_Ha=1e-12)
            problem = ReducedMonovalentSD(core, core_result, valence_keys=virtual)
            control = BoundCUDAValenceControl(
                problem.valence, current / "cuda", memory_bytes=device_cap
            )
            public_solve = problem.solve

            def private_solve(key, bound=problem, **kwargs):
                return solve(bound, key, residual_tolerance_Ha=1e-14)

            problem.solve = private_solve
            case = core, core_result, problem
            gates = 0
            for mode in ("cpu", "cuda"):
                control.select(mode)
                for key in virtual:
                    for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                        helper[
                            "test_complete_reduced_signed_map_against_fock_and_primary_anchor"
                        ](case, key, m)
                        helper[
                            "test_reduced_feedback_root_matches_complete_finite_reference"
                        ](case, key, m)
                        gates += 2
            problem.solve = public_solve
            helper["test_preflight_identity_and_original_fixed_energy_consumer"](case)
            problem.solve = private_solve
            paired = []
            for key in virtual:
                prefix = variant + "_" + str(key[0]) + "_" + str(key[1])
                rng = np.random.default_rng(7823 + 10 * key[0] + key[1])
                vector = rng.normal(0, 0.001, plan["amplitudes"][key])
                for linear in (False, True):
                    maps = []
                    for mode in ("cpu", "cuda"):
                        control.select(mode)
                        energy, action = problem.evaluate(key, vector, linear=linear)
                        value = np.r_[energy, action]
                        arrays[prefix + f"_{linear}_{mode}"] = value
                        maps.append(value)
                    if np.max(abs(maps[0] - maps[1])) > 1e-11:
                        raise AssertionError(
                            "complete paired CPU/CUDA signed map differs"
                        )
                roots, seconds = [], []
                for mode in ("cpu", "cuda"):
                    control.select(mode)
                    start = time.perf_counter()
                    root = problem.solve(key)
                    seconds.append(time.perf_counter() - start)
                    arrays[prefix + "_" + mode + "_root"] = root.amplitudes
                    roots.append(root)
                a, b = roots
                error = max(
                    abs(a.correlation_energy_Ha - b.correlation_energy_Ha),
                    float(np.max(abs(a.amplitudes - b.amplitudes))),
                    max(
                        abs(x[1] - y[1])
                        for x, y in zip(
                            a.continuation_path, b.continuation_path, strict=True
                        )
                    ),
                )
                if error > 1e-11:
                    raise AssertionError(
                        "complete paired CPU/CUDA feedback root differs"
                    )
                paired.append(
                    dict(
                        key=list(key),
                        maximum_error_Ha=error,
                        cpu_actions=a.matrix_vector_products,
                        cuda_actions=b.matrix_vector_products,
                        cpu_seconds=seconds[0],
                        cuda_seconds=seconds[1],
                        cuda_raw_Ha=b.residual_max_Ha,
                        cuda_energy_Ha=b.energy_residual_Ha,
                    )
                )
            if control.calls == 0:
                raise AssertionError(
                    "explicit CUDA policy did not execute native callback"
                )
            records.append(
                dict(
                    variant=variant,
                    independent_gates=gates + 1,
                    paired_roots=paired,
                    cuda_calls=control.calls,
                    epochs=control.epoch,
                )
            )
            save("progress.json", records)
            print(json.dumps(records[-1]), flush=True)
            control.select("cpu")
            del problem, control, core, core_result, case
        np.savez(out / "complete_arrays.npz", **arrays)
        changed = [path for path, sha in hashes.items() if digest(ROOT / path) != sha]
        if changed:
            raise ValueError("actual source changed during independent CUDA control")
        save(
            "result.json",
            dict(
                status="pass",
                variants=records,
                independent_gates=sum(row["independent_gates"] for row in records),
                arrays=len(arrays),
                arrays_sha256=digest(out / "complete_arrays.npz"),
                source_postcheck="pass",
                highest_level_reached=False,
            ),
        )
    except BaseException as error:
        save(
            "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--term41-blas", action="store_true")
    parser.add_argument("--terma4-blas", action="store_true")
    main(parser.parse_args())
