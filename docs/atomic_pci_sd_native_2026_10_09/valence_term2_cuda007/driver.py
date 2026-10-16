"""Same-space complete original/streamed term2 plus energy-response control."""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from build_pci_valence_backend import build_valence  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.valence_sd import plan_valence_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    if args.compare_value_policy and (not args.response_policy or args.value_only):
        raise ValueError("paired value/response check requires the unified full kernel")
    if args.cuda_density and (not args.response_policy or args.value_only):
        raise ValueError("CUDA control requires the unified complete response kernel")
    if args.cuda_fortran_output and not args.cuda_density:
        raise ValueError("CUDA native output layout requires the explicit CUDA path")
    if args.cuda_contractions and (not args.cuda_density or args.cuda_fortran_output):
        raise ValueError(
            "whole CUDA contractions require explicit CUDA and no host pair layout"
        )
    if args.paired_cpu and (
        not args.cuda_density or not args.full or args.threads != 8
    ):
        raise ValueError(
            "paired CPU/CUDA control requires the complete eight-worker Yb case"
        )
    if args.threads not in (1, 8) or args.replay_case and not args.full:
        raise ValueError(
            "declared serial/eight-worker control; replay only full source-bound case"
        )
    kernel = OWNER / (
        "valence_term2_response_policy.f90"
        if args.response_policy
        else "valence_term2_value.f90"
        if args.value_only
        else "valence_term2_parallel.f90"
        if args.threads > 1
        else "valence_term2_streamed.f90"
    )
    options = dict(
        term21_threads=8 if args.full else 2,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    contract = dict(
        hypothesis="Exact term2 can reuse complete virtual-pair potentials across core channels and project before angular recoupling, including full analytic denominator and supplied numerator response",
        fixed="Original full signed term2, native weighted/tagged Coulomb kernel and physical orbitals; same angular/radial pool, no screening or altered denominator",
        system="complete original Yb68core414virtual/ns123/Fermi/l<=3/5Ha"
        if args.full
        else "Na+ Ne-like all10core/ns38 point nucleus; all declared 3s4s3p4p3d virtuals with every magnetic partner",
        gate_Ha=1e-11,
        seed=314521,
        cap_bytes=32 * 1024**3 if args.full else 2 * 1024**3,
        load=os.getloadavg(),
        stop="one complete signed random input per declared column; candidate response evaluated only when requested; retain any failure; no physical result/precision claim",
        performance_scope="single diagnostic maps under observed load; no paired speedup claim",
        options=options,
        driver_sha256=digest(Path(__file__)),
        kernel_sha256=digest(kernel),
        valence_term2_workers=args.threads,
        value_only=args.value_only,
        unified_response_policy=args.response_policy,
        compare_value_policy=args.compare_value_policy,
        cuda_density=args.cuda_density,
        cuda_fortran_output=args.cuda_fortran_output,
        cuda_contractions=args.cuda_contractions,
        host_angular_plan_cap_bytes=128 * 1024**2 if args.cuda_contractions else None,
        device_cap_bytes=(2 if args.cuda_contractions else 1) * 1024**3
        if args.cuda_density
        else None,
        paired_cpu=args.paired_cpu,
        paired_load_guard=27 if args.paired_cpu else None,
        paired_timing_scope="warm complete term2 plus C copies, eight CPU workers/inner-one BLAS for both; GPU density transfers included; excludes preparation, compilation and full SD solve",
        response_scope="not evaluated by value-only candidate; full companion remains separate"
        if args.value_only
        else "complete supplied numerator and explicit denominator response",
        replay_case=args.replay_case,
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    shutil.copy2(__file__, out / "driver.py")
    identities = {}
    private_sources = [kernel]
    if args.cuda_density:
        private_sources.extend(
            OWNER / name
            for name in ("cuda_density_control.py", "cuda_term2_control.py")
        )
    if args.cuda_contractions:
        private_sources.append(OWNER / "cuda_full_term2_control.py")
    for path in (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx/methods").glob("_pci*.f90"))
        + sorted((ROOT / "tools").glob("build_pci*.py"))
        + private_sources
    ):
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        identities[str(relative)] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(identities, indent=2) + "\n")
    if args.full:
        source = OWNER / "yb_full_run008"
        p = json.loads((source / "yb_full_plan.json").read_text())
        occupied, virtual = (
            tuple(map(tuple, p[k])) for k in ("occupied_keys", "virtual_keys")
        )
        with np.load(source / "yb_full_inputs.npz", allow_pickle=False) as data:
            workspace = {
                k.removeprefix("workspace_"): data[k].copy()
                for k in data
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
        keys = ((6, -1),)
        shutil.copytree(OWNER / "yb_full_static003/backend", out / "backend")
        input_identity = dict(
            input_sha256=digest(source / "yb_full_inputs.npz"),
            parent_manifest_sha256=digest(
                OWNER / "yb_full_static003/backend/manifest.json"
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
        plan = plan_core_sd(
            occupied,
            virtual,
            PolynomialCellCoulombFeatures(workspace).identity,
            **options,
        )
        valence = plan_valence_sd(plan, keys)
        build_valence(
            REF / "pCI",
            out / "backend",
            plan["dimensions"],
            REF / "gfortran",
            nxv=valence["nxv"],
            nvh=valence["nvh"],
            runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
            blas_library=OWNER
            / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
            **options,
        )
        input_identity = dict(occupied=occupied, virtual=virtual)
    backend = out / "backend"
    manifest = json.loads((backend / "manifest.json").read_text())
    template = kernel.read_text().replace("CASE_THREADS", str(args.threads))
    if args.cuda_density and not args.cuda_contractions:
        from cuda_term2_control import BRIDGE, kernel_source

        template = kernel_source(template)
        (backend / "cuda_density_bridge.f90").write_text(BRIDGE)
    if args.cuda_contractions:
        from cuda_full_term2_control import plan_source

        (backend / "cuda_term2_plan.f90").write_text(plan_source(kernel.read_text()))
    if args.paired_cpu:
        cpu_template = (
            kernel.read_text()
            .split("subroutine atomx_val_term2_original_map")[0]
            .replace("CASE_THREADS", str(args.threads))
            .replace("valence_term2_streamed", "valence_term2_cpu")
            .replace("atomx_val_term2_streamed_map", "atomx_val_term2_cpu_map")
            .replace("atomx_val_term2_value_map", "atomx_val_term2_cpu_value_map")
        )
        (backend / "valence_term2_cpu.f90").write_text(cpu_template)
    (backend / "valence_term2_streamed.f90").write_text(template)
    command = manifest["command"][:]
    units = ["valence_term2_streamed.f90"]
    if args.cuda_density and not args.cuda_contractions:
        units.insert(0, "cuda_density_bridge.f90")
        command.insert(command.index("-o"), "cuda_density_bridge.o")
    if args.cuda_contractions:
        units.insert(0, "cuda_term2_plan.f90")
        command.insert(command.index("-o"), "cuda_term2_plan.o")
    if args.paired_cpu:
        units.append("valence_term2_cpu.f90")
        command.insert(command.index("-o"), "valence_term2_cpu.o")
    compile_command = command[: command.index("-shared")] + ["-c", *units]
    command.insert(command.index("-o"), "valence_term2_streamed.o")
    if args.threads > 1:
        compile_command.insert(compile_command.index("-c"), "-fopenmp")
    with (backend / "term2_compiler.log").open("w") as log:
        for invocation in (compile_command, command):
            subprocess.run(
                invocation,
                cwd=backend,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    manifest["compile_commands"].extend([compile_command, command])
    manifest["command"] = command
    manifest["diagnostic_term2"] = contract
    manifest["files"] = {
        p.name: digest(p)
        for p in backend.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (backend / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    core = CoreSD(
        waves,
        workspace,
        z,
        ns,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=nuclear,
        backend_directory=backend,
        memory_bytes=contract["cap_bytes"],
        radial_representation="polynomial",
        **options,
    )
    valence = plan_valence_sd(core.plan, keys)
    d = core.plan["dimensions"]
    estimated_scratch = 8 * (
        d["NHF"] * (d["NX"] * d["NK"] + d["NXX"] ** 2 + 4 * d["NXX"] + 2)
        + 4 * d["NXX"] ** 2
    )
    estimated_scratch += (
        (args.threads - 1) * 8 * (d["NHF"] * (4 * d["NXX"] + 2) + 4 * d["NXX"] ** 2)
    )
    bound = (
        valence["complete_payload_bound_bytes"]
        + estimated_scratch
        + 8 * 12 * d["NXX"] ** 2 * valence["nvh"]
    )
    if bound > contract["cap_bytes"]:
        raise MemoryError("complete diagnostic arrays/scratch exceed predeclared cap")
    (out / "resource.json").write_text(
        json.dumps(
            dict(
                bound_bytes=bound,
                extra_kernel_scratch_bytes=estimated_scratch,
                dimensions=dict(d),
                input_identity=input_identity,
            ),
            indent=2,
        )
        + "\n"
    )
    save_inputs = {f"P_{n}_{k}": waves[n, k][0] for n, k in occupied + virtual}
    save_inputs.update({f"Q_{n}_{k}": waves[n, k][1] for n, k in occupied + virtual})
    np.savez_compressed(out / "orbitals.npz", **save_inputs)
    lib = core._backend
    cuda_control = None
    if args.cuda_density and not args.cuda_contractions:
        from cuda_term2_control import NativeCUDADensityControl

        cuda_control = NativeCUDADensityControl(
            PolynomialCellCoulombFeatures(workspace),
            d,
            out,
            memory_bytes=contract["device_cap_bytes"],
            fortran_output=args.cuda_fortran_output,
        )
        cuda_control.bind(lib)
    if args.cuda_contractions:
        from cuda_full_term2_control import NativeCUDAFullTerm2Control

        cuda_control = NativeCUDAFullTerm2Control(
            PolynomialCellCoulombFeatures(workspace),
            d,
            out,
            lib,
            memory_bytes=contract["device_cap_bytes"],
        )
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    init = lib.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
    init.restype = None
    original, streamed = (
        lib.atomx_val_term2_original_map,
        lib.atomx_val_term2_value_map
        if args.value_only
        else lib.atomx_val_term2_streamed_map,
    )
    for fn, count in ((original, 4), (streamed, 2 if args.value_only else 4)):
        fn.argtypes = [ctypes.c_double] + [pointer] * count
        fn.restype = None
    if args.cuda_contractions:
        streamed = cuda_control.compute
    rng = np.random.default_rng(contract["seed"])
    results = []
    seed = np.zeros(core.plan["raw_amplitudes"])
    for key in keys:
        slot = next(g.index(key) for g in core.plan["groups"] if key in g)
        init(slot + 1, key[1], len(seed), seed)
        shape = (d["NXX"], d["NXX"], manifest["valence_sd"]["nvh"])
        numerator, prime, reference, derivative, actual, actual_prime = [
            np.zeros(shape, order="F") for _ in range(6)
        ]
        for channel, (_, im, inn, _) in enumerate(valence["channels"][key]):
            startm = sum(k in occupied for k in core.plan["groups"][im])
            startn = sum(k in occupied for k in core.plan["groups"][inn])
            endm, endn = len(core.plan["groups"][im]), len(core.plan["groups"][inn])
            numerator[startm:endm, startn:endn, channel] = (
                rng.normal(size=(endm - startm, endn - startn)) * 1e-4
            )
            prime[startm:endm, startn:endn, channel] = (
                rng.normal(size=(endm - startm, endn - startn)) * 1e-4
            )
        index = next(
            i for i, label in enumerate(core.integrals.labels) if label[:2] == key
        )
        energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
        print(
            json.dumps(
                dict(
                    stage="enter_original_term2", key=key, energy_Ha=energy, shape=shape
                )
            ),
            flush=True,
        )
        before = time.perf_counter()
        if args.replay_case:
            parent = OWNER / args.replay_case
            old_contract = json.loads((parent / "contract.json").read_text())
            if (
                old_contract["system"] != contract["system"]
                or old_contract["seed"] != contract["seed"]
            ):
                raise AssertionError("retained original physical input/seed differs")
            old_resource = json.loads((parent / "resource.json").read_text())
            if old_resource["input_identity"] != input_identity or old_resource[
                "dimensions"
            ] != dict(d):
                raise AssertionError(
                    "retained complete workspace/source dimensions differ"
                )
            with np.load(parent / "orbitals.npz", allow_pickle=False) as old_waves:
                if set(old_waves.files) != set(save_inputs) or any(
                    not np.array_equal(old_waves[name], value)
                    for name, value in save_inputs.items()
                ):
                    raise AssertionError("retained complete physical orbitals differ")
            old_rows = json.loads((parent / "partial_result.json").read_text())
            old_energy = next(
                row["energy_Ha"] for row in old_rows if row["key"] == list(key)
            )
            if energy != old_energy:
                raise AssertionError("retained physical denominator energy differs")
            data_path = parent / f"term2_{key[0]}_{key[1]}.npz"
            with np.load(data_path, allow_pickle=False) as data:
                if not np.array_equal(
                    numerator, data["numerator"]
                ) or not np.array_equal(prime, data["prime"]):
                    raise AssertionError("retained complete signed inputs differ")
                reference[:] = data["original"]
                derivative[:] = data["original_prime"]
            (out / "reference_binding.json").write_text(
                json.dumps(
                    dict(
                        parent_case=args.replay_case,
                        data_sha256=digest(data_path),
                        contract_sha256=digest(parent / "contract.json"),
                        orbitals_sha256=digest(parent / "orbitals.npz"),
                        resource_sha256=digest(parent / "resource.json"),
                        energy_Ha=energy,
                        purpose="replay actual unchanged complete signed inputs against retained original map; no recomputed reference or speedup claim",
                    ),
                    indent=2,
                )
                + "\n"
            )
            seconds_original = None
        else:
            original(energy, numerator, prime, reference, derivative)
            seconds_original = time.perf_counter() - before
        print(
            json.dumps(
                dict(stage="enter_streamed_term2", seconds_original=seconds_original)
            ),
            flush=True,
        )
        before = time.perf_counter()
        if args.value_only:
            streamed(energy, numerator, actual)
        else:
            streamed(energy, numerator, prime, actual, actual_prime)
        seconds_streamed = time.perf_counter() - before
        errors = [
            float(np.max(abs(a - b)))
            for a, b in (
                ((reference, actual),)
                if args.value_only
                else ((reference, actual), (derivative, actual_prime))
            )
        ]
        finite = all(
            np.all(np.isfinite(a))
            for a in (
                (reference, actual)
                if args.value_only
                else (reference, derivative, actual, actual_prime)
            )
        )
        label = f"{key[0]}_{key[1]}"
        arrays = dict(numerator=numerator, original=reference, streamed=actual)
        value_policy_error, policy_restore_error = None, None
        if args.compare_value_policy:
            value = np.zeros(shape, order="F")
            if args.cuda_contractions:
                cuda_control.compute(
                    energy, numerator, prime, value, actual_prime, response=False
                )
            else:
                value_function = lib.atomx_val_term2_value_map
                value_function.argtypes = [ctypes.c_double, pointer, pointer]
                value_function.restype = None
                value_function(energy, numerator, value)
            value_policy_error = float(np.max(abs(value - actual)))
            if not np.array_equal(value, actual):
                raise AssertionError("one-kernel value policy changed signed output")
            restored, restored_prime = [np.zeros(shape, order="F") for _ in range(2)]
            streamed(energy, numerator, prime, restored, restored_prime)
            policy_restore_error = max(
                float(np.max(abs(restored - actual))),
                float(np.max(abs(restored_prime - actual_prime))),
            )
            if not np.array_equal(restored, actual) or not np.array_equal(
                restored_prime, actual_prime
            ):
                raise AssertionError(
                    "value policy contaminated requested full response"
                )
            arrays.update(
                value_policy=value, restored=restored, restored_prime=restored_prime
            )
        if not args.value_only:
            arrays.update(
                prime=prime, original_prime=derivative, streamed_prime=actual_prime
            )
        np.savez_compressed(out / f"term2_{label}.npz", **arrays)
        record = dict(
            key=key,
            energy_Ha=energy,
            raw_signed_error_Ha=errors[0],
            raw_response_error_Ha=None if args.value_only else errors[1],
            finite=bool(finite),
            seconds_original=seconds_original,
            seconds_streamed=seconds_streamed,
            value_policy_error_Ha=value_policy_error,
            policy_restore_error_Ha=policy_restore_error,
        )
        results.append(record)
        (out / "partial_result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if not finite or max(errors) > contract["gate_Ha"]:
            raise AssertionError("complete signed original term2/response gate failed")
        if args.paired_cpu:
            cpu = lib.atomx_val_term2_cpu_map
            cpu.argtypes = [ctypes.c_double] + [pointer] * 4
            cpu.restype = None
            observations = []
            functions = {"cpu": cpu, "cuda": streamed}
            paired_value, paired_prime = (
                np.zeros_like(actual),
                np.zeros_like(actual_prime),
            )
            sequence = ("cpu", "cuda", "cpu", "cuda", "cuda", "cpu", "cpu", "cuda")
            for ordinal, variant in enumerate(sequence):
                load_before = os.getloadavg()
                started = time.perf_counter()
                functions[variant](energy, numerator, prime, paired_value, paired_prime)
                seconds = time.perf_counter() - started
                load_after = os.getloadavg()
                errors = [
                    float(np.max(abs(a - b)))
                    for a, b in (
                        (paired_value, reference),
                        (paired_prime, derivative),
                        (paired_value, actual),
                        (paired_prime, actual_prime),
                    )
                ]
                finite = all(
                    np.all(np.isfinite(a)) for a in (paired_value, paired_prime)
                )
                row = dict(
                    ordinal=ordinal,
                    variant=variant,
                    warmup=ordinal < 2,
                    seconds=seconds,
                    load_before=load_before,
                    load_after=load_after,
                    full_signed_errors_Ha=errors,
                    finite=bool(finite),
                )
                observations.append(row)
                (out / "paired_observations.json").write_text(
                    json.dumps(observations, indent=2) + "\n"
                )
                print(
                    json.dumps(dict(stage="paired_complete_term2", **row)), flush=True
                )
                if not finite or max(errors) > contract["gate_Ha"]:
                    np.savez_compressed(
                        out / "paired_failure.npz",
                        value=paired_value,
                        prime=paired_prime,
                    )
                    raise AssertionError(
                        "same-source complete CPU/CUDA paired signed gate failed"
                    )
            pairs = []
            for first in range(2, len(observations), 2):
                pair = {
                    row["variant"]: row["seconds"]
                    for row in observations[first : first + 2]
                }
                pairs.append(
                    dict(
                        pair=first // 2 - 1,
                        **pair,
                        cpu_over_cuda=pair["cpu"] / pair["cuda"],
                    )
                )
            guarded = all(
                max(row["load_before"] + row["load_after"])
                <= contract["paired_load_guard"]
                for row in observations
                if not row["warmup"]
            )
            (out / "paired_result.json").write_text(
                json.dumps(
                    dict(
                        full_signed_gate_passed=True,
                        load_guard_passed=guarded,
                        pairs=pairs,
                        observed_median_cpu_over_cuda=float(
                            np.median([row["cpu_over_cuda"] for row in pairs])
                        ),
                        controlled_speedup=None
                        if not guarded
                        else float(np.median([row["cpu_over_cuda"] for row in pairs])),
                        scope=contract["paired_timing_scope"],
                    ),
                    indent=2,
                )
                + "\n"
            )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="complete signed value-only kernel control passed"
                if args.value_only
                else "complete signed kernel/analytic-response control passed",
                rows=results,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                scope="original kernel equivalence only; no full solve or controlled speedup claim",
                cuda_complete_pair_calls=None
                if cuda_control is None
                else cuda_control.calls,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replay-case")
    parser.add_argument("--value-only", action="store_true")
    parser.add_argument("--response-policy", action="store_true")
    parser.add_argument("--compare-value-policy", action="store_true")
    parser.add_argument("--cuda-density", action="store_true")
    parser.add_argument("--cuda-fortran-output", action="store_true")
    parser.add_argument("--cuda-contractions", action="store_true")
    parser.add_argument("--paired-cpu", action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        out = OWNER / args.run_name
        if out.is_dir():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
