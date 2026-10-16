"""Four finite primary/Fock-space controls for private standalone term31 GEMM."""

import argparse
import ctypes
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(OWNER)]

from monovalent_davidson_control import solve as davidson_solve  # noqa: E402
from monovalent_term31_blas_control import install, scratch_bytes  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd  # noqa: E402
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

    save(
        "contract.json",
        dict(
            hypothesis="Private complete terminal P/Q term31 GEMM equals the original signed radial loop and independent finite primary/Fock equations for every valence column/M",
            fixed="Original/streamed finite native DC twelve-spinor Be+ space; .37rad complete virtual and .08rad occupied-virtual rotations are separately matched changed inputs; no amplitude/channel removal",
            variants=[
                "unchanged",
                "virtual_rotated",
                "occupied_rotated",
                "combined_rotated",
            ],
            maps_and_roots_gate_Ha=1e-11,
            raw_gate_Ha=1e-10,
            state_gate=1e-10,
            cap_bytes=256 * 1024**2,
            private_extra_allowance_bytes=8 * 1024**2,
            inverse_policy="Unchanged public standalone continuation/GMRES/raw feedback gates",
            stop="Any finite signed/root/raw/state/resource/source failure rejects; no gate/space changes",
            load=os.getloadavg(),
            scope="Private finite arithmetic/root comparison; no large-space, end-to-end performance, GPU or public adoption claim",
            private_davidson=args.davidson,
            davidson_contract="Exact directed augmented pencil with complete native bare-ev image reuse,16-vector numerical subspace/MGS2/restart and unchanged8couplings; actual complete public raw/energy each step and final affine; finite numerical target1e-14Ha, physical comparison gates unchanged"
            if args.davidson
            else None,
        ),
    )
    files = (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx").rglob("*.f90"))
        + sorted((ROOT / "tools").glob("build_pci_*.py"))
        + [
            ROOT / "tests/test_monovalent_sd.py",
            ROOT / "tests/test_reduced_monovalent_sd.py",
            Path(__file__),
            OWNER / "monovalent_term31_blas_control.py",
            OWNER / "monovalent_davidson_control.py",
        ]
    )
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    helper = runpy.run_path(str(ROOT / "tests/test_reduced_monovalent_sd.py"))
    real_init = CoreSD.__init__

    class Factory:
        def __init__(self, variant, mode):
            self.variant, self.mode = variant, mode

        def mktemp(self, name):
            path = out / self.variant / self.mode / name
            path.mkdir(parents=True, exist_ok=False)
            return path

    records, arrays = [], {}
    for variant in (
        "unchanged",
        "virtual_rotated",
        "occupied_rotated",
        "combined_rotated",
    ):
        cases = []
        for mode in ("original", "blas"):

            def prepare(current, *ctor_args, variant=variant, mode=mode, **kwargs):
                orbitals = ctor_args[0]
                rotations = []
                if variant in ("virtual_rotated", "combined_rotated"):
                    rotations.append(((2, -1), (3, -1), 0.37))
                if variant in ("occupied_rotated", "combined_rotated"):
                    rotations.append(((1, -1), (2, -1), 0.08))
                for left, right, angle in rotations:
                    a, b = orbitals[left], orbitals[right]
                    c, s = np.cos(angle), np.sin(angle)
                    orbitals[left] = tuple(
                        c * x + s * y for x, y in zip(a, b, strict=True)
                    )
                    orbitals[right] = tuple(
                        -s * x + c * y for x, y in zip(a, b, strict=True)
                    )
                if mode == "blas":
                    feature = PolynomialCellCoulombFeatures(ctor_args[1])
                    core_plan = plan_core_sd(
                        kwargs["occupied_keys"],
                        kwargs["virtual_keys"],
                        feature.identity,
                        **{
                            name: kwargs[name]
                            for name in (
                                "term21_threads",
                                "term21_mode",
                                "core_singles_blas",
                                "core_single_to_doubles_blas",
                            )
                        },
                    )
                    plan = plan_reduced_monovalent_sd(
                        core_plan,
                        kwargs["virtual_keys"],
                        valence_term2_blas=True,
                        valence_term2_threads=2,
                    )
                    bound = (
                        plan["complete_payload_bound_bytes"]
                        + scratch_bytes(core_plan["dimensions"])
                        + 8 * (40 * max(plan["amplitudes"].values()) + 4 * 21**2)
                    )
                    if bound > kwargs["memory_bytes"]:
                        raise MemoryError(
                            "complete private control payload exceeds cap before load"
                        )
                    install(kwargs["backend_directory"], memory_bytes=8 * 1024**2)
                real_init(current, *ctor_args, **kwargs)

            with patch.object(CoreSD, "__init__", prepare):
                case = helper["reduced_core"].__wrapped__(
                    Factory(variant, mode), SimpleNamespace(param="streamed")
                )
            core, core_result, problem = case
            if (
                problem.plan["complete_payload_bound_bytes"]
                + scratch_bytes(core.plan["dimensions"])
                + 40 * 8 * max(problem.plan["amplitudes"].values())
                > core.memory_bytes
            ):
                raise MemoryError(
                    "complete current finite standalone plus private scratch exceeds cap"
                )
            cases.append(case)
            if mode == "blas":
                public_solve = problem.solve
                if args.davidson:

                    def private_solve(key, bound=problem, **kwargs):
                        return davidson_solve(
                            bound,
                            key,
                            residual_tolerance_Ha=min(
                                kwargs.get("residual_tolerance_Ha", 1e-10), 1e-14
                            ),
                        )

                    problem.solve = private_solve
                gates = 0
                for key in problem.plan["valence_keys"]:
                    for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                        helper[
                            "test_complete_reduced_signed_map_against_fock_and_primary_anchor"
                        ](case, key, m)
                        helper[
                            "test_reduced_feedback_root_matches_complete_finite_reference"
                        ](case, key, m)
                        gates += 2
                problem.solve = public_solve
                helper["test_preflight_identity_and_original_fixed_energy_consumer"](
                    case
                )
                if args.davidson:
                    problem.solve = private_solve
                records.append(
                    dict(
                        variant=variant,
                        independent_gates=gates + 1,
                        private_extra_bytes=scratch_bytes(core.plan["dimensions"]),
                    )
                )
        original, optimized = (case[2] for case in cases)
        for key in original.plan["valence_keys"]:
            prefix = variant + "_" + str(key[0]) + "_" + str(key[1])
            rng = np.random.default_rng(915 + key[0] * 10 + key[1])
            vector = rng.normal(0, 0.001, original.plan["amplitudes"][key])
            with optimized.core._lock:
                prepared = optimized._prepare(key)
                s = np.zeros(optimized.core.plan["dimensions"]["NX"])
                s[prepared[1]] = vector[: len(prepared[1])]
                native = optimized.core._backend.atomx_monovalent_term31_control
                pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
                native.argtypes = [ctypes.c_int, pointer, pointer]
                native.restype = None
                component_a, component_b = (
                    np.zeros(prepared[0][2], order="F") for _ in range(2)
                )
                native(0, s, component_a)
                native(1, s, component_b)
                if np.max(abs(component_a - component_b)) > 1e-11:
                    raise AssertionError(
                        "complete callable original/BLAS term31 component differs"
                    )
                arrays[prefix + "_original_term31"] = component_a
                arrays[prefix + "_blas_term31"] = component_b
            for linear in (False, True):
                a, x = original.evaluate(key, vector, linear=linear)
                b, y = optimized.evaluate(key, vector, linear=linear)
                if max(abs(a - b), float(np.max(abs(x - y)))) > 1e-11:
                    raise AssertionError(
                        "complete private/original signed term31 map differs"
                    )
                arrays[prefix + "_vector"] = vector
                arrays[prefix + "_" + str(linear) + "_original"] = np.r_[a, x]
                arrays[prefix + "_" + str(linear) + "_blas"] = np.r_[b, y]
            a, b = (
                original.solve(key, residual_tolerance_Ha=1e-14),
                optimized.solve(key, residual_tolerance_Ha=1e-14),
            )
            if (
                max(
                    abs(a.correlation_energy_Ha - b.correlation_energy_Ha),
                    float(np.max(abs(a.amplitudes - b.amplitudes))),
                )
                > 1e-11
            ):
                raise AssertionError("complete private/original feedback root differs")
            arrays[prefix + "_original_root"] = a.amplitudes
            arrays[prefix + "_blas_root"] = b.amplitudes
            if args.davidson:
                if (
                    max(
                        abs(x[1] - y[1])
                        for x, y in zip(
                            a.continuation_path, b.continuation_path, strict=True
                        )
                    )
                    > 1e-11
                ):
                    raise AssertionError(
                        "private/public reference-connected eight-step branch differs"
                    )
                records[-1].setdefault("roots", []).append(
                    dict(
                        key=key,
                        original_actions=a.matrix_vector_products,
                        private_actions=b.matrix_vector_products,
                        private_raw_Ha=b.residual_max_Ha,
                        private_energy_Ha=b.energy_residual_Ha,
                    )
                )
        save("progress.json", records)
        print(json.dumps(records[-1]), flush=True)
    np.savez(out / "complete_arrays.npz", **arrays)
    changed = [path for path, sha in hashes.items() if digest(ROOT / path) != sha]
    if changed:
        raise ValueError("actual source changed: " + str(changed))
    save(
        "result.json",
        dict(
            status="pass",
            variants=records,
            independent_gates=sum(r["independent_gates"] for r in records),
            arrays=len(arrays),
            arrays_sha256=digest(out / "complete_arrays.npz"),
            sources_match=True,
            highest_level_reached=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--davidson", action="store_true")
    main(parser.parse_args())
