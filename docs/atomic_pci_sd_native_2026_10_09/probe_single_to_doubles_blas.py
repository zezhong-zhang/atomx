"""Full signed original-action gates for prototype or public term31/32 BLAS."""

import argparse
import ctypes
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from build_pci_sd_backend import build  # noqa: E402
from build_single_to_doubles_control import build_control  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def main(args):
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    source = OWNER / "yb_full_run008"
    finite = OWNER / "yb_full_core_equation001"
    full = json.loads((source / "yb_full_plan.json").read_text())
    selected = (
        full
        if args.space == "full"
        else json.loads((finite / "plan_4.json").read_text())
    )
    occupied, virtual = (
        tuple(map(tuple, selected[k])) for k in ("occupied_keys", "virtual_keys")
    )
    cap = (32 if args.space == "full" else 4) * 1024**3
    native = source / "yb_full_inputs.npz"
    original = (
        OWNER / "yb_full_run007/yb_full_first_action.npz"
        if args.space == "full"
        else finite / "signed_actions.npz"
    )
    contract = dict(
        hypothesis="Weighted native BLAS term31/32 preserve every signed core SD output",
        fixed="All68core Fermi/V68/ns123, original scalar angular channels and full declared radial pool",
        space=args.space,
        virtual_radial_count=len(virtual),
        workers=args.workers,
        public_kernel=args.public_kernel,
        native_input_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        original_action_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        signed_action_gate_Ha=1e-11,
        raw_residual_gate_Ha=1e-10,
        core_energy_gate_Ha=1e-11,
        cap_bytes=cap,
        load=os.getloadavg(),
        stop="Any full signed action failure rejects kernel; no selected coordinates or physics changes",
        scope="exact equation optimization gate; no space convergence/experiment/timing claim",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    archive = out / "source"
    names = tuple(
        OWNER / n
        for n in (
            Path(__file__).name,
            "build_single_to_doubles_control.py",
            "core_single_to_doubles_blas.f90",
        )
    ) + tuple(
        ROOT / n
        for n in (
            "tools/build_pci_sd_backend.py",
            "src/atomx/methods/core_sd.py",
            "src/atomx/methods/_pci_core_singles_blas.f90",
            "src/atomx/methods/_pci_term21_streamed.f90",
            "src/atomx/methods/_pci_core_blas.f90",
            "src/atomx/methods/_pci_core_single_to_doubles_blas.f90",
            "src/atomx/numerics/bspline/coulomb_features.py",
        )
    )
    hashes = {}
    for p in names:
        q = archive / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with np.load(native, allow_pickle=False) as data:
        workspace = {
            k.removeprefix("workspace_"): data[k].copy()
            for k in data
            if k.startswith("workspace_")
        }
        workspace = {k: v.item() if v.ndim == 0 else v for k, v in workspace.items()}
        waves = {
            key: (
                data[f"P_{key[0]}_{key[1]}"].copy(),
                data[f"Q_{key[0]}_{key[1]}"].copy(),
            )
            for key in occupied + virtual
        }
        nuclear = make_nuclear_potential(70, "fermi")
        if (
            np.max(
                abs(
                    np.vectorize(nuclear, otypes=[float])(workspace["gr"])
                    - data["nuclear_on_outer_gauss"]
                )
            )
            > 1e-11
        ):
            raise ValueError("saved Fermi operator mismatch")
    features = PolynomialCellCoulombFeatures(workspace)
    plan = plan_core_sd(
        occupied,
        virtual,
        features.identity,
        term21_threads=args.workers,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=args.public_kernel,
    )
    d = plan["dimensions"]
    prototype_scratch = (
        8
        * d["NHF"]
        * (d["NXX"] * (d["NK"] + 1 + args.workers) + d["NN"] + 2 * args.workers)
        + 16 * d["NV"] * args.workers
    )
    extra = 0 if args.public_kernel else prototype_scratch
    if (
        plan["complete_payload_bound_bytes"] + extra + 40 * plan["krylov_vector_bytes"]
        > cap
    ):
        raise MemoryError(
            "complete unchanged model plus all kernel/Krylov scratch exceeds cap"
        )
    (out / "resource.json").write_text(
        json.dumps(dict(plan=plan, extra_kernel_bytes=extra), indent=2) + "\n"
    )
    builder = build if args.public_kernel else build_control
    options = {"core_single_to_doubles_blas": True} if args.public_kernel else {}
    builder(
        REF / "pCI",
        out / "backend",
        d,
        REF / "gfortran",
        runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
        blas_library=OWNER
        / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
        term21_threads=args.workers,
        term21_mode="streamed",
        core_singles_blas=True,
        **options,
    )
    before = time.perf_counter()
    problem = CoreSD(
        waves,
        workspace,
        70,
        workspace["ns"],
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=nuclear,
        backend_directory=out / "backend",
        term21_threads=args.workers,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=args.public_kernel,
        memory_bytes=cap,
        radial_representation="polynomial",
    )
    preparing = time.perf_counter() - before
    problem.set_diagram_progress(True)
    with np.load(original, allow_pickle=False) as old:
        controls = (
            {"initial": (old["amplitudes"].copy(), old["residual"] + old["source"])}
            if args.space == "full"
            else {
                name: (old[name].copy(), old[f"{name}_0"].copy())
                for name in ("initial", "random")
            }
        )
        if args.space == "full" and (
            np.max(abs(problem.source - old["source"])) > 1e-11
            or not np.array_equal(problem.denominators, old["denominators"])
        ):
            raise AssertionError("full original source or denominator mismatch")
    records = []
    for name, (vector, expected) in controls.items():
        start = time.perf_counter()
        actual = problem.action(vector)
        maximum = float(np.max(abs(actual - expected)))
        np.savez_compressed(
            out / f"{name}_signed.npz", vector=vector, original=expected, actual=actual
        )
        records.append(
            dict(
                name=name, action_error_Ha=maximum, seconds=time.perf_counter() - start
            )
        )
        print(json.dumps(records[-1]), flush=True)
        if maximum > contract["signed_action_gate_Ha"]:
            raise AssertionError("complete original signed action failed")
    # In the full initial vector the DHF single source may be tiny. Stimulate
    # every allowed single with a signed random vector and directly compare
    # both full native diagrams, including all padding, in the SAME library.
    size = d["NXX"] ** 2 * d["NCH"]
    array_size = size + d["NXX"] * d["NN"]
    live_in = (ctypes.c_double * array_size).in_dll(problem._backend, "rhocin_")
    live_out = (ctypes.c_double * array_size).in_dll(problem._backend, "rhocout_")
    inside, outside = np.ctypeslib.as_array(live_in), np.ctypeslib.as_array(live_out)
    singles = inside[size:].reshape((d["NXX"], d["NN"]), order="F")
    singles[:] = 0
    rng = np.random.default_rng(835173)
    for i, key in enumerate(occupied):
        partial = next(p for p, group in enumerate(plan["groups"]) if key in group)
        for row, target in enumerate(plan["groups"][partial]):
            if target in virtual:
                singles[row, i] = rng.normal() * 1e-3
    for which in (31, 32):
        original_term = getattr(problem._backend, f"term{which}_")
        original_term.argtypes = []
        original_term.restype = None
        outside[:] = 0
        original_term()
        expected = outside.copy()
        optimized_term = problem._backend.core_single_to_doubles_blas_
        optimized_term.argtypes = [ctypes.POINTER(ctypes.c_int)]
        optimized_term.restype = None
        outside[:] = 0
        optimized_term(ctypes.byref(ctypes.c_int(which)))
        maximum = float(np.max(abs(outside - expected)))
        np.savez_compressed(
            out / f"term{which}_stimulated.npz",
            single_input=singles,
            original=expected,
            optimized=outside,
        )
        record = dict(
            name=f"term{which}_stimulated",
            action_error_Ha=maximum,
            complete_common_coordinates=array_size,
            input_single_max=float(np.max(abs(singles))),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        if maximum > contract["signed_action_gate_Ha"]:
            raise AssertionError("complete stimulated single-to-double diagram failed")
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="pass",
                records=records,
                prepare_seconds=preparing,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                provenance=problem.provenance,
                physical_space_converged=False,
                experiment_validated=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--space", choices=("finite", "full"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--public-kernel", action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        out = OWNER / args.run_name
        if out.exists():
            (out / "failure.json").write_text(
                json.dumps(dict(type=type(exc).__name__, message=str(exc)), indent=2)
                + "\n"
            )
        raise
