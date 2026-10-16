"""Alternating one/eight-worker complete Yb term2; retained original oracle."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from atomx.methods.core_sd import CoreSD  # noqa: E402
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run_name, *, value_only_comparison=False):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    original = OWNER / "valence_term2_full003"
    parent = OWNER / "valence_term2_parallel_full006"
    physical = OWNER / "yb_full_run008"
    kernel = ROOT / "src/atomx/methods/_pci_valence_term2_streamed.f90"
    value_kernel = OWNER / "valence_term2_value.f90"
    baseline, candidate = (
        ("response", "value_only") if value_only_comparison else (1, 8)
    )
    contract = dict(
        hypothesis="Disjoint complete core-channel heap workers improve exact term2 wall time; no reduction or omitted intermediate",
        fixed="Original complete Yb68core414virtual/ns123/Fermi/l<=3/<5Ha, complete signed random numerator/response seed314521; one current kernel expanded for one/eight workers; same library/core/radial/angular input and inner-one BLAS",
        reference_case="valence_term2_full003",
        original_reference_sha256=digest(original / "term2_6_-1.npz"),
        physical_input_sha256=digest(physical / "yb_full_inputs.npz"),
        parent_manifest_sha256=digest(parent / "backend/manifest.json"),
        kernel_sha256=digest(kernel),
        signed_gate_Ha=1e-11,
        bitwise_worker_equivalence_required=True,
        cap_bytes=32 * 1024**3,
        warmups=(baseline, candidate),
        alternating_pairs=(
            (baseline, candidate),
            (candidate, baseline),
            (baseline, candidate),
        ),
        max_load_guard=27.0 if value_only_comparison else 18.0,
        required_pair_speedup=1.0,
        timing_scope="warm complete term2 plus C input/output copies; excludes integral preparation, compilation, full residual solve and screening",
        stop="Any input/source/cap/signed mismatch rejects; load failure suppresses controlled timing claim but signed controls still run; no tolerance/space change",
    )
    if value_only_comparison:
        contract.update(
            hypothesis="Skip unconsumed derivative contractions in value-only Krylov actions; full signed values stay bitwise to the original companion",
            fixed="same full original physical/signed input and bound backend; both variants use8 heap workers and inner-one BLAS; derivative input affects only the full companion's derivative output",
            value_kernel_sha256=digest(value_kernel),
            concurrent_workload="unchanged production five-column003, up to8 workers; unrelated9CPU baseline retained",
            guard_reason="32 logical CPUs; threshold27 retains5CPU headroom for the predeclared concurrently active8-worker production task; no after-result guard adjustment",
            timing_scope="warm signed value request: full-response companion versus value-only candidate including each C copy; no complete response or end-to-end speedup claim",
            baseline=baseline,
            candidate=candidate,
        )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    shutil.copy2(__file__, out / "driver.py")
    archive = out / "source"
    identities = {}
    for path in (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + [kernel]
        + ([value_kernel] if value_only_comparison else [])
    ):
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        identities[str(path.relative_to(ROOT))] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(identities, indent=2) + "\n")
    backend = out / "backend"
    shutil.copytree(parent / "backend", backend)
    private = (
        (backend / "valence_term2_streamed.f90")
        .read_text()
        .split("subroutine atomx_val_term2_streamed_map")[0]
    )
    if private != kernel.read_text().replace("CASE_THREADS", "8"):
        raise AssertionError("retained eight-worker kernel differs from public body")
    body = (
        kernel.read_text()
        .replace("CASE_THREADS", "1")
        .replace("valence_term2_streamed", "valence_term2_one")
    )
    body += """
subroutine atomx_val_term2_one_map(energy,d,dd,rd,pd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
  xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
  call valence_term2_one(energy)
  rd=xvo; pd=dxvo
end subroutine
"""
    if value_only_comparison:
        body = (
            value_kernel.read_text()
            .split("subroutine atomx_val_term2_original_map")[0]
            .replace("CASE_THREADS", "8")
        )
    (backend / "valence_term2_one.f90").write_text(body)
    manifest = json.loads((backend / "manifest.json").read_text())
    command = manifest["command"][:]
    compile_command = command[: command.index("-shared")] + [
        "-fopenmp",
        "-c",
        "valence_term2_one.f90",
    ]
    command.insert(command.index("-o"), "valence_term2_one.o")
    with (backend / "paired_compiler.log").open("w") as log:
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
    manifest["paired_term2"] = contract
    manifest["files"] = {
        p.name: digest(p)
        for p in backend.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (backend / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plan = json.loads((physical / "yb_full_plan.json").read_text())
    occupied, virtual = (
        tuple(map(tuple, plan[k])) for k in ("occupied_keys", "virtual_keys")
    )
    bound = json.loads((parent / "resource.json").read_text())["bound_bytes"]
    if bound > contract["cap_bytes"]:
        raise MemoryError("complete retained diagnostic payload exceeds cap")
    (out / "resource.json").write_text(
        json.dumps(
            dict(
                bound_bytes=bound,
                parent_resource_sha256=digest(parent / "resource.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    with np.load(physical / "yb_full_inputs.npz", allow_pickle=False) as data:
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
    with np.load(parent / "orbitals.npz", allow_pickle=False) as data:
        for key, (p, q) in waves.items():
            if not np.array_equal(
                p, data[f"P_{key[0]}_{key[1]}"]
            ) or not np.array_equal(q, data[f"Q_{key[0]}_{key[1]}"]):
                raise AssertionError("complete retained physical waves differ")
    core = CoreSD(
        waves,
        workspace,
        70,
        123,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=make_nuclear_potential(70, "fermi"),
        backend_directory=backend,
        memory_bytes=contract["cap_bytes"],
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
        radial_representation="polynomial",
    )
    if core._blas_runtime["parallel"] != 2 or core._blas_runtime["threads"] != 1:
        raise ValueError("bound OpenMP BLAS must have one inner thread")
    lib = core._backend
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    init = lib.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
    init.restype = None
    key = (6, -1)
    slot = next(g.index(key) for g in core.plan["groups"] if key in g)
    init(
        slot + 1, -1, core.plan["raw_amplitudes"], np.zeros(core.plan["raw_amplitudes"])
    )
    with np.load(original / "term2_6_-1.npz", allow_pickle=False) as data:
        numerator, prime, reference, reference_prime, serial = (
            np.array(data[k], order="F")
            for k in ("numerator", "prime", "original", "original_prime", "streamed")
        )
        serial_prime = np.array(data["streamed_prime"], order="F")
    index = next(i for i, label in enumerate(core.integrals.labels) if label[:2] == key)
    energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
    if (
        energy
        != json.loads((original / "result.json").read_text())["rows"][0]["energy_Ha"]
    ):
        raise AssertionError("complete retained physical energy differs")
    functions = (
        {
            baseline: lib.atomx_val_term2_streamed_map,
            candidate: lib.atomx_val_term2_value_map,
        }
        if value_only_comparison
        else {1: lib.atomx_val_term2_one_map, 8: lib.atomx_val_term2_streamed_map}
    )
    for variant, function in functions.items():
        count = 2 if variant == "value_only" else 4
        function.argtypes = [ctypes.c_double] + [pointer] * count
        function.restype = None
    actual, actual_prime = np.zeros_like(numerator), np.zeros_like(prime)
    observations = []
    sequence = list(contract["warmups"]) + [
        w for pair in contract["alternating_pairs"] for w in pair
    ]
    for ordinal, variant in enumerate(sequence):
        response_requested = variant != "value_only"
        load_before = os.getloadavg()
        started = time.perf_counter()
        if response_requested:
            functions[variant](energy, numerator, prime, actual, actual_prime)
        else:
            functions[variant](energy, numerator, actual)
        seconds = time.perf_counter() - started
        load_after = os.getloadavg()
        errors = [
            float(np.max(abs(a - b)))
            for a, b in (
                ((actual, reference), (actual_prime, reference_prime))
                if response_requested
                else ((actual, reference),)
            )
        ]
        equal = np.array_equal(actual, serial) and (
            not response_requested or np.array_equal(actual_prime, serial_prime)
        )
        record = dict(
            ordinal=ordinal,
            workers=8 if value_only_comparison else variant,
            variant=variant,
            response_requested=response_requested,
            warmup=ordinal < 2,
            seconds=seconds,
            load_before=load_before,
            load_after=load_after,
            signed_errors_Ha=errors,
            bitwise_equal=bool(equal),
        )
        observations.append(record)
        arrays = dict(signed=actual)
        if response_requested:
            arrays["response"] = actual_prime
        np.savez_compressed(out / f"map_{ordinal}_{variant}.npz", **arrays)
        (out / "observations.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
        if (
            max(errors) > contract["signed_gate_Ha"]
            or not equal
            or not all(
                np.all(np.isfinite(a))
                for a in ((actual, actual_prime) if response_requested else (actual,))
            )
        ):
            raise AssertionError(
                "complete original signed/response or worker equivalence gate failed"
            )
    measured = observations[2:]
    ratios = []
    for a, b in zip(measured[::2], measured[1::2], strict=True):
        pair = {a["variant"]: a["seconds"], b["variant"]: b["seconds"]}
        ratios.append(pair[baseline] / pair[candidate])
    guarded = all(
        max(r["load_before"] + r["load_after"]) <= contract["max_load_guard"]
        for r in observations
    )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="complete signed controls passed",
                paired_speedups=ratios,
                median_speedup=float(np.median(ratios)),
                timing_guard_pass=guarded,
                controlled_kernel_improvement=guarded
                and min(ratios) > contract["required_pair_speedup"],
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                scope=contract["timing_scope"],
                end_to_end_claim=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--value-only-comparison", action="store_true")
    arguments = parser.parse_args()
    try:
        main(arguments.run_name, value_only_comparison=arguments.value_only_comparison)
    except BaseException as error:
        out = OWNER / arguments.run_name
        if out.exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
