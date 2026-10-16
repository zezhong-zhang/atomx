"""Four-core-electron full signed CCSDT projected equations, without FCI fitting."""

import argparse
import os
import resource
import time
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402 -- limit BLAS threads before importing NumPy
from audit import (  # noqa: E402
    BACKEND,
    CASE,
    REPO,
    archive_scientific_inputs,
    audit_case,
    dump,
    sha,
    solve,
    source_archive,
)
from native import build  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--saved-n10",
        action="store_true",
        help="Earlier zero-Fov 10-spinor truncation control",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = (
        [CASE / n for n in ("integrals.json", "ATOMX_ONE", "ATOMX_TWO")]
        if args.saved_n10
        else []
    )
    binding = source_archive(
        output,
        [
            Path(__file__),
            Path(__file__).with_name("native.py"),
            REPO / "tests/test_fscc_scope_projection.py",
            *inputs,
        ],
    )
    n, nc, na = (10, 210, 252) if args.saved_n10 else (12, 495, 792)
    contract = dict(
        hypothesis="Optional Fov-T2 repair extends beyond the exact two-core anchor to retained four-core CCSDT projections.",
        fixed=f"{n} complete native magnetic spinors, fixed two-core Z4 ns30 ks7 VN orbitals; 1s2 2s2 reference. Complete 2p virtuals plus 3s doublet in n12; symmetric two-vertex finite-frequency transverse H fixed before solve. {nc} core / {na} attachment determinants and all {na} x {n - 4} signed residual entries.",
        gates=dict(
            projected_equation_Ha=1e-11,
            signed_heff_projection_Ha=1e-10,
            normalization=1e-12,
            fci_comparison="reported, never an automatic required equality at truncated CCSDT",
        ),
        iteration_control="Jacobi update 1e-13, raw denominator bound <=1e-11 Ha; full independent projected gates unchanged. Earlier 1e-14 numerical-floor failure preserved.",
        resources=dict(
            threads=1,
            address_cap_bytes=4 * 1024**3,
            backend_memory_bytes=512 * 1024**2,
            timeout_seconds=120,
            rank6_dense_bytes=16 * n**6,
            complete_matrix_bytes=16 * (nc**2 + na**2),
        ),
        stop="One complete declared solve; preserve failed projected equations and omitted-rank residuals; do not change H, threshold or production backend.",
        source_and_inputs=binding,
        load=os.getloadavg(),
    )
    dump(output / "contract.json", contract)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.perf_counter()
    try:
        from atomx.methods.fscc import load_integrals
        from atomx.physics.csf.spinor_integrals import SpinorIntegrals

        if args.saved_n10:
            old = load_integrals(CASE)
            fock = old.one_body_Ha.copy()
            for p in range(10):
                for q in range(10):
                    fock[p, q] += sum(old.eri(p, i, q, i) for i in range(4))
            fock = (fock + fock.conj().T) / 2
            energy = (
                old.frozen_energy_Ha
                + np.trace(old.one_body_Ha[:4, :4])
                + 0.5 * sum(old.eri(i, j, i, j) for i in range(4) for j in range(4))
            )
            a = SpinorIntegrals(
                old.labels,
                4,
                old.one_body_Ha,
                fock,
                old.packed_eri_Ha,
                old.frozen_energy_Ha,
                float(energy.real),
                dict(
                    old.identity,
                    reference_change="1s2 2s2; original two-core orbitals; no reoptimization",
                ),
            )
        else:
            _, orbitals, a = build()
            np.savez_compressed(
                output / "native_orbitals.npz",
                **{
                    f"{part}_{key[0]}_{key[1]}": array
                    for key, pair in orbitals.items()
                    for part, array in zip(("P", "Q"), pair, strict=True)
                },
            )
        backend = solve(a, output / "fscc")
        metrics, arrays = audit_case(output / "fscc")
        metrics["backend_raw_equation_bound_Ha"] = backend.equation_residual_bound_Ha
        gates = dict(
            core_retained_equations=metrics["core_projected_rank123_max_Ha"] <= 1e-11,
            attachment_retained_equations=metrics["attachment_projected_rank012_max_Ha"]
            <= 1e-11,
            signed_heff_projection=metrics["heff_signed_projection_max_Ha"] <= 1e-10,
            intermediate_normalization=metrics["intermediate_normalization_max"]
            <= 1e-12,
            complete_model_metric=metrics["physical_metric_min_eigenvalue"] > 0,
            nonzero_fov_stress=metrics["fock_occupied_virtual_max_Ha"] > 1e-8
            if not args.saved_n10
            else True,
            source_inputs_unchanged=all(sha(REPO / p) == s for p, s in binding.items()),
        )
        np.savez_compressed(output / "full_arrays.npz", **arrays)
        result = dict(
            metrics,
            gates=gates,
            highest_level_reached=False,
            status="finite_projected_scope_pass"
            if all(gates.values())
            else "unclosed_larger_core_equation_gap",
            card_summary=f"4-core CCSDT: retained core {metrics['core_projected_rank123_max_Ha']:.3g} Ha; attachment {metrics['attachment_projected_rank012_max_Ha']:.3g} Ha; complete FCI difference separately {metrics['attachment_fci_method_error_max_Ha']:.3g} Ha.",
            source_and_inputs=binding,
            executable_sha256=sha(BACKEND),
            actual_inputs={
                str(p.relative_to(output)): sha(p)
                for p in (output / "fscc").iterdir()
                if p.is_file()
            },
            arrays_sha256=sha(output / "full_arrays.npz"),
            archive_sha256=sha(output / "source_inputs.tar.gz"),
            inputs_archive_sha256=archive_scientific_inputs(output),
            wall_seconds=time.perf_counter() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
    except Exception as exc:
        result = dict(
            status="unclosed_scope_execution_failure",
            error=type(exc).__name__,
            message=str(exc),
            gates=dict(execution=False),
            highest_level_reached=False,
            card_summary="4-core signed CCSDT probe failed; raw failure preserved.",
            source_and_inputs=binding,
        )
    dump(output / "result.json", result)
    print(result["status"], result["gates"])


if __name__ == "__main__":
    main()
