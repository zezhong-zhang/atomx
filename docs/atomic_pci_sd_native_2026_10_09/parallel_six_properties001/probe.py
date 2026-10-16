"""Actual native E1/E2/M1 from complete candidate e^T Omega versus same-H CI."""

# ruff: noqa: E402 -- BLAS thread limits and diagnostic helper path precede imports.

import argparse
import os
import resource
import sys
import time
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

SCOPE = Path(__file__).resolve().parent.parent / "parallel_six_fscc_scope001"
sys.path.insert(0, str(SCOPE))

import numpy as np  # noqa: E402 -- thread limit and diagnostic helper path first
from audit import (  # noqa: E402
    BACKEND,
    REPO,
    archive_scientific_inputs,
    audit_case,
    dump,
    sha,
    solve,
    source_archive,
)
from native import build  # noqa: E402
from operators import determinant_operator, single_particle  # noqa: E402

from atomx.methods.ci_effective import (  # noqa: E402
    FiniteCIModel,
    FiniteCITransitionOperator,
)
from atomx.properties.transitions.fscc_effective import (
    FiniteCorrelatedWave,  # noqa: E402
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path(__file__).with_name("operators.py"),
        SCOPE / "native.py",
        REPO / "tests/test_fscc_effective_properties.py",
    ]
    binding = source_archive(output, sources)
    contract = dict(
        hypothesis="Actual complete FS-CCSDT wave e^T Omega and its full physical metric reproduce every signed native E1/E2/M1 spherical component in the same full-CI invariant subspace.",
        fixed="Fresh native Z4 ns30 ks7 fixed two-core DC orbitals, 1s/2s/2p complete 10 magnetic spinors; fixed symmetric two-vertex dynamic transverse H. Complete 120 x 8 actual wave and 8 x 8 effective matrices; all 3 E1, 5 E2 and 3 M1 components.",
        scientific_gates=dict(
            full_bloch_Ha=1e-11,
            heff_Ha=1e-10,
            full_signed_operator_au=1e-11,
            wave_metric_norm=1e-12,
        ),
        operator_contract="Public angular_eL/angular_mL and radial_eL/radial_mL, explicit long wavelength, length E gauge; magnetic spherical components, not reduced line amplitudes. No rates/lifetimes, finite photon-frequency accuracy or experimental certificate.",
        distinguishing_outcomes=[
            "actual wave metric versus bare P O P",
            "complete signed off-diagonal matrices versus diagonal strengths",
            "native physical operators versus random operator controls",
        ],
        resources=dict(
            threads=1,
            address_bytes=4 * 1024**3,
            backend_memory_bytes=512 * 1024**2,
            timeout_s=120,
            determinants=120,
            dense_rank6_bytes=16 * 10**6,
        ),
        source_and_inputs=binding,
        load=os.getloadavg(),
        stop="One same-H solve and all complete matrix comparisons; failures and input hashes preserved.",
    )
    dump(output / "contract.json", contract)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.perf_counter()
    try:
        core, orbitals, a = build(2, 1)
        radial = {
            f"{part}_{key[0]}_{key[1]}": array
            for key, pair in orbitals.items()
            for part, array in zip(("P", "Q"), pair, strict=True)
        }
        radial.update(
            {f"workspace_{key}": value for key, value in core._bsp_data.items()}
        )
        np.savez_compressed(output / "native_radial_inputs.npz", **radial)
        backend = solve(a, output / "fscc")
        metrics, arrays = audit_case(output / "fscc")
        h, wave, total = (
            arrays["full_attachment_hamiltonian_Ha"],
            arrays["wave_operator"],
            arrays["total_heff_Ha"],
        )
        if np.max(abs(h.imag)) != 0 or not np.array_equal(h.real, h.real.T):
            raise ValueError(
                "public finite CI parent requires exact real symmetric same H"
            )
        parent = FiniteCIModel(
            h.real,
            tuple(int(i) for i in arrays["model_indices"]),
            reference_identity="native Z4 dynamic transverse full magnetic 3e",
            csf_identities=tuple(
                f"magnetic determinant:{m}" for m in arrays["attachment_masks"]
            ),
            radial_binding=(sha(output / "native_radial_inputs.npz"),),
            hamiltonian_terms=("dirac-coulomb", "real-finite-frequency-transverse"),
        )
        actual = FiniteCorrelatedWave(
            parent,
            wave,
            total,
            source_identity=sha(output / "fscc/model_space_matrix.npz"),
            metric=arrays["physical_metric"],
            residual_tolerance_Ha=1e-11,
        )
        exact = parent.bloch_reduction(tuple(range(8)), residual_tolerance_Ha=1e-11)
        comparison, full_operators = {}, {}
        for kind, rank in (("E", 1), ("E", 2), ("M", 1)):
            for q in range(-rank, rank + 1):
                name = f"{kind}{rank}_q{q}"
                one = single_particle(a.labels, orbitals, core._bsp_data, kind, rank, q)
                full = determinant_operator(one, arrays["attachment_masks"])
                operator = FiniteCITransitionOperator(
                    full,
                    parent,
                    parent,
                    operator_identity=f"native long-wavelength {name}; signed magnetic component",
                    unit="au",
                )
                bloch = actual.effective_transition(
                    operator, actual, representation="bloch"
                )
                orthonormal = actual.effective_transition(
                    operator, actual, representation="orthonormal"
                )
                reference_bloch = exact.effective_transition(
                    operator, exact, representation="bloch"
                )
                reference = exact.effective_transition(
                    operator, exact, representation="orthonormal"
                )
                comparison[name] = dict(
                    full_signed_orthonormal_max_au=float(
                        np.max(abs(orthonormal - reference))
                    ),
                    full_signed_bloch_max_au=float(
                        np.max(abs(bloch - reference_bloch))
                    ),
                    correlated_strength_max_au=float(np.max(abs(orthonormal))),
                    bare_model_defect_max_au=float(
                        np.max(
                            abs(
                                orthonormal
                                - full[np.ix_(parent.p_indices, parent.p_indices)]
                            )
                        )
                    ),
                )
                for label, array in (
                    ("one", one),
                    ("full", full),
                    ("bloch", bloch),
                    ("orthonormal", orthonormal),
                    ("ci_reference", reference),
                ):
                    arrays[f"{name}_{label}"] = array
                full_operators[kind, rank, q] = full
        reversal = max(
            float(
                np.max(abs(full_operators[kind, rank, -q] - (-1) ** q * full.conj().T))
            )
            for (kind, rank, q), full in full_operators.items()
        )
        max_property = max(
            max(v["full_signed_orthonormal_max_au"], v["full_signed_bloch_max_au"])
            for v in comparison.values()
        )
        metric_difference = float(np.max(abs(actual.metric - exact.metric)))
        wave_difference = float(np.max(abs(actual.wave_operator - exact.wave_operator)))
        gates = dict(
            full_signed_native_properties=max_property <= 1e-11,
            actual_full_H_wave_equation=actual.intertwining_residual_Ha <= 1e-11,
            metric_adjoint_equation=actual.metric_adjoint_residual_Ha <= 1e-11,
            physical_metric_matches_ci=metric_difference <= 1e-12,
            complete_wave_matches_ci=wave_difference <= 1e-12,
            native_spherical_adjoint=reversal <= 1e-12,
            core_retained_equations=metrics["core_projected_rank123_max_Ha"] <= 1e-11,
            full_signed_heff=metrics["heff_signed_projection_max_Ha"] <= 1e-10,
            all_multipole_families_nonzero=all(
                any(
                    v["correlated_strength_max_au"] > 1e-8
                    for k, v in comparison.items()
                    if k.startswith(kind)
                )
                for kind in ("E1", "E2", "M1")
            ),
            source_inputs_unchanged=all(sha(REPO / p) == s for p, s in binding.items()),
        )
        arrays.update(
            ci_exact_wave=exact.wave_operator,
            ci_exact_metric=exact.metric,
            actual_isometry=actual.orthonormal_wave_operator,
            ci_exact_isometry=exact.orthonormal_wave_operator,
        )
        np.savez_compressed(output / "full_arrays.npz", **arrays)
        result = dict(
            status="finite_native_properties_pass"
            if all(gates.values())
            else "unclosed_native_property_gap",
            gates=gates,
            highest_level_reached=False,
            comparisons=comparison,
            card_summary=f"Same-H complete native E1/E2/M1: 11 signed 8 x 8 matrices, defect {max_property:.3g} au; actual e^T Omega full-H residual {actual.intertwining_residual_Ha:.3g} Ha. Fixed finite orbitals/long wavelength only.",
            physical_metric_max_defect=metric_difference,
            complete_wave_max_defect=wave_difference,
            native_spherical_adjoint_max_au=reversal,
            full_signed_property_max_au=max_property,
            backend_raw_equation_bound_Ha=backend.equation_residual_bound_Ha,
            actual_wave_full_H_residual_Ha=actual.intertwining_residual_Ha,
            source_and_inputs=binding,
            executable_sha256=sha(BACKEND),
            actual_inputs={
                str(p.relative_to(output)): sha(p)
                for p in (output / "fscc").iterdir()
                if p.is_file()
            },
            native_radial_inputs_sha256=sha(output / "native_radial_inputs.npz"),
            arrays_sha256=sha(output / "full_arrays.npz"),
            archive_sha256=sha(output / "source_inputs.tar.gz"),
            inputs_archive_sha256=archive_scientific_inputs(output),
            wall_seconds=time.perf_counter() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
    except Exception as exc:
        result = dict(
            status="unclosed_properties_execution_failure",
            gates=dict(execution=False),
            highest_level_reached=False,
            error=type(exc).__name__,
            message=str(exc),
            source_and_inputs=binding,
            card_summary="Native property probe failed; no normalization/rate accuracy claim.",
        )
    dump(output / "result.json", result)
    print(result["status"], result["gates"])


if __name__ == "__main__":
    main()
