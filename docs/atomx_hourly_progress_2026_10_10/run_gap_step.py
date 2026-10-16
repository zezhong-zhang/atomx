"""Registered gap diagnoses; scheduling supplies data, never physical formulas.

The partition is an exact finite matrix identity. Whole-interaction controls
explicitly change H and compare each H to its own independent complete FCI.
Neither action fixes missing production diagrams or certifies experiment.
"""

import argparse
import hashlib
import itertools
import json
import os
import resource
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OWNER = REPO / "docs/atomic_pci_sd_native_2026_10_09"
sys.path.insert(0, str(OWNER))
from probe_bloch_oracle import diagnose  # noqa: E402

from atomx.methods.fscc import compute, load_integrals  # noqa: E402
from atomx.physics.csf.spinor_integrals import SpinorIntegrals  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_case(relative, expected):
    directory = (REPO / relative).resolve()
    if not directory.is_relative_to((OWNER / "hourly_iterations").resolve()):
        raise ValueError("gap inputs require the declared finite evidence owner")
    path = directory / "result.json"
    if sha(path) != expected:
        raise ValueError("selection result identity changed before execution")
    result = json.loads(path.read_text())
    if not result["oracle_gate_pass"] or not result["source_and_inputs_unchanged"]:
        raise ValueError(
            "gap action requires validated unchanged source-bound evidence"
        )
    return directory, result


def partition(arrays, nspinors=10):
    """Keep all P and Q rows, complex signs and complete two-core multiplets."""
    if nspinors != 10:
        raise ValueError("declared oracle has exactly ten spinors")
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(nspinors), 3)]
    holes = np.array([2 - (mask & 3).bit_count() for mask in masks])
    selected = np.asarray(arrays["model_determinant_indices"], dtype=int)
    expected = np.flatnonzero(holes == 0)
    if not np.array_equal(selected, expected):
        raise ValueError("model determinant order differs from complete P identity")
    hbar, omega = arrays["hbar3_Ha"], arrays["omega"]
    if hbar.shape != (120, 120) or omega.shape != (120, 8):
        raise ValueError("incomplete finite Hamiltonian or wave operator")
    if not all(np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError("nonfinite scientific arrays")
    energy = float(np.linalg.eigvalsh(arrays["full_h2_Ha"])[0])
    pieces = [hbar[np.ix_(selected, selected)] - energy * np.eye(8)]
    sectors = []
    for rank in (1, 2):
        rows = np.flatnonzero(holes == rank)
        piece = hbar[np.ix_(selected, rows)] @ omega[rows, :]
        pieces.append(piece)
        sectors.append(
            dict(
                core_holes=rank,
                determinant_rows=len(rows),
                contribution_max_Ha=float(np.max(abs(piece))),
                wave_operator_max=float(np.max(abs(omega[rows, :]))),
            )
        )
    closure = float(np.max(abs(sum(pieces) - arrays["exact_heff_Ha"])))
    backend = arrays["exact_heff_Ha"] + arrays["backend_minus_exact_heff_Ha"]
    # This uses the exact Omega, not unexported backend amplitudes. It is an
    # inconsistency diagnosis of the saved target effective operator only.
    target_residual = hbar @ omega - omega @ (backend + energy * np.eye(8))
    residual_by_sector = [
        dict(
            core_holes=rank,
            maximum_Ha=float(np.max(abs(target_residual[holes == rank]))),
        )
        for rank in (0, 1, 2)
    ]
    metrics = dict(
        partition_closure_Ha=closure,
        partition_gate_pass=bool(closure <= 1e-11),
        sector_contributions=sectors,
        exact_omega_with_backend_heff_residual=residual_by_sector,
        backend_heff_max_Ha=float(np.max(abs(arrays["backend_minus_exact_heff_Ha"]))),
        backend_strict_1e10_gate_pass=bool(
            np.max(abs(arrays["backend_minus_exact_heff_Ha"])) <= 1e-10
        ),
        interpretation="Exact signed rank partition closes; residual uses exact Omega with saved backend HEFF. It does not identify the backend amplitude or a missing diagram.",
    )
    payload = dict(
        core_holes_by_row=holes,
        model_determinant_indices=selected,
        model_block_Ha=pieces[0],
        one_core_hole_Ha=pieces[1],
        two_core_hole_Ha=pieces[2],
        exact_omega_backend_heff_residual_Ha=target_residual,
    )
    return metrics, payload


def scaled_interaction(dc, transverse, scale):
    """Scale the complete signed transverse difference, including the Fock ledger."""
    if scale not in (0.5, 1.5):
        raise ValueError("only declared whole-interaction controls are executable")
    if (
        dc.labels != transverse.labels
        or dc.noccupied != transverse.noccupied
        or dc.identity["binding"] != transverse.identity["binding"]
        or not dc.identity["physical_source_sha256"].items()
        <= transverse.identity["physical_source_sha256"].items()
    ):
        raise ValueError(
            "DC and transverse controls require the same full radial/source identity"
        )

    def blend(a, b):
        return a + scale * (b - a)

    return SpinorIntegrals(
        labels=dc.labels,
        noccupied=dc.noccupied,
        one_body_Ha=blend(dc.one_body_Ha, transverse.one_body_Ha),
        fock_Ha=blend(dc.fock_Ha, transverse.fock_Ha),
        packed_eri_Ha=blend(dc.packed_eri_Ha, transverse.packed_eri_Ha),
        frozen_energy_Ha=blend(dc.frozen_energy_Ha, transverse.frozen_energy_Ha),
        reference_energy_Ha=blend(
            dc.reference_energy_Ha, transverse.reference_energy_Ha
        ),
        identity=dict(
            transverse.identity,
            hamiltonian="Declared H_DC + lambda*(H_transverse-H_DC)",
            whole_transverse_scale=scale,
            diagnostic_changed_H=True,
            note="Whole signed operator scaling with fixed source orbitals/frequency prescription. No Fov deletion, fitted correction or production accuracy claim.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job["kind"] not in ("bloch_partition", "whole_transverse_control"):
        raise ValueError("unsupported registered scientific action")
    args.output.mkdir(parents=True, exist_ok=False)
    directory, prior = bound_case(job["source"], job["result_sha256"])
    files = [
        Path(__file__),
        args.job,
        OWNER / "probe_bloch_oracle.py",
        REPO / "tests/test_transverse_spinor_integrals.py",
        REPO / "tests/test_spinor_integrals_fscc.py",
        directory / "result.json",
        directory / "full_arrays.npz",
        *sorted((REPO / "src/atomx").rglob("*.py")),
    ]
    dc_dir = None
    if job["kind"] == "whole_transverse_control":
        dc_dir, _ = bound_case(job["dc_source"], job["dc_result_sha256"])
        for source in (directory, dc_dir):
            source_contract = json.loads((source / "contract.json").read_text())
            for name in ("methods/fscc.py", "physics/csf/spinor_integrals.py"):
                current = (REPO / "src/atomx" / name).resolve()
                if sha(current) != source_contract["sources"][str(current)]:
                    raise ValueError(
                        "public integral decoder source changed from the saved input contract"
                    )
            for name in ("integrals.json", "ATOMX_ONE", "ATOMX_TWO"):
                files.append(source / "fscc" / name)
            files.append(source / "contract.json")
    binding = {str(p.resolve()): sha(p) for p in files}
    contract = dict(
        job=job,
        hypothesis="Full core-hole partition localizes the unchanged effective operator; complete transverse scaling then distinguishes response order from radial error.",
        fixed="Complete ten-spinor, 120-determinant, eight-root retained spaces; original source inputs immutable. Scaling explicitly changes the entire transverse H and uses its own FCI.",
        distinguishing_outcomes="Signed partition must close; whole-H controls measure changes in HEFF defects. A negative target is retained and never a fitted correction.",
        gates=dict(partition_Ha=1e-11, oracle_Ha=1e-11, backend_strict_Ha=1e-10),
        caps=dict(
            address_bytes=4 * 1024**3,
            backend_bytes=512 * 1024**2,
            threads=1,
            seconds=600,
        ),
        estimate=dict(
            determinants=120, roots=8, finite_array_payload_upper_bytes=2 * 1024**2
        ),
        stop="One selected registered action; preserve all signed rows and target failures; no production source or validated backend changes.",
        sources=binding,
        load=os.getloadavg(),
        highest_level_reached=False,
    )
    backend = OWNER / "expt_compact_public_source001-build/expt.x"
    if job["kind"] == "whole_transverse_control":
        contract["backend_sha256"] = sha(backend)
        original_contract = json.loads((directory / "contract.json").read_text())
        if contract["backend_sha256"] != original_contract["backend_sha256"]:
            raise ValueError("validated backend differs from the selected reference")
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    with tarfile.open(args.output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(
                path,
                arcname=str(path.relative_to(REPO))
                if path.is_relative_to(REPO)
                else "input/" + path.name,
            )
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.monotonic()
    try:
        if job["kind"] == "bloch_partition":
            arrays_path = directory / "full_arrays.npz"
            if sha(arrays_path) != job["arrays_sha256"]:
                raise ValueError("selected array identity changed")
            with zipfile.ZipFile(arrays_path) as archive:
                if sum(x.file_size for x in archive.infolist()) > 2 * 1024**2:
                    raise ValueError(
                        "finite arrays exceed predeclared payload estimate"
                    )
            with np.load(arrays_path, allow_pickle=False) as arrays:
                metrics, payload = partition(dict(arrays))
            if not metrics["partition_gate_pass"]:
                raise RuntimeError("complete signed partition failed its fixed gate")
            metrics.update(
                bloch_equation_max_Ha=prior["bloch_equation_max_Ha"],
                oracle_gate_pass=prior["oracle_gate_pass"],
                card_summary=f"完整算符一／二核孔分解闭合 {metrics['partition_closure_Ha']:.3e} Ha；更严后端门仍未通过。下一步检验全横向强度响应；尚未确定缺失图项。",
            )
        else:
            integrals = scaled_interaction(
                load_integrals(dc_dir / "fscc"),
                load_integrals(directory / "fscc"),
                job["scale"],
            )
            backend_hash = contract["backend_sha256"]
            solved = compute(
                integrals,
                executable=backend,
                directory=args.output / "fscc",
                model="CCSDT",
                sector="0h1p",
                active_indices=range(2, 10),
                tolerance=1e-14,
                residual_bound_tolerance_Ha=1e-11,
                memory_bytes=512 * 1024**2,
                tile_size=4,
                threads=1,
                timeout_seconds=120,
            )
            metrics, payload = diagnose(integrals, solved.model_space_matrix_Ha)
            if not metrics["oracle_gate_pass"] or sha(backend) != backend_hash:
                raise RuntimeError(
                    "independent oracle or pinned backend identity failed"
                )
            metrics.update(
                backend_sha256=backend_hash,
                backend_raw_bound_Ha=solved.equation_residual_bound_Ha,
                diagnostic_changed_H=True,
                original_source_defect_Ha=prior["backend_heff_max_Ha"],
                defect_response_ratio=metrics["backend_heff_max_Ha"]
                / prior["backend_heff_max_Ha"],
                card_summary=f"全横向 λ={job['scale']:g}：完整 FCI／CCSDT 差 {metrics['backend_heff_max_Ha']:.3e} Ha；每个改变的 H 使用独立 FCI，未作为生产修复。",
            )
        metrics.update(
            job=job,
            sources_and_inputs_unchanged=all(
                sha(Path(p)) == expected for p, expected in binding.items()
            ),
            highest_level_reached=False,
            observed_wall_seconds=time.monotonic() - start,
            observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            performance_claim="Actual finite one-CPU task observation; no controlled or GPU/end-to-end speedup claim",
        )
        if not metrics["sources_and_inputs_unchanged"]:
            raise RuntimeError("source or input changed during selected task")
        np.savez_compressed(args.output / "full_arrays.npz", **payload)
        metrics["arrays_sha256"] = sha(args.output / "full_arrays.npz")
        (args.output / "result.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error=type(error).__name__, message=str(error)))
        )
        raise


if __name__ == "__main__":
    main()
