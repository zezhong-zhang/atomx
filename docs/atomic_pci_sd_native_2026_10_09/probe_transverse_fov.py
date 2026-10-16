"""Declared Fov-only Hamiltonian intervention; never a production repair."""

import hashlib
import json
import os
import resource
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
from test_transverse_spinor_integrals import determinant_matrix  # noqa: E402

from atomx.methods.fscc import compute, load_integrals, resource_estimate  # noqa: E402
from atomx.physics.csf.spinor_integrals import SpinorIntegrals  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    owner = Path(__file__).resolve().parent
    run = owner / sys.argv[1]
    run.mkdir()
    original = (
        owner
        / "transverse_integrals004/test_tmp/test_live_finite_frequency_fsc0/fsccsdt"
    )
    backend = owner / "expt_compact_public_source001-build/expt.x"
    a = load_integrals(original)
    source_files = sorted((REPO / "src/atomx").rglob("*.py")) + [
        Path(__file__),
        REPO / "tests/test_transverse_spinor_integrals.py",
    ]
    sources = {str(p.relative_to(REPO)): sha(p) for p in source_files}
    contract = {
        "hypothesis": "The persistent 2s-only three-electron defect is associated with nonzero Fov introduced by transverse exchange. Zeroing and doubling only Fov, with consistent one-body reconstruction and complete new-H FCI, separates this from a core constant or iteration tolerance error.",
        "fixed": "Complete signed 10spinor/120det/8modelroot tensors; original ERI, labels, reference, nucleus/spinors, real backend bytes and tight 1e-14 iteration. Changed-H factor controls are diagnostics, not physical predictions or a recommended canonicalization.",
        "factors": [0.0, 2.0],
        "gate_Ha": 1e-10,
        "stop": "Two declared full solves only, each 120s and 512MiB backend; 2GiB process address cap. No fit, physical production adoption or reference source edits.",
        "estimate": resource_estimate(a, model="CCSDT", sector="0h1p", active_count=8),
        "source_sha256": sources,
        "backend_sha256": sha(backend),
        "input_sha256": {p.name: sha(p) for p in original.iterdir() if p.is_file()},
        "load": os.getloadavg(),
    }
    (run / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    with tarfile.open(run / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in source_files:
            archive.add(p, arcname=str(p.relative_to(REPO)))
    if contract["estimate"]["estimated_working_bytes"] > 512 * 1024**2:
        raise MemoryError("complete tensor estimate exceeds cap")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    rows, arrays = [], {}
    try:
        for factor in contract["factors"]:
            delta = np.zeros_like(a.fock_Ha)
            delta[:2, 2:] = (factor - 1) * a.fock_Ha[:2, 2:]
            delta[2:, :2] = delta[:2, 2:].conj().T
            new = SpinorIntegrals(
                a.labels,
                a.noccupied,
                a.one_body_Ha + delta,
                a.fock_Ha + delta,
                a.packed_eri_Ha,
                a.frozen_energy_Ha,
                a.reference_energy_Ha,
                dict(a.identity, diagnosis_only_Fov_scale=factor),
            )
            _, full = determinant_matrix(new, 3)
            exact = np.linalg.eigvalsh(full)[:8]
            result = compute(
                new,
                executable=backend,
                directory=run / f"factor{factor:g}",
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
            defects = np.asarray(result.sector_energies_Ha) - exact
            arrays[f"factor{factor:g}_full_H_Ha"] = full
            arrays[f"factor{factor:g}_defects_Ha"] = defects
            rows.append(
                {
                    "factor": factor,
                    "complete_root_defects_Ha": defects.tolist(),
                    "max_Ha": float(np.max(abs(defects))),
                    "stricter_diagnostic_gate_pass": bool(
                        np.max(abs(defects)) <= 1e-10
                    ),
                    "raw_equation_bound_Ha": result.equation_residual_bound_Ha,
                }
            )
        for p, expected in sources.items():
            assert sha(REPO / p) == expected
        for name, expected in contract["input_sha256"].items():
            assert sha(original / name) == expected
        assert sha(backend) == contract["backend_sha256"]
        np.savez_compressed(run / "full_arrays.npz", **arrays)
        data = {
            "status": "Completed declared diagnostic interventions; per-H FCI targets explicit",
            "rows": rows,
            "source_and_inputs_unchanged": True,
            "scope": "Changed-H diagnostic factors only; zero Fov is not adopted as a repair",
        }
        (run / "result.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data))
    except Exception as error:
        (run / "failure.json").write_text(
            json.dumps({"error": str(error)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
