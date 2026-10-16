"""Same-Hamiltonian finite FS-CCSDT stopping-condition diagnosis."""

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
    sources = sorted((REPO / "src/atomx").rglob("*.py")) + [
        Path(__file__),
        REPO / "tests/test_transverse_spinor_integrals.py",
    ]
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    inputs = {p.name: sha(p) for p in original.iterdir() if p.is_file()}
    estimate = resource_estimate(a, model="CCSDT", sector="0h1p", active_count=8)
    contract = {
        "hypothesis": "If the 2s-only FCI defect is an amplitude stopping error, tighter raw convergence at unchanged H/space/backend reduces it. A persistent defect rules out this tolerance setting as a sufficient repair, without identifying an omitted diagram.",
        "fixed_conditions": "Same saved real finite transverse Fock/ERI/frozen reference, complete 10spinor/120det/8root space, same exact source-bound backend. No reference shift or root matching change.",
        "controls": {"original_raw_tolerance": 1e-12, "new_raw_tolerance": 1e-14},
        "gates_Ha": {"original_FCI": 1e-9, "stricter_diagnostic_FCI": 1e-10},
        "stop": "One tighter solve only, 120s process timeout, 512MiB backend cap and 2GiB process address cap. Preserve failures and do not scan tolerances or alter H.",
        "resource_estimate": estimate,
        "source_sha256": source_hashes,
        "input_sha256": inputs,
        "actual_executable_sha256": sha(backend),
        "backend_manifest_sha256": sha(backend.parent / "atomx_backend.json"),
        "load": os.getloadavg(),
    }
    (run / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    with tarfile.open(run / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(REPO)))
    if estimate["estimated_working_bytes"] > 512 * 1024**2:
        raise MemoryError("complete tensor estimate exceeds declared cap")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    try:
        _, h2 = determinant_matrix(a, 2)
        _, h3 = determinant_matrix(a, 3)
        exact = np.linalg.eigvalsh(h3)[:8]
        original_result = json.loads((original / "result.json").read_text())
        result = compute(
            a,
            executable=backend,
            directory=run / "tight",
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
        old_defect = np.asarray(original_result["sector_energies_Ha"]) - exact
        new_defect = np.asarray(result.sector_energies_Ha) - exact
        maximum = float(np.max(abs(new_defect)))
        assert maximum <= contract["gates_Ha"]["original_FCI"]
        for relative, expected in source_hashes.items():
            assert sha(REPO / relative) == expected
        for name, expected in inputs.items():
            assert sha(original / name) == expected
        assert sha(backend) == contract["actual_executable_sha256"]
        np.savez_compressed(
            run / "full_arrays.npz",
            complete_two_electron_H_Ha=h2,
            complete_three_electron_H_Ha=h3,
            exact_low_roots_Ha=exact,
            original_defects_Ha=old_defect,
            tighter_defects_Ha=new_defect,
            tighter_model_matrix_Ha=result.model_space_matrix_Ha,
        )
        data = {
            "status": "Original finite gate passes; stricter diagnostic target evaluated separately",
            "original_root_defects_Ha": old_defect.tolist(),
            "tighter_root_defects_Ha": new_defect.tolist(),
            "core_minus_complete_FCI_Ha": float(
                result.core_energy_Ha - np.linalg.eigvalsh(h2)[0]
            ),
            "tighter_raw_update": result.amplitude_update_max,
            "tighter_equation_residual_bound_Ha": result.equation_residual_bound_Ha,
            "original_FCI_gate_pass": maximum <= 1e-9,
            "stricter_diagnostic_FCI_gate_pass": maximum <= 1e-10,
            "source_and_inputs_unchanged": True,
            "arrays_sha256": sha(run / "full_arrays.npz"),
            "highest_level_achieved": False,
            "scope": "Finite unchanged-H stopping-condition intervention; no physical root shift or production precision bound",
        }
        (run / "result.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data))
    except Exception as error:
        (run / "failure.json").write_text(
            json.dumps({"error": type(error).__name__, "message": str(error)}, indent=2)
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
