"""Read-only complete determinant/HEFF confirmation of retained transverse runs."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
from test_transverse_spinor_integrals import determinant_matrix  # noqa: E402

from atomx.methods.fscc import load_integrals  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    contract = json.loads((run / "saved_confirmation_contract.json").read_text())
    paths = [
        run / "test_tmp" / f"test_live_finite_frequency_fsc{i}" / "fsccsdt"
        for i in (0, 1)
    ]
    inputs = {
        str(p.relative_to(run)): sha(p)
        for directory in paths
        for p in directory.iterdir()
        if p.is_file()
    }
    arrays, rows = {}, []
    for symmetry, path in zip(("C1", "axial"), paths, strict=True):
        integrals = load_integrals(path)
        if len(integrals.labels) != 10 or integrals.noccupied != 2:
            raise ValueError("confirmation requires the original complete 10 spinors")
        masks, full = determinant_matrix(integrals, 3)
        roots = np.linalg.eigvalsh(full)
        result = json.loads((path / "result.json").read_text())
        with np.load(path / "model_space_matrix.npz") as data:
            arrays[symmetry + "_HEFF_Ha"] = data["matrix_Ha"]
            arrays[symmetry + "_active_indices"] = data["global_active_indices"]
        arrays[symmetry + "_FCI_Ha"] = full
        arrays[symmetry + "_FCI_roots_Ha"] = roots
        arrays[symmetry + "_determinant_masks"] = np.asarray(masks)
        defect = float(np.max(abs(roots[:8] - result["sector_energies_Ha"])))
        assert defect <= contract["gates_Ha"]["all_eight_FCI_roots"]
        rows.append(
            {
                "symmetry": symmetry,
                "determinants": len(masks),
                "complete_model_roots": 8,
                "FSCC_minus_FCI_max_Ha": defect,
                "raw_equation_residual_bound_Ha": result["equation_residual_bound_Ha"],
            }
        )
    np.testing.assert_array_equal(
        arrays["C1_active_indices"], arrays["axial_active_indices"]
    )
    fci_defect = float(np.max(abs(arrays["C1_FCI_Ha"] - arrays["axial_FCI_Ha"])))
    heff_defect = float(np.max(abs(arrays["C1_HEFF_Ha"] - arrays["axial_HEFF_Ha"])))
    assert (
        max(fci_defect, heff_defect) <= contract["gates_Ha"]["complete_C1_axial_matrix"]
    )
    for relative, expected in inputs.items():
        assert sha(run / relative) == expected
    for relative, expected in contract["source_sha256"].items():
        assert sha(REPO / relative) == expected
    np.savez_compressed(run / "saved_confirmation_arrays.npz", **arrays)
    record = {
        "status": "pass",
        "rows": rows,
        "complete_signed_FCI_C1_axial_max_Ha": fci_defect,
        "complete_directed_HEFF_C1_axial_max_Ha": heff_defect,
        "input_sha256": inputs,
        "source_unchanged": True,
        "driver_sha256": sha(Path(__file__)),
        "contract_sha256": sha(run / "saved_confirmation_contract.json"),
        "arrays_sha256": sha(run / "saved_confirmation_arrays.npz"),
        "scope": "Read-only finite Hamiltonian confirmation; no new solve or timing",
    }
    (run / "saved_confirmation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "input_sha256"}))


if __name__ == "__main__":
    main()
