"""Hash every saved input array coordinate without changing retained archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
INPUTS = (
    OWNER / "finite_action001/physical_inputs.npz",
    OWNER / "finite_action001/numerator_inputs.npz",
    ROOT / "docs/atomic_pci_sd_native_2026_10_09/yb_full_run008/yb_full_inputs.npz",
    ROOT / "docs/atomic_pci_sd_native_2026_10_09/valence_term2_full003/term2_6_-1.npz",
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    files = []
    for path in INPUTS:
        arrays = {}
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                value = data[name]
                numeric = value.dtype.kind in "biufc"
                if numeric and not np.all(np.isfinite(value)):
                    raise AssertionError(f"nonfinite physical input: {path}/{name}")
                arrays[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "elements": int(value.size),
                    "numeric_finite_gate": True if numeric else None,
                    "full_F_order_sha256": hashlib.sha256(value.tobytes(order="F")).hexdigest(),
                }
        files.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "arrays": arrays})
    proof = {
        "private_verifier_sha256": digest(Path(__file__)),
        "scope": "Every complete physical input and signed numerator/response coordinate. Whole NPZ identities are also checked against actual frozen input manifests by verify_and_aggregate.py.",
        "files": files,
        "whole_arrays": sum(len(item["arrays"]) for item in files),
    }
    (OWNER / "input_coordinates_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"files": len(files), "whole_arrays": proof["whole_arrays"]}))


if __name__ == "__main__":
    main()
