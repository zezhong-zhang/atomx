"""Independent dense and branch-boundary anchors for the trial polar map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from atomx.numerics.bspline.orbital_optimization import _polar_retraction

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
SOURCE = ROOT / "src/atomx/numerics/bspline/orbital_optimization.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_polar(matrix):
    """Full spectral construction; no polynomial or production Gram used."""
    with mp.workdps(75):
        wide = mp.matrix(matrix.tolist())
        values, vectors = mp.eigsy(wide.T * wide)
        inverse = vectors * mp.diag([1 / mp.sqrt(x) for x in values]) * vectors.T
        return np.asarray((wide * inverse).tolist(), dtype=np.float64)


def old_polar(matrix):
    values, vectors = np.linalg.eigh(matrix.T @ matrix)
    return matrix @ ((vectors / np.sqrt(values)) @ vectors.T)


def main():
    before = sha(SOURCE)
    rng = np.random.default_rng(147)
    base, _ = np.linalg.qr(rng.normal(size=(64, 8)))
    random = rng.normal(size=base.shape)
    mixed = base.T @ random
    tangent = random - base @ ((mixed + mixed.T) / 2)
    tangent /= np.linalg.norm(tangent)
    controls = [("dense_unit_baseline", base)]
    controls += [(f"tangent_{step:.0e}", base + step * tangent)
                 for step in (1e-3, 1e-8, 1e-13, 1e-16)]
    for defect in (.00999999, .01000001):
        factor = np.ones(8)
        factor[0] = np.sqrt(1 + defect)
        controls.append((f"threshold_{defect:.8f}", base * factor))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    controls.append(("right_orthogonal_rotation", controls[1][1] @ rotation))
    phase = np.ones(8)
    phase[2] = -1
    controls.append(("column_phase_flip", controls[1][1] * phase))
    rows = []
    first_polar = None
    for name, matrix in controls:
        actual = _polar_retraction(matrix)
        reference = independent_polar(matrix)
        legacy = old_polar(matrix)
        identity = np.eye(matrix.shape[1])
        gram_norm = float(np.linalg.norm(matrix.T @ matrix - identity, ord=np.inf))
        oracle_error = float(np.max(abs(actual - reference)))
        gram_error = float(np.max(abs(actual.T @ actual - identity)))
        row = {
            "control": name,
            "input_gram_infinity_norm": gram_norm,
            "expected_branch": "near-unit series" if gram_norm < .01 else "spectral fallback",
            "max_abs_same_map_error": oracle_error,
            "max_abs_gram_error": gram_error,
            "legacy_max_abs_same_map_error": float(np.max(abs(legacy - reference))),
            "actual_minus_legacy_max_abs": float(np.max(abs(actual - legacy))),
            "passed": oracle_error < 1.5e-15 and gram_error < 4e-15,
        }
        if name == "tangent_1e-03":
            first_polar = actual
        if name == "right_orthogonal_rotation":
            error = float(np.max(abs(actual - first_polar @ rotation)))
            row["equivariance_error"] = error
            row["passed"] = row["passed"] and error < 1.5e-15
        if name == "column_phase_flip":
            error = float(np.max(abs(actual - first_polar * phase)))
            row["phase_error"] = error
            row["passed"] = row["passed"] and error < 1.5e-15
        rows.append(row)
    try:
        _polar_retraction(np.zeros((64, 8)))
        rank_rejected = False
    except ValueError:
        rank_rejected = True
    source = SOURCE.read_text()
    seed_unchanged = ("gram_values, gram_vectors = np.linalg.eigh(y.T @ y)" in source
                      and "y = y @ ((gram_vectors / np.sqrt(gram_values)) @ gram_vectors.T)" in source
                      and source.count("_polar_retraction(y)") == 2)
    report = {
        "contract_sha256": sha(OWNER / "polar_review_contract.json"),
        "script_sha256": sha(Path(__file__)),
        "source_sha256": before,
        "source_unchanged_during_run": before == sha(SOURCE),
        "input_seed": 147,
        "precision_decimal_digits": 75,
        "long_double_mantissa_bits_on_host": int(np.finfo(np.longdouble).nmant),
        "rows": rows,
        "rank_deficient_rejected": rank_rejected,
        "original_seed_spectral_polar_block_retained": seed_unchanged,
        "passed": before == sha(SOURCE) and rank_rejected and seed_unchanged
                  and all(row["passed"] for row in rows),
        "limitations": "No new native Yb roots or physical improvement certified; preconditioner/metric fingerprints must still match in production continuation.",
    }
    (OWNER / "polar_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
