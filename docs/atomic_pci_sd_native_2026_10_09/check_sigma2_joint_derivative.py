"""Full signed, source-bound re-solved energy-response gate for Sigma2."""

import hashlib
import json
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check():
    base = OWNER / "sigma2_control003"
    source = read(base / "control_source_sha256.json")
    plan = read(base / "plan.json")
    manifest = read(base / "backend/manifest.json")
    raw = np.load(base / "sigma2_raw.npz")
    magnetic = np.load(base / "sigma2_magnetic.npz")
    derivative = np.load(base / "sigma2_joint_derivative.npz")
    analytic = derivative["native"]
    if analytic.shape != (16,) * 4 or not np.all(np.isfinite(analytic)):
        raise AssertionError("complete finite joint derivative required")
    if np.max(abs(analytic - derivative["oracle"])) > 1e-11:
        raise AssertionError("independent signed analytic derivative failed")
    identities = {}
    comparisons = []
    finite = {}
    for suffix, step in (("1e3", 1e-3), ("3e4", 3e-4), ("1e4", 1e-4)):
        pair = []
        for sign, direction in ((-1, "minus"), (1, "plus")):
            run = OWNER / f"sigma2_control004_{direction}_{suffix}"
            if read(run / "control_source_sha256.json") != source:
                raise AssertionError("source-bound controls must match")
            if read(run / "plan.json") != plan:
                raise AssertionError("complete spaces or payload plan differ")
            current_manifest = read(run / "backend/manifest.json")
            for field in ("pin", "dimensions", "upstream_files_sha256", "files"):
                if current_manifest[field] != manifest[field]:
                    raise AssertionError(f"compiled reference identity differs: {field}")
            current_raw = np.load(run / "sigma2_raw.npz")
            for field in (
                "channels", "core_physical", "core_residual", "direct",
                "core_s", "core_d", "eps",
            ):
                if not np.array_equal(raw[field], current_raw[field]):
                    raise AssertionError(f"fixed physical inputs changed: {field}")
            current_magnetic = np.load(run / "sigma2_magnetic.npz")
            if not np.array_equal(current_magnetic["labels"], magnetic["labels"]):
                raise AssertionError("full magnetic identities differ")
            # Core columns do not shift; all 16 virtual columns must shift together.
            energy_change = (
                current_magnetic["energy_by_column"] - magnetic["energy_by_column"]
            )
            if np.any(energy_change[:2]) or np.max(abs(energy_change[2:] - sign * step)) > 1e-14:
                raise AssertionError("joint energy variable differs from contract")
            columns = read(run / "valence_columns.json")
            if len(columns) != 6 or sum(len(c["magnetic_gates"]) for c in columns) != 16:
                raise AssertionError("complete radial and magnetic columns required")
            if max(c["raw_residual_Ha"] for c in columns) > 1e-10:
                raise AssertionError("re-solved original equation residual failed")
            if read(run / "sigma2_comparison.json")["complete_signed_error_Ha"] > 1e-11:
                raise AssertionError("independent re-solved full values failed")
            identities[run.name] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (run / "sigma2_raw.npz", run / "sigma2_magnetic.npz")
            }
            value = current_magnetic["native"]
            if value.shape != analytic.shape or not np.all(np.isfinite(value)):
                raise AssertionError("complete finite re-solved tensor required")
            pair.append(value)
        difference = (pair[1] - pair[0]) / (2 * step)
        error = abs(difference - analytic)
        tolerance = 1e-9 + 1e-6 * abs(analytic)
        comparison = dict(
            step_Ha=step,
            max_signed_error=float(np.max(error)),
            max_error_over_tolerance=float(np.max(error / tolerance)),
            passed=bool(np.all(error <= tolerance)),
        )
        comparisons.append(comparison)
        finite[f"difference_{suffix}"] = difference
        finite[f"error_{suffix}"] = error
    result = dict(
        contract="sigma2_joint_derivative_contract.json",
        full_signed_coordinates=int(analytic.size),
        comparisons=comparisons,
        identities=identities,
        fixed_input_identity="bitwise core, integral, label and compiled-backend equality",
        scope="finite canonical control, total joint energy response; not a QED or production precision claim",
    )
    target = OWNER / "sigma2_joint_resolved_comparison.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        OWNER / "sigma2_joint_resolved_vectors.npz", analytic=analytic, **finite
    )
    print(json.dumps(result, indent=2), flush=True)
    if not all(c["passed"] for c in comparisons):
        raise AssertionError("predeclared full signed re-solved derivative gate failed")


if __name__ == "__main__":
    check()
