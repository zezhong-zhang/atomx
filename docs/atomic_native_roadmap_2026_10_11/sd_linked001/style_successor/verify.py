"""Saved-array replay with independent mask-derived cluster generators."""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    if os.environ.get(name) != "1":
        raise RuntimeError("one-thread verification requires explicit environment")
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))

import itertools  # noqa: E402

import numpy as np  # noqa: E402

from atomx.methods.linearized_sd import solve  # noqa: E402

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def op(mask, operations):
    sign = 1
    for index, create in reversed(operations):
        bit = 1 << index
        if bool(mask & bit) == create:
            return None, 0
        if (mask & (bit-1)).bit_count() % 2:
            sign = -sign
        mask ^= bit
    return mask, sign


def replay():
    contract = json.loads((OWNER / "contract.json").read_text())
    result = json.loads((OWNER / "result.json").read_text())
    if result["status"] != "passed" or result["contract_sha256"] != digest(OWNER / "contract.json"):
        raise AssertionError("original executed contract/result not accepted")
    if digest(OWNER / "arrays.npz") != result["arrays_sha256"]:
        raise AssertionError("actual saved arrays changed")
    original_input = ROOT / contract["input_path"]
    for path in (original_input, OWNER / "inputs.npz"):
        if digest(path) != contract["input_sha256"]:
            raise AssertionError("fixed original/archived input changed")
    for path, expected in contract["sources"].items():
        for actual in (ROOT / path, OWNER / "source" / path):
            if digest(actual) != expected:
                raise AssertionError(f"source identity failed {actual}")
    arrays = np.load(OWNER / "arrays.npz", allow_pickle=False)
    masks = tuple(sum(1 << i for i in row) for row in itertools.combinations(range(12), 4))
    lookup = {mask: row for row, mask in enumerate(masks)}
    ref = 15
    cases = {}
    for mode in contract["modes"]:
        h = arrays[f"{mode}_H_raw"]
        c = arrays[f"{mode}_scalar_vectors"]
        source = c[:, 1:].conj().T @ h @ c[:, 0]
        href = h @ c[:, 0]
        columns = []
        for coordinate in range(1, 13):
            # Derive each generator from its signed determinant vector, not
            # from the pair-stencil construction used in the original probe.
            generators = []
            for row, mask in enumerate(masks):
                coefficient = c[row, coordinate]
                if coefficient != 0:
                    holes = tuple(i for i in range(12) if ref & ~mask & (1 << i))
                    particles = tuple(i for i in range(12) if mask & ~ref & (1 << i))
                    operations = tuple((i, True) for i in particles) + tuple((i, False) for i in reversed(holes))
                    target, phase = op(ref, operations)
                    if target != mask or not phase:
                        raise AssertionError("reference-defined generator invalid")
                    generators.append((operations, coefficient / phase))
            thref = np.zeros(495, dtype=np.complex128)
            for operations, coefficient in generators:
                for row, mask in enumerate(masks):
                    if href[row] != 0:
                        target, sign = op(mask, operations)
                        if sign:
                            thref[lookup[target]] += coefficient * sign * href[row]
            columns.append(c[:, 1:].conj().T @ (h @ c[:, coordinate] - thref))
        k = np.column_stack(columns)
        amplitudes = arrays[f"{mode}_amplitudes"]
        projected = c.real.T @ ((h.real + h.real.T)/2) @ c.real
        projected = (projected + projected.T)/2
        vectors = tuple(tuple((mask, float(c[row, j].real)) for row, mask in enumerate(masks) if c[row, j] != 0)
                        for j in range(13))
        solved = solve(projected, vectors, ref, reference_identity=f"saved-array-native-{mode}",
                       residual_tolerance_Ha=1e-12)
        checks = dict(
            mask_derived_K_vs_saved_oracle_Ha=float(np.max(abs(k - arrays[f"{mode}_K"]))),
            mask_derived_K_vs_actual_native_Ha=float(np.max(abs(k - arrays[f"{mode}_shared_K"]))),
            replay_source_vs_actual_source_Ha=float(np.max(abs(source - arrays[f"{mode}_source"]))),
            replay_root_raw_Ha=float(np.max(abs(source + k @ amplitudes))),
            fresh_native_root_raw_Ha=solved.equation_residual_Ha,
            fresh_native_amplitudes_vs_actual_max=float(np.max(abs(solved.amplitudes-amplitudes))),
            actual_nonzero_Fov_T2_max_Ha=float(np.max(abs(arrays[f"{mode}_Fov_T2"]))),
            actual_missing_offdiagonal_after_Fock_PP_HH_Ha=float(np.max(abs(
                arrays[f"{mode}_missing_after_Fock_PP_HH"] - np.diag(np.diag(arrays[f"{mode}_missing_after_Fock_PP_HH"]))))))
        for name, value in checks.items():
            if name.startswith("actual_"):
                if value <= 1e-11:
                    raise AssertionError("requested discriminator accidentally zero")
            else:
                limit = 1e-12 if "root_raw" in name else 1e-11
                if not math.isfinite(value) or value > limit:
                    raise AssertionError(f"original replay gate {name}: {value}")
        cases[mode] = checks
    return dict(status="passed", cases=cases, source_files=len(contract["sources"]),
                source_snapshots_verified=True, actual_original_and_archived_input_verified=True,
                result_sha256=digest(OWNER / "result.json"), arrays_sha256=digest(OWNER / "arrays.npz"),
                verifier_sha256=digest(Path(__file__)),
                scope="Independent mask-derived generator replay and fresh public finite native solve; no external production solver.")


if __name__ == "__main__":
    result = replay()
    (OWNER / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
