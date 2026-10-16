"""Private resident-vector arithmetic gate on the complete retained64x64 operator."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator

OWNER = Path(__file__).resolve().parent
sys.path.insert(0, str(OWNER / "cuda_python_runtime"))
import cupy as cp  # noqa: E402
from cuda_krylov_control import CUDAKrylovControl  # noqa: E402


def main(out):
    out.mkdir(exist_ok=False)
    parent = OWNER / "valence_recycling_invariants026"
    paths = [
        Path(__file__),
        OWNER / "cuda_krylov_control.py",
        OWNER / "reorthogonal_gcrot.py",
        OWNER / "scipy_private_gcrot_LICENSE.txt",
        parent / "complete_operator.npz",
        parent / "native.npz",
    ]
    archive = out / "source"
    archive.mkdir()
    identities = {}
    for path in paths:
        name = path.name
        shutil.copy2(path, archive / name)
        identities[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "contract.json").write_text(
        json.dumps(
            dict(
                hypothesis="Resident full FP64 vectors and pivoted two-pass image QR reproduce a complete physical64-coordinate operator inverse within original signed/raw1e-11 gates",
                fixed="full native026 matrix/source and retained native-root response rhs, same m30/k10/maxiter100/three refinement budgets; no changed physical map or vector screening",
                resource="64MiB complete host reference,8GiB owned device pool, explicit1024-byte matrix bridge reserve",
                scope="complete finite matrix arithmetic only, not the root-dependent full SD/independent magnetic proof or controlled timing",
                stop="any resource, complete signed or raw failure rejects; no fallback/relaxed gate",
                source_sha256=identities,
            ),
            indent=2,
        )
        + "\n"
    )
    with np.load(parent / "complete_operator.npz", allow_pickle=False) as data:
        matrix, source = data["matrix"], data["source"]
    with np.load(parent / "native.npz", allow_pickle=False) as data:
        rhs = data["rhs"]
    if matrix.shape != (64, 64) or matrix.nbytes * 8 > 64 * 1024**2:
        raise MemoryError("complete finite declared operator/cap differs")
    before = cp.cuda.get_allocator()
    limit = cp.get_default_memory_pool().get_limit()
    solver = CUDAKrylovControl(cp, out / "runtime", operator_device_bound_bytes=1024)
    operator = LinearOperator(
        matrix.shape, matvec=lambda vector: matrix @ vector, dtype=float
    )
    recycled = []

    def inverse(b):
        options = dict(
            m=30,
            k=10,
            maxiter=100,
            atol=1e-14,
            rtol=0,
            CU=recycled,
            discard_C=True,
            truncate="oldest",
        )
        result, info = solver.solve(operator, b, **options)
        for _ in range(3):
            defect = b - matrix @ result
            if np.max(abs(defect)) <= 1e-14:
                break
            delta, info = solver.solve(operator, defect, **dict(options, atol=1e-15))
            result += delta
        return result, info

    root, root_info = inverse(source)
    prime, prime_info = inverse(rhs)
    record = dict(
        root_raw_Ha=float(np.max(abs(matrix @ root - source))),
        prime_raw_Ha=float(np.max(abs(matrix @ prime - rhs))),
        signed_root_Ha=float(np.max(abs(root - np.linalg.solve(matrix, source)))),
        signed_prime_Ha=float(np.max(abs(prime - np.linalg.solve(matrix, rhs)))),
        root_info=root_info,
        prime_info=prime_info,
        calls=solver.calls,
        allocator_restored=cp.cuda.get_allocator() is before,
        default_pool_limit_unchanged=cp.get_default_memory_pool().get_limit() == limit,
    )
    reject = CUDAKrylovControl(
        cp,
        out / "resource_rejection",
        operator_device_bound_bytes=1024,
        memory_bytes=8,
    )
    options = dict(
        m=30,
        k=10,
        maxiter=100,
        atol=1e-14,
        rtol=0,
        CU=[],
        discard_C=True,
        truncate="oldest",
    )
    try:
        reject.solve(operator, source, **options)
    except MemoryError:
        record["resource_reject_before_allocation"] = reject.pool.total_bytes() == 0
    else:
        raise AssertionError("declared undersized resident pool did not reject")

    def failing_callback(vector):
        if vector.shape != (64,) or vector.dtype != np.float64:
            raise AssertionError("incomplete callback vector")
        raise RuntimeError("retained complete callback rejection")

    try:
        solver.solve(operator, source, **dict(options, callback=failing_callback))
    except RuntimeError as error:
        if str(error) != "retained complete callback rejection":
            raise
        record["callback_reject_restores_allocator"] = cp.cuda.get_allocator() is before
        record["callback_reject_preserves_default_pool"] = (
            cp.get_default_memory_pool().get_limit() == limit
        )
    else:
        raise AssertionError("callback rejection did not propagate")
    np.savez_compressed(
        out / "vectors.npz",
        root=root,
        prime=prime,
        rhs=rhs,
        source=source,
        matrix=matrix,
    )
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if any(
        record[k] > 1e-11 or not np.isfinite(record[k])
        for k in ("root_raw_Ha", "prime_raw_Ha", "signed_root_Ha", "signed_prime_Ha")
    ):
        raise AssertionError("complete resident FP64 signed/raw gate failed")
    if not record["allocator_restored"] or not record["default_pool_limit_unchanged"]:
        raise AssertionError("caller allocator/default-pool state changed")
    if not all(
        record[name]
        for name in (
            "resource_reject_before_allocation",
            "callback_reject_restores_allocator",
            "callback_reject_preserves_default_pool",
        )
    ):
        raise AssertionError("complete resident failure/resource ownership gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if (
        not args.run_name.startswith("cuda_krylov")
        or not args.run_name.replace("_", "").isalnum()
    ):
        parser.error("explicit unique owner-local run name required")
    out = OWNER / args.run_name
    if out.exists():
        parser.error("preserve existing evidence; select a new run name")
    try:
        main(out)
    except BaseException as error:
        if out.exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
