"""Private resident FP64 GCROT vectors with two full orthogonalization passes.

Clone the source-bound arithmetic control into an owned execution namespace;
no SciPy/module monkeypatch. Only small Hessenberg/QR coefficients use CPU
LAPACK. Every complete physical action still calls the declared native operator.
"""

import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import reorthogonal_gcrot as algorithm


class CUDAKrylovControl:
    def __init__(
        self, cp, out, *, operator_device_bound_bytes, memory_bytes=8 * 1024**3
    ):
        self.cp, self.out = cp, Path(out)
        self.out.mkdir(exist_ok=False)
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (operator_device_bound_bytes, memory_bytes)
        ):
            raise ValueError("explicit positive complete device caps required")
        self.cap, self.operator_bound = memory_bytes, operator_device_bound_bytes
        self.device = cp.cuda.Device(0)
        self.pool = cp.cuda.MemoryPool()
        with self.device:
            self.pool.set_limit(size=self.cap)
            cp.cuda.get_current_stream().synchronize()
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.calls = 0
        namespace = dict(algorithm.__dict__)
        namespace.update(
            make_system=self._system,
            get_blas_funcs=self._blas,
            qr=self._qr,
            np=SimpleNamespace(
                **{
                    name: getattr(np, name)
                    for name in ("zeros", "ones", "nan", "finfo", "errstate")
                },
                empty=cp.empty,
                isfinite=lambda x: (
                    cp.isfinite(x) if isinstance(x, cp.ndarray) else np.isfinite(x)
                ),
            ),
        )
        fgmres = FunctionType(
            algorithm._fgmres.__code__,
            namespace,
            argdefs=algorithm._fgmres.__defaults__,
        )
        namespace["_fgmres"] = fgmres
        self.algorithm = FunctionType(
            algorithm.gcrotmk.__code__,
            namespace,
            argdefs=algorithm.gcrotmk.__defaults__,
        )
        self.algorithm.__kwdefaults__ = dict(algorithm.gcrotmk.__kwdefaults__)
        self.identity = dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cpu_algorithm_sha256=hashlib.sha256(
                Path(algorithm.__file__).read_bytes()
            ).hexdigest(),
            cupy_version=cp.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            own_pool_cap_bytes=self.cap,
            operator_device_bound_bytes=self.operator_bound,
            dtype="float64 only",
            physical_actions="complete native operator through explicit full-vector copies",
            coefficient_scope="small Hessenberg/QR/least-squares CPU only; complete vectors and recycled image QR remain on-device",
            public_adopted=False,
        )
        (self.out / "contract.json").write_text(
            json.dumps(self.identity, indent=2) + "\n"
        )

    def _blas(self, names, arrays):
        cp = self.cp
        if any(not isinstance(a, cp.ndarray) or a.dtype != cp.float64 for a in arrays):
            raise ValueError("complete resident FP64 BLAS vectors required")

        def axpy(x, y, n=None, a=1.0):
            if n is not None and n != x.size:
                raise ValueError("complete vector AXPY required")
            y += a * x
            return y

        def scal(a, x):
            x *= a
            return x

        functions = dict(
            axpy=axpy,
            scal=scal,
            dot=lambda x, y: float(cp.dot(x, y).item()),
            nrm2=lambda x: float(cp.linalg.norm(x).item()),
        )
        return [functions[name] for name in names]

    def _qr(self, matrix, *, overwrite_a, mode, pivoting):
        """Full pivoted two-pass MGS; no full-vector host transfers or screening."""
        if not (overwrite_a and mode == "economic" and pivoting):
            raise ValueError("explicit complete pivoted image QR required")
        cp = self.cp
        rows, count = matrix.shape
        q = cp.zeros_like(matrix)
        r, permutation = np.zeros((count, count)), np.arange(count)
        for j in range(count):
            norms = cp.linalg.norm(matrix[:, j:], axis=0)
            pivot = j + int(cp.argmax(norms).item())
            if pivot != j:
                temporary = matrix[:, j].copy()
                matrix[:, j], matrix[:, pivot] = matrix[:, pivot], temporary
                r[:, [j, pivot]] = r[:, [pivot, j]]
                permutation[[j, pivot]] = permutation[[pivot, j]]
            norm = float(cp.linalg.norm(matrix[:, j]).item())
            if not np.isfinite(norm):
                raise FloatingPointError("nonfinite complete recycled image norm")
            if norm == 0:
                # Original GCROT rank test discards the zero R diagonal.
                if j == 0:
                    raise FloatingPointError("all recycled images are exactly zero")
                break
            r[j, j] = norm
            q[:, j] = matrix[:, j] / norm
            if j + 1 < count:
                for _ in range(2):
                    coefficients = q[:, j] @ matrix[:, j + 1 :]
                    r[j, j + 1 :] += cp.asnumpy(coefficients)
                    matrix[:, j + 1 :] -= q[:, j, None] * coefficients[None, :]
        return q, r, permutation

    def _system(self, operator, preconditioner, initial, rhs):
        cp = self.cp
        if preconditioner is not None or operator.dtype != np.dtype(float):
            raise ValueError(
                "declared complete real unpreconditioned native operator required"
            )

        def action(vector):
            self.calls += 1
            result = np.asarray(operator.matvec(cp.asnumpy(vector)))
            if (
                result.shape != rhs.shape
                or result.dtype != np.float64
                or not np.all(np.isfinite(result))
            ):
                raise ValueError("complete finite actual native action required")
            return cp.asarray(result)

        b = cp.array(rhs, dtype=cp.float64, copy=True)
        x = (
            cp.zeros_like(b)
            if initial is None
            else cp.array(initial, dtype=cp.float64, copy=True)
        )
        a = SimpleNamespace(shape=operator.shape, dtype=np.dtype(float), matvec=action)
        identity = SimpleNamespace(matvec=lambda v: v)
        return a, identity, x, b, cp.asnumpy

    def solve(self, operator, rhs, **options):
        cp = self.cp
        m, k = options["m"], options["k"]
        count = operator.shape[0]
        if (
            operator.shape != (count, count)
            or rhs.shape != (count,)
            or rhs.dtype != np.float64
        ):
            raise ValueError("complete square native vector problem required")
        if options.get("truncate") != "oldest" or not options.get("discard_C"):
            raise ValueError("explicit oldest/actual-image refresh policy required")
        reserve = 2 * (min(m + k, count) + 1) + 6 * (k + 1) + 24 + 16
        bound = 8 * count * reserve + self.operator_bound
        with self.device, self.stream, cp.cuda.using_allocator(self.pool.malloc):
            free, _ = cp.cuda.runtime.memGetInfo()
            if bound > self.cap or bound > free + self.pool.free_bytes():
                raise MemoryError(
                    "complete resident Krylov/operator cap/free preflight failed"
                )
            callback = options.get("callback")
            if callback is not None:
                options["callback"] = lambda vector: callback(cp.asnumpy(vector))
            try:
                result = self.algorithm(operator, rhs, **options)
            finally:
                # Drain this owned stream before returning caller resources,
                # including when a native action or checkpoint rejects.
                self.stream.synchronize()
            with (self.out / "observations.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(
                            calls=self.calls,
                            count=count,
                            device_bound_bytes=bound,
                            used_bytes=self.pool.used_bytes(),
                            reserved_bytes=self.pool.total_bytes(),
                            scope="owned resident complete FP64 numerical vectors; no controlled timing claim",
                        )
                    )
                    + "\n"
                )
            return result
