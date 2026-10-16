"""PRIVATE native Coulomb density-band executor; not a qualified public API.

Imports are standard-library only until a numerical factory is explicitly
called. Public Atomx owns factors and signed density construction. This engine
only executes the existing semiseparable scan/row equation; it does not select
physics, compute angular tables, assemble H, or contract/normalize CI states.
No installation, backend fallback, automatic source inference, or array casts.
"""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

COMPONENTS = ("pppp", "pqpq", "qpqp", "qqqq")
MAX_OWNED_BYTES = 512 * 1024 * 1024


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("a complete lowercase SHA256 identity is required")


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"unsupported {label}")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Complete declared source union, preserved in caller insertion order.

    Completeness with respect to the actual full signed RDM is a consuming
    qualification gate, not something a numerical primitive can infer.
    ``jobs`` contain (rank, key_a, key_b); each generates all four components.
    ``keys``/``jobs`` must be immutable tuples; keys are native (n, kappa).
    """

    source_sha256: str
    workspace_sha256: str
    signed_source_union_sha256: str
    ns: int
    orders: tuple
    extents: tuple
    keys: tuple
    jobs: tuple
    chunk_sources: int = 128

    def __post_init__(self):
        for value in (self.source_sha256, self.workspace_sha256,
                      self.signed_source_union_sha256):
            _sha(value)
        _integer(self.ns, 96, 123, "ns; no dense-bank fallback")
        _integer(self.chunk_sources, 1, 128, "chunk size")
        for label, values in (("orders", self.orders), ("extents", self.extents)):
            if type(values) is not tuple or len(values) != 2:
                raise ValueError(f"immutable two-component {label} required")
        for order, extent in zip(self.orders, self.extents, strict=True):
            _integer(order, 1, 9, "component order")
            _integer(extent, order, self.ns, "component extent")
        if type(self.keys) is not tuple or not 1 <= len(self.keys) <= 31:
            raise ValueError("immutable complete radial keys required")
        for key in self.keys:
            if (type(key) is not tuple or len(key) != 2
                    or any(type(v) is not int for v in key)
                    or key[0] < 1 or key[1] == 0):
                raise ValueError("native (n,kappa) key required")
        if self.keys != tuple(sorted(set(self.keys))):
            raise ValueError("radial key order must be sorted and unique")
        if type(self.jobs) is not tuple or not 1 <= len(self.jobs) <= 3472:
            raise ValueError("complete immutable source jobs required")
        seen = set()
        for job in self.jobs:
            if type(job) is not tuple or len(job) != 3:
                raise ValueError("immutable (rank,key_a,key_b) job required")
            rank, a, b = job
            _integer(rank, 0, 6, "nonnegative Coulomb multipole")
            if a not in self.keys or b not in self.keys or a > b or job in seen:
                raise ValueError("duplicate, undeclared, or noncanonical source pair")
            seen.add(job)
        if _digest(self.jobs) != self.signed_source_union_sha256:
            raise ValueError("complete signed source-union declaration mismatch")

    @property
    def pairs(self):
        return tuple(dict.fromkeys((a, b) for _, a, b in self.jobs))

    @property
    def ranks(self):
        return tuple(sorted({rank for rank, _, _ in self.jobs}))

    @property
    def identity(self):
        return _digest((self.source_sha256, self.workspace_sha256,
                        self.signed_source_union_sha256, self.ns, self.orders,
                        self.extents, self.keys, self.jobs, self.chunk_sources))


def resource_preflight(plan):
    """Pure-integer bounds, including padding/copies and each component scan."""
    ns, k = plan.ns, max(plan.orders)
    f = len(plan.ranks) * 4
    densities = len(plan.pairs) * 2
    batch = min(plan.chunk_sources, len(plan.jobs)) * 4
    factors = f * (k * k * ns * (2 * k - 1) + 4 * k * ns) * 8
    density_bytes = densities * ns * k * 8
    scans = batch * 2 * k * (ns + 1) * 8
    output = batch * ns * k * 8
    indices = batch * 4 * 8
    dense_cpu_pairs = min(plan.chunk_sources, len(plan.jobs)) * 2 * ns * ns * 8
    # Device arrays plus corresponding immutable host objects and construction
    # temporaries. The caller-owned public factors/densities are INCLUDED once.
    device = factors + density_bytes + scans + output + indices
    host = 3 * factors + 3 * density_bytes + scans + 3 * output + indices + dense_cpu_pairs
    if max(device, host) > MAX_OWNED_BYTES:
        raise ValueError("owned payload budget exceeded; no model reduction")
    chunks = (len(plan.jobs) + plan.chunk_sources - 1) // plan.chunk_sources
    return {
        "factor_bytes": factors, "unique_density_bytes": density_bytes,
        "chunk_prefix_suffix_bytes": scans, "chunk_band_bytes": output,
        "chunk_index_bytes": indices, "CPU_dense_postprocess_bytes": dense_cpu_pairs,
        "owned_device_bytes_upper": device, "owned_host_bytes_upper": host,
        "all_band_D2H_bytes_upper": len(plan.jobs) * 4 * ns * k * 8,
        "source_jobs": len(plan.jobs), "component_jobs": 4 * len(plan.jobs),
        "chunks": chunks, "scan_kernel_calls_per_trial": chunks,
        "row_kernel_calls_per_trial": chunks,
        "scan_owned_rows": 4 * len(plan.jobs) * k,
        "result_owned_rows": 4 * len(plan.jobs) * ns * k,
        "scope": "array payload only; runtime/context/JIT/RSS/free device admission separate",
    }


def require_descriptor(array, shape):
    """Metadata-only check, also callable with standard-library mocks."""
    if (tuple(array.shape) != tuple(shape) or str(array.dtype) != "float64"
            or not array.flags.c_contiguous):
        raise ValueError("exact native contiguous float64 shape required; no cast")


def _owned(array, shape, np):
    require_descriptor(array, shape)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite input")
    # Bytes-backed arrays cannot be made writeable by setflags. Producer
    # mutation after this copy cannot change the owned resident input.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, slots=True, init=False)
class DensitySnapshot:
    plan_identity: str
    trial_sha256: str
    value_sha256: str
    bands: object

    def __init__(self, *args, **kwargs):
        raise TypeError("use from_public_bands; arbitrary snapshot construction is unsupported")

    @classmethod
    def from_public_bands(cls, plan, bands, *, trial_sha256):
        """Own already constructed complete P/Q source bands; no delta adapter."""
        _sha(trial_sha256)
        resource_preflight(plan)
        import numpy as np

        expected = tuple((pair, right) for pair in plan.pairs for right in range(2))
        if set(bands) != set(expected):
            raise ValueError("missing/extra complete P/Q source density")
        k = max(plan.orders)
        packed = np.zeros((len(expected), plan.ns, k), dtype=np.float64)
        for index, (pair, right) in enumerate(expected):
            order, extent = plan.orders[right], plan.extents[right]
            value = _owned(bands[(pair, right)], (plan.ns, order), np)
            for offset in range(order):
                if np.any(value[max(0, extent - offset):, offset] != 0.0):
                    raise ValueError("nonzero structural RKB density tail; never clip")
            packed[index, :, :order] = value
        owned = _owned(packed, packed.shape, np)
        result = object.__new__(cls)
        for name, value in (("plan_identity", plan.identity), ("trial_sha256", trial_sha256),
                            ("value_sha256", hashlib.sha256(owned.tobytes()).hexdigest()),
                            ("bands", owned)):
            object.__setattr__(result, name, value)
        return result


def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


def _ordered_scalar_functions(compile_scalar, add, mul):
    """Same scalar scan/row on both targets; exactly the public loop order."""

    def scan(rise, fall, density, prefix, suffix, factor, source, job, jp, ns, kr):
        for j in range(ns + 1):
            prefix[job, jp, j] = 0.0
            suffix[job, jp, j] = 0.0
        if jp < kr:
            end = ns - jp
            for j in range(end):
                prefix[job, jp, j + 1] = add(prefix[job, jp, j],
                                            mul(rise[factor, jp, j], density[source, j, jp]))
            for j in range(end - 1, -1, -1):
                suffix[job, jp, j] = add(suffix[job, jp, j + 1],
                                        mul(fall[factor, jp, j], density[source, j, jp]))

    def row(near, rise, fall, density, prefix, suffix,
            factor, source, job, ip, i, ns, kl, kr):
        value = 0.0
        if ip < kl and i < ns - ip:
            first_near = i + ip - kl + 1
            for jp in range(kr):
                end = ns - jp
                before = min(end, max(0, first_near))
                after = min(end, max(0, i + kr - jp))
                value = add(value, mul(fall[factor, ip, i], prefix[job, jp, before]))
                value = add(value, mul(rise[factor, ip, i], suffix[job, jp, after]))
                for j in range(max(0, first_near), after):
                    value = add(value, mul(near[factor, ip, jp, i, j - first_near],
                                           density[source, j, jp]))
        return value

    return compile_scalar(scan), compile_scalar(row)


def _cpu_kernels(nb):
    scalar = nb.njit(inline="always", fastmath=False)
    scan, row = _ordered_scalar_functions(scalar, scalar(_add), scalar(_mul))

    @nb.njit(parallel=True, fastmath=False)
    def scans(rise, fall, density, prefix, suffix, jobs, ns, k):
        for tid in nb.prange(jobs.shape[0] * k):
            job, jp = tid // k, tid % k
            scan(rise, fall, density, prefix, suffix,
                 jobs[job, 0], jobs[job, 1], job, jp, ns, jobs[job, 3])

    @nb.njit(parallel=True, fastmath=False)
    def rows(near, rise, fall, density, prefix, suffix, jobs, result, ns, k):
        for tid in nb.prange(jobs.shape[0] * ns * k):
            job, local = tid // (ns * k), tid % (ns * k)
            i, ip = local // k, local % k
            result[job, i, ip] = row(near, rise, fall, density, prefix, suffix,
                                    jobs[job, 0], jobs[job, 1], job, ip, i,
                                    ns, jobs[job, 2], jobs[job, 3])

    return scans, rows


def _cuda_kernels(cuda):
    from numba.cuda import libdevice

    scan, row = _ordered_scalar_functions(
        cuda.jit(device=True, inline=True, fastmath=False),
        libdevice.dadd_rn, libdevice.dmul_rn)

    @cuda.jit(fastmath=False)
    def scans(rise, fall, density, prefix, suffix, jobs, ns, k):
        tid = cuda.grid(1)
        if tid < jobs.shape[0] * k:
            job, jp = tid // k, tid % k
            scan(rise, fall, density, prefix, suffix,
                 jobs[job, 0], jobs[job, 1], job, jp, ns, jobs[job, 3])

    @cuda.jit(fastmath=False)
    def rows(near, rise, fall, density, prefix, suffix, jobs, result, ns, k):
        tid = cuda.grid(1)
        if tid < jobs.shape[0] * ns * k:
            job, local = tid // (ns * k), tid % (ns * k)
            i, ip = local // k, local % k
            result[job, i, ip] = row(near, rise, fall, density, prefix, suffix,
                                    jobs[job, 0], jobs[job, 1], job, ip, i,
                                    ns, jobs[job, 2], jobs[job, 3])

    return scans, rows


@contextmanager
def _worker_mask(nb, threads):
    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        yield
    finally:
        nb.set_num_threads(previous)


class PrivateBatchExecutor:
    """Resident exact factors, a per-trial density owner, and reusable scratch.

    Non-reentrant, owner-managed stream/context. Each yielded four-band packet
    is immutable; downstream CPU expansion/actions/full signed accumulation
    remain the existing public consumer's responsibility and timing scope.
    Concurrent mutation of internal/device objects is unsupported. Numerical
    construction is not authorized by importing this prepared module.
    """

    def __init__(self, plan, factors, *, backend, threads=1):
        if backend not in ("cpu", "cuda"):
            raise ValueError("explicit cpu/cuda backend required; no fallback")
        _integer(threads, 1, 64, "CPU worker count")
        if backend == "cuda" and threads != 1:
            raise ValueError("CPU worker mask does not select CUDA parallelism")
        self.preflight = resource_preflight(plan)
        expected = tuple((rank, comp) for rank in plan.ranks for comp in COMPONENTS)
        if set(factors) != set(expected):
            raise ValueError("complete rank/four-component factor set required")
        import numba as nb
        import numpy as np

        self.plan, self.backend, self.threads = plan, backend, threads
        self.np, self.nb, self._busy = np, nb, False
        ns, k = plan.ns, max(plan.orders)
        self.factor_ids = {key: i for i, key in enumerate(expected)}
        near = np.zeros((len(expected), k, k, ns, 2 * k - 1), dtype=np.float64)
        moments = [np.zeros((len(expected), k, ns), dtype=np.float64) for _ in range(4)]
        hashes = []
        for index, (rank, comp) in enumerate(expected):
            left, right = divmod(COMPONENTS.index(comp), 2)
            kl, kr = plan.orders[left], plan.orders[right]
            actual = factors[(rank, comp)]
            if type(actual) not in (tuple, list) or len(actual) != 5:
                raise ValueError("public five-factor tuple required")
            shapes = ((kl, kr, ns, kl + kr - 1), (kl, ns), (kl, ns),
                      (kr, ns), (kr, ns))
            values = tuple(_owned(a, shape, np) for a, shape in zip(actual, shapes, strict=True))
            hashes.append((rank, comp, tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in values)))
            near[index, :kl, :kr, :, :kl + kr - 1] = values[0]
            for target, value in zip(moments, values[1:], strict=True):
                target[index, :value.shape[0], :] = value
        self.factor_value_sha256 = _digest(hashes)
        self.resident_identity = _digest((plan.identity, self.factor_value_sha256))
        self.geometry = tuple(_owned(a, a.shape, np) for a in (near, *moments))
        size = min(plan.chunk_sources, len(plan.jobs)) * 4
        self.prefix = np.empty((size, k, ns + 1), dtype=np.float64)
        self.suffix = np.empty_like(self.prefix)
        self.output = np.empty((size, ns, k), dtype=np.float64)
        self.counters = {"trials": 0, "scan_kernel": 0, "row_kernel": 0,
                         "factor_H2D_bytes": 0, "density_H2D_bytes": 0,
                         "job_H2D_bytes": 0, "band_D2H_bytes": 0,
                         "stream_sync": 0}
        if backend == "cpu":
            if threads > nb.config.NUMBA_NUM_THREADS:
                raise ValueError("worker count exceeds configured Numba limit")
            self.kernels = _cpu_kernels(nb)
        else:
            from numba import cuda

            if not cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU fallback or install")
            self.cuda = cuda
            self.context = cuda.current_context()
            self.device_id = self.context.device.id
            free, _ = self.context.get_memory_info()
            if free < self.preflight["owned_device_bytes_upper"]:
                raise MemoryError("insufficient device payload memory")
            self.stream = cuda.stream()
            self.device_geometry = tuple(cuda.to_device(a, stream=self.stream) for a in self.geometry)
            self.device_prefix = cuda.device_array(self.prefix.shape, dtype=np.float64, stream=self.stream)
            self.device_suffix = cuda.device_array(self.suffix.shape, dtype=np.float64, stream=self.stream)
            self.device_output = cuda.device_array(self.output.shape, dtype=np.float64, stream=self.stream)
            self.device_density = cuda.device_array((len(plan.pairs) * 2, ns, k),
                                                    dtype=np.float64, stream=self.stream)
            self.counters["factor_H2D_bytes"] = sum(a.nbytes for a in self.geometry)
            self.stream.synchronize()
            self.counters["stream_sync"] += 1
            self.kernels = _cuda_kernels(cuda)

    def chunks(self, snapshot):
        """Execute each exact source once and yield (source_jobs, four_bands)."""
        if (type(snapshot) is not DensitySnapshot or snapshot.plan_identity != self.plan.identity
                or self._busy):
            raise ValueError("wrong/stale trial owner or concurrent execution")
        np, plan = self.np, self.plan
        _sha(snapshot.trial_sha256)
        require_descriptor(snapshot.bands, (len(plan.pairs) * 2, plan.ns, max(plan.orders)))
        if (snapshot.bands.flags.writeable
                or hashlib.sha256(snapshot.bands.tobytes()).hexdigest() != snapshot.value_sha256):
            raise ValueError("trial density bytes changed")
        if self.backend == "cuda" and (self.cuda.current_context() is not self.context
                                       or self.cuda.current_context().device.id != self.device_id):
            raise ValueError("CUDA context/device changed")
        pair_ids = {pair: i for i, pair in enumerate(plan.pairs)}
        self._busy = True
        try:
            self.counters["trials"] += 1
            if self.backend == "cuda":
                self.device_density.copy_to_device(snapshot.bands, stream=self.stream)
                self.counters["density_H2D_bytes"] += snapshot.bands.nbytes
            for start in range(0, len(plan.jobs), plan.chunk_sources):
                source_jobs = plan.jobs[start:start + plan.chunk_sources]
                entries = []
                for rank, a, b in source_jobs:
                    for component_index, comp in enumerate(COMPONENTS):
                        left, right = divmod(component_index, 2)
                        entries.append((self.factor_ids[(rank, comp)], pair_ids[(a, b)] * 2 + right,
                                        plan.orders[left], plan.orders[right]))
                jobs = np.array(entries, dtype=np.int64)
                count = len(entries)
                near, rise_left, fall_left, rise_right, fall_right = self.geometry
                k = max(plan.orders)
                if self.backend == "cpu":
                    with _worker_mask(self.nb, self.threads):
                        self.kernels[0](rise_right, fall_right, snapshot.bands,
                                        self.prefix, self.suffix, jobs, plan.ns, k)
                        self.kernels[1](near, rise_left, fall_left, snapshot.bands,
                                        self.prefix, self.suffix, jobs, self.output, plan.ns, k)
                else:
                    d_jobs = self.cuda.to_device(jobs, stream=self.stream)
                    self.counters["job_H2D_bytes"] += jobs.nbytes
                    self.kernels[0][(count * k + 127) // 128, 128, self.stream](
                        self.device_geometry[3], self.device_geometry[4], self.device_density,
                        self.device_prefix, self.device_suffix, d_jobs, plan.ns, k)
                    self.kernels[1][(count * plan.ns * k + 127) // 128, 128, self.stream](
                        self.device_geometry[0], self.device_geometry[1], self.device_geometry[2],
                        self.device_density, self.device_prefix, self.device_suffix,
                        d_jobs, self.device_output, plan.ns, k)
                    self.device_output[:count].copy_to_host(self.output[:count], stream=self.stream)
                    self.stream.synchronize()
                    self.counters["stream_sync"] += 1
                    self.counters["band_D2H_bytes"] += count * plan.ns * k * 8
                self.counters["scan_kernel"] += 1
                self.counters["row_kernel"] += 1
                shape = (len(source_jobs), 4, plan.ns, k)
                result = _owned(self.output[:count].reshape(shape), shape, np)
                yield source_jobs, result
        finally:
            self._busy = False
