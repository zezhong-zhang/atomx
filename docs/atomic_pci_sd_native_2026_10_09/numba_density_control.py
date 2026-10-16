"""Private complete FP64 density batch with shared Numba CPU/CUDA arithmetic.

Backends are explicit. Geometry/rank weights belong to the public feature
service; this control changes execution only. CPU workers and a CUDA stream
are scoped to this owner. CUDA is optional and imported only when selected.
The resident CUDA output is borrowed until the next action, not a snapshot.
"""

from contextlib import contextmanager

import numba as nb
import numpy as np


def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


def _scalar_functions(compile_function, add, mul):
    """Compile the same ordered scalar algorithms for each target."""

    def moment(x, outer, rise, fall, cell, b):
        up, down = 0.0, 0.0
        for component in range(2):
            for m in range(outer.shape[2]):
                value = x[outer[component, cell, m], b]
                up = add(up, mul(rise[cell, m], value))
                down = add(down, mul(fall[cell, m], value))
        return up, down

    def scan(up, down, prefix, suffix, b):
        u, d = 0.0, 0.0
        for cell in range(up.shape[0]):
            prefix[cell, b] = u
            u = add(u, up[cell, b])
        for cell in range(up.shape[0] - 1, -1, -1):
            suffix[cell, b] = d
            d = add(d, down[cell, b])

    def scatter(
        x,
        cell_map,
        node_map,
        rise,
        fall,
        prefix,
        suffix,
        rowptr,
        partner,
        edge,
        weights,
        i,
        b,
    ):
        cell, m = cell_map[i], node_map[i]
        value = 0.0
        if cell >= 0:
            value = add(value, mul(fall[cell, m], prefix[cell, b]))
            value = add(value, mul(rise[cell, m], suffix[cell, b]))
        # Stable incident-edge order, including both diagonal orientations.
        for e in range(rowptr[i], rowptr[i + 1]):
            value = add(value, mul(weights[edge[e]], x[partner[e], b]))
        return value

    return tuple(compile_function(f) for f in (moment, scan, scatter))


def _cpu_kernels():
    compile_scalar = nb.njit(inline="always", fastmath=False)
    moment, scan, scatter = _scalar_functions(
        compile_scalar, compile_scalar(_add), compile_scalar(_mul)
    )

    @nb.njit(parallel=True, fastmath=False)
    def moments(x, outer, rise, fall, up, down):
        batch = x.shape[1]
        for tid in nb.prange(up.size):
            cell, b = tid // batch, tid % batch
            up[cell, b], down[cell, b] = moment(x, outer, rise, fall, cell, b)

    @nb.njit(parallel=True, fastmath=False)
    def scans(up, down, prefix, suffix):
        for b in nb.prange(up.shape[1]):
            scan(up, down, prefix, suffix, b)

    @nb.njit(parallel=True, fastmath=False)
    def scatters(
        x,
        cell_map,
        node_map,
        rise,
        fall,
        prefix,
        suffix,
        rowptr,
        partner,
        edge,
        weights,
        y,
    ):
        batch = x.shape[1]
        for tid in nb.prange(y.size):
            i, b = tid // batch, tid % batch
            y[i, b] = scatter(
                x,
                cell_map,
                node_map,
                rise,
                fall,
                prefix,
                suffix,
                rowptr,
                partner,
                edge,
                weights,
                i,
                b,
            )

    return moments, scans, scatters


def _cuda_kernels(cuda):
    from numba.cuda import libdevice

    # Explicit IEEE round-to-nearest operations prevent CUDA's contraction
    # of multiply/add; fastmath=False alone is not a PTX no-FMA certificate.
    moment, scan, scatter = _scalar_functions(
        cuda.jit(device=True, inline=True, fastmath=False),
        libdevice.dadd_rn,
        libdevice.dmul_rn,
    )

    @cuda.jit(fastmath=False)
    def moments(x, outer, rise, fall, up, down):
        tid = cuda.grid(1)
        if tid < up.size:
            cell, b = tid // x.shape[1], tid % x.shape[1]
            up[cell, b], down[cell, b] = moment(x, outer, rise, fall, cell, b)

    @cuda.jit(fastmath=False)
    def scans(up, down, prefix, suffix):
        b = cuda.grid(1)
        if b < up.shape[1]:
            scan(up, down, prefix, suffix, b)

    @cuda.jit(fastmath=False)
    def scatters(
        x,
        cell_map,
        node_map,
        rise,
        fall,
        prefix,
        suffix,
        rowptr,
        partner,
        edge,
        weights,
        y,
    ):
        tid = cuda.grid(1)
        if tid < y.size:
            i, b = tid // x.shape[1], tid % x.shape[1]
            y[i, b] = scatter(
                x,
                cell_map,
                node_map,
                rise,
                fall,
                prefix,
                suffix,
                rowptr,
                partner,
                edge,
                weights,
                i,
                b,
            )

    return moments, scans, scatters


@contextmanager
def _workers(threads):
    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        yield
    finally:
        nb.set_num_threads(previous)


class NumbaDensityControl:
    """Same host ``apply_into`` contract with explicit CPU/CUDA execution.

    Fixed full batch and explicit memory cap are required. Real C-contiguous
    float64 only: complex C-product and other layouts reject rather than cast.
    CUDA allocations are owner-held arrays on a dedicated stream, with no
    caller allocator, default-stream or global pool modification.
    """

    def __init__(self, feature, *, backend, batch, memory_bytes, threads=8):
        if backend not in ("cpu", "cuda"):
            raise ValueError("explicit cpu or cuda backend required")
        for label, value in (
            ("batch", batch),
            ("memory_bytes", memory_bytes),
            ("threads", threads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"positive integer {label} required")
        if threads > nb.config.NUMBA_NUM_THREADS:
            raise ValueError("thread request exceeds Numba's configured maximum")
        self.feature, self.backend, self.batch = feature, backend, batch
        self.cap, self.threads = memory_bytes, threads
        self.nodes = len(feature.radii)
        self.nv, self.ng = feature.outer_indices.shape[1:]
        self.shape = (self.nodes, batch)
        edges = len(feature.edge_left)
        # Conservative construction + resident geometry + rank-buffer bound.
        # Includes CSR sort/control scratch before any CSR allocation.
        self.geometry_bound = 8 * (
            8 * self.nodes
            + 3 * feature.outer_indices.size
            + 12 * edges
            + 6 * self.nv * self.ng
        )
        self.bound = self.geometry_bound + 8 * batch * (2 * self.nodes + 4 * self.nv)
        if self.bound > self.cap:
            raise MemoryError("complete Numba geometry/batch exceeds explicit cap")
        self.cuda = None
        if backend == "cuda":
            from numba import cuda

            if not cuda.is_available():
                raise RuntimeError("selected CUDA backend unavailable; no CPU fallback")
            self.cuda = cuda
            free, _ = cuda.current_context().get_memory_info()
            if self.bound > free:
                raise MemoryError("complete Numba batch exceeds free device memory")
        row = np.empty(2 * edges, dtype=np.int64)
        partner = np.empty_like(row)
        edge = np.repeat(np.arange(edges, dtype=np.int64), 2)
        row[0::2], row[1::2] = feature.edge_left, feature.edge_right
        partner[0::2], partner[1::2] = feature.edge_right, feature.edge_left
        order = np.argsort(row, kind="stable")
        rowptr = np.concatenate(
            ([0], np.cumsum(np.bincount(row, minlength=self.nodes)))
        )
        cell_map = np.full(self.nodes, -1, dtype=np.int64)
        node_map = np.full(self.nodes, -1, dtype=np.int64)
        for component in range(2):
            for cell in range(self.nv):
                for m in range(self.ng):
                    i = feature.outer_indices[component, cell, m]
                    if cell_map[i] != -1:
                        raise ValueError(
                            "duplicate outer coordinate requires explicit handling"
                        )
                    cell_map[i], node_map[i] = cell, m
        self.host_geometry = tuple(
            np.ascontiguousarray(a)
            for a in (
                feature.outer_indices,
                cell_map,
                node_map,
                rowptr,
                partner[order],
                edge[order],
            )
        )
        self.rank, self.weights = None, None
        if self.cuda is None:
            self.geometry = self.host_geometry
            self.work = tuple(np.empty((self.nv, batch)) for _ in range(4))
            self.kernels = _cpu_kernels()
        else:
            self.stream = self.cuda.stream()
            self.geometry = tuple(
                self.cuda.to_device(a, stream=self.stream) for a in self.host_geometry
            )
            self.x, self.y = (
                self.cuda.device_array(self.shape, dtype=np.float64, stream=self.stream)
                for _ in range(2)
            )
            self.work = tuple(
                self.cuda.device_array(
                    (self.nv, batch), dtype=np.float64, stream=self.stream
                )
                for _ in range(4)
            )
            self.kernels = _cuda_kernels(self.cuda)
            self.stream.synchronize()
        self.input_ready = False

    def _validate(self, x):
        if (
            not isinstance(x, np.ndarray)
            or x.dtype != np.float64
            or x.shape != self.shape
            or not x.flags.c_contiguous
        ):
            raise ValueError("complete C-contiguous real float64 batch required")
        if not np.all(np.isfinite(x)):
            raise ValueError("finite complete density batch required")

    def prepare_rank(self, rank):
        arrays = self.feature.rank_arrays(rank)  # public rank validation/weights
        if self.rank == rank:
            return
        if self.cuda is None:
            self.weights = arrays
        elif self.weights is None:
            self.weights = tuple(
                self.cuda.to_device(a, stream=self.stream) for a in arrays
            )
        else:
            # Reuse the same three buffers; no retained old rank/device cache.
            for device, host in zip(self.weights, arrays, strict=True):
                device.copy_to_device(host, stream=self.stream)
        self.rank = rank

    def set_input(self, x):
        if self.cuda is None:
            raise ValueError("resident upload requires selected CUDA backend")
        self._validate(x)
        self.x.copy_to_device(x, stream=self.stream)
        self.input_ready = True

    def _execute(self, x, y):
        rise, fall, weights = self.weights
        outer, cell, node, rowptr, partner, edge = self.geometry
        up, down, prefix, suffix = self.work
        arguments = (
            (x, outer, rise, fall, up, down),
            (up, down, prefix, suffix),
            (
                x,
                cell,
                node,
                rise,
                fall,
                prefix,
                suffix,
                rowptr,
                partner,
                edge,
                weights,
                y,
            ),
        )
        if self.cuda is None:
            with _workers(self.threads):
                for kernel, args in zip(self.kernels, arguments, strict=True):
                    kernel(*args)
        else:
            for kernel, args, size in zip(
                self.kernels,
                arguments,
                (self.nv * self.batch, self.batch, self.nodes * self.batch),
                strict=True,
            ):
                kernel[(size + 127) // 128, 128, self.stream](*args)

    def apply_resident(self, rank):
        if self.cuda is None or not self.input_ready:
            raise ValueError(
                "selected CUDA backend with uploaded complete input required"
            )
        self.prepare_rank(rank)
        self._execute(self.x, self.y)
        return self.y

    def synchronize(self):
        if self.cuda is not None:
            self.stream.synchronize()

    def apply_into(self, x, rank, y):
        """Complete host action; GPU timing includes input and output transfers."""
        self._validate(x)
        if (
            not isinstance(y, np.ndarray)
            or y.dtype != np.float64
            or y.shape != self.shape
            or not y.flags.c_contiguous
            or not y.flags.writeable
            or np.shares_memory(x, y)
        ):
            raise ValueError("distinct writable complete float64 output required")
        self.prepare_rank(rank)
        if self.cuda is None:
            self._execute(x, y)
        else:
            self.set_input(x)
            self._execute(self.x, self.y)
            self.y.copy_to_host(y, stream=self.stream)
            self.synchronize()
        return y
