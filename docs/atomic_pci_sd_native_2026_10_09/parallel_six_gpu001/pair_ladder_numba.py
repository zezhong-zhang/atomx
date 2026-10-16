"""Private complete core PP/HH FP64 control with shared CPU/CUDA algorithm.

This is a contraction control, not a new all-SD GPU backend. Real full FP64
only; complex J interactions stay in the production complex CPU operator.
"""

import time

import numba as nb
import numpy as np
from numba_density_control import _add, _mul, _workers


def scalar_factory(compile_scalar, add, mul):
    def scalar(pp, hh, sizes, offsets, p_offsets, h_offsets, x, coordinate, batch):
        block = 0
        while coordinate >= offsets[block + 1]:
            block += 1
        nv, no = sizes[block, 0], sizes[block, 1]
        local = coordinate - offsets[block]
        v, o = local // no, local % no
        p_value, h_value = 0.0, 0.0
        for u in range(nv):
            p_value = add(
                p_value,
                mul(
                    pp[p_offsets[block] + v * nv + u],
                    x[offsets[block] + u * no + o, batch],
                ),
            )
        for j in range(no):
            h_value = add(
                h_value,
                mul(
                    x[offsets[block] + v * no + j, batch],
                    hh[h_offsets[block] + j * no + o],
                ),
            )
        return p_value, h_value

    return compile_scalar(scalar)


def cpu_kernel():
    compile_scalar = nb.njit(inline="always", fastmath=False)
    scalar = scalar_factory(compile_scalar, compile_scalar(_add), compile_scalar(_mul))

    @nb.njit(parallel=True, fastmath=False)
    def action(pp, hh, sizes, offsets, p_offsets, h_offsets, x, p, h):
        for tid in nb.prange(x.size):
            coordinate, batch = tid // x.shape[1], tid % x.shape[1]
            p[coordinate, batch], h[coordinate, batch] = scalar(
                pp, hh, sizes, offsets, p_offsets, h_offsets, x, coordinate, batch
            )

    return action


def cuda_kernel(cuda):
    from numba.cuda import libdevice

    scalar = scalar_factory(
        cuda.jit(device=True, inline=True, fastmath=False),
        libdevice.dadd_rn,
        libdevice.dmul_rn,
    )

    @cuda.jit(fastmath=False)
    def action(pp, hh, sizes, offsets, p_offsets, h_offsets, x, p, h):
        tid = cuda.grid(1)
        if tid < x.size:
            coordinate, batch = tid // x.shape[1], tid % x.shape[1]
            p[coordinate, batch], h[coordinate, batch] = scalar(
                pp, hh, sizes, offsets, p_offsets, h_offsets, x, coordinate, batch
            )

    return action


class NumbaPairLadderControl:
    def __init__(self, ladders, *, backend, batch, memory_bytes, threads=1):
        if backend not in ("cpu", "cuda"):
            raise ValueError("explicit CPU or CUDA target required")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (batch, memory_bytes, threads)
        ):
            raise ValueError("positive batch/cap/threads required")
        if threads > nb.config.NUMBA_NUM_THREADS:
            raise ValueError("configured worker ceiling exceeded")
        coordinates = len(ladders.source_Ha)
        numeric = sum(v.size for v in ladders.pp_Ha.values()) + sum(
            v.size for v in ladders.hh_Ha.values()
        )
        self.bound = 8 * (
            2 * numeric + 6 * coordinates * batch + 12 * len(ladders.layout)
        )
        if self.bound > memory_bytes:
            raise MemoryError("whole ladder batch exceeds cap before copies or CUDA")
        pp, hh, sizes, offsets, p_offsets, h_offsets = [], [], [], [0], [], []
        for block, occupied, virtual, _first, last in ladders.layout:
            if np.any(ladders.pp_Ha[block].imag) or np.any(ladders.hh_Ha[block].imag):
                raise ValueError(
                    "real FP64 control rejects complete complex interaction"
                )
            p_offsets.append(len(pp))
            h_offsets.append(len(hh))
            pp.extend(ladders.pp_Ha[block].real.ravel())
            hh.extend(ladders.hh_Ha[block].real.ravel())
            sizes.append((len(virtual), len(occupied)))
            offsets.append(last)
        self.host_geometry = tuple(
            np.ascontiguousarray(value, dtype=np.float64 if i < 2 else np.int64)
            for i, value in enumerate((pp, hh, sizes, offsets, p_offsets, h_offsets))
        )
        self.shape, self.threads, self.backend = (coordinates, batch), threads, backend
        self.input_ready = False
        self.cuda = None
        if backend == "cpu":
            self.geometry, self.kernel = self.host_geometry, cpu_kernel()
        else:
            from numba import cuda

            if not cuda.is_available():
                raise RuntimeError("explicit CUDA target unavailable, no CPU fallback")
            self.cuda = cuda
            free, total = cuda.current_context().get_memory_info()
            if self.bound > free:
                raise MemoryError("whole ladder batch exceeds free device payload")
            self.stream = cuda.stream()
            started = time.perf_counter()
            self.geometry = tuple(
                cuda.device_array_like(a, stream=self.stream)
                for a in self.host_geometry
            )
            self.x, self.p, self.h = (
                cuda.device_array(self.shape, dtype=np.float64, stream=self.stream)
                for _ in range(3)
            )
            self.synchronize()
            allocation_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for device, host in zip(self.geometry, self.host_geometry, strict=True):
                device.copy_to_device(host, stream=self.stream)
            self.synchronize()
            self.setup_layers = dict(
                allocation_seconds=allocation_seconds,
                complete_geometry_upload_seconds=time.perf_counter() - started,
            )
            self.kernel = cuda_kernel(cuda)
            self.synchronize()

    def _validate(self, x):
        if (
            not isinstance(x, np.ndarray)
            or x.dtype != np.float64
            or x.shape != self.shape
            or not x.flags.c_contiguous
            or not np.all(np.isfinite(x))
        ):
            raise ValueError("complete C real FP64 batch required")

    def apply_into(self, x, p, h):
        self._validate(x)
        for y in (p, h):
            if (
                not isinstance(y, np.ndarray)
                or y.shape != self.shape
                or y.dtype != np.float64
                or not y.flags.c_contiguous
                or not y.flags.writeable
                or np.shares_memory(x, y)
            ):
                raise ValueError("distinct full real FP64 writable output required")
        if np.shares_memory(p, h):
            raise ValueError("separated PP and HH outputs required")
        if self.cuda is None:
            with _workers(self.threads):
                self.kernel(*self.geometry, x, p, h)
        else:
            self.set_input(x)
            self.apply_resident()
            self.p.copy_to_host(p, stream=self.stream)
            self.h.copy_to_host(h, stream=self.stream)
            self.synchronize()
        return p, h

    def set_input(self, x):
        if self.cuda is None:
            raise ValueError("resident upload requires CUDA")
        self._validate(x)
        self.x.copy_to_device(x, stream=self.stream)
        self.input_ready = True

    def apply_resident(self):
        if self.cuda is None or not self.input_ready:
            raise ValueError(
                "resident contraction requires uploaded complete CUDA input"
            )
        self.kernel[(self.x.size + 127) // 128, 128, self.stream](
            *self.geometry, self.x, self.p, self.h
        )

    def synchronize(self):
        if self.cuda is not None:
            self.stream.synchronize()
