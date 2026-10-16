"""Private zero-copy Numba density kernels inside the existing full CuPy map.

Reuse the proven original geometry, complete preflight, rank buffers and
caller stream. Only the three arithmetic kernels change. Every CuPy allocation
remains in the original capped private diagram pool. No array travels through
host memory at this adapter boundary, and there is no backend fallback.
"""

from numba_density_control import _cuda_kernels


class NumbaCupyDensityAdapter:
    def __init__(self, original):
        from numba import cuda

        self.original, self.cp, self.cuda = original, original.cp, cuda
        if not cuda.is_available():
            raise RuntimeError("selected Numba CUDA target unavailable")
        self.kernels = _cuda_kernels(cuda)
        self.calls = 0
        self.stream_handles = set()
        self.batch_shapes = set()

    def preflight(self, batch):
        return self.original.preflight(batch)

    def apply(self, x, rank):
        cp, cuda, source = self.cp, self.cuda, self.original
        if (
            not isinstance(x, cp.ndarray)
            or x.dtype != cp.float64
            or x.ndim != 2
            or x.shape[0] != source.nodes
            or not x.flags.c_contiguous
        ):
            raise ValueError("complete resident real float64 C batch required")
        if int(x.device.id) != int(cuda.get_current_device().id):
            raise ValueError("caller CuPy and Numba device/context identity differs")
        batch = x.shape[1]
        source.preflight(batch)  # unchanged full-space/free-memory rejection
        if source.rank != rank:
            source.weights = tuple(
                cp.asarray(a) for a in source.feature.rank_arrays(rank)
            )
            source.rank = rank
        rising, falling, weights = source.weights
        outer, cell, node, rowptr, partner, edge = source.geometry
        up, down, prefix, suffix = (
            cp.empty((source.nv, batch), dtype=cp.float64) for _ in range(4)
        )
        y = cp.empty_like(x)
        arguments = (
            (x, outer, rising, falling, up, down),
            (up, down, prefix, suffix),
            (
                x,
                cell,
                node,
                rising,
                falling,
                prefix,
                suffix,
                rowptr,
                partner,
                edge,
                weights,
                y,
            ),
        )
        # Producer, all three Numba kernels and downstream cuBLAS/RawKernels
        # share exactly the current caller stream, including explicit nonzero
        # streams. Importing views with sync=False is valid only under this
        # enforced same-stream ordering; a different stream needs events.
        handle = int(cp.cuda.get_current_stream().ptr)
        stream = cuda.external_stream(handle)
        for kernel, args, size in zip(
            self.kernels,
            arguments,
            (source.nv * batch, batch, source.nodes * batch),
            strict=True,
        ):
            views = tuple(cuda.as_cuda_array(a, sync=False) for a in args)
            if any(
                int(v.device_ctypes_pointer.value) != int(a.data.ptr)
                for v, a in zip(views, args, strict=True)
            ):
                raise AssertionError("CUDA array-interface import copied storage")
            kernel[(size + 127) // 128, 128, stream](*views)
        self.calls += 1
        self.stream_handles.add(handle)
        self.batch_shapes.add((source.nodes, batch))
        return y
