"""Exact matrix action of the shared cell-Coulomb pair-density kernel.

Cross-cell moments and same-cell edges are those of CellCoulombFeatures.
No four-orbital tensor, fitted auxiliary basis or amplitude truncation is
introduced. BLAS contracts complete density matrices on bounded edge blocks.
"""

import hashlib
from types import MappingProxyType

import numpy as np


def _owned(a):
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


class CellPairContraction:
    """M[m,n] = sum[r,s] R^k(mr,sn) * rho[r,s], with C-products.

    Four operands are complete component-tagged samples with columns ordered
    as target-left, source-left, source-right and target-right. They are owned
    immutable snapshots. Caller-supplied arrays are excluded from the work
    cap; all owned operands, moments and conservative complex scratch are
    included. The kernel retains every source coordinate and quadrature edge.
    """

    def __init__(
        self,
        features,
        target_left,
        source_left,
        source_right,
        target_right,
        rank,
        *,
        edge_batch_size=1024,
        max_work_bytes=128 * 1024**2,
    ):
        for name, value in (
            ("edge_batch_size", edge_batch_size),
            ("max_work_bytes", max_work_bytes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"positive integer {name} required")
        arrays = tuple(
            np.asarray(a, dtype=np.result_type(a, float))
            for a in (target_left, source_left, source_right, target_right)
        )
        if any(
            a.ndim != 2
            or a.shape[0] != len(features.radii)
            or a.shape[1] < 1
            or not np.all(np.isfinite(a))
            for a in arrays
        ):
            raise ValueError("finite complete auxiliary-node column operands required")
        rising, falling, edges = features.rank_arrays(rank)
        m, r, s, n = (a.shape[1] for a in arrays)
        nv, ng = features.identity["quadrature_nodes"]
        batch = min(int(edge_batch_size), len(edges))
        # Complex scratch covers either a real or complex trial density.
        # Include four moments, two cumulative moments and einsum temporaries.
        bound = sum(a.nbytes for a in arrays) + 16 * (
            nv * (4 * m * r + 6 * n * s + 2 * m * s)
            + 4 * m * n
            + r * s
            + batch * (2 * m + 2 * r + 4 * s + 2 * n + 4)
            + 2 * nv * ng * (m + r + s + n)
        )
        if bound > max_work_bytes:
            raise MemoryError(
                f"complete cell-pair contraction needs {bound} bytes; cap={max_work_bytes}"
            )
        self._arrays = tuple(_owned(a) for a in arrays)
        self._batch = batch
        self._edges = (features.edge_left, features.edge_right, edges)
        outer = features.outer_indices.transpose(1, 0, 2).reshape(nv, 2 * ng)
        wr, wf = (
            np.tile(w[:, None, :], (1, 2, 1)).reshape(nv, 2 * ng)
            for w in (rising, falling)
        )
        a, b, c, d = self._arrays

        def moment(left, right, weights):
            return np.einsum(
                "cgm,cgr,cg->cmr", left[outer], right[outer], weights, optimize=True
            )

        self._left_rise = _owned(moment(a, b, wr))
        self._left_fall = _owned(moment(a, b, wf))
        right_rise, right_fall = moment(d, c, wr), moment(d, c, wf)
        prefix = np.zeros_like(right_rise)
        suffix = np.zeros_like(right_fall)
        prefix[1:] = np.cumsum(right_rise[:-1], axis=0)
        suffix[:-1] = np.cumsum(right_fall[:0:-1], axis=0)[::-1]
        self._right_prefix = _owned(prefix)
        self._right_suffix = _owned(suffix)
        self._density_shape = (r, s)
        self._output_shape = (m, n)
        h = hashlib.sha256()
        for a in self._arrays:
            h.update(repr((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        self.identity = MappingProxyType(
            dict(
                representation="exact cell pair-density matrix contraction",
                feature_geometry_sha256=features.identity["geometry_sha256"],
                operand_sha256=h.hexdigest(),
                rank=int(rank),
                density_shape=(r, s),
                output_shape=(m, n),
                edge_batch_size=batch,
                complete_work_bound_bytes=bound,
                max_work_bytes=int(max_work_bytes),
                operator="weighted native Coulomb C-product; no amplitude threshold",
            )
        )

    def __call__(self, density):
        rho = np.asarray(density, dtype=np.result_type(density, float))
        if rho.shape != self._density_shape or not np.all(np.isfinite(rho)):
            raise ValueError("finite complete source density matrix required")
        # Only completed cells enter prefix/suffix; same-cell contributions
        # use the original triangular edges below, including both directions.
        result = np.einsum(
            "cmr,rs,cns->mn", self._left_rise, rho, self._right_suffix, optimize=True
        )
        result += np.einsum(
            "cmr,rs,cns->mn", self._left_fall, rho, self._right_prefix, optimize=True
        )
        a, b, c, d = self._arrays
        left, right, weights = self._edges
        for start in range(0, len(weights), self._batch):
            i = left[start : start + self._batch]
            j = right[start : start + self._batch]
            w = weights[start : start + self._batch]
            scalar = np.einsum("es,es->e", b[i] @ rho, c[j])
            result += a[i].T @ (d[j] * (w * scalar)[:, None])
            scalar = np.einsum("es,es->e", b[j] @ rho, c[i])
            result += a[j].T @ (d[i] * (w * scalar)[:, None])
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite native cell-pair matrix action")
        return result
