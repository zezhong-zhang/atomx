"""Sparse auxiliary-node action of the native cell Coulomb quadrature.

This is a factorization of the existing convolution integral, not a new
quadrature or a fitted density basis. Outer Gauss nodes and all original
triangular subnodes are retained. Dirac components have separate tags:
their same-cell quadrature orders differ in a kinetic-balance workspace.
Only exact cross-cell semiseparability and same-cell edges are used.
"""

import hashlib
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType

import numba as nb
import numpy as np

from atomx.numerics.bspline.basis import evaluate_bsplines_on_grid
from atomx.numerics.bspline.convol import _gauss_legendre_01
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

_import_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _owned(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@nb.njit(cache=True)
def _apply(
    rho, outer, rising_weights, falling_weights, edge_left, edge_right, edge_weights
):
    out = np.zeros_like(rho)
    _, nv, ng = outer.shape
    rising = np.zeros(nv, dtype=rho.dtype)
    falling = np.zeros(nv, dtype=rho.dtype)
    for cell in range(nv):
        for component in range(2):
            for m in range(ng):
                index = outer[component, cell, m]
                rising[cell] += rising_weights[cell, m] * rho[index]
                falling[cell] += falling_weights[cell, m] * rho[index]
    # Completed-cell moments; the current cell belongs to the edge action.
    prefix = rho[0] * 0
    for cell in range(nv):
        for component in range(2):
            for m in range(ng):
                out[outer[component, cell, m]] += falling_weights[cell, m] * prefix
        prefix += rising[cell]
    suffix = rho[0] * 0
    for cell in range(nv - 1, -1, -1):
        for component in range(2):
            for m in range(ng):
                out[outer[component, cell, m]] += rising_weights[cell, m] * suffix
        suffix += falling[cell]
    for edge in range(len(edge_left)):
        i, j, weight = edge_left[edge], edge_right[edge], edge_weights[edge]
        out[i] += weight * rho[j]
        out[j] += weight * rho[i]
    return out


class CellCoulombFeatures:
    """Exact sparse radial action on the native B-spline quadrature samples.

    ``sample(P,Q)`` embeds both components in distinct auxiliary nodes.
    ``dot(sample(a)*sample(b), apply_density(sample(c)*sample(d), k))``
    equals the native scalar ``R^k(ab,cd)`` including all four component
    terms. There is no conjugation: complex values use a c-product.
    ``apply_density`` returns a *weighted* action; do not integrate it with
    another set of Gauss weights or interpret it as a pointwise potential.

    The payload cap is checked before constructing sample tables. It covers
    owned arrays and the declared rank-cache capacity, not process/JIT RSS.
    Coefficient vectors follow the padded native Dirac basis convention.
    """

    def __init__(self, workspace, *, memory_bytes=128 * 1024**2, rank_cache_size=8):
        for label, value in (
            ("memory_bytes", memory_bytes),
            ("rank_cache_size", rank_cache_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"positive integer {label} required")
        gr = np.asarray(workspace["gr"], dtype=float)
        weights = np.asarray(workspace["grw"], dtype=float)
        nv = int(workspace["nv"])
        ns = int(workspace["ns"])
        orders = (
            int(workspace.get("ksp", workspace["ks"])),
            int(workspace.get("ksq", workspace["ks"])),
        )
        sizes = (int(workspace.get("nsp", ns)), int(workspace.get("nsq", ns)))
        knots = (
            np.asarray(workspace.get("tp", workspace["t"]), dtype=float),
            np.asarray(workspace["t"], dtype=float),
        )
        if (
            gr.ndim != 2
            or gr.shape[0] != nv
            or weights.shape != gr.shape
            or not np.all(np.isfinite(gr))
            or not np.all(np.isfinite(weights))
            or np.any(gr <= 0)
            or np.any(weights <= 0)
            or min(orders) < 1
        ):
            raise ValueError("finite positive native cell quadrature required")
        breaks = [t[k - 1 : k + nv] for t, k in zip(knots, orders, strict=True)]
        if (
            any(len(b) != nv + 1 or np.any(np.diff(b) <= 0) for b in breaks)
            or not np.array_equal(breaks[0], breaks[1])
            or np.any(gr <= breaks[0][:-1, None])
            or np.any(gr >= breaks[0][1:, None])
        ):
            raise ValueError("P/Q must have identical ordered cell boundaries")
        ng = gr.shape[1]
        suborders = [tuple(sorted({order, max(orders)})) for order in orders]
        widths = tuple(ng * (1 + sum(values)) for values in suborders)
        nodes = nv * sum(widths)
        edges = nv * ng * (sum(orders) + 2 * max(orders))
        # Values tables, radii, outer maps, edge geometry and cached actions.
        estimate = (
            8 * nv * sum(w * k for w, k in zip(widths, orders, strict=True))
            + 8 * nodes
            + 32 * nv * ng
            + 32 * edges
            + rank_cache_size * 8 * (2 * nv * ng + edges)
        )
        if estimate > memory_bytes:
            raise MemoryError(
                f"complete auxiliary Coulomb payload needs {estimate} bytes; cap={memory_bytes}"
            )
        inner_maps, tables = [], []
        outer = np.empty((2, nv, ng), dtype=np.int64)
        offset = 0
        radii = []
        for c in range(2):
            grid = np.empty((nv, widths[c]))
            grid[:, :ng] = gr
            outer[c] = offset + np.arange(nv)[:, None] * widths[c] + np.arange(ng)
            maps = {}
            begin = ng
            for order in suborders[c]:
                x, w = np.empty(order), np.empty(order)
                # Use precisely the native triangular kernel's subnodes.
                _gauss_legendre_01(order, x, w)
                span = gr - breaks[c][:-1, None]
                sub = breaks[c][:-1, None, None] + span[:, :, None] * x
                grid[:, begin : begin + ng * order] = sub.reshape(nv, ng * order)
                indices = (
                    offset
                    + np.arange(nv)[:, None, None] * widths[c]
                    + begin
                    + np.arange(ng * order).reshape(ng, order)
                )
                maps[order] = (indices, span[:, :, None] * w, sub)
                begin += ng * order
            table = evaluate_bsplines_on_grid(
                knots[c], sizes[c], orders[c], grid, nderiv=1
            )[..., 0]
            native = np.asarray(
                workspace["bsp_p"]
                if c == 0
                else workspace.get("bsp_q", workspace["bsp_p"])
            )
            if (
                native.shape != (nv, ng, orders[c])
                or np.max(abs(table[:, :ng] - native)) > 1e-11
            ):
                raise ValueError(
                    "native basis tables do not match declared spline geometry"
                )
            # Preserve original outer-node operands and roundoff exactly.
            table[:, :ng] = native
            tables.append(_owned(table))
            radii.append(grid.ravel())
            inner_maps.append(maps)
            offset += nv * widths[c]
        left, right, base, ratios = [], [], [], []
        for a in range(2):
            for b in range(2):
                order = orders[a] if a == b else max(orders)
                indices, subweights, subradii = inner_maps[b][order]
                left.append(
                    np.broadcast_to(outer[a, :, :, None], indices.shape).ravel()
                )
                right.append(indices.ravel())
                base.append((weights[:, :, None] * subweights / gr[:, :, None]).ravel())
                ratios.append((subradii / gr[:, :, None]).ravel())
        self._tables = tuple(tables)
        self._orders, self._sizes, self._ns = orders, sizes, ns
        self._nv, self._ng = nv, ng
        self._gr, self._weights = _owned(gr), _owned(weights)
        self.radii, self.outer_indices = _owned(np.concatenate(radii)), _owned(outer)
        self.edge_left, self.edge_right = (
            _owned(np.concatenate(left)),
            _owned(np.concatenate(right)),
        )
        self.edge_base, self.edge_ratio = (
            _owned(np.concatenate(base)),
            _owned(np.concatenate(ratios)),
        )
        self._rank_cache, self._rank_capacity = OrderedDict(), int(rank_cache_size)
        h = hashlib.sha256()
        for value in (gr, weights, *knots, *tables):
            h.update(repr((value.shape, value.dtype.str)).encode())
            h.update(value.tobytes())
        self.identity = MappingProxyType(
            dict(
                representation="native cell Coulomb auxiliary nodes",
                geometry_sha256=h.hexdigest(),
                coefficient_sizes=sizes,
                component_orders=orders,
                auxiliary_nodes=nodes,
                same_cell_edges=edges,
                quadrature_nodes=(nv, ng),
                payload_estimate_bytes=estimate,
                rank_cache_capacity=rank_cache_size,
                contraction="weighted bilinear c-product; no new quadrature or screening",
            )
        )

    def sample(self, P, Q):
        """Embed one native Dirac spinor in component-tagged exact nodes."""
        result = []
        for c, coefficients in enumerate((P, Q)):
            values = np.asarray(coefficients)
            if values.shape != (self._ns,) or not np.all(np.isfinite(values)):
                raise ValueError("finite padded native coefficient vector required")
            result.append(
                eval_orbital_on_grid(
                    values, self._tables[c], self._nv, self._orders[c], self._sizes[c]
                ).ravel()
            )
        return np.concatenate(result)

    def rank_arrays(self, rank):
        """Owned weighted moments/edges for one nonnegative integer multipole."""
        if (
            isinstance(rank, bool)
            or not isinstance(rank, (int, np.integer))
            or rank < 0
        ):
            raise ValueError("nonnegative integer multipole required")
        if rank not in self._rank_cache:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                rising = self._weights * self._gr**rank
                falling = self._weights * self._gr ** (-rank - 1)
                edges = self.edge_base * self.edge_ratio**rank
            self._rank_cache[rank] = tuple(_owned(x) for x in (rising, falling, edges))
            while len(self._rank_cache) > self._rank_capacity:
                self._rank_cache.popitem(last=False)
        else:
            self._rank_cache.move_to_end(rank)
        return self._rank_cache[rank]

    def apply_density(self, density, rank):
        """Return weighted K_k density in O(nodes + same-cell edges) work."""
        rho = np.asarray(density, dtype=np.result_type(density, float))
        if rho.shape != self.radii.shape or not np.all(np.isfinite(rho)):
            raise ValueError("finite complete auxiliary-node density required")
        rising, falling, edges = self.rank_arrays(rank)
        return _apply(
            rho,
            self.outer_indices,
            rising,
            falling,
            self.edge_left,
            self.edge_right,
            edges,
        )

    def integral(self, orbitals, rank):
        """R^k(ab,cd) for four (P,Q) coefficient pairs, without conjugation."""
        if len(orbitals) != 4:
            raise ValueError("four native coefficient pairs required")
        a, b, c, d = (self.sample(*pair) for pair in orbitals)
        return np.dot(a * b, self.apply_density(c * d, rank))


def _polynomial_interpolation(points, nodes):
    """Barycentric cardinal polynomials, including exact node coincidences."""
    difference = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(difference, 1.0)
    weights = 1.0 / np.prod(difference, axis=1)
    delta = np.asarray(points)[..., None] - nodes
    exact = delta == 0
    safe = np.where(exact, 1.0, delta)
    cardinal = weights / safe
    cardinal = np.where(
        exact.any(axis=-1, keepdims=True), exact.astype(float), cardinal
    )
    cardinal /= cardinal.sum(axis=-1, keepdims=True)
    return np.where(exact.any(axis=-1, keepdims=True), exact.astype(float), cardinal)


class PolynomialCellCoulombFeatures(CellCoulombFeatures):
    """Exact polynomial-coordinate transform of the native cell quadrature.

    In each cell, every native spinor component is a polynomial of degree
    order-1, and each pair density has degree at most 2*(order-1). Values at
    degree+1 distinct tagged nodes therefore determine the complete density.
    The operator is A.T @ K_native @ A, including all native triangular
    quadrature weights and cross-component terms. New nodes are interpolation
    coordinates, never replacement integration nodes or fitted density data.
    The weighted action and c-product contract match CellCoulombFeatures.
    """

    def __init__(self, workspace, *, memory_bytes=128 * 1024**2, rank_cache_size=8):
        super().__init__(
            workspace, memory_bytes=memory_bytes, rank_cache_size=rank_cache_size
        )
        native_identity = dict(self.identity)
        original_tables = self._tables
        original_radii, original_outer = self.radii, self.outer_indices
        original_left, original_right = self.edge_left, self.edge_right
        m = 2 * max(self._orders) - 1
        nv = self._nv
        nodes = 2 * nv * m
        edges_per_cell = (2 * m) * (2 * m + 1) // 2
        edges = nv * edges_per_cell
        estimate = (
            native_identity["payload_estimate_bytes"]
            # Bound construction maps, cardinal-polynomial scratch and the
            # concatenation/owned copy, rather than just the final map.
            + 32 * len(original_radii) * m
            + 8 * nv * m * sum(self._orders)
            + 8 * nodes
            + 16 * nv * m
            + 16 * edges
            + rank_cache_size * 8 * (2 * nv * m + edges)
            + 24 * nv * (2 * m) ** 2
            + 16 * nv * self._ng * max(self._orders) * m
        )
        if estimate > memory_bytes:
            raise MemoryError(
                f"complete polynomial Coulomb payload needs {estimate} bytes; cap={memory_bytes}"
            )
        x, _ = np.polynomial.legendre.leggauss(m)
        x = (x + 1) / 2
        knots = (
            np.asarray(workspace.get("tp", workspace["t"]), dtype=float),
            np.asarray(workspace["t"], dtype=float),
        )
        breaks = knots[0][self._orders[0] - 1 : self._orders[0] + nv]
        grid = breaks[:-1, None] + np.diff(breaks)[:, None] * x
        maps, tables = [], []
        offset = 0
        for c in range(2):
            width = original_tables[c].shape[1]
            old = original_radii[offset : offset + nv * width].reshape(nv, width)
            local = (old - breaks[:-1, None]) / np.diff(breaks)[:, None]
            maps.append(_polynomial_interpolation(local, x).reshape(nv * width, m))
            tables.append(
                _owned(
                    evaluate_bsplines_on_grid(
                        knots[c], self._sizes[c], self._orders[c], grid, nderiv=1
                    )[..., 0]
                )
            )
            offset += nv * width
        self._interpolation = _owned(np.concatenate(maps))
        self._native_outer = original_outer
        self._native_edge_groups = tuple(
            _owned(
                np.flatnonzero(
                    ((original_left < nv * original_tables[0].shape[1]) == (a == 0))
                    & ((original_right < nv * original_tables[0].shape[1]) == (b == 0))
                ).reshape(nv, -1)
            )
            for a in range(2)
            for b in range(2)
        )
        self._native_edge_left, self._native_edge_right = original_left, original_right
        # Retain the original weights and geometry; they produce the compressed
        # operator at each requested rank without approximating its kernel.
        self._tables = tuple(tables)
        self.radii = _owned(np.concatenate((grid.ravel(), grid.ravel())))
        self.outer_indices = _owned(np.arange(nodes).reshape(2, nv, m))
        left, right = np.triu_indices(2 * m)
        self._local_left, self._local_right = _owned(left), _owned(right)
        self._diagonal_edge = _owned(left == right)
        # Cell-local component coordinates map into globally tagged samples.
        indexed = self.outer_indices.transpose(1, 0, 2).reshape(nv, 2 * m)
        self.edge_left, self.edge_right = (
            _owned(indexed[:, left].ravel()),
            _owned(indexed[:, right].ravel()),
        )
        self._ng = m
        h = hashlib.sha256(native_identity["geometry_sha256"].encode())
        for value in (
            *tables,
            self.radii,
            self._interpolation,
            self.edge_left,
            self.edge_right,
        ):
            h.update(repr((value.shape, value.dtype.str)).encode())
            h.update(value.tobytes())
        self.identity = MappingProxyType(
            dict(
                native_identity,
                representation="native cell Coulomb polynomial density coordinates",
                native_geometry_sha256=native_identity["geometry_sha256"],
                geometry_sha256=h.hexdigest(),
                auxiliary_nodes=nodes,
                same_cell_edges=edges,
                quadrature_nodes=(nv, m),
                payload_estimate_bytes=estimate,
                density_polynomial_degree=m - 1,
                interpolation_nodes_per_component_cell=m,
                contraction="A.T K_native A weighted bilinear c-product; original quadrature retained",
            )
        )

    def rank_arrays(self, rank):
        if (
            isinstance(rank, bool)
            or not isinstance(rank, (int, np.integer))
            or rank < 0
        ):
            raise ValueError("nonnegative integer multipole required")
        if rank not in self._rank_cache:
            m, nv = self._ng, self._nv
            lagrange = self._interpolation[self._native_outer[0]]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                rising = np.einsum(
                    "cem,ce->cm", lagrange, self._weights * self._gr**rank
                )
                falling = np.einsum(
                    "cem,ce->cm", lagrange, self._weights * self._gr ** (-rank - 1)
                )
                native_weights = self.edge_base * self.edge_ratio**rank
            local = np.zeros((nv, 2 * m, 2 * m))
            for (a, b), indices in zip(
                ((0, 0), (0, 1), (1, 0), (1, 1)), self._native_edge_groups, strict=True
            ):
                left = self._interpolation[self._native_edge_left[indices]]
                right = self._interpolation[self._native_edge_right[indices]]
                term = left.transpose(0, 2, 1) @ (
                    native_weights[indices][..., None] * right
                )
                local[:, a * m : (a + 1) * m, b * m : (b + 1) * m] += term
                local[:, b * m : (b + 1) * m, a * m : (a + 1) * m] += term.transpose(
                    0, 2, 1
                )
            compressed = local[:, self._local_left, self._local_right].copy()
            compressed[:, self._diagonal_edge] *= 0.5
            self._rank_cache[rank] = tuple(
                _owned(a) for a in (rising, falling, compressed.ravel())
            )
            while len(self._rank_cache) > self._rank_capacity:
                self._rank_cache.popitem(last=False)
        else:
            self._rank_cache.move_to_end(rank)
        return self._rank_cache[rank]
