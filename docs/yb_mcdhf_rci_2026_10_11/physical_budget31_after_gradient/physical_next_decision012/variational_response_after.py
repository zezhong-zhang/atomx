"""Real isolated-eigenstate response and derivatives of a polar orbital path.

The array caller owns state/basis/Hamiltonian/metric and parameter units.
These routines neither choose physical states nor create CI result ownership.
The third-order response uses the variational 2n+1 identity (Gonze, PRA52,
1086, 1995). Derivatives here are derivatives, not Taylor coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def polar_path_derivatives(point, direction):
    """Return Y(0), Y'(0), Y''(0) for polar(point+t*direction).

    Differentiate the positive matrix square root via two Sylvester equations;
    no finite differences or assumption of exactly unit Gram/exact tangency.
    Small positive Gram matrices, including repeated eigenvalues, are supported.
    The caller must keep its radial metric/projector when unpacking the arrays.
    """
    y, d = np.asarray(point), np.asarray(direction)
    if (y.ndim != 2 or y.shape != d.shape or not y.size or y.shape[0] < y.shape[1]
            or np.iscomplexobj(y) or np.iscomplexobj(d)
            or not np.isfinite(y).all() or not np.isfinite(d).all()):
        raise ValueError('polar path requires matching finite real full-column arrays')
    wide = np.longdouble
    y, d = y.astype(wide), d.astype(wide)
    s = y.T @ y
    values, q = np.linalg.eigh(np.asarray(s, dtype=float))
    if values[0] <= 0:
        raise ValueError('polar path requires positive definite Gram')
    roots = np.sqrt(values.astype(wide))
    q = q.astype(wide)
    # Keep the common identity out of the reconstruction near repeated unit roots.
    inverse = np.eye(len(roots), dtype=wide) + (q * (1 / roots - 1)) @ q.T
    s1, s2 = y.T @ d + d.T @ y, 2 * d.T @ d

    def sylvester(rhs):
        return q @ ((q.T @ rhs @ q) / (roots[:, None] + roots[None, :])) @ q.T

    r1 = sylvester(s1)
    r2 = sylvester(s2 - 2 * r1 @ r1)
    f1 = -inverse @ r1 @ inverse
    f2 = 2 * inverse @ r1 @ inverse @ r1 @ inverse - inverse @ r2 @ inverse
    return tuple(np.asarray(v, dtype=float) for v in
                 (y @ inverse, d @ inverse + y @ f1, 2 * d @ f1 + y @ f2))


@dataclass(frozen=True)
class RealEigenstateResponse:
    """Local real normalized response; no omitted-spectrum or orbital certificate.

    ``ci_third_coefficient`` is E'''/6 minus <c|H'''|c>/6. It predicts the
    leading h^2 *coefficient-response* part of a central FD, not its whole
    truncation error. H', H'' must be independent derivatives of the same H path.
    """

    first: float
    second: float
    ci_third_coefficient: float
    coefficient_derivative: np.ndarray
    nearest_gap: float | None
    response_residual: float
    gauge_defect: float


def real_eigenstate_response(hamiltonian, first_derivative, second_derivative,
                            energies, vectors, root, *, max_dimension=2048,
                            residual_tolerance=1e-8, gram_tolerance=1e-10,
                            gap_tolerance=1e-10):
    """Complete-spectrum first response and variational second/third actions.

    Real symmetric ordinary-metric arrays only; a selected/partial spectrum or
    degenerate target rejects. Complete eigenvectors are checked against H.
    All external states and signed off-diagonal matrix elements are retained.
    This dense diagnostic is not a matrix-free response solver for large CI.
    """
    h, h1, h2, e, v = map(np.asarray, (hamiltonian, first_derivative,
                                     second_derivative, energies, vectors))
    if (isinstance(max_dimension, (bool, np.bool_)) or not isinstance(max_dimension, int)
            or max_dimension <= 0 or h.ndim != 2 or h.shape[0] != h.shape[1]
            or not 0 < len(h) <= max_dimension):
        raise ValueError('complete response dimension exceeds the declared dense budget')
    n = len(h)
    if (isinstance(root, (bool, np.bool_)) or not isinstance(root, (int, np.integer))
            or not 0 <= root < n or h1.shape != h.shape or h2.shape != h.shape
            or e.shape != (n,) or v.shape != (n, n)
            or any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (h, h1, h2, e, v))
            or any(not np.array_equal(x, x.T) for x in (h, h1, h2))):
        raise ValueError('response requires complete finite real symmetric operators/eigensystem')
    if any(not np.isfinite(t) or t <= 0 for t in
           (residual_tolerance, gram_tolerance, gap_tolerance)):
        raise ValueError('response tolerances must be finite and positive')
    if (np.max(abs(h @ v - v * e)) > residual_tolerance
            or np.max(abs(v.T @ v - np.eye(n))) > gram_tolerance):
        raise ValueError('response eigensystem does not match complete H and unit metric')
    external = np.arange(n) != root
    gaps = e[root] - e[external]
    if len(gaps) and np.min(abs(gaps)) <= gap_tolerance:
        raise ValueError('rank-one response requires an isolated target; include the degenerate subspace')
    wide = np.longdouble
    c = v[:, root].astype(wide)
    c /= np.sqrt(c @ c)
    h1, h2 = h1.astype(wide), h2.astype(wide)
    action = h1 @ c
    e1 = c @ action
    coefficients = (v[:, external].astype(wide).T @ action) / gaps.astype(wide)
    c1 = v[:, external].astype(wide) @ coefficients
    c1 -= c * (c @ c1)
    e2 = c @ h2 @ c + 2 * c1 @ action
    third = c1 @ h2 @ c + c1 @ (h1 @ c1 - e1 * c1)
    centered = h.astype(wide)
    centered.flat[::n+1] -= wide(e[root])
    defect = centered @ c1 + action - e1 * c
    response_residual = float(np.linalg.norm(defect))
    if response_residual > residual_tolerance:
        raise ValueError('projected response equation failed against the original complete H')
    result = np.frombuffer(np.asarray(c1, dtype=float).tobytes(), dtype=float)
    return RealEigenstateResponse(float(e1), float(e2), float(third), result,
                                 float(np.min(abs(gaps))) if len(gaps) else None,
                                 response_residual, float(abs(c @ c1)))
