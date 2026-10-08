"""Real orthonormal coefficient geometry inside an already fixed projector.

These maps do not construct or change a radial metric/no-pair space. Complex
Hilbert or c-product orbital geometries require a separate explicit contract.
"""

import numpy as np
from scipy.linalg import solve_sylvester


def real_polar_retraction(matrix, *, direction=None):
    """Return the real rectangular polar factor and optional exact velocity.

    For B=Y S, S positive, dY=(I-Y Y.T) K S^-1+Y Omega, with
    S Omega+Omega S=Y.T K-K.T Y. The solve is only orbital-column order.
    The rank gate matches the existing orbital optimizer's polar retraction.
    A supplied direction returns (Y, dY); otherwise only Y is returned.
    """
    b = np.asarray(matrix)
    if (
        b.ndim != 2
        or not b.shape[1]
        or b.shape[0] < b.shape[1]
        or np.iscomplexobj(b)
        or not np.isfinite(b).all()
    ):
        raise ValueError("finite real full-column-rank orbital matrix required")
    b = np.asarray(b, dtype=float)
    k = None
    if direction is not None:
        k = np.asarray(direction)
        if k.shape != b.shape or np.iscomplexobj(k) or not np.isfinite(k).all():
            raise ValueError("finite real polar direction of the same shape required")
        k = np.asarray(k, dtype=float)
    values, vectors = np.linalg.eigh(b.T @ b)
    if values.min() <= 1e-14:
        raise ValueError("rank-deficient orbital polar trial")
    inverse = (vectors / np.sqrt(values)) @ vectors.T
    y = b @ inverse
    if k is None:
        return y
    positive = (vectors * np.sqrt(values)) @ vectors.T
    omega = solve_sylvester(positive, positive, y.T @ k - k.T @ y)
    velocity = (k - y @ (y.T @ k)) @ inverse + y @ omega
    return y, velocity
