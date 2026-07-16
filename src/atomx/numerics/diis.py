"""Pulay DIIS extrapolation (method-agnostic convergence acceleration).

Generic direct-inversion-in-the-iterative-subspace extrapolator shared
by every method that runs a self-consistent loop (Dirac-Fock, future
HF-in-bspline).  Pure linear algebra on iterate/error histories — no
knowledge of the equation being solved.

The outer SCF loop itself still lives inside
:class:`~atomx.methods.dirac_fock.DFSolver`.  Extracting that loop
into a pure function is tracked as a follow-up to the
physics/numerics reorg (it needs a small Fock-build callback
interface and is not blocking the reorganisation).
"""

from __future__ import annotations

import numpy as np


#: Threshold on DIIS coefficient magnitude beyond which extrapolation
#: is rejected as numerically unstable.
_DIIS_COEFFICIENT_CUTOFF: float = 5.0

#: Tikhonov regularisation strength for the DIIS B matrix diagonal,
#: expressed as a fraction of the mean |diag(B)|.
_DIIS_TIKHONOV_FRACTION: float = 1e-12


def diis_extrapolate(orb_history, err_history):
    """Pulay DIIS extrapolation of orbital vectors.

    Solves the DIIS linear system to find optimal mixing coefficients
    that minimise the error-vector norm.

    Parameters
    ----------
    orb_history
        List of past orbital vectors (each a 1-D array of the same
        length).
    err_history
        List of past error vectors (orbital changes at each iteration),
        same length as ``orb_history``.

    Returns
    -------
    np.ndarray or None
        Extrapolated orbital vector, or ``None`` if DIIS fails
        (singular B matrix or coefficients that exceed
        :data:`_DIIS_COEFFICIENT_CUTOFF`).
    """
    n = len(err_history)
    if n < 2:
        return None

    # DIIS B matrix:  B[i, j] = <e_i | e_j>.
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(i, n):
            B[i, j] = np.dot(err_history[i], err_history[j])
            B[j, i] = B[i, j]

    # Tikhonov regularisation for numerical stability.
    diag_mean = np.mean(np.abs(np.diag(B[:n, :n])))
    if diag_mean > 0:
        for i in range(n):
            B[i, i] += _DIIS_TIKHONOV_FRACTION * diag_mean

    # Lagrange constraint row/column enforcing Σ c_i = 1.
    B[:n, n] = -1.0
    B[n, :n] = -1.0
    B[n, n] = 0.0

    rhs = np.zeros(n + 1)
    rhs[n] = -1.0

    try:
        c = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return None

    coeffs = c[:n]
    if np.any(np.abs(coeffs) > _DIIS_COEFFICIENT_CUTOFF):
        return None

    result = np.zeros_like(orb_history[0])
    for i in range(n):
        result += coeffs[i] * orb_history[i]
    return result


__all__ = ["diis_extrapolate"]
