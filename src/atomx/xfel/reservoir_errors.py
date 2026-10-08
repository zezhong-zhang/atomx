"""Conservative generator bounds for signed Lindblad operator substitutions.

These compare explicitly defined Markov models with fixed bath membership.
They do not certify microscopic bath memory, secularization or omitted shifts.
"""

import numpy as np
from scipy.sparse import bmat, csc_matrix


def lindblad_difference_bound(reference, approximate):
    """Bound ||D(C)-D(B)||_(1->1) by 2||C-B|| (||C||+||B||).

    Frobenius norms upper-bound spectral norms, retaining all signed paths.
    Rectangular matrices are allowed: embedding their disjoint initial/final
    sectors in the joint square basis leaves these norms unchanged.
    Units are the operator's squared units (s^-1 or cm² for PI).
    """
    c, b = np.asarray(reference, complex), np.asarray(approximate, complex)
    if (
        c.shape != b.shape
        or c.ndim != 2
        or not np.isfinite(c).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("matching finite signed bath matrices required")
    eta = 2 * np.linalg.norm(c - b) * (np.linalg.norm(c) + np.linalg.norm(b))
    if not np.isfinite(eta):
        raise ValueError("nonfinite Lindblad generator difference bound")
    return float(eta)


def real_residual_jacobian(generator, difference, state, *, dimension, eta_fs):
    """Real Jacobian for d rho=L rho and d e=eta(e+1)+sqrt(n)||Delta rho||F.

    The norm is nonholomorphic, so a complex stiff Jacobian is inappropriate.
    At an exactly zero residual use the zero norm subgradient; no smoothing
    or cutoff changes the residual used by the ODE.
    """
    matrix = csc_matrix(generator)
    error = csc_matrix(difference)
    residual = error @ state
    norm = np.linalg.norm(residual)
    gradient = (
        np.sqrt(dimension) * (error.conj().T @ residual) / norm
        if norm
        else np.zeros(len(state), complex)
    )
    real_linear = bmat(
        [[matrix.real, -matrix.imag], [matrix.imag, matrix.real]], format="csc"
    )
    row = csc_matrix(np.r_[gradient.real, gradient.imag][None, :])
    return bmat([[real_linear, None], [row, csc_matrix([[eta_fs]])]], format="csc")
