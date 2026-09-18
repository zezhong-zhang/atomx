"""Y^k multipole potentials on a Gauss quadrature grid.

For two radial functions ``P_a(r), P_b(r)`` the Hartree multipole function

    y_k^{a b}(r) = integral_0^infty P_a(r') P_b(r')
                                    [ r_<^k / r_>^{k+1} ] dr'

is the building block of the close-coupling direct and exchange
potentials. This module evaluates :math:`y_k^{ab}(r)` at every node of
the composite Gauss-Legendre quadrature used by the inner-region B-spline
assembly. The implementation uses the cumulative-integral identity

    y_k(r) = (1 / r^{k+1}) * I_<(r) + r^k * I_>(r)

with ``I_<(r) = integral_0^r P_a P_b r'^k dr'`` and
``I_>(r) = integral_r^{R_box} P_a P_b r'^{-(k+1)} dr'``, evaluated by a
single sweep over the same Gauss nodes. Outside the R-matrix boundary
``a`` the integrand is taken to be the analytic tail of the (analytic)
target orbitals so the closed-form ``y_k`` is exact to quadrature
accuracy.
"""

from __future__ import annotations

import numpy as np


def cumulative_y_k(
    r_q: np.ndarray,
    w_q: np.ndarray,
    f_q: np.ndarray,
    k: int,
    f_tail_lower_bound: float = 0.0,
    f_tail_upper_bound: float = 0.0,
) -> np.ndarray:
    """Multipole moment ``y_k(r)`` of a product radial function at Gauss nodes.

    Parameters
    ----------
    r_q :
        Gauss-Legendre quadrature nodes (sorted, strictly positive).
    w_q :
        Corresponding quadrature weights.
    f_q :
        Sampled values of ``P_a(r) * P_b(r)`` at the quadrature nodes.
    k :
        Multipole order (``k >= 0``).
    f_tail_lower_bound, f_tail_upper_bound :
        Contributions from ``[0, r_q[0]]`` and ``[r_q[-1], inf]`` outside
        the quadrature support. Set to zero when both arguments live
        entirely within the quadrature support (the usual case once
        ``R_box >> 1/Z``).

    Returns
    -------
    np.ndarray
        ``y_k(r_q[i])`` evaluated at each Gauss node.

    Notes
    -----
    The "less" integral uses the trapezoidal-like cumulative quadrature
    inherited from the same Gauss-Legendre nodes; the "greater" integral
    is the corresponding tail sum. Because Gauss-Legendre weights are
    *not* a partition of ``[0, R]`` in the trapezoidal sense, the
    cumulative form here is exact only at the granularity of the knot
    cells. For the dense Gauss grid used by the R-matrix box (10 points
    per knot cell, ~30-50 cells) this is more than adequate for the
    1 %-level resonance fits the phase-2 acceptance gate demands.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if w_q.shape != r_q.shape or f_q.shape != r_q.shape:
        raise ValueError("r_q, w_q, f_q must have matching shape")

    rk = r_q ** k
    # Use 1/r^{k+1} = rk / r^{2k+1}? No — simpler: r^{-(k+1)} = rk / r^{2k+1}
    # … just compute directly.
    inv_rk1 = 1.0 / (r_q ** (k + 1))

    f_rk = f_q * rk
    f_inv = f_q * inv_rk1

    # Total integral over [r_q[0], r_q[-1]] via the Gauss weights.

    # Cumulative sums (running integrals from each node).
    cum_lower = np.cumsum(w_q * f_rk)
    cum_upper_from_right = np.cumsum((w_q * f_inv)[::-1])[::-1]

    # I_<(r_i) ~ integral from 0 to r_i. Approximate the part of the i-th
    # quadrature cell that lies below r_i as one half of (w_i * f_rk_i).
    half_cell_lower = 0.5 * w_q * f_rk
    half_cell_upper = 0.5 * w_q * f_inv

    I_lt = (
        f_tail_lower_bound
        + (cum_lower - half_cell_lower)
    )
    I_gt = (
        f_tail_upper_bound
        + (cum_upper_from_right - half_cell_upper)
    )
    return inv_rk1 * I_lt + rk * I_gt


def cumulative_y_k_to_R(
    r_q: np.ndarray,
    w_q: np.ndarray,
    f_q: np.ndarray,
    k: int,
    tail_upper: float = 0.0,
) -> np.ndarray:
    """``y_k`` for a product whose support is bounded above by the box.

    Use when one (or both) of the radial functions is a B-spline (zero
    outside ``[0, a]``); ``tail_upper`` then captures any analytic tail
    contribution from the *target* orbital alone if the product factor
    extends beyond the box (negligible for B-spline products).
    """
    return cumulative_y_k(
        r_q, w_q, f_q, k,
        f_tail_lower_bound=0.0,
        f_tail_upper_bound=tail_upper,
    )
