"""B-spline evaluation via Cox-de Boor recursion with Numba acceleration.

Core routines for evaluating B-splines and their derivatives at
Gauss-Legendre quadrature points within each knot interval.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def bsplvb(t, ks, x, left):
    """Evaluate all nonzero B-splines of order ks at point x.

    Uses the Cox-de Boor triangular recursion (de Boor, 1978).

    Parameters
    ----------
    t : ndarray, shape (n_knots,)
        Knot sequence.
    ks : int
        B-spline order (degree + 1).
    x : float
        Evaluation point.
    left : int
        Index such that t[left] <= x < t[left+1].

    Returns
    -------
    b : ndarray, shape (ks,)
        Values of B_{left-ks+1,ks}(x), ..., B_{left,ks}(x).
    """
    b = np.zeros(ks)
    b[0] = 1.0

    dr = np.zeros(ks)
    dl = np.zeros(ks)

    for j in range(1, ks):
        dr[j] = t[left + j] - x
        dl[j] = x - t[left + 1 - j]
        saved = 0.0
        for r in range(j):
            term = b[r] / (dr[r + 1] + dl[j - r])
            b[r] = saved + dr[r + 1] * term
            saved = dl[j - r] * term
        b[j] = saved

    return b


@nb.njit(cache=True)
def bsplvd(t, ks, x, left, nderiv):
    """Evaluate B-splines and their derivatives at point x.

    Parameters
    ----------
    t : ndarray, shape (n_knots,)
        Knot sequence.
    ks : int
        B-spline order.
    x : float
        Evaluation point.
    left : int
        Knot interval index: t[left] <= x < t[left+1].
    nderiv : int
        Number of derivatives (1 = values only, 2 = values + first derivative, ...).

    Returns
    -------
    db : ndarray, shape (ks, nderiv)
        db[i, 0] = B_{left-ks+1+i}(x)
        db[i, 1] = d/dx B_{left-ks+1+i}(x)
        etc.
    """
    db = np.zeros((ks, nderiv))

    # Compute splines of reduced order for derivatives
    # Start from order ks - nderiv + 1 and build up
    mhigh = min(nderiv, ks)
    kstart = ks + 1 - mhigh

    # Base splines of order kstart
    b0 = bsplvb(t, kstart, x, left)
    db[:kstart, 0] = b0[:kstart]

    if mhigh == 1:
        # Only values needed, build up to full order
        # Actually we need full order values
        b_full = bsplvb(t, ks, x, left)
        db[:, 0] = b_full
        return db

    # For derivatives: use the identity
    # d/dx B_{i,k}(x) = (k-1) * [B_{i,k-1}(x)/(t_{i+k-1}-t_i) - B_{i+1,k-1}(x)/(t_{i+k}-t_{i+1})]

    # Get full-order B-spline values
    b_full = bsplvb(t, ks, x, left)
    db[:, 0] = b_full

    # Compute derivatives using finite-difference formula on lower-order splines
    for d in range(1, min(nderiv, ks)):
        kk = ks - d  # order of the lower splines
        b_low = bsplvb(t, kk, x, left)
        # b_low has kk nonzero splines: indices left-kk+1 to left

        deriv = np.zeros(ks)
        for i in range(ks):
            # Global B-spline index: left - ks + 1 + i
            gi = left - ks + 1 + i
            # d^d/dx^d B_{gi,ks}(x) uses B-splines of order kk = ks - d

            # Position of gi in the lower-order array
            # Lower order splines span indices left - kk + 1 to left
            # i.e., gi_low ranges from left - kk + 1 to left

            val = 0.0
            # Using recurrence: d/dx B_{i,k} = (k-1) [B_{i,k-1}/(t_{i+k-1}-t_i) - B_{i+1,k-1}/(t_{i+k}-t_{i+1})]
            # For d-th derivative we apply this d times
            # But simplest to just do first derivative from order ks-1 splines


            deriv[i] = val

        # Use the more robust matrix-based approach
        # d/dx B_{i,k}(x) = (k-1)/(t_{i+k-1}-t_i) * B_{i,k-1}(x)
        #                  - (k-1)/(t_{i+k}-t_{i+1}) * B_{i+1,k-1}(x)
        if d == 1:
            kk = ks - 1
            b_low = bsplvb(t, kk, x, left)
            fac = float(kk)  # = ks - 1
            for i in range(ks):
                gi = left - ks + 1 + i
                # B_{gi,kk}(x) is at position gi - (left - kk + 1) = gi - left + kk - 1
                #                            = i - ks + 1 + kk - 1 + 1 = i - 1
                #   (since kk = ks - 1, so i + kk - ks = i - 1)
                idx_lo = i - 1  # index of B_{gi,kk} in b_low
                idx_hi = i      # index of B_{gi+1,kk} in b_low

                b_lo = 0.0
                if 0 <= idx_lo < kk:
                    denom = t[gi + kk] - t[gi]
                    if abs(denom) > 1e-30:
                        b_lo = b_low[idx_lo] / denom

                b_hi = 0.0
                if 0 <= idx_hi < kk:
                    denom = t[gi + ks] - t[gi + 1]
                    if abs(denom) > 1e-30:
                        b_hi = b_low[idx_hi] / denom

                db[i, 1] = fac * (b_lo - b_hi)

    return db


@nb.njit(cache=True)
def find_span(t, ns, ks, x):
    """Find knot span index: return left such that t[left] <= x < t[left+1].

    For x at the right boundary, returns ns - 1.

    Parameters
    ----------
    t : ndarray
        Knot sequence.
    ns : int
        Number of B-splines.
    ks : int
        B-spline order.
    x : float
        Evaluation point.

    Returns
    -------
    left : int
    """
    if x >= t[ns]:
        return ns - 1
    if x <= t[ks - 1]:
        return ks - 1

    # Binary search
    lo = ks - 1
    hi = ns
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < t[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def gauss_legendre_points(n):
    """Return Gauss-Legendre nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    xi, wi : ndarray, shape (n,)
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    return xi, wi


def setup_gauss_points(t, ns, ks, n_gauss=None):
    """Set up Gauss-Legendre quadrature points and weights in each knot interval.

    Parameters
    ----------
    t : ndarray, shape (ns + ks,)
        Knot sequence.
    ns : int
        Number of B-splines.
    ks : int
        B-spline order.
    n_gauss : int, optional
        Number of Gauss points per interval. Default: ks (exact for polynomials
        of degree 2*ks - 2, sufficient for overlap integrals).

    Returns
    -------
    gr : ndarray, shape (nv, n_gauss)
        Radial coordinates of Gauss points in each interval.
    grw : ndarray, shape (nv, n_gauss)
        Gauss weights (including Jacobian for interval mapping).
    """
    if n_gauss is None:
        n_gauss = ks
    nv = ns - ks + 1  # number of intervals
    xi, wi = gauss_legendre_points(n_gauss)

    gr = np.zeros((nv, n_gauss))
    grw = np.zeros((nv, n_gauss))

    for iv in range(nv):
        a = t[ks + iv - 1]  # knot interval start: t[ks-1+iv] = t[ks+iv-1]
        b = t[ks + iv]      # knot interval end: t[ks+iv]
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        gr[iv, :] = mid + half * xi
        grw[iv, :] = half * wi

    return gr, grw


@nb.njit(cache=True)
def _evaluate_bsplines_batch(t, ks, gr_flat, lefts, n_points, nderiv):
    """Batch-evaluate B-splines at many points. Full Numba kernel."""
    bsp = np.zeros((n_points, ks, nderiv))

    for p in range(n_points):
        x = gr_flat[p]
        left = lefts[p]

        # Values via de Boor recursion
        b = np.zeros(ks)
        b[0] = 1.0
        dr = np.zeros(ks)
        dl = np.zeros(ks)
        for j in range(1, ks):
            dr[j] = t[left + j] - x
            dl[j] = x - t[left + 1 - j]
            saved = 0.0
            for r in range(j):
                term = b[r] / (dr[r + 1] + dl[j - r])
                b[r] = saved + dr[r + 1] * term
                saved = dl[j - r] * term
            b[j] = saved

        bsp[p, :, 0] = b

        if nderiv >= 2:
            # First derivative via finite-difference formula on (ks-1) order splines
            kk = ks - 1
            b_low = np.zeros(kk)
            b_low[0] = 1.0
            dr2 = np.zeros(kk)
            dl2 = np.zeros(kk)
            for j in range(1, kk):
                dr2[j] = t[left + j] - x
                dl2[j] = x - t[left + 1 - j]
                saved = 0.0
                for r in range(j):
                    term = b_low[r] / (dr2[r + 1] + dl2[j - r])
                    b_low[r] = saved + dr2[r + 1] * term
                    saved = dl2[j - r] * term
                b_low[j] = saved

            fac = float(kk)
            for i in range(ks):
                gi = left - ks + 1 + i
                idx_lo = i - 1
                idx_hi = i
                b_lo = 0.0
                if 0 <= idx_lo < kk:
                    denom = t[gi + kk] - t[gi]
                    if abs(denom) > 1e-30:
                        b_lo = b_low[idx_lo] / denom
                b_hi = 0.0
                if 0 <= idx_hi < kk:
                    denom = t[gi + ks] - t[gi + 1]
                    if abs(denom) > 1e-30:
                        b_hi = b_low[idx_hi] / denom
                bsp[p, i, 1] = fac * (b_lo - b_hi)

    return bsp


def evaluate_bsplines_on_grid(t, ns, ks, gr, nderiv=1):
    """Evaluate B-splines (and derivatives) at all Gauss points — Numba batch.

    Parameters
    ----------
    t : ndarray, shape (ns + ks,)
    ns, ks : int
    gr : ndarray, shape (nv, n_gauss)
    nderiv : int  (1=values, 2=values+derivatives)

    Returns
    -------
    bsp : ndarray, shape (nv, n_gauss, ks, nderiv)
    """
    nv, n_gauss = gr.shape
    n_points = nv * n_gauss
    gr_flat = gr.ravel()

    # Pre-compute left indices for each point
    lefts = np.empty(n_points, dtype=np.int64)
    for iv in range(nv):
        left = ks - 1 + iv
        for m in range(n_gauss):
            lefts[iv * n_gauss + m] = left

    bsp_flat = _evaluate_bsplines_batch(t, ks, gr_flat, lefts, n_points, nderiv)
    return bsp_flat.reshape(nv, n_gauss, ks, nderiv)


# ---------------------------------------------------------------------------
# B-spline data preparation (moved from dirac/matrix.py)
# ---------------------------------------------------------------------------

def _build_knot_sequence(t_master, ks_master, ns_master, k_target):
    """Build knot sequence for order k_target from master knot sequence.

    Same interior knots, different boundary multiplicity.
    """
    nv = ns_master - ks_master + 1
    n_target = nv + k_target - 1
    interior = t_master[ks_master:ns_master]  # interior knots

    t_new = np.zeros(n_target + k_target)
    t_new[:k_target] = t_master[0]         # boundary: k_target copies
    t_new[k_target:k_target + len(interior)] = interior
    t_new[n_target:] = t_master[ns_master]  # boundary: k_target copies of r_max
    return t_new, n_target


def prepare_bspline_data(t, ns, ks, n_gauss=None, kinetic_balance=False):
    """Prepare all B-spline data for Dirac matrix assembly.

    Creates the ``bsp_data`` dict — the central data structure that flows
    through the entire calculation pipeline (dirac → scf → mcdhf → transitions).

    Parameters
    ----------
    t : ndarray, shape (ns + ks,)
        Knot sequence (for the full ks order).
    ns : int
        Number of B-splines.
    ks : int
        B-spline order.
    n_gauss : int, optional
        Number of Gauss-Legendre points per interval.
    kinetic_balance : bool
        If True, use ksp=ks-1 for P, ksq=ks for Q (eliminates spurious states).

    Returns
    -------
    dict
        B-spline data with keys: nv, ks, ksp, ksq, nsp, nsq, gr, grw,
        bsp_p, bsp_q, bsp_q_d, bsp_p_d, t, tp, ns.
    """
    nv = ns - ks + 1
    gr, grw = setup_gauss_points(t, ns, ks, n_gauss)

    if kinetic_balance:
        ksp = ks - 1
        ksq = ks
    else:
        ksp = ks
        ksq = ks

    nsp = nv + ksp - 1
    nsq = nv + ksq - 1  # = ns when ksq = ks

    # Evaluate Q-component B-splines (full order ksq = ks)
    bsp_q_full = evaluate_bsplines_on_grid(t, nsq, ksq, gr, nderiv=2)
    bsp_q = bsp_q_full[:, :, :ksq, 0]
    bsp_q_d = bsp_q_full[:, :, :ksq, 1]

    tp = t  # default: same knot sequence for P and Q
    if ksp == ksq:
        # Same order: share B-spline arrays
        bsp_p = bsp_q.copy()
        bsp_p_d = bsp_q_d.copy()
    else:
        # RKB: build separate P-component B-splines on dedicated knot sequence
        tp, nsp_check = _build_knot_sequence(t, ks, ns, ksp)
        assert nsp_check == nsp
        bsp_p_full = evaluate_bsplines_on_grid(tp, nsp, ksp, gr, nderiv=2)
        bsp_p = bsp_p_full[:, :, :ksp, 0]
        bsp_p_d = bsp_p_full[:, :, :ksp, 1]

    ns_full = max(nsp, nsq)

    return {
        "nv": nv,
        "ks": ks,
        "ksp": ksp,
        "ksq": ksq,
        "nsp": nsp,
        "nsq": nsq,
        "gr": gr,
        "grw": grw,
        "bsp_p": bsp_p,
        "bsp_q": bsp_q,
        "bsp_q_d": bsp_q_d,
        "bsp_p_d": bsp_p_d,
        "t": t,
        "tp": tp,      # P-component knot sequence (differs from t when KB active)
        "ns": ns_full,  # max(nsp, nsq) for backward compatibility
    }
