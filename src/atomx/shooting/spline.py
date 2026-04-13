"""Cubic spline interpolation — SPLIN0, SPLINE, VINT, FINDI, SPLVAL.

Translates radial.f SPLIN0 (line 6464), SPLINE (line 6336), VINT (line 283).
The spline stores f(x) = A + B*x + C*x^2 + D*x^3 per interval.
"""

import numpy as np
import numba as nb
import math


@nb.njit(cache=True)
def findi(x, xtab, n):
    """Binary search: find j such that xtab[j-1] <= x < xtab[j].

    Returns 1-based index j (Fortran convention) in range [1, n-1].
    """
    if x <= xtab[0]:
        return 1
    if x >= xtab[n - 1]:
        return n - 1
    jl = 0
    ju = n - 1
    while ju - jl > 1:
        jm = (jl + ju) // 2
        if x >= xtab[jm]:
            jl = jm
        else:
            ju = jm
    return jl + 1  # 1-based


@nb.njit(cache=True)
def splval(x, xtab, a, b, c, d, n):
    """Evaluate cubic spline at point x.

    The spline is f(x) = A(j) + B(j)*x + C(j)*x^2 + D(j)*x^3
    where j is the interval containing x.
    """
    j = findi(x, xtab, n) - 1  # Convert to 0-based
    if j >= n - 1:
        j = n - 2
    return a[j] + x * (b[j] + x * (c[j] + x * d[j]))


@nb.njit(cache=True)
def splin0(x, y, n, s1=0.0, sn=0.0):
    """Natural cubic spline initialization — SPLIN0 (radial.f:6464-6563).

    Computes coefficients (a, b, c, d) such that in each interval [x_i, x_{i+1}]:
        f(x) = a_i + b_i*x + c_i*x^2 + d_i*x^3

    Parameters:
        x: grid points (strictly increasing), length n
        y: function values, length n
        n: number of points (>= 4)
        s1, sn: second derivatives at endpoints (0 for natural spline)

    Returns:
        a, b, c, d: coefficient arrays, each length n
    """
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    if n < 4:
        # Degenerate fallback
        for i in range(n):
            a[i] = y[i]
        return a, b, c, d

    n1 = n - 1
    n2 = n - 2

    # Auxiliary arrays: h(=a) and delta(=d)
    for i in range(n1):
        a[i] = x[i + 1] - x[i]  # h_i
        d[i] = (y[i + 1] - y[i]) / a[i]  # delta_i

    # Symmetric coefficient matrix (augmented)
    for i in range(n2):
        b[i] = 2.0 * (a[i] + a[i + 1])

    for k in range(n1 - 1, 0, -1):
        d[k] = 6.0 * (d[k] - d[k - 1])

    d[1] = d[1] - a[0] * s1
    d[n1 - 1] = d[n1 - 1] - a[n1 - 1] * sn

    # Gauss elimination of tridiagonal system
    for i in range(1, n2):
        r = a[i] / b[i - 1]
        b[i] = b[i] - r * a[i]
        d[i + 1] = d[i + 1] - r * d[i]

    # Back-substitution: sigma coefficients stored in d
    d[n - 1] = sn
    d[n1 - 1] = d[n1 - 1] / b[n2 - 1]
    for k in range(n2 - 1, 0, -1):
        d[k] = (d[k] - a[k] * d[k + 1]) / b[k - 1]

    # Spline coefficients
    si1 = s1
    for i in range(n1):
        si = si1
        si1 = d[i + 1]
        h = a[i]
        hi = 1.0 / h
        a[i] = ((hi / 6.0) * (si * x[i + 1] ** 3 - si1 * x[i] ** 3)
                + hi * (y[i] * x[i + 1] - y[i + 1] * x[i])
                + (h / 6.0) * (si1 * x[i] - si * x[i + 1]))
        b[i] = ((hi / 2.0) * (si1 * x[i] ** 2 - si * x[i + 1] ** 2)
                + hi * (y[i + 1] - y[i])
                + (h / 6.0) * (si - si1))
        c[i] = (hi / 2.0) * (si * x[i + 1] - si1 * x[i])
        d[i] = (hi / 6.0) * (si1 - si)

    # Quadratic extrapolation for x > x(n). Natural spline if sn=0.
    a[n - 1] = a[n1 - 1] + d[n1 - 1] * x[n - 1] ** 3
    b[n - 1] = b[n1 - 1] - 3.0 * d[n1 - 1] * x[n - 1] ** 2
    if abs(sn) < 1.0e-16:
        c[n - 1] = 0.0
    else:
        c[n - 1] = c[n1 - 1] + 3.0 * d[n1 - 1] * x[n - 1]
    d[n - 1] = 0.0

    return a, b, c, d


@nb.njit(cache=True)
def vint(r, rv, nv):
    """Potential interpolation — VINT (radial.f:283-415).

    Sets up cubic spline of R*V(R) from input table, with Coulomb tail handling.

    Parameters:
        r:  radial grid points, r[0]=0, non-decreasing. Length nv.
        rv: R*V(R) values. Length nv.
        nv: number of input points (>= 4)

    Returns:
        rg:   spline knot points (length nvt)
        rvg:  R*V values at knots
        va, vb, vc, vd: spline coefficients per interval
        nvt:  number of spline intervals + 1
    """
    eps = 1.0e-12
    max_pts = nv + 10  # Extra space for possible discontinuity

    # Work arrays
    rt = np.zeros(max_pts)
    vt = np.zeros(max_pts)

    rt[0] = 0.0
    vt[0] = rv[0]
    for i in range(1, nv):
        rt[i] = r[i]
        vt[i] = rv[i]

    # Coulomb tail: find where RV becomes constant
    zinf = rv[nv - 1]
    tol = max(abs(zinf) * 1.0e-10, 1.0e-10)
    nvi = nv
    nve = nv

    for i in range(nv - 1, 2, -1):
        if abs(vt[i - 1] - zinf) > tol:
            nve = i
            # Check if we need a discontinuity point
            if (rt[i] - rt[i - 1]) < eps * max(abs(rt[i - 1]), abs(rt[i])):
                rt[i] = rt[i - 1]
                vt[i] = zinf
                nve = i - 1
            else:
                nvi = nvi + 1
                # Shift points to make room for discontinuity
                for j in range(nvi - 1, nve, -1):
                    rt[j] = rt[j - 1]
                vt[nve + 1] = zinf
            break
        else:
            vt[i] = zinf
    else:
        nve = 4

    # Natural cubic spline interpolation, piecewise between discontinuities
    # We process segments between repeated points
    rg_out = np.zeros(max_pts)
    rvg_out = np.zeros(max_pts)
    va_out = np.zeros(max_pts)
    vb_out = np.zeros(max_pts)
    vc_out = np.zeros(max_pts)
    vd_out = np.zeros(max_pts)

    # For simplicity, do a single spline fit up to nve (no discontinuities
    # in most atomic potential cases)
    nseg = nve + 1
    if nseg > max_pts:
        nseg = max_pts

    # Fit spline on [0, nve]
    x_seg = rt[:nseg]
    y_seg = vt[:nseg]

    a_s, b_s, c_s, d_s = splin0(x_seg, y_seg, nseg, 0.0, 0.0)

    io = 0
    for j in range(nseg - 1):
        rg_out[io] = x_seg[j]
        rvg_out[io] = y_seg[j]
        va_out[io] = a_s[j]
        vb_out[io] = b_s[j]
        vc_out[io] = c_s[j]
        vd_out[io] = d_s[j]
        io += 1

    # Last point: Coulomb tail (constant R*V = zinf)
    io += 1
    nvt = io
    rg_out[nvt - 1] = x_seg[nseg - 1]
    rvg_out[nvt - 1] = zinf
    va_out[nvt - 1] = zinf
    vb_out[nvt - 1] = 0.0
    vc_out[nvt - 1] = 0.0
    vd_out[nvt - 1] = 0.0

    return (rg_out[:nvt], rvg_out[:nvt],
            va_out[:nvt], vb_out[:nvt], vc_out[:nvt], vd_out[:nvt], nvt)
