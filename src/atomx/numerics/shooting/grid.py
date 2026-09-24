"""SGRID: Salvat's non-uniform radial grid — radial.f lines 7253-7452.

Grid equation: A*R + B*log(R+C) + D = I
Properties: R(1)=0, R(N)=RN, R(2)≈R2, spacing increases monotonically.
"""

import math

import numba as nb
import numpy as np


@nb.njit(cache=True)
def sgrid(rn, r2, drn, n, fast=False):
    """Generate Salvat's non-uniform radial grid.

    Parameters:
        rn:   outer grid point (grid extends from 0 to rn)
        r2:   approximate value of R(2) (controls small-r spacing)
        drn:  R(N)-R(N-1) (controls large-r spacing)
        n:    tentative number of grid points
        fast: use safeguarded local Newton inversion for continuum grids;
              the default retains the original bisection sequence

    Returns:
        r:    radial grid points r(0:n-1), r[0]=0, r[n-1]=rn
        dr:   Jacobian dR/dI at each point
        n_out: actual number of grid points used
    """
    rrn = rn
    rr2 = r2
    if rr2 < 1.0e-8:
        rr2 = 1.0e-8
    if rr2 >= 1.0e-2 * rrn:
        rr2 = 1.0e-2
    if rr2 >= rrn:
        # Degenerate case
        rr2 = 0.01 * rrn

    rdrn = drn
    if rdrn <= rr2:
        rdrn = max(2.0 * (rrn - rr2) / max(n, 10), rr2)

    nr = max(n, 10)
    tst = 5.0 * (rrn - rr2) / nr
    if rdrn > tst:
        rdrn = tst

    # Salvat's original NDIM was 25000; _shooting_points sizes grids up
    # to 300k so that h*p stays ~0.5 for interpolation of the fastest
    # waves (Xe 200 keV: capping at 25k degraded sampling to ~8 pts per
    # wavelength -> percent-level interp wiggle on interp_grid consumers).
    nmax = 400000
    nlow = int(min(rrn / rdrn + 10.0, float(nmax)))
    if nr < nlow:
        nr = nlow
    if nr > nmax:
        nr = nmax

    nhigh_f = 0.5 * ((rrn / rr2) + (rrn / rdrn)) - 20.0
    nhigh = min(int(nhigh_f), nmax)

    if nr > nhigh:
        # Try linear grid
        a_lin = (nr - 1) / rrn
        aa = 1.0 / a_lin
        if aa < rdrn and aa < rr2:
            # Linear grid
            r = np.zeros(nr)
            dr_arr = np.zeros(nr)
            r[0] = 0.0
            dr_arr[0] = aa
            for i in range(1, nr):
                r[i] = i * aa
                dr_arr[i] = aa
            r[nr - 1] = rrn
            return r, dr_arr, nr
        # Fallback: increase nr
        nr = nhigh

    # Grid parameters (ME-8.6 to ME-8.10)
    ag = (rrn - (nr - 1) * rr2) * rdrn / (rrn * (rdrn - rr2))

    if ag > 1.0 or ag < 0.5002:
        # Try linear grid
        a_lin = (nr - 1) / rrn
        aa = 1.0 / a_lin
        if aa < rdrn and aa < rr2:
            r = np.zeros(nr)
            dr_arr = np.zeros(nr)
            r[0] = 0.0
            dr_arr[0] = aa
            for i in range(1, nr):
                r[i] = i * aa
                dr_arr[i] = aa
            r[nr - 1] = rrn
            return r, dr_arr, nr
        # Should not happen for reasonable inputs
        nr = max(nr, int(rrn / rr2) + 10)
        ag = (rrn - (nr - 1) * rr2) * rdrn / (rrn * (rdrn - rr2))

    # Bisection to find x such that f(x) = (1+x)*(1-x*log((1+x)/x)) = ag
    xl = 0.0
    xu = 1000.0
    for _ in range(200):
        x = 0.5 * (xl + xu)
        f = (1.0 + x) * (1.0 - x * math.log((1.0 + x) / x))
        if f > ag:
            xl = x
        else:
            xu = x
        if xu - xl <= 1.0e-15:
            break

    c = x * rrn
    b = x * (c + rrn) * (rdrn - rr2) / (rdrn * rr2)
    a = (c - b * rr2) / (c * rr2)
    d = 1.0 - b * math.log(c)

    r = np.zeros(nr)
    dr_arr = np.zeros(nr)

    r[0] = 0.0
    dr_arr[0] = (r[0] + c) / (a * (r[0] + c) + b)

    rr = 1.0e-35
    for i in range(1, nr):
        target = float(i + 1)  # Fortran uses 1-based index
        if fast:
            # The previous root plus its local Jacobian is already very
            # close to the next root. Newton needs only a few iterations,
            # while independent bisection takes ~50 per grid point.
            rl = r[i - 1]
            ru = rrn
            rr = min(ru, rl + dr_arr[i - 1])
            for _ in range(16):
                fr = a * rr + b * math.log(rr + c) + d - target
                if abs(fr) <= 2.0e-13:
                    break
                if fr > 0.0:
                    ru = rr
                else:
                    rl = rr
                trial = rr - fr / (a + b / (rr + c))
                rr = 0.5 * (rl + ru) if trial <= rl or trial >= ru else trial
            # Pathological near-linear parameters may need more iterations;
            # keep the original monotone bracket as the accuracy fallback.
            fr = a * rr + b * math.log(rr + c) + d - target
            if abs(fr) > 2.0e-12:
                for _ in range(100):
                    rr = 0.5 * (rl + ru)
                    fr = a * rr + b * math.log(rr + c) + d - target
                    if fr > 0.0:
                        ru = rr
                    else:
                        rl = rr
                    if abs(fr) <= 2.0e-13:
                        break
        else:
            rl = rr
            ru = rrn
            # Original Salvat/Fortran bisection for structure grids.
            for _ in range(200):
                rr = 0.5 * (ru + rl)
                fr = a * rr + b * math.log(rr + c) + d - target
                if fr > 0.0:
                    ru = rr
                else:
                    rl = rr
                if ru - rl <= 1.0e-15 * rr:
                    break
        r[i] = rr
        dr_arr[i] = (rr + c) / (a * (rr + c) + b)

    r[nr - 1] = rrn
    return r, dr_arr, nr
