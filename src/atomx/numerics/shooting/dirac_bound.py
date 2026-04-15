"""DBOUND: Dirac bound-state eigenvalue solver — radial.f lines 781-1110.

Solves the radial Dirac equation for bound states using shooting method
with power-series integration and eigenvalue refinement from Q-discontinuity.
"""

import numpy as np
import numba as nb
import math
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.spline import findi, splin0, splval
from atomx.numerics.shooting.dirac_integrator import doutw, dinw


@nb.njit(cache=True)
def _merge_grids(rg, nvt, rad, ngp, eps_tol):
    """Merge potential grid (rg) with user grid (rad), sort, return (r, ind, nrt)."""
    max_pts = nvt + ngp
    r = np.zeros(max_pts)
    ind = np.zeros(max_pts, dtype=nb.int64)

    # Copy potential grid
    for i in range(nvt):
        r[i] = rg[i]
        ind[i] = i
    nrt = nvt

    # Add user grid points not already present
    for i in range(ngp):
        rloc = rad[i]
        found = False
        for j in range(nrt):
            if abs(rloc - r[j]) < eps_tol:
                found = True
                break
        if not found:
            r[nrt] = rloc
            # Find spline interval
            j_idx = 0
            for j in range(nvt - 1):
                if rg[j] <= rloc:
                    j_idx = j
            ind[nrt] = j_idx
            nrt += 1

    # Sort by r value (insertion sort for stability)
    for i in range(1, nrt):
        key_r = r[i]
        key_ind = ind[i]
        j = i - 1
        while j >= 0 and r[j] > key_r:
            r[j + 1] = r[j]
            ind[j + 1] = ind[j]
            j -= 1
        r[j + 1] = key_r
        ind[j + 1] = key_ind

    return r, ind, nrt


@nb.njit(cache=True)
def dbound(e_in, eps, n, k, rg, rvg, va, vb, vc, vd, nvt, rad, ngp):
    """Solve Dirac radial equation for bound state.

    Parameters:
        e_in: initial energy estimate
        eps:  global tolerance (> 1e-15)
        n:    principal quantum number
        k:    relativistic angular momentum quantum number kappa
        rg, rvg, va, vb, vc, vd: spline data from vint/splin0
        nvt:  number of spline intervals
        rad:  user radial grid (0-based)
        ngp:  number of user grid points

    Returns:
        e:    converged eigenvalue
        ier:  error code (0 = success)
        p:    large component on user grid (length ngp)
        q:    small component on user grid (length ngp)
        ilast: index of practical infinity on user grid
    """
    ier = 0

    if k < 0:
        l = -k - 1
    else:
        l = k
    nr = n - l - 1  # Radial quantum number
    ak = float(k)

    dell = 10.0 * eps
    e = e_in
    if e > -1.0e-1:
        e = -1.0e-1
    fl1 = 0.5 * l * (l + 1)

    # Merge grids
    eps_tol = max(0.5 * eps, 1.0e-10)
    r, ind, nrt = _merge_grids(rg, nvt, rad, ngp, eps_tol)

    # Allocate working arrays
    pt = np.zeros(nrt)
    qt = np.zeros(nrt)

    # Minimum of effective radial potential (ME-5.1)
    emin = 1.0
    for i in range(1, nrt):
        rn = r[i]
        j = ind[i]
        rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
        emin = min(emin, (rvn + fl1 / rn) / rn)

    if emin > -1.0e-35:
        ier = 1
        p_out = np.zeros(ngp)
        q_out = np.zeros(ngp)
        return e, ier, p_out, q_out, 0

    rn = r[nrt - 1]
    j = ind[nrt - 1]
    rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
    esup = (rvn + fl1 / rn) / rn
    if esup > 0.0:
        esup = 0.0

    if l == 0:
        tehydr = -(rvg[0] / n) ** 2 if abs(rvg[0]) > 1.0e-30 else -10.0
        emin = min(e, tehydr, -10.0)

    if emin < -SL * SL:
        emin = -SL * SL
    if e > esup or e < emin:
        e = 0.5 * (esup + emin)

    emax = esup
    icmin = 0
    icmax = 0
    isum = 0

    # Shooting loop
    max_shots = 200
    for shot in range(max_shots):
        if e > -1.0e-16:
            ier = 2
            break

        # Outer turning point (ME-5.2) — scan from outside in
        iotp = nrt - 1
        for jj in range(1, nrt):
            idx = nrt - 1 - jj
            rn = r[idx]
            if rn < 1.0e-30:
                continue
            j_ind = ind[idx]
            rvn = va[j_ind] + rn * (vb[j_ind] + rn * (vc[j_ind] + rn * vd[j_ind]))
            ekin = e - (rvn + fl1 / rn) / rn
            if ekin > 0.0:
                iotp = idx
                break

        iotp = iotp + 1  # One past the turning point
        if iotp > nrt - 2:
            ier = 3
            break
        if iotp < 2:
            iotp = 2

        # Outward solution
        nzero, sumout, iotp_upd, nzmax = doutw(
            e, eps, ak, nr, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)
        iotp = iotp_upd

        if nzmax > 1 and nzero <= nr:
            ier = 4
            break

        # Too many nodes
        if nzero > nr:
            if icmin == 0:
                emin = emin - 2.0 * (emax - emin)
            emax = e
            icmax = 1
            e = 0.5 * (emin + e)
            if emax - emin < dell * abs(emin):
                ier = 5
                break
            continue

        # Too few nodes
        if nzero < nr:
            icmin = 1
            emin = e
            e = 0.5 * (e + emax)
            if emax - emin < dell * abs(emin):
                ier = 5
                break
            continue

        # Correct number of nodes found
        if isum == 0:
            isum = 1
            nzero, sumout, iotp_upd, nzmax = doutw(
                e, eps, ak, nr, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)
            iotp = iotp_upd

        po = pt[iotp]
        qo = qt[iotp]

        # Inward solution
        sumin, ilast = dinw(e, eps, ak, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)

        if ilast < 0:
            ier = 6
            break

        # Match outward and inward solutions at iotp
        if abs(pt[iotp]) > 1.0e-30:
            fact = po / pt[iotp]
        else:
            fact = 1.0
        for i in range(iotp, nrt):
            pt[i] *= fact
            qt[i] *= fact
        sumin = sumin * fact * fact

        qi = qt[iotp]

        # Normalization integral
        total_sum = sumin + sumout

        # Eigenvalue correction (ME-5.15)
        if total_sum < 1.0e-15:
            total_sum = 1.0
        de = SL * po * (qo - qi) / total_sum
        ep = e + de

        if de < 0.0:
            icmax = 1
            emax = e
        if de > 0.0:
            icmin = 1
            emin = e

        if icmin == 0 and ep < emin:
            emin = 1.1 * emin
            if ep < emin:
                ep = 0.5 * (e + emin)
        if icmin == 1 and ep < emin:
            ep = 0.5 * (e + emin)
        if icmax == 1 and ep > emax:
            ep = 0.5 * (e + emax)
        if ep > esup:
            ep = 0.5 * (esup + e)

        if ep > esup and abs(e - esup) < dell * abs(esup):
            ier = 5
            break

        eo = e
        e = ep
        if min(abs(de), abs(e - eo)) <= abs(e * dell):
            break  # Converged

    # Normalization
    if total_sum > 1.0e-30:
        fact = 1.0 / math.sqrt(total_sum)
        for i in range(ilast + 1):
            pt[i] *= fact
            qt[i] *= fact
            if abs(pt[i]) < 1.0e-99:
                pt[i] = 0.0
            if abs(qt[i]) < 1.0e-99:
                qt[i] = 0.0

    # Zero beyond practical infinity
    for i in range(ilast + 1, nrt):
        pt[i] = 0.0
        qt[i] = 0.0

    # Extract to user grid (RAD)
    p_out = np.zeros(ngp)
    q_out = np.zeros(ngp)
    for i in range(ngp):
        rloc = rad[i]
        # Find nearest point in merged grid
        j_found = 0
        for j in range(nrt - 1):
            if r[j] <= rloc:
                j_found = j
        if j_found < nrt - 1:
            if rloc - r[j_found] < r[j_found + 1] - rloc:
                p_out[i] = pt[j_found]
                q_out[i] = qt[j_found]
            else:
                p_out[i] = pt[j_found + 1]
                q_out[i] = qt[j_found + 1]
        else:
            p_out[i] = pt[j_found]
            q_out[i] = qt[j_found]

    # Convert ilast to user grid index
    rlast = r[ilast] if ilast < nrt else r[nrt - 1]
    ilast_user = ngp - 1
    for i in range(ngp):
        if rad[i] >= rlast:
            ilast_user = min(i + 1, ngp - 1)
            break

    return e, ier, p_out, q_out, ilast_user
