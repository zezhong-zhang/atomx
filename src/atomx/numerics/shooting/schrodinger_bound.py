"""SBOUND: Schrödinger bound-state eigenvalue solver — radial.f lines 460-777.

Solves the radial Schrödinger equation for bound states using shooting method
with power-series integration and eigenvalue refinement from derivative discontinuity.
"""

import math

import numba as nb
import numpy as np

from atomx.numerics.shooting.schrodinger_integrator import sinw, soutw
from atomx.numerics.shooting.schrodinger_power_series import sch0
from atomx.numerics.shooting.spline import merge_grids as _merge_grids_s


@nb.njit(cache=True)
def sbound(e_in, eps, n, l, rg, rvg, va, vb, vc, vd, nvt, rad, ngp):
    """Solve Schrödinger radial equation for bound state.

    Parameters:
        e_in: initial energy estimate
        eps:  global tolerance (> 1e-15)
        n:    principal quantum number
        l:    orbital angular momentum quantum number
        rg, rvg, va, vb, vc, vd: spline data
        nvt:  number of spline intervals
        rad:  user radial grid
        ngp:  number of user grid points

    Returns:
        e:    converged eigenvalue
        ier:  error code (0 = success)
        p:    radial function on user grid
        q:    derivative dP/dr on user grid
        ilast: practical infinity index
    """
    ier = 0
    nr = n - l - 1  # Radial quantum number
    al = float(l)

    dell = 10.0 * eps
    e = e_in
    if e > -1.0e-1:
        e = -1.0e-1
    fl1 = 0.5 * l * (l + 1)

    # Merge grids
    eps_tol = max(0.5 * eps, 1.0e-10)
    r, ind, nrt = _merge_grids_s(rg, nvt, rad, ngp, eps_tol)

    pt = np.zeros(nrt)
    qt = np.zeros(nrt)

    # Minimum of effective potential (ME-5.1)
    emin = 1.0
    for i in range(1, nrt):
        rn = r[i]
        j = ind[i]
        rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
        emin = min(emin, (rvn + fl1 / rn) / rn)

    if emin > -1.0e-35:
        ier = 1
        return e, ier, np.zeros(ngp), np.zeros(ngp), 0

    rn = r[nrt - 1]
    j = ind[nrt - 1]
    rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
    esup = (rvn + fl1 / rn) / rn
    if esup > 0.0:
        esup = 0.0

    if l == 0:
        tehydr = -(rvg[0] / n) ** 2 if abs(rvg[0]) > 1.0e-30 else -10.0
        emin = min(e, tehydr, -10.0)

    if e > esup or e < emin:
        e = 0.5 * (esup + emin)

    emax = esup
    icmin = 0
    icmax = 0
    isum = 0
    total_sum = 1.0

    max_shots = 200
    for _ in range(max_shots):
        if e > -1.0e-16:
            ier = 2
            break

        # Outer turning point (ME-5.2)
        # A bracket-search trial can lie below the effective-potential
        # minimum. Then there is no classical turning point: match near
        # the origin and let the derivative correction move the energy
        # back into the spectrum, as SBOUND does. Choosing the outer edge
        # instead falsely reports a small box and abandons a valid root.
        iotp = 1
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

        iotp = iotp + 1
        if iotp > nrt - 2:
            ier = 3
            break
        if iotp < 2:
            iotp = 2

        # Outward solution
        nzero, sumout, nzmax = soutw(
            e, eps, al, nr, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)

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

        # Correct nodes
        if isum == 0:
            isum = 1
            nzero, sumout, nzmax = soutw(
                e, eps, al, nr, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)

        po = pt[iotp]
        qo = qt[iotp]

        # Inward solution
        sumin, ilast, ier_inw = sinw(
            e, eps, al, iotp, isum, r, ind, nrt, va, vb, vc, vd, pt, qt)
        if ier_inw > 0:
            ier = ier_inw
            break

        # Match
        fact = po / pt[iotp] if abs(pt[iotp]) > 1e-30 else 1.0
        for i in range(iotp, ilast + 1):
            pt[i] *= fact
            qt[i] *= fact
        sumin = sumin * fact * fact

        qi = qt[iotp]
        total_sum = sumin + sumout

        # Eigenvalue correction (ME-5.11)
        # Note: Schrödinger has DE = PO*(QO-QI)/(2*SUM), no SL factor
        if total_sum < 1.0e-15:
            total_sum = 1.0
        de = po * (qo - qi) / (total_sum + total_sum)
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

        if ep >= esup and abs(e - esup) < dell * abs(esup):
            ier = 5
            break

        eo = e
        e = ep
        if min(abs(de), abs(e - eo)) <= abs(e * dell):
            break
    else:
        ier = 7  # eigenvalue refinement exhausted its iteration limit

    if ier != 0:
        # An error can precede inward integration, so ilast need not exist.
        # Failed roots carry no normalized orbital payload.
        return e, ier, np.zeros(ngp), np.zeros(ngp), 0

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

    for i in range(ilast + 1, nrt):
        pt[i] = 0.0
        qt[i] = 0.0

    if l == 0 and r[0] == 0.0:
        # SOUTW's Q(0)=0 initializes the regular solution. It is not its
        # physical derivative: an s wave has P(r)=A*r+O(r²). Recover A
        # from the same origin series and the normalized first sample.
        first = ind[0]
        _, _, coefficients, _, terms = sch0(e, al, eps, 0., 0., 0., r[1],
            va[first], vb[first], vc[first], vd[first])
        if terms >= 60:
            return e, 8, np.zeros(ngp), np.zeros(ngp), 0
        qt[0] = pt[1]/(r[1]*np.sum(coefficients[:terms]))

    # Extract to user grid
    p_out = np.zeros(ngp)
    q_out = np.zeros(ngp)
    for i in range(ngp):
        rloc = rad[i]
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

    rlast = r[ilast] if ilast < nrt else r[nrt - 1]
    ilast_user = ngp - 1
    for i in range(ngp):
        if rad[i] >= rlast:
            ilast_user = min(i + 1, ngp - 1)
            break

    return e, ier, p_out, q_out, ilast_user
