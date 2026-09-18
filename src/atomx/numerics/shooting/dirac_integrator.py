"""DIR, DOUTW, DINW: Dirac radial equation integration with adaptive stepping.

Translates radial.f DIR (line 2118), DOUTW (line 1733), DINW (line 1808).
"""

import math

import numba as nb

from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.dirac_power_series import dir0


@nb.njit(cache=True, nogil=True)
def dir_integrate(e, ak, eps, isum, ppi, qqi, ra, rb,
                  rv0, rv1, rv2, rv3):
    """Adaptive step-size Dirac integrator — DIR (radial.f:2118-2226).

    Integrates from RA to RB using power series with adaptive step doubling/halving.

    Returns:
        pf, qf: solution at rb
        rln:    log of renormalizing factor
        rsum:   normalization integral contribution
        nstep:  number of steps taken
        nchs:   number of zeros of P in (ra, rb)
        ca2:    ``ca[2]`` coefficient from the final ``dir0`` substep — the
                power-series coefficient of the ``(r-r0)^2`` term for the
                large component.  Needed by :func:`doutw` for the Fortran
                DOUTW ``TCONV = 2*CA(3)*PPI`` turning-point blow-up check,
                which is far more reliable than a 3-point finite-difference
                approximation of ``P''`` (the FD version misfires for sharp
                heavy-Z 1s / 2p₁/₂ orbitals where successive grid values are
                very close in magnitude).
    """
    nchs = 0
    rln = 0.0
    rsum = 0.0

    h = rb - ra
    direct = -1.0 if h < 0.0 else 1.0

    k_ctrl = -2
    nstep = 0

    r1 = ra
    p1 = ppi
    q1 = qqi

    while True:
        r0 = r1
        p0 = p1
        q0 = q1

        while True:
            iout = 0
            r1 = r0 + h
            if direct * (rb - r1) < direct * 1.0e-1 * h:
                r1 = rb
                h = rb - r0
                iout = 1

            p1, q1, ca, cb, s, t, nsum = dir0(
                e, ak, eps, p0, q0, r0, r1, rv0, rv1, rv2, rv3)
            hp = h

            k_ctrl += 1
            if nsum > 15:
                # Slow convergence
                if nsum < 60:
                    break  # Accept this step
                # Too slow, halve step
                h = 0.5 * h
                k_ctrl = -4
                continue
            # Fast convergence
            if k_ctrl < 0:
                break  # Accept, but don't double yet
            # Can try doubling
            h = h + h
            k_ctrl = 0
            break

        # Normalization integral (ME-5.18)
        if isum == 1:
            rsumi = 0.0
            cdems = 2.0 * s - 1.0
            cdemt = 2.0 * (s + t) - 1.0
            for i1 in range(nsum):
                for i2 in range(nsum):
                    denom_p = cdems + float(i1 + i2 + 2)
                    denom_q = cdemt + float(i1 + i2 + 2)
                    if abs(denom_p) > 1.0e-30:
                        rsumi += ca[i1] * ca[i2] / denom_p
                    if abs(denom_q) > 1.0e-30:
                        rsumi += cb[i1] * cb[i2] / denom_q
            rsum = rsum + direct * hp * rsumi

        nstep += 1

        # Renormalization
        tst_amp = abs(p1)
        if tst_amp > 1.0e2:
            rln = rln + math.log(tst_amp)
            p1 = p1 / tst_amp
            q1 = q1 / tst_amp
            rsum = rsum / (tst_amp * tst_amp)

        if p0 * p1 < 0.0 and r0 > 0.0:
            nchs += 1

        if iout != 0:
            break

    pf = p1
    qf = q1
    ca2 = ca[2]  # power-series (r-r0)² coefficient — used by DOUTW TCONV check
    return pf, qf, rln, rsum, nstep, nchs, ca2


@nb.njit(cache=True, nogil=True)
def doutw(e, eps, ak, nr, iotp_in, isum,
          r, ind, nrt, va, vb, vc, vd,
          pt, qt):
    """Outward Dirac integration — DOUTW (radial.f:1733-1804).

    Integrates outward from r=0 to the turning point.

    Parameters:
        e:    energy
        eps:  tolerance
        ak:   kappa (float)
        nr:   radial quantum number (= n - l - 1)
        iotp_in: initial outer turning point index (1-based Fortran)
        isum: if 1, compute normalization integral
        r:    merged grid (0-based)
        ind:  interval index for each grid point (0-based into spline arrays)
        nrt:  total number of merged grid points
        va, vb, vc, vd: spline coefficients (0-based)
        pt, qt: output arrays for P(r), Q(r)

    Returns:
        nzero: number of nodes of P
        sumout: normalization integral (outward part)
        iotp:   updated turning point (0-based)
        nzmax: max zeros in single interval
    """
    nzero = 0
    nzmax = 0
    iotp = iotp_in  # 0-based

    n1 = nrt if e < 0.0 else iotp

    pt[0] = 0.0
    qt[0] = 0.0
    sumout = 0.0

    for i in range(n1 - 1):
        ra = r[i]
        rb = r[i + 1]
        in_idx = ind[i]
        rv0 = va[in_idx]
        rv1 = vb[in_idx]
        rv2 = vc[in_idx]
        rv3 = vd[in_idx]
        ppi = pt[i]
        qqi = qt[i]

        pf, qf, rln_val, rsum_val, nstep, nchs, ca2 = dir_integrate(
            e, ak, eps, isum, ppi, qqi, ra, rb, rv0, rv1, rv2, rv3)

        nzero += nchs
        if nchs > nzmax:
            nzmax = nchs
        if nzero > nr and e < 0.0:
            pt[i + 1] = pf
            qt[i + 1] = qf
            return nzero, sumout, iotp, nzmax

        pt[i + 1] = pf
        qt[i + 1] = qf

        # Past-turning-point blow-up detection — Fortran DOUTW
        # line 1777-1786.  ``TCONV = 2*CA(3)*PPI`` is the
        # analytical second derivative of P (from the power-series
        # ``(r-r0)^2`` coefficient returned by ``dir_integrate``)
        # times the boundary value ``PPI``.  Positive TCONV past
        # ``iotp`` means P is becoming convex (picking up the
        # exponentially-growing irregular solution) — stop
        # integration there so spurious sign flips don't corrupt
        # the node count.
        #
        # The earlier 3-point finite-difference proxy
        # ``(P[i+1] - 2·P[i] + P[i-1]) * PPI`` underestimated and
        # sometimes flipped sign on the sharp heavy-Z inner
        # region, letting doutw count a spurious node — which
        # drove ``dbound`` into the "too many nodes" branch and
        # out to the wrong eigenvalue.  See trace.md 2026-04-20
        # and benchmarks/REPORT.md §6.4 for the diagnosis.
        if e < 0.0 and i >= iotp and ppi != 0.0:
            tconv = 2.0 * ca2 * ppi
            if tconv > 1.0e-15:
                if isum == 1:
                    sumout = sumout + rsum_val
                iotp = i + 1
                return nzero, sumout, iotp, nzmax

        if i == 0:
            if isum == 1:
                sumout = sumout + rsum_val
            continue

        # Renormalization
        if rln_val > 0.0:
            fact = math.exp(-rln_val)
            for j in range(i + 1):
                pt[j] *= fact
                qt[j] *= fact
            if isum == 1:
                sumout = sumout * fact * fact + rsum_val
        else:
            if isum == 1:
                sumout = sumout + rsum_val

    return nzero, sumout, iotp, nzmax


@nb.njit(cache=True, nogil=True)
def dinw(e, eps, ak, iotp, isum,
         r, ind, nrt, va, vb, vc, vd,
         pt, qt):
    """Inward Dirac integration — DINW (radial.f:1808-1891).

    Integrates inward from practical infinity to the turning point.

    Returns:
        sumin:  normalization integral (inward part)
        ilast:  index of practical infinity (0-based)
    """
    kappa_int = int(ak)
    l = -kappa_int - 1 if kappa_int < 0 else kappa_int
    al = float(l)
    trinf = 22500.0  # 150^2

    # WKB solution at outer grid point (ME-5.13, ME-5.14)
    fact = (e + 2.0 * SL * SL) / (SL * SL)
    n = nrt
    ilast = n - 1

    while True:
        n1_idx = ind[n - 2]  # Spline interval for point n-1
        rn = r[n - 1]
        rvn = va[n1_idx] + rn * (vb[n1_idx] + rn * (vc[n1_idx] + rn * vd[n1_idx]))
        rvnp = vb[n1_idx] + rn * (2.0 * vc[n1_idx] + rn * 3.0 * vd[n1_idx])
        cmu = fact * rn * (rvn - e * rn) + al * (al + 1.0)

        if cmu <= 0.0:
            # Error: should not happen for bound states
            pt[n - 1] = 1.0
            qt[n - 1] = 0.0
            ilast = n - 1
            break

        # Practical infinity check
        if cmu < trinf or n == iotp + 2:
            crat = ((0.5 - math.sqrt(cmu)) / rn
                    - 0.25 * fact * (rvn + rn * rvnp - 2.0 * rn * e) / cmu)
            pt[n - 1] = 1.0
            qt[n - 1] = SL * (crat + ak / rn) / (e + 2.0 * SL * SL)
            ilast = n - 1
            break
        else:
            pt[n - 1] = 0.0
            qt[n - 1] = 0.0
            n -= 1
            if n <= iotp + 1:
                pt[n - 1] = 1.0
                qt[n - 1] = 0.0
                ilast = n - 1
                break

    # Inward integration from ilast to iotp
    sumin = 0.0
    n1_range = ilast - iotp

    for j in range(1, n1_range + 1):
        i = ilast - j  # Goes from ilast-1 down to iotp
        i1 = i + 1
        ra = r[i1]
        rb = r[i]
        in_idx = ind[i]
        rv0 = va[in_idx]
        rv1 = vb[in_idx]
        rv2 = vc[in_idx]
        rv3 = vd[in_idx]
        ppi = pt[i1]
        qqi = qt[i1]

        pf, qf, rln_val, rsum_val, nstep, nchs, ca2 = dir_integrate(
            e, ak, eps, isum, ppi, qqi, ra, rb, rv0, rv1, rv2, rv3)

        pt[i] = pf
        qt[i] = qf

        # Renormalization
        if rln_val > 0.0:
            fact_r = math.exp(-rln_val)
            for m in range(i1, ilast + 1):
                pt[m] *= fact_r
                qt[m] *= fact_r
            if isum == 1:
                sumin = sumin * fact_r * fact_r + rsum_val
        else:
            if isum == 1:
                sumin = sumin + rsum_val

    return sumin, ilast
