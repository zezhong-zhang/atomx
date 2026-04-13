"""SCH, SOUTW, SINW: Schrödinger radial equation integration with adaptive stepping.

Translates radial.f SCH (line 1895), SOUTW (line 1598), SINW (line 1655).
"""

import numpy as np
import numba as nb
import math
from atomx.shooting.schrodinger_power_series import sch0


@nb.njit(cache=True)
def sch_integrate(e, al, eps, isum, ppi, qqi, ra, rb,
                  rv0, rv1, rv2, rv3):
    """Adaptive step-size Schrödinger integrator — SCH (radial.f:1895-1999).

    Returns:
        pf, qf: solution (value, derivative) at rb
        rln:    log of renormalizing factor
        rsum:   normalization integral contribution
        nstep:  number of steps
        nchs:   number of zeros of P in (ra, rb)
    """
    nchs = 0
    rln = 0.0
    rsum = 0.0

    h = rb - ra
    if h < 0.0:
        direct = -1.0
    else:
        direct = 1.0

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

            p1, q1, ca, s, nsum = sch0(
                e, al, eps, p0, q0, r0, r1, rv0, rv1, rv2, rv3)
            hp = h

            k_ctrl += 1
            if nsum > 15:
                if nsum < 60:
                    break
                h = 0.5 * h
                k_ctrl = -4
                continue
            if k_ctrl < 0:
                break
            h = h + h
            k_ctrl = 0
            break

        # Normalization integral (ME-5.17) — only P component for Schrödinger
        if isum == 1:
            rsumi = 0.0
            cdems = 2.0 * s - 1.0
            for i1 in range(nsum):
                for i2 in range(nsum):
                    denom = cdems + float(i1 + i2 + 2)
                    if abs(denom) > 1.0e-30:
                        rsumi += ca[i1] * ca[i2] / denom
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
    return pf, qf, rln, rsum, nstep, nchs


@nb.njit(cache=True)
def soutw(e, eps, al, nr, iotp, isum,
          r, ind, nrt, va, vb, vc, vd,
          pt, qt):
    """Outward Schrödinger integration — SOUTW (radial.f:1598-1651).

    Returns:
        nzero: number of nodes
        sumout: normalization integral (outward)
        nzmax: max zeros in single interval
    """
    nzero = 0
    nzmax = 0
    # Fortran N1=IOTP-1, loop I=1..N1 sets PT(2..IOTP).
    # In 0-based: loop i=0..iotp-1 sets pt[1..iotp].
    n1 = iotp

    pt[0] = 0.0
    qt[0] = 0.0
    sumout = 0.0

    for i in range(n1):
        ra = r[i]
        rb = r[i + 1]
        in_idx = ind[i]
        rv0 = va[in_idx]
        rv1 = vb[in_idx]
        rv2 = vc[in_idx]
        rv3 = vd[in_idx]
        ppi = pt[i]
        qqi = qt[i]

        pf, qf, rln_val, rsum_val, nstep, nchs = sch_integrate(
            e, al, eps, isum, ppi, qqi, ra, rb, rv0, rv1, rv2, rv3)

        nzero += nchs
        if nchs > nzmax:
            nzmax = nchs
        if nzero > nr and e < 0.0:
            pt[i + 1] = pf
            qt[i + 1] = qf
            return nzero, sumout, nzmax

        pt[i + 1] = pf
        qt[i + 1] = qf

        if i == 0:
            if isum == 1:
                sumout += rsum_val
            continue

        # Renormalization
        if rln_val > 0.0:
            fact = math.exp(-rln_val)
            for jj in range(i + 1):
                pt[jj] *= fact
                qt[jj] *= fact
            if isum == 1:
                sumout = sumout * fact * fact + rsum_val
        else:
            if isum == 1:
                sumout += rsum_val

    return nzero, sumout, nzmax


@nb.njit(cache=True)
def sinw(e, eps, al, iotp, isum,
         r, ind, nrt, va, vb, vc, vd,
         pt, qt):
    """Inward Schrödinger integration — SINW (radial.f:1655-1729).

    Returns:
        sumin: normalization integral (inward)
        ilast: practical infinity index (0-based)
        ier:   error code
    """
    trinf = 22500.0  # 150^2
    ier = 0

    # WKB at outer grid point (ME-5.4, ME-5.5)
    n = nrt
    ilast = n - 1

    while True:
        n1_idx = ind[n - 2]
        rn = r[n - 1]
        rvn = va[n1_idx] + rn * (vb[n1_idx] + rn * (vc[n1_idx] + rn * vd[n1_idx]))
        rvnp = vb[n1_idx] + rn * (2.0 * vc[n1_idx] + rn * 3.0 * vd[n1_idx])
        cmu = 2.0 * rn * (rvn - e * rn) + al * (al + 1.0)

        if cmu <= 0.0:
            ier = 6
            return 0.0, ilast, ier

        if cmu < trinf or n == iotp + 2:
            crat = 1.0 - rn * (rvn + rn * rvnp - 2.0 * e * rn) / cmu
            pt[n - 1] = 1.0
            qt[n - 1] = (0.5 / rn) * crat - math.sqrt(cmu) / rn
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

    sumin = 0.0
    n1_range = ilast - iotp

    for j in range(1, n1_range + 1):
        i = ilast - j
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

        pf, qf, rln_val, rsum_val, nstep, nchs = sch_integrate(
            e, al, eps, isum, ppi, qqi, ra, rb, rv0, rv1, rv2, rv3)

        pt[i] = pf
        qt[i] = qf

        if rln_val > 0.0:
            fact_r = math.exp(-rln_val)
            for m in range(i1, ilast + 1):
                pt[m] *= fact_r
                qt[m] *= fact_r
            if isum == 1:
                sumin = sumin * fact_r * fact_r + rsum_val
        else:
            if isum == 1:
                sumin += rsum_val

    return sumin, ilast, ier
