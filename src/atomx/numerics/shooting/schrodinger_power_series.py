"""SCH0: Schrödinger radial equation power series integrator — radial.f lines 2003-2114.

Solves the Schrödinger radial equation for R*V(R) = RV0 + RV1*R + RV2*R^2 + RV3*R^3.
P is the radial function, Q = dP/dr (derivative).
"""


import numba as nb
import numpy as np

OVER = 1.0e15  # Overflow level


@nb.njit(cache=True)
def sch0(e, al, eps, p0, q0, r0, r1, rv0, rv1, rv2, rv3):
    """Single-interval Schrödinger power series integrator (SCH0).

    Parameters:
        e:    kinetic energy
        al:   orbital angular momentum quantum number (float)
        eps:  global tolerance
        p0, q0: function value and derivative at r0
        r0, r1: interval endpoints
        rv0, rv1, rv2, rv3: cubic spline coefficients of R*V(R)

    Returns:
        p1, q1: solution (value and derivative) at r1
        ca:     power series coefficients (length 60)
        s:      power series exponent
        nsum:   number of terms used (100 = overflow)
    """
    ca = np.zeros(60)

    rve = rv1 - e
    h = r1 - r0
    h2 = h * h

    if r0 > 1.0e-10:
        # ===== Middle region (ME-4.10 to ME-4.13) =====
        s = 0.0
        rho = h / r0
        u0 = al * (al + 1.0) + 2.0 * r0 * (rv0 + r0 * (rve + r0 * (rv2 + r0 * rv3)))
        u1 = 2.0 * (rv0 + r0 * (2.0 * rve + r0 * (3.0 * rv2 + r0 * 4.0 * rv3))) * h
        u2 = 2.0 * (rve + r0 * (3.0 * rv2 + r0 * 6.0 * rv3)) * h2
        u3 = 2.0 * (rv2 + r0 * 4.0 * rv3) * h2 * h
        u4 = 2.0 * rv3 * h2 * h2
        ut = u0 + u1 + u2 + u3 + u4

        ca[0] = p0
        ca[1] = q0 * h
        ca[2] = rho * rho * u0 * ca[0] / 2.0
        ca[3] = rho * (rho * (u0 * ca[1] + u1 * ca[0]) - 4.0 * ca[2]) / 6.0
        cak = (u0 - 2.0) * ca[2] + u1 * ca[1] + u2 * ca[0]
        ca[4] = rho * (rho * cak - 12.0 * ca[3]) / 12.0
        cak = (u0 - 6.0) * ca[3] + u1 * ca[2] + u2 * ca[1] + u3 * ca[0]
        ca[5] = rho * (rho * cak - 24.0 * ca[4]) / 20.0

        p1 = ca[0] + ca[1] + ca[2] + ca[3] + ca[4] + ca[5]
        q1 = ca[1] + 2.0 * ca[2] + 3.0 * ca[3] + 4.0 * ca[4] + 5.0 * ca[5]
        p2p1 = 2.0 * ca[2] + 6.0 * ca[3] + 12.0 * ca[4] + 20.0 * ca[5]

        nsum = 6
        for i in range(6, 60):
            k = i  # 0-based i, Fortran I=i+1, K=I-1=i
            km1 = k - 1
            km2 = k - 2
            # Fortran: CAK = (U0-(K-2)*(K-3))*CA(I-2) + ...
            # In 0-based: CA(I-2) = ca[i-2]
            cak = (u0 - float(km1 - 1) * float(km1 - 2)) * ca[i - 2]
            if i >= 3:
                cak += u1 * ca[i - 3]
            if i >= 4:
                cak += u2 * ca[i - 4]
            if i >= 5:
                cak += u3 * ca[i - 5]
            if i >= 6:
                cak += u4 * ca[i - 6]
            ca[i] = rho * (rho * cak - 2.0 * km1 * km2 * ca[k - 1]) / (k * km1)

            p1 += ca[i]
            dq1 = float(k) * ca[i]
            q1 += dq1
            p2p1 += float(k) * float(km1) * ca[i]

            # Overflow check
            tst = max(abs(p1), abs(q1), abs(p2p1))
            if tst > OVER:
                nsum = 100
                return p1, q1, ca, s, nsum

            t1 = abs(ca[i])
            t2 = abs(r1 * r1 * p2p1 - h2 * ut * p1)
            tst1 = eps * max(abs(p1), abs(q1) / (i + 1))
            tst2 = eps * max(abs(p1), abs(q1))
            if t1 < tst1 and t2 < tst2:
                nsum = k + 1
                q1 = q1 / h
                return p1, q1, ca, s, nsum

        nsum = 60
        q1 = q1 / h
        return p1, q1, ca, s, nsum

    # ===== First interval (r0 < 1e-10) (ME-4.15 to ME-4.18) =====
    s = al + 1.0
    u0 = al * s
    u1 = 2.0 * rv0 * r1
    u2 = 2.0 * rve * r1 ** 2
    u3 = 2.0 * rv2 * r1 ** 3
    u4 = 2.0 * rv3 * r1 ** 4
    ut = u0 + u1 + u2 + u3 + u4

    ca[0] = 1.0
    ca[1] = u1 * ca[0] / ((s + 1.0) * s - u0)
    ca[2] = (u1 * ca[1] + u2 * ca[0]) / ((s + 2.0) * (s + 1.0) - u0)
    ca[3] = (u1 * ca[2] + u2 * ca[1] + u3 * ca[0]) / ((s + 3.0) * (s + 2.0) - u0)

    p1 = ca[0] + ca[1] + ca[2] + ca[3]
    q1 = s * ca[0] + (s + 1.0) * ca[1] + (s + 2.0) * ca[2] + (s + 3.0) * ca[3]
    p2p1 = (s * (s - 1.0) * ca[0] + (s + 1.0) * s * ca[1]
            + (s + 2.0) * (s + 1.0) * ca[2] + (s + 3.0) * (s + 2.0) * ca[3])

    nsum = 4
    for i in range(4, 60):
        k = i
        val = u1 * ca[k - 1]
        if k >= 2:
            val += u2 * ca[k - 2]
        if k >= 3:
            val += u3 * ca[k - 3]
        if k >= 4:
            val += u4 * ca[k - 4]
        ca[i] = val / ((s + k) * (s + k - 1.0) - u0)

        p1 += ca[i]
        dq1 = (s + k) * ca[i]
        q1 += dq1
        p2p1 += (s + k - 1.0) * dq1

        tst = max(abs(p1), abs(q1), abs(p2p1))
        if tst > OVER:
            nsum = 100
            return p1, q1, ca, s, nsum

        t1 = abs(ca[i])
        t2 = abs(r1 * r1 * (p2p1 - ut * p1))
        tst1 = eps * max(abs(p1), abs(q1) / (i + 1))
        tst2 = eps * max(abs(p1), abs(q1))
        if t1 < tst1 and t2 < tst2:
            nsum = k + 1
            # Renormalize (ME-4.18)
            q1 = q1 / (abs(p1) * r1)
            p1 = p1 / abs(p1)
            return p1, q1, ca, s, nsum

    nsum = 60
    q1 = q1 / (abs(p1) * r1)
    p1 = p1 / abs(p1)
    return p1, q1, ca, s, nsum
