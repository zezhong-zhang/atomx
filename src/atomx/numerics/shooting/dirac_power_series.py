"""DIR0: Dirac radial equation power series integrator — radial.f lines 2230-2450.

Solves the Dirac radial equation for R*V(R) = RV0 + RV1*R + RV2*R^2 + RV3*R^3
(piecewise cubic spline coefficients) using power series expansion.

Given boundary conditions (P0, Q0) at RA, computes (P1, Q1) at RB.
"""

import math

import numba as nb
import numpy as np

from atomx.numerics.shooting.constants import SL

OVER = 1.0e15  # Overflow level


@nb.njit(cache=True)
def dir0(e, ak, eps, p0, q0, r0, r1, rv0, rv1, rv2, rv3):
    """Single-interval Dirac power series integrator (DIR0).

    Parameters:
        e:    particle kinetic energy
        ak:   kappa (relativistic angular momentum quantum number, float)
        eps:  global tolerance for series convergence
        p0, q0: large and small components at r0
        r0, r1: interval endpoints
        rv0, rv1, rv2, rv3: cubic spline coefficients of R*V(R) in this interval

    Returns:
        p1, q1: solution at r1
        ca, cb: power series coefficient arrays (length 60)
        s, t:   power series exponents (P ~ r^s, Q ~ r^(s+t))
        nsum:   number of terms used (100 = overflow)
    """
    ca = np.zeros(60)
    cb = np.zeros(60)

    isig = 1
    if ak > 0.0:
        isig = -1

    h = r1 - r0
    h2 = h * h
    rve = rv1 - e  # RV1 - E

    if r0 > 1.0e-10:
        # ===== Middle region (ME-4.23 to ME-4.27) =====
        s = 0.0
        t = 0.0
        # The homogeneous system preserves an exactly zero initial state.
        # A relative residual test would otherwise require 0 < eps*0 and
        # report nonconvergence forever. The origin branch remains distinct:
        # it constructs the nonzero regular Frobenius solution from P(0)=Q(0)=0.
        if p0 == 0.0 and q0 == 0.0:
            return 0.0, 0.0, ca, cb, s, t, 1
        rho = h / r0
        u0 = (rv0 + r0 * (rve + r0 * (rv2 + r0 * rv3))) / SL
        u1 = (rve + r0 * (2.0 * rv2 + r0 * 3.0 * rv3)) * h / SL
        u2 = (rv2 + r0 * 3.0 * rv3) * h2 / SL
        u3 = rv3 * h * h2 / SL
        ub = u0 - 2.0 * SL * r0
        uh = u1 - 2.0 * SL * h
        ut = u0 + u1 + u2 + u3
        uq = ut - 2.0 * SL * r1

        ca[0] = p0
        cb[0] = q0
        ca[1] = -rho * (ak * ca[0] + ub * cb[0])
        cb[1] = rho * (ak * cb[0] + u0 * ca[0])
        ca[2] = -rho * ((ak + 1.0) * ca[1] + ub * cb[1] + uh * cb[0]) / 2.0
        cb[2] = rho * ((ak - 1.0) * cb[1] + u0 * ca[1] + u1 * ca[0]) / 2.0
        ca[3] = -rho * ((ak + 2.0) * ca[2] + ub * cb[2] + uh * cb[1] + u2 * cb[0]) / 3.0
        cb[3] = rho * ((ak - 2.0) * cb[2] + u0 * ca[2] + u1 * ca[1] + u2 * ca[0]) / 3.0

        p1 = ca[0] + ca[1] + ca[2] + ca[3]
        pp1 = ca[1] + 2.0 * ca[2] + 3.0 * ca[3]
        q1 = cb[0] + cb[1] + cb[2] + cb[3]
        qp1 = cb[1] + 2.0 * cb[2] + 3.0 * cb[3]

        nsum = 4
        for i in range(4, 60):
            k = i  # 0-based i corresponds to Fortran I=i+1, K=I-1=i
            # Fortran: CA(I) where I=5..60, K=I-1=4..59
            # In 0-based: ca[i] where i=4..59
            ca_i2 = cb[i - 2] if i >= 2 else 0.0
            ca_i3 = cb[i - 3] if i >= 3 else 0.0
            ca_i4 = cb[i - 4] if i >= 4 else 0.0
            ca[i] = -rho * ((ak + k - 1.0) * ca[k - 1] + ub * cb[k - 1]
                             + uh * ca_i2 + u2 * ca_i3 + u3 * ca_i4) / k
            # For cb: use ca values
            cb_i2 = ca[i - 2] if i >= 2 else 0.0
            cb_i3 = ca[i - 3] if i >= 3 else 0.0
            cb_i4 = ca[i - 4] if i >= 4 else 0.0
            cb[i] = rho * ((ak - k + 1.0) * cb[k - 1] + u0 * ca[k - 1]
                            + u1 * cb_i2 + u2 * cb_i3 + u3 * cb_i4) / k

            p1 += ca[i]
            pp1 += k * ca[i]
            q1 += cb[i]
            qp1 += k * cb[i]

            # Overflow check
            tst = max(abs(p1), abs(q1), abs(pp1), abs(qp1))
            if tst > OVER:
                nsum = 100
                return p1, q1, ca, cb, s, t, nsum

            # Convergence check (Fortran residuals)
            t1a = abs(r1 * pp1 + h * (ak * p1 + uq * q1))
            t1b = abs(r1 * qp1 - h * (ak * q1 + ut * p1))
            t1 = max(t1a, t1b)
            t2 = max(abs(ca[i]), abs(cb[i]))
            tst_conv = eps * max(abs(p1), abs(q1))
            if t1 < tst_conv and t2 < tst_conv:
                nsum = k + 1
                return p1, q1, ca, cb, s, t, nsum

        nsum = 60
        return p1, q1, ca, cb, s, t, nsum

    # ===== First interval (r0 < 1e-10) =====
    u0v = rv0 / SL
    u1v = rve * r1 / SL
    u2v = rv2 * r1 ** 2 / SL
    u3v = rv3 * r1 ** 3 / SL
    ut = u0v + u1v + u2v + u3v
    uq = ut - 2.0 * SL * r1
    uh = u1v - 2.0 * SL * r1

    if abs(u0v) >= 1.0e-10:
        # ===== Case A: U0 != 0 (ME-4.30 to ME-4.34) =====
        s = math.sqrt(ak * ak - u0v * u0v)
        t = 0.0
        ds = s + s

        ca[0] = 1.0
        cb[0] = -(s + ak) / u0v
        cai = u1v * ca[0]
        cbi = uh * cb[0]
        ca[1] = (-u0v * cai - (s + 1.0 - ak) * cbi) / (ds + 1.0)
        cb[1] = ((s + 1.0 + ak) * cai - u0v * cbi) / (ds + 1.0)
        cai = u1v * ca[1] + u2v * ca[0]
        cbi = uh * cb[1] + u2v * cb[0]
        ca[2] = (-u0v * cai - (s + 2.0 - ak) * cbi) / (2.0 * (ds + 2.0))
        cb[2] = ((s + 2.0 + ak) * cai - u0v * cbi) / (2.0 * (ds + 2.0))

        p1 = ca[0] + ca[1] + ca[2]
        pp1 = s * ca[0] + (s + 1.0) * ca[1] + (s + 2.0) * ca[2]
        q1 = cb[0] + cb[1] + cb[2]
        qp1 = s * cb[0] + (s + 1.0) * cb[1] + (s + 2.0) * cb[2]

        nsum = 3
        for i in range(3, 60):
            k = i  # 0-based
            cai = u1v * ca[k - 1]
            if k >= 2:
                cai += u2v * ca[k - 2]
            if k >= 3:
                cai += u3v * ca[k - 3]
            cbi = uh * cb[k - 1]
            if k >= 2:
                cbi += u2v * cb[k - 2]
            if k >= 3:
                cbi += u3v * cb[k - 3]
            ca[i] = (-u0v * cai - (s + k - ak) * cbi) / (k * (ds + k))
            cb[i] = ((s + k + ak) * cai - u0v * cbi) / (k * (ds + k))

            p1 += ca[i]
            pp1 += (s + k) * ca[i]
            q1 += cb[i]
            qp1 += (s + k) * cb[i]

            tst = max(abs(p1), abs(q1), abs(pp1), abs(qp1))
            if tst > OVER:
                nsum = 100
                return p1, q1, ca, cb, s, t, nsum

            t1a = abs(r1 * pp1 + h * (ak * p1 + uq * q1))
            t1b = abs(r1 * qp1 - h * (ak * q1 + ut * p1))
            t1 = max(t1a, t1b)
            t2 = max(abs(ca[i]), abs(cb[i]))
            tst_conv = eps * max(abs(p1), abs(q1))
            if t1 < tst_conv and t2 < tst_conv:
                nsum = k + 1
                # Renormalize (ME-4.39)
                q1 = q1 / abs(p1)
                p1 = p1 / abs(p1)
                return p1, q1, ca, cb, s, t, nsum

        nsum = 60
        q1 = q1 / abs(p1)
        p1 = p1 / abs(p1)
        return p1, q1, ca, cb, s, t, nsum

    # U0 == 0
    if isig > 0:
        # ===== Case B: U0=0, SIG=+1 (kappa<0) (ME-4.35, ME-4.36) =====
        s = abs(ak)
        t = 1.0
        ds1 = s + s + 1.0

        ca[0] = 1.0
        cb[0] = u1v * ca[0] / ds1
        ca[1] = 0.0
        cb[1] = u2v * ca[0] / (ds1 + 1.0)
        ca[2] = -uh * cb[0] / 2.0
        cb[2] = (u1v * ca[2] + u3v * ca[0]) / (ds1 + 2.0)
        ca[3] = -(uh * cb[1] + u2v * cb[0]) / 3.0
        cb[3] = (u1v * ca[3] + u2v * ca[2]) / (ds1 + 3.0)

        p1 = ca[0] + ca[1] + ca[2] + ca[3]
        pp1 = s * ca[0] + (s + 1.0) * ca[1] + (s + 2.0) * ca[2] + (s + 3.0) * ca[3]
        q1 = cb[0] + cb[1] + cb[2] + cb[3]
        qp1 = (s + 1.0) * cb[0] + (s + 2.0) * cb[1] + (s + 3.0) * cb[2]

        nsum = 4
        for i in range(4, 60):
            k = i
            cb_im2 = cb[i - 2] if i >= 2 else 0.0
            cb_im3 = cb[i - 3] if i >= 3 else 0.0
            cb_im4 = cb[i - 4] if i >= 4 else 0.0
            ca[i] = -(uh * cb_im2 + u2v * cb_im3 + u3v * cb_im4) / k

            ca_im1 = ca[i - 1] if i >= 1 else 0.0
            ca_im2 = ca[i - 2] if i >= 2 else 0.0
            cb[i] = (u1v * ca[i] + u2v * ca_im1 + u3v * ca_im2) / (ds1 + k)

            p1 += ca[i]
            pp1 += (s + k) * ca[i]
            q1 += cb[i]
            qp1 += (s + i) * cb[i]

            tst = max(abs(p1), abs(q1), abs(pp1), abs(qp1))
            if tst > OVER:
                nsum = 100
                return p1, q1, ca, cb, s, t, nsum

            t1a = abs(r1 * pp1 + h * (ak * p1 + uq * q1))
            t1b = abs(r1 * qp1 - h * (ak * q1 + ut * p1))
            t1 = max(t1a, t1b)
            t2 = max(abs(ca[i]), abs(cb[i]))
            tst_conv = eps * max(abs(p1), abs(q1))
            if t1 < tst_conv and t2 < tst_conv:
                nsum = k + 1
                q1 = q1 / abs(p1)
                p1 = p1 / abs(p1)
                return p1, q1, ca, cb, s, t, nsum

        nsum = 60
        q1 = q1 / abs(p1)
        p1 = p1 / abs(p1)
        return p1, q1, ca, cb, s, t, nsum

    else:
        # ===== Case C: U0=0, SIG=-1 (kappa>0) (ME-4.37, ME-4.38) =====
        s = abs(ak) + 1.0
        t = -1.0
        ds1 = s + abs(ak)

        if uh > 0.0:
            cb[0] = -1.0
        else:
            cb[0] = 1.0
        ca[0] = -uh * cb[0] / ds1
        cb[1] = 0.0
        ca[1] = -u2v * cb[0] / (ds1 + 1.0)
        cb[2] = u1v * ca[0] / 2.0
        ca[2] = -(uh * cb[2] + u3v * cb[0]) / (ds1 + 2.0)
        cb[3] = (u1v * ca[1] + u2v * ca[0]) / 3.0
        ca[3] = -(uh * cb[3] + u2v * cb[2]) / (ds1 + 3.0)

        p1 = ca[0] + ca[1] + ca[2] + ca[3]
        pp1 = s * ca[0] + (s + 1.0) * ca[1] + (s + 2.0) * ca[2] + (s + 3.0) * ca[3]
        q1 = cb[0] + cb[1] + cb[2] + cb[3]
        qp1 = (s - 1.0) * cb[0] + s * cb[1] + (s + 1.0) * cb[2]

        nsum = 4
        for i in range(4, 60):
            k = i
            ca_im2 = ca[i - 2] if i >= 2 else 0.0
            ca_im3 = ca[i - 3] if i >= 3 else 0.0
            ca_im4 = ca[i - 4] if i >= 4 else 0.0
            cb[i] = (u1v * ca_im2 + u2v * ca_im3 + u3v * ca_im4) / k

            cb_im1 = cb[i - 1] if i >= 1 else 0.0
            cb_im2 = cb[i - 2] if i >= 2 else 0.0
            ca[i] = -(uh * cb[i] + u2v * cb_im1 + u3v * cb_im2) / (ds1 + k)

            p1 += ca[i]
            pp1 += (s + k) * ca[i]
            q1 += cb[i]
            qp1 += (s + k - 1.0) * cb[i]

            tst = max(abs(p1), abs(q1), abs(pp1), abs(qp1))
            if tst > OVER:
                nsum = 100
                return p1, q1, ca, cb, s, t, nsum

            t1a = abs(r1 * pp1 + h * (ak * p1 + uq * q1))
            t1b = abs(r1 * qp1 - h * (ak * q1 + ut * p1))
            t1 = max(t1a, t1b)
            t2 = max(abs(ca[i]), abs(cb[i]))
            tst_conv = eps * max(abs(p1), abs(q1))
            if t1 < tst_conv and t2 < tst_conv:
                nsum = k + 1
                q1 = q1 / abs(p1)
                p1 = p1 / abs(p1)
                return p1, q1, ca, cb, s, t, nsum

        nsum = 60
        q1 = q1 / abs(p1)
        p1 = p1 / abs(p1)
        return p1, q1, ca, cb, s, t, nsum
