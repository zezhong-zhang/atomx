"""Radial Coulomb integrals, independent of SCF orchestration."""
import math
import numba as nb
import numpy as np
from .quadrature import slag6
from .spline import splin0

@nb.njit(cache=True)
def _slater_integral(f1, f2, l, r, difr, np_):
    """Slater integral — DHFS.f SLATER function (line 1191-1351).

    Computes ∫∫ F1(r)*F2(s) * r_<^l / r_>^(l+1) dr ds
    using Hartree's ODE method with power series integration.

    Parameters:
        f1, f2: radial density-like arrays on the grid (length np_)
        l:      multipole order
        r:      radial grid
        difr:   grid Jacobian (dR/dI)
        np_:    number of grid points
    """
    eps_sl = 1.0e-9
    al = float(l)

    # Spline F1 for outward integration of Z(r)
    # Z(r) = ∫_0^r F1(r') * (r'/r)^l dr'  (outward ODE)
    a1 = np.zeros(np_)
    b1 = np.zeros(np_)
    c1 = np.zeros(np_)
    d1 = np.zeros(np_)
    # Simple spline: use SPLIN0 on (r, f1)
    a1, b1, c1, d1 = splin0(r, f1, np_, 0.0, 0.0)

    z_arr = np.zeros(np_)
    z_arr[0] = 0.0
    # First interval: exact polynomial integration
    if np_ > 1:
        r2v = r[1]
        z_arr[1] = (a1[0] / (al + 1.0) * r2v
                    + b1[0] / (al + 2.0) * r2v ** 2
                    + c1[0] / (al + 3.0) * r2v ** 3
                    + d1[0] / (al + 4.0) * r2v ** 4)

    # Outward integration for Z(r) via power series ODE.
    #
    # The inner substep-halving loop has no natural cap — if the
    # power series stops converging (e.g. heavy-Z densities with
    # sharp inner peaks), ``nsteps`` doubles forever.  We cap at
    # MAX_NSTEPS so one pathological (r, L) can't stall ``_expval``
    # for minutes.  The cap of 4096 is empirically large enough for
    # Z ≤ 92 to reproduce FAC's local-exchange total energies to
    # 10⁻⁴ relative (on the Task 0 calibration panel) while keeping
    # pathological grid points from doubling forever.
    MAX_NSTEPS = 4096
    for i in range(1, np_ - 1):
        ra = r[i]
        if ra < 1.0e-30:
            z_arr[i + 1] = 0.0
            continue
        h = r[i + 1] - r[i]
        nsteps = 1
        converged_step = False

        while not converged_step and nsteps <= MAX_NSTEPS:
            hstep = h / nsteps
            rc = ra - hstep
            zc = z_arr[i]
            for _ in range(nsteps):
                rc = rc + hstep
                if rc < 1.0e-30:
                    continue
                # Evaluate spline coefficients at this point
                g1 = rc * (a1[i] + rc * (b1[i] + rc * (c1[i] + rc * d1[i])))
                g2 = (a1[i] + rc * (2.0 * b1[i] + rc * (3.0 * c1[i] + rc * 4.0 * d1[i]))) * hstep
                g3 = (b1[i] + rc * (3.0 * c1[i] + rc * 6.0 * d1[i])) * hstep ** 2
                g4 = (c1[i] + rc * 4.0 * d1[i]) * hstep ** 3
                g5 = d1[i] * hstep ** 4
                hr = hstep / rc

                dz = zc
                dz = hr * (g1 - (al + 0.0) * dz) / 1.0
                zc += dz
                dz = hr * (g2 - (al + 1.0) * dz) / 2.0
                zc += dz
                dz = hr * (g3 - (al + 2.0) * dz) / 3.0
                zc += dz
                dz = hr * (g4 - (al + 3.0) * dz) / 4.0
                zc += dz
                dz = hr * (g5 - (al + 4.0) * dz) / 5.0
                zc += dz

                conv_inner = False
                for j in range(5, 50):
                    dz = -hr * ((al + float(j)) * dz) / float(j + 1)
                    zc += dz
                    if abs(dz) < eps_sl * abs(zc):
                        conv_inner = True
                        break

                if not conv_inner:
                    # Need more substeps
                    nsteps *= 2
                    break
            else:
                converged_step = True

        z_arr[i + 1] = zc

    # Spline Z(r) for inward integration → Y(r)
    az = np.zeros(np_)
    bz = np.zeros(np_)
    cz = np.zeros(np_)
    dz_arr = np.zeros(np_)
    az, bz, cz, dz_arr = splin0(r, z_arr, np_, 0.0, 0.0)

    y_arr = np.zeros(np_)
    y_arr[np_ - 1] = z_arr[np_ - 1]

    itst = 0
    for i in range(np_ - 2, 0, -1):
        ra = r[i + 1]
        h = r[i] - r[i + 1]  # Negative
        nsteps = 1
        converged_step = False

        while not converged_step and nsteps <= MAX_NSTEPS:
            hstep = h / nsteps
            rc = ra - hstep
            yc = y_arr[i + 1]
            for _ in range(nsteps):
                rc = rc + hstep
                if abs(rc) < 1.0e-30:
                    break
                g1 = az[i] + rc * (bz[i] + rc * (cz[i] + rc * dz_arr[i]))
                g2 = (bz[i] + rc * (2.0 * cz[i] + rc * 3.0 * dz_arr[i])) * hstep
                g3 = (cz[i] + rc * 3.0 * dz_arr[i]) * hstep ** 2
                g4 = dz_arr[i] * hstep ** 3
                hr = hstep / rc
                twoL1 = 2.0 * al + 1.0

                dy = yc
                dy = hr * ((al + 1.0) * dy - twoL1 * g1) / 1.0
                yc += dy
                dy = hr * ((al + 0.0) * dy - twoL1 * g2) / 2.0
                yc += dy
                dy = hr * ((al - 1.0) * dy - twoL1 * g3) / 3.0
                yc += dy
                dy = hr * ((al - 2.0) * dy - twoL1 * g4) / 4.0
                yc += dy

                conv_inner = False
                for j in range(4, 50):
                    dy = hr * (al + 2.0 - float(j)) * dy / float(j + 1)
                    yc += dy
                    if abs(dy) < eps_sl * abs(yc):
                        conv_inner = True
                        break

                if not conv_inner and r[i] > 1.0e-6:
                    nsteps *= 2
                    break
            else:
                converged_step = True

        y_arr[i] = yc

        # Empirical Y correction for small r (DHFS.f:1307-1319)
        if itst == 0:
            if abs(y_arr[i]) < abs(y_arr[i + 1]) and r[i] < 1.0e-4:
                itst = 1
        else:
            if abs(y_arr[i]) > abs(y_arr[i + 1]) or abs(y_arr[i]) < 1.0e-14:
                for kk in range(i, -1, -1):
                    y_arr[kk] = 0.0
                break

    # Final integral: ∫ F2(r) * Y(r) / r dr via SLAG6
    integrand = np.zeros(np_)
    for i in range(1, np_):
        if r[i] > 1.0e-10:
            integrand[i] = (y_arr[i] * f2[i] / r[i]) * difr[i]
    if r[0] >= 1.0e-10:
        integrand[0] = (y_arr[0] * f2[0] / r[0]) * difr[0]

    result = slag6(1.0, integrand, np_)
    return result[np_ - 1]


@nb.njit(cache=True)
def _yk_over_r(f1, l, r, difr, np_):
    """Hartree screening function ``Y^L(f1; r)/r`` on the grid via SLAG6.

        Y_L(r)/r = (1/r^(L+1)) · ∫_0^r f1(s) s^L ds
                   +  r^L · ∫_r^∞ f1(s)/s^(L+1) ds

    Each inner integral is evaluated with the same 6-point Lagrange
    routine (:func:`slag6`) that the outer SCF uses for the electronic
    potential.  Inner-boundary treatment mirrors ``_sfield``: the
    innermost grid point (``r[0] = 0``) is dropped from the backward
    integrand since ``f1/r^(L+1)`` is formally singular there; the
    returned array is 0 wherever ``r <= 1e-30`` (the physical density
    vanishes there anyway).

    Shared by :func:`_slater_integral_slag6` (which contracts it with a
    second density) and the non-local HF exchange source assembly in
    :mod:`atomx.numerics.shooting.hartree_fock`.
    """
    # Forward: I1(r) = ∫_0^r f1(s) · s^L ds
    aux_fwd = np.zeros(np_)
    for i in range(np_):
        aux_fwd[i] = f1[i] * r[i] ** l * difr[i]
    i1 = slag6(1.0, aux_fwd, np_)

    # Backward: I2_tail(r) = ∫_r^R f1(s)/s^(L+1) ds
    # Build the integrand reversed so SLAG6 integrates outer → inner.
    aux_bwd = np.zeros(np_)
    for i in range(np_ - 1):
        idx = np_ - 1 - i  # idx goes np_-1 down to 1
        if r[idx] > 1.0e-30:
            aux_bwd[i] = f1[idx] / r[idx] ** (l + 1) * difr[idx]
    # aux_bwd[np_ - 1] corresponds to r[0] (= 0 or ~1e-30); leave it 0.
    bwd = slag6(1.0, aux_bwd, np_)
    # bwd[k] = ∫_{r[np_-1-k]}^{r[np_-1]} f1/s^(L+1) ds
    # hence I2_tail(r[i]) = bwd[np_ - 1 - i]

    y = np.zeros(np_)
    for i in range(np_):
        if r[i] > 1.0e-30:
            y[i] = i1[i] / r[i] ** (l + 1) + r[i] ** l * bwd[np_ - 1 - i]
    return y


@nb.njit(cache=True)
def _slater_integral_slag6(f1, f2, l, r, difr, np_):
    """Slater integral via direct SLAG6 forward+backward quadrature.

    Computes ``F^L(f1, f2) = ∫∫ f1(r)·f2(s) · r_<^L / r_>^(L+1) dr ds``
    by assembling the Hartree Y^L function (see :func:`_yk_over_r`)
    and returning ``∫_0^∞ f2(r)·Y_L(r) dr``.  The quadrature error is
    bounded by the SCF's own tolerance even for the sharp inner peaks
    of heavy-Z 1s densities, where the Salvat/DHFS.f power-series ODE
    (:func:`_slater_integral`) hits its ``MAX_NSTEPS`` cap and returns
    with only ~10⁻³ precision.
    """
    y = _yk_over_r(f1, l, r, difr, np_)

    # Outer integral: ∫ f2(r) · Y_L(r) · dr, with Jacobian difr built in.
    integrand = np.zeros(np_)
    for i in range(np_):
        integrand[i] = f2[i] * y[i] * difr[i]
        # at r[0]=0 the physical density (and therefore f2) is zero and
        # y[0] = 0 by construction — the first point contributes nothing.
    result = slag6(1.0, integrand, np_)
    return result[np_ - 1]
