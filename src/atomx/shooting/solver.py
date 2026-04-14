"""DHFS SCF solver — follows DHFS.f exactly.

Self-consistent field calculation for atoms using ODE shooting with local
Slater exchange. Supports both Dirac and Schrödinger equations.

Total energy computed via EXPVAL (DHFS.f:984-1103) using:
- Salvat's SLATER integral function (power-series ODE method)
- DCJ3 angular coefficients (reuses atomx.scf.angular_coeffs.exchange_lambda)
"""

import numpy as np
import numba as nb
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from atomx.shooting.constants import SL, HREV, A0B, PI, FOURPI, CSLATE, FR1O3
from atomx.shooting.grid import sgrid
from atomx.shooting.spline import splin0, findi
from atomx.shooting.quadrature import slag6
from atomx.shooting.dirac_bound import dbound
from atomx.shooting.schrodinger_bound import sbound


@nb.njit(cache=True)
def _sfield(r, difr, rden, vnuc, np_, alpha, qelec, nshell,
            va_work, vb_work, vc_work, vd_work):
    """Compute screened potential — SFIELD (DHFS.f:892-980).

    Uses SLAG6 for integration (not trapezoidal).

    Returns:
        rvt: trial potential r*V(r) array
        vel: electronic potential r*V_el
        vex: exchange potential r*V_ex
    """
    rvt = np.zeros(np_)
    vel = np.zeros(np_)
    vex = np.zeros(np_)

    # Electronic potential via forward/backward SLAG6 integrals
    aux1 = np.zeros(np_)
    for i in range(np_):
        aux1[i] = rden[i] * difr[i]
    fwd = slag6(1.0, aux1, np_)

    # Backward integral: RDEN(NP+1-I)*DIFR(NP+1-I)/R(NP+1-I)
    aux = np.zeros(np_)
    np1 = np_
    for i in range(np_ - 1):
        idx = np1 - 1 - i  # Goes from np_-1 down to 1
        if r[idx] > 1.0e-30:
            aux[i] = rden[idx] * difr[idx] / r[idx]
        else:
            aux[i] = 0.0
    aux[np_ - 1] = 0.0
    bwd = slag6(1.0, aux, np_)

    for i in range(np_):
        vel[i] = fwd[i] + bwd[np1 - 1 - i] * r[i]

    # Exchange potential
    if qelec < 2.1 and nshell == 1:
        # Self-interaction correction (DHFS.f:948-956)
        fact = 1.0 / qelec
        for i in range(np_):
            vex[i] = -vel[i] * fact
            rvt[i] = vnuc[i] + vel[i] + vex[i]
        return rvt, vel, vex

    # Slater exchange (DHFS.f:960-968)
    for i in range(np_):
        if i > 0:
            arg = CSLATE * r[i] * rden[i]
            if arg > 1.0e-30:
                vex[i] = -alpha * arg ** FR1O3
        rvt[i] = vnuc[i] + vel[i] + vex[i]

    # Latter tail correction (DHFS.f:970-978)
    z = -vnuc[1] if np_ > 1 else 0.0  # vnuc = -Z
    rvinf = -1.0 - z + qelec
    for i in range(np_ - 1, -1, -1):
        if rvt[i] > rvinf:
            rvt[i] = rvinf
            vex[i] = rvinf - vnuc[i] - vel[i]
        else:
            break

    return rvt, vel, vex


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

    # Outward integration for Z(r) via power series ODE
    for i in range(1, np_ - 1):
        ra = r[i]
        if ra < 1.0e-30:
            z_arr[i + 1] = 0.0
            continue
        h = r[i + 1] - r[i]
        nsteps = 1
        converged_step = False

        while not converged_step:
            hstep = h / nsteps
            rc = ra - hstep
            zc = z_arr[i]
            for kk in range(nsteps):
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

        while not converged_step:
            hstep = h / nsteps
            rc = ra - hstep
            yc = y_arr[i + 1]
            for kk in range(nsteps):
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
def _dcj3(l, k1, k2):
    """Angular coefficient D^L(K1,K2) — DHFS.f DCJ3 (line 1113-1187).

    D^L(K1,K2) = [1/(2J1+1)] * <K1||C^L||K2>^2

    Parameters:
        l:  multipole order
        k1, k2: relativistic kappa quantum numbers
    """
    if l < 0 or k1 == 0 or k2 == 0:
        return 0.0

    j1d = abs(k1 + k1) - 1  # 2*j1
    if k1 > 0:
        l1d = j1d + 1  # 2*l1
    else:
        l1d = j1d - 1

    j2d = abs(k2 + k2) - 1  # 2*j2
    if k2 > 0:
        l2d = j2d + 1
    else:
        l2d = j2d - 1

    ld = l + l  # 2*L

    # Parity selection rule
    if (ld + l1d + l2d) % 4 != 0:
        return 0.0

    # Triangle rule
    it1 = ld + j1d - j2d
    it2 = ld - j1d + j2d
    it3 = -ld + j1d + j2d
    if min(it1, it2, it3) < 0:
        return 0.0

    jsumd = ld + j1d + j2d
    jsum = jsumd // 2
    if jsumd - 2 * jsum != 0:
        return 0.0

    if jsum % 2 == 0:
        fact = float(jsum - j2d) * float(jsum - j1d + 1) / float(j1d * (j2d + 1))
        l1 = (j1d - 1) // 2
        l2 = (j2d + 1) // 2
    else:
        fact = float(jsum - ld) * float(jsum + 1) / float(j1d * (j2d + 1))
        l1 = (j1d - 1) // 2
        l2 = (j2d - 1) // 2

    it1 = l + l1 - l2
    it2 = l - l1 + l2
    it3 = -l + l1 + l2
    if min(it1, it2, it3) < 0:
        return 0.0

    lsumd = l + l1 + l2
    lsum = lsumd // 2

    # Compute TAU ratios: TAU(n) = n! / (2n-1)!!
    # TAU(0)=1, TAU(1)=1, TAU(n) = TAU(n-1)*n/(2n-1)
    max_idx = max(lsum, lsum - l, lsum - l1, lsum - l2) + 1
    if max_idx < 2:
        max_idx = 2
    tau = np.zeros(max_idx + 1)
    tau[0] = 1.0
    tau[1] = 1.0
    for ii in range(2, max_idx + 1):
        tau[ii] = (tau[ii - 1] * ii) / float(2 * ii - 1)

    idx_ls = lsum
    idx_ls_l = lsum - l
    idx_ls_l1 = lsum - l1
    idx_ls_l2 = lsum - l2
    if idx_ls < 0 or idx_ls_l < 0 or idx_ls_l1 < 0 or idx_ls_l2 < 0:
        return 0.0

    cg2 = fact * (float(2 * l1 + 1) / float(lsumd + 1)) * \
          (tau[idx_ls] / tau[idx_ls_l]) / (tau[idx_ls_l1] * tau[idx_ls_l2])

    return cg2 * float(j2d + 1) / float(j1d + 1)


@nb.njit(cache=True)
def _expval(nshell, shell_kappa, shell_occ, shell_2j,
            pa, qa, ev, rv, vnuc, r, difr, np_, equation):
    """Total energy via EXPVAL (DHFS.f:984-1103).

    Uses Slater integrals and DCJ3 angular coefficients for exact
    configuration-average total energy.

    equation: 0 = Dirac (density = P²+Q²), 1 = Schrödinger (density = P²)
    """
    # Single-particle kinetic energies and nuclear interaction
    ek = np.zeros(nshell)
    en = np.zeros(nshell)

    for s in range(nshell):
        f1 = np.zeros(np_)
        f2 = np.zeros(np_)
        f1[0] = 0.0
        f2[0] = 0.0
        for i in range(1, np_):
            if equation == 0:
                dens_i = pa[s, i] ** 2 + qa[s, i] ** 2
            else:
                dens_i = pa[s, i] ** 2
            f1[i] = dens_i * rv[i] * difr[i] / r[i]
            f2[i] = dens_i * vnuc[i] * difr[i] / r[i]
        f1_int = slag6(1.0, f1, np_)
        ek[s] = ev[s] - f1_int[np_ - 1]  # Kinetic = eigenvalue - <V>
        f2_int = slag6(1.0, f2, np_)
        en[s] = f2_int[np_ - 1]  # Nuclear

    # Two-electron interaction (ES matrix)
    es = np.zeros((nshell, nshell))
    for is1 in range(nshell):
        rj1 = 0.5 * shell_2j[is1]
        ff0 = float(shell_2j[is1] + 1) / float(max(shell_2j[is1], 1))

        for is2 in range(is1, nshell):
            # F^0 integral: direct Coulomb
            f1_s = np.zeros(np_)
            f2_s = np.zeros(np_)
            for i in range(np_):
                if equation == 0:
                    f1_s[i] = pa[is1, i] ** 2 + qa[is1, i] ** 2
                    f2_s[i] = pa[is2, i] ** 2 + qa[is2, i] ** 2
                else:
                    f1_s[i] = pa[is1, i] ** 2
                    f2_s[i] = pa[is2, i] ** 2
            es[is1, is2] = _slater_integral(f1_s, f2_s, 0, r, difr, np_)
            if is1 == is2:
                es[is1, is2] /= ff0

            # G^L integrals: exchange
            f1_ex = np.zeros(np_)
            f2_ex = np.zeros(np_)
            for i in range(np_):
                if equation == 0:
                    f1_ex[i] = pa[is1, i] * pa[is2, i] + qa[is1, i] * qa[is2, i]
                else:
                    f1_ex[i] = pa[is1, i] * pa[is2, i]
                f2_ex[i] = f1_ex[i]

            rj2 = 0.5 * shell_2j[is2]
            lmin_ex = int(abs(rj1 - rj2))
            if is1 == is2:
                lmin_ex = 1
            lmax_ex = int(rj1 + rj2 + 0.5)

            for ll in range(lmin_ex, lmax_ex + 1):
                coef = _dcj3(ll, shell_kappa[is1], shell_kappa[is2])
                if abs(coef) > 1.0e-12:
                    gl = _slater_integral(f1_ex, f2_ex, ll, r, difr, np_)
                    es[is1, is2] -= gl * coef / float(shell_2j[is2] + 1)

            if is2 != is1:
                es[is2, is1] = es[is1, is2]

        if is1 < nshell:
            es[is1, is1] *= ff0

    # Total energy
    ekin_total = 0.0
    ebin = 0.0
    for s in range(nshell):
        ekin_total += shell_occ[s] * ek[s]
        ebin += shell_occ[s] * (ek[s] + en[s])
        for s2 in range(nshell):
            if s2 == s:
                ebin += 0.5 * shell_occ[s] * (shell_occ[s] - 1.0) * es[s, s]
            else:
                ebin += 0.5 * shell_occ[s] * shell_occ[s2] * es[s, s2]

    return ebin, ekin_total


@nb.njit(cache=True)
def _dhfs_scf(z, nshell, shell_n, shell_kappa, shell_occ, shell_l, shell_2j,
              np_, rn, r2, drn, alpha, max_iter, epsc, aw, equation):
    """Full DHFS SCF loop — DHFS.f lines 700-888.

    Parameters:
        equation: 0 = Dirac, 1 = Schrödinger
    """
    eps = 1.0e-11  # Internal solver tolerance (DHFS.f:739)
    qelec = 0.0
    for s in range(nshell):
        qelec += shell_occ[s]

    # Generate grid
    r, difr, np_actual = sgrid(rn, r2, drn, np_)
    np_ = np_actual

    # Nuclear potential: r*V_nuc = -Z (point nucleus)
    vnuc = np.zeros(np_)
    for i in range(np_):
        vnuc[i] = -z

    # Finite nuclear size (Fermi distribution) if aw > 0.5
    if aw > 0.5:
        f2bohr = 1.0e-13 / A0B
        r1_nuc = 1.07 * aw ** FR1O3 * f2bohr
        r2_nuc = 0.546 * f2bohr
        denn = np.zeros(np_)
        vnuc_work = np.zeros(np_)
        for i in range(np_):
            xx = math.exp((r1_nuc - r[i]) / r2_nuc) if (r1_nuc - r[i]) / r2_nuc < 500 else 1.0e200
            denn[i] = xx / (1.0 + xx)
            vnuc_work[i] = denn[i] * r[i] ** 2 * difr[i]
        fwd_nuc = slag6(1.0, vnuc_work, np_)
        aux_nuc = np.zeros(np_)
        for i in range(np_ - 1):
            idx = np_ - 1 - i
            aux_nuc[i] = denn[idx] * r[idx] * difr[idx]
        aux_nuc[np_ - 1] = 0.0
        bwd_nuc = slag6(1.0, aux_nuc, np_)
        fnorm = z / fwd_nuc[np_ - 1] if abs(fwd_nuc[np_ - 1]) > 1.0e-30 else z
        for i in range(np_):
            vnuc[i] = -fnorm * (fwd_nuc[i] + bwd_nuc[np_ - 1 - i] * r[i])

    # Thomas-Fermi-Moliere starting density (DHFS.f:377-393)
    rtf = 0.88534 / z ** 0.33333333333
    al1 = -6.0 / rtf
    al2 = -1.2 / rtf
    al3 = -0.3 / rtf
    rden = np.zeros(np_)
    aux1 = np.zeros(np_)
    for i in range(np_):
        rr = r[i]
        rden[i] = rr * (0.10 * al1 * al1 * math.exp(al1 * rr)
                        + 0.55 * al2 * al2 * math.exp(al2 * rr)
                        + 0.35 * al3 * al3 * math.exp(al3 * rr))
        aux1[i] = rden[i] * difr[i]
    den_int = slag6(1.0, aux1, np_)
    rnorm = qelec / den_int[np_ - 1] if abs(den_int[np_ - 1]) > 1.0e-30 else 1.0
    for i in range(np_):
        rden[i] *= rnorm

    # Spline work arrays
    rg = r.copy()
    rvg = np.zeros(np_)
    va = np.zeros(np_)
    vb_arr = np.zeros(np_)
    vc = np.zeros(np_)
    vd = np.zeros(np_)

    # Orbital storage
    pa = np.zeros((nshell, np_))
    qa = np.zeros((nshell, np_))
    ev = np.zeros(nshell)

    # Initial SFIELD (DHFS.f:771)
    rvt, vel, vex = _sfield(r, difr, rden, vnuc, np_, alpha, qelec, nshell,
                             va, vb_arr, vc, vd)
    rv = rvt.copy()

    # SCF iteration (DHFS.f:776-846)
    factor = 0.0
    chi = 1.0
    chiant = 1.0e30
    converged = False
    iteration = 0

    for it in range(1, max_iter + 1):
        iteration = it

        # Adaptive mixing (DHFS.f:777-784)
        if chiant > chi:
            factor += 0.025
            if factor > 0.5:
                factor = 0.5
        else:
            factor -= 0.05
            if factor < 0.025:
                factor = 0.025
        faccom = 1.0 - factor
        chiant = chi

        # Mix potential (DHFS.f:790-794)
        for i in range(np_):
            rv[i] = faccom * rv[i] + factor * rvt[i]
            rden[i] = 0.0
            rvg[i] = rv[i]

        # Refit cubic spline (DHFS.f:795) — CRITICAL for DBOUND accuracy
        va, vb_arr, vc, vd = splin0(rg, rvg, np_, 0.0, 0.0)

        # Solve each shell (DHFS.f:799-827)
        for s in range(nshell):
            nnn = shell_n[s]
            kkk = shell_kappa[s]
            e_guess = ev[s] if abs(ev[s]) > 1.0e-15 else -0.1

            if equation == 0:
                # Dirac
                e_val, ier, p_sol, q_sol, ilast = dbound(
                    e_guess, eps, nnn, kkk,
                    rg, rvg, va, vb_arr, vc, vd, np_,
                    r, np_)
            else:
                # Schrödinger
                lll = shell_l[s]
                e_val, ier, p_sol, q_sol, ilast = sbound(
                    e_guess, eps, nnn, lll,
                    rg, rvg, va, vb_arr, vc, vd, np_,
                    r, np_)

            # Renormalize with SLAG6 (DHFS.f:810-818)
            # Schrödinger: norm = ∫P² dr; Dirac: norm = ∫(P²+Q²) dr
            f1 = np.zeros(np_)
            for i in range(np_):
                if equation == 0:
                    f1[i] = (p_sol[i] ** 2 + q_sol[i] ** 2) * difr[i]
                else:
                    f1[i] = p_sol[i] ** 2 * difr[i]
            f1_int = slag6(1.0, f1, np_)
            fnorm_val = 1.0 / math.sqrt(f1_int[np_ - 1]) if f1_int[np_ - 1] > 1.0e-30 else 1.0

            for i in range(np_):
                pa[s, i] = p_sol[i] * fnorm_val
                qa[s, i] = q_sol[i] * fnorm_val

            ev[s] = e_val

            # Density: Schrödinger uses P² only; Dirac uses P²+Q²
            for i in range(np_):
                if equation == 0:
                    rden[i] += shell_occ[s] * (pa[s, i] ** 2 + qa[s, i] ** 2)
                else:
                    rden[i] += shell_occ[s] * pa[s, i] ** 2

        # New trial potential (DHFS.f:829)
        rvt, vel, vex = _sfield(r, difr, rden, vnuc, np_, alpha, qelec, nshell,
                                 va, vb_arr, vc, vd)

        # Convergence check (DHFS.f:830-834)
        chi = -1.0e10
        for i in range(np_):
            chip = abs(rv[i] - rvt[i]) / max(abs(rv[i]), 1.0e-18)
            if chip > chi:
                chi = chip

        # Test (DHFS.f:839)
        if chi <= epsc and factor >= 0.25:
            converged = True
            break

        if qelec <= 1.0:
            # Single-electron: no SCF needed beyond first iteration
            converged = True
            break

    # Final potential mix (DHFS.f:851-856)
    for i in range(np_):
        rv[i] = faccom * rv[i] + factor * rvt[i]
        rvg[i] = rv[i]
        rden[i] = 0.0
    va, vb_arr, vc, vd = splin0(rg, rvg, np_, 0.0, 0.0)

    # Final solve (DHFS.f:862-885)
    for s in range(nshell):
        nnn = shell_n[s]
        kkk = shell_kappa[s]
        e_guess = ev[s]

        if equation == 0:
            e_val, ier, p_sol, q_sol, ilast = dbound(
                e_guess, eps, nnn, kkk,
                rg, rvg, va, vb_arr, vc, vd, np_, r, np_)
        else:
            lll = shell_l[s]
            e_val, ier, p_sol, q_sol, ilast = sbound(
                e_guess, eps, nnn, lll,
                rg, rvg, va, vb_arr, vc, vd, np_, r, np_)

        f1 = np.zeros(np_)
        for i in range(np_):
            if equation == 0:
                f1[i] = (p_sol[i] ** 2 + q_sol[i] ** 2) * difr[i]
            else:
                f1[i] = p_sol[i] ** 2 * difr[i]
        f1_int = slag6(1.0, f1, np_)
        fnorm_val = 1.0 / math.sqrt(f1_int[np_ - 1]) if f1_int[np_ - 1] > 1.0e-30 else 1.0

        for i in range(np_):
            pa[s, i] = p_sol[i] * fnorm_val
            qa[s, i] = q_sol[i] * fnorm_val
        ev[s] = e_val

        for i in range(np_):
            if equation == 0:
                rden[i] += shell_occ[s] * (pa[s, i] ** 2 + qa[s, i] ** 2)
            else:
                rden[i] += shell_occ[s] * pa[s, i] ** 2

    # Compute total energy via EXPVAL (DHFS.f:984-1103)
    # Uses Slater integrals + DCJ3 angular coefficients for exact result
    e_total, ekin_total = _expval(
        nshell, shell_kappa, shell_occ, shell_2j,
        pa, qa, ev, rv, vnuc, r, difr, np_, equation)

    return ev, e_total, converged, iteration, r, difr, np_


@dataclass
class ShootingResult:
    """Result of an ODE shooting calculation.

    Satisfies the ``AtomicResult`` protocol from ``atomx.results``.
    """
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    method: str = "shooting"
    energy_components: Dict[str, float] = field(default_factory=dict)
    _grid: Optional[np.ndarray] = field(default=None, repr=False)
    _orbital_data: Optional[Dict] = field(default=None, repr=False)
    _equation: str = field(default="dirac", repr=False)

    @property
    def energy_breakdown(self) -> Dict[str, float]:
        return self.energy_components

    @property
    def theory_level(self):
        from atomx.types import TheoryLevel
        if self._equation == "schrodinger":
            return TheoryLevel.HARTREE_FOCK
        return TheoryLevel.DIRAC_FOCK_SLATER

    @property
    def method_name(self) -> str:
        eq = "Schrodinger" if self._equation == "schrodinger" else "Dirac"
        return f"{eq}-Fock-Slater (shooting)"

    @property
    def coupling_scheme(self):
        from atomx.types import CouplingScheme
        if self._equation == "schrodinger":
            return CouplingScheme.LS
        return CouplingScheme.JJ

    @property
    def wavefunctions(self):
        from atomx.results import Wavefunctions, OrbitalWavefunction
        if self._grid is None or self._orbital_data is None:
            return Wavefunctions(grid=np.array([]), basis_type="grid")

        orbitals = {}
        for label, data in self._orbital_data.items():
            orbitals[label] = OrbitalWavefunction(
                P=data["P"], Q=data.get("Q"),
                n=data["n"], l=data["l"],
                kappa=data.get("kappa"),
                occupation=data["occupation"],
                energy=self.orbital_energies.get(label, 0.0),
                label=label)
        return Wavefunctions(
            grid=self._grid, orbitals=orbitals, basis_type="grid")


class ShootingSolver:
    """DFS SCF using ODE shooting — Salvat/Radial method.

    Parameters:
        Z: nuclear charge
        subshells: list of Subshell objects with n, kappa, occupation
        n_grid: number of radial grid points (default 1000)
        r_max: outer radius in Bohr (default 75.0)
        alpha_x: Slater exchange parameter (default 1.5)
        aw: atomic weight in g/mol (0 = point nucleus)
        equation: 'dirac' or 'schrodinger'
    """

    def __init__(self, Z, subshells, n_grid=1000, r_max=75.0, alpha_x=1.5,
                 aw=0.0, equation='dirac'):
        self.Z = Z
        self.subshells = subshells
        self.n_grid = n_grid
        self.r_max = r_max
        self.alpha_x = alpha_x
        self.aw = aw
        self.equation = equation

    def solve(self, max_iter=300, tol=1e-9):
        n_shells = len(self.subshells)
        shell_n = np.array([sh.n for sh in self.subshells], dtype=np.int64)
        shell_kappa = np.array([sh.kappa for sh in self.subshells], dtype=np.int64)
        shell_occ = np.array([float(sh.occupation) for sh in self.subshells])

        # Also need l and 2j for Schrödinger
        shell_l = np.zeros(n_shells, dtype=np.int64)
        shell_2j = np.zeros(n_shells, dtype=np.int64)
        for i, sh in enumerate(self.subshells):
            k = sh.kappa
            if k < 0:
                shell_l[i] = -k - 1
                shell_2j[i] = -2 * k - 1
            else:
                shell_l[i] = k
                shell_2j[i] = 2 * k - 1

        eq_flag = 0 if self.equation == 'dirac' else 1

        # Grid parameters matching DHFS defaults
        r2 = 1.0e-7
        drn = max(2.0 * self.r_max / self.n_grid, 0.5)

        orb_E, E_total, converged, iters, r, difr, np_actual = _dhfs_scf(
            float(self.Z), n_shells, shell_n, shell_kappa, shell_occ,
            shell_l, shell_2j,
            self.n_grid, self.r_max, r2, drn,
            self.alpha_x, max_iter, tol, self.aw, eq_flag)

        oe = {}
        for i, sh in enumerate(self.subshells):
            oe[sh.label] = orb_E[i]

        return ShootingResult(
            converged=converged, iterations=iters,
            total_energy=E_total, orbital_energies=oe)
