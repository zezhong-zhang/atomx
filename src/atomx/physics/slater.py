"""Local Slater mean field and configuration-average energy operators.

The caller supplies radial arrays and quadrature geometry. No bound-state
search or SCF iteration is performed here.
"""
import math

import numba as nb
import numpy as np

from atomx.numerics.shooting.constants import CSLATE, FR1O3
from atomx.numerics.shooting.coulomb import _slater_integral, _slater_integral_slag6
from atomx.numerics.shooting.quadrature import slag6


@nb.njit(cache=True)
def _sfield(r, difr, rden, vnuc, np_, alpha, qelec, nshell, z,
            va_work, vb_work, vc_work, vd_work):
    """Compute screened potential — SFIELD (DHFS.f:892-980).

    Uses SLAG6 for integration (not trapezoidal).

    Parameters:
        z: actual nuclear charge (parameter to Latter's correction).
           Previously inferred from ``-vnuc[1]`` which is only valid for
           a point nucleus.  For the Fermi distribution, ``vnuc`` vanishes
           at the origin (finite V(0)), so inferring Z from the innermost
           grid point gave Z ~ 0 and effectively disabled Latter's
           correction — leading to a systematic ~1e-2 Ha bias in total
           binding energy for multi-electron atoms.

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
    rvinf = -1.0 - z + qelec
    for i in range(np_ - 1, -1, -1):
        if rvt[i] > rvinf:
            rvt[i] = rvinf
            vex[i] = rvinf - vnuc[i] - vel[i]
        else:
            break

    return rvt, vel, vex


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
    l1d = j1d + 1 if k1 > 0 else j1d - 1  # 2*l1

    j2d = abs(k2 + k2) - 1  # 2*j2
    l2d = j2d + 1 if k2 > 0 else j2d - 1

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
            pa, qa, ev, rv, vnuc, r, difr, np_, equation,
            slater_method):
    """Total energy via EXPVAL (DHFS.f:984-1103).

    Uses Slater integrals and DCJ3 angular coefficients for exact
    configuration-average total energy.

    equation: 0 = Dirac (density = P²+Q²), 1 = Schrödinger (density = P²)
    slater_method: 0 = Salvat/DHFS.f power-series ODE (legacy,
        bit-accurate at low Z), 1 = direct SLAG6 forward+backward
        (accurate for heavy Z where the power series stops
        converging — see :func:`_slater_integral_slag6`).
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
            dens_i = pa[s, i] ** 2 + qa[s, i] ** 2 if equation == 0 else pa[s, i] ** 2
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
            if slater_method == 1:
                es[is1, is2] = _slater_integral_slag6(f1_s, f2_s, 0, r, difr, np_)
            else:
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
                    if slater_method == 1:
                        gl = _slater_integral_slag6(
                            f1_ex, f2_ex, ll, r, difr, np_)
                    else:
                        gl = _slater_integral(
                            f1_ex, f2_ex, ll, r, difr, np_)
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
def _local_field(r, difr, density, vnuc, parameters):
    """Compiled Slater-family field supplied to the numerical SCF engine."""
    alpha, qelec, nshell, z = parameters
    # Historical workspace arguments are unused by SFIELD.
    return _sfield(r, difr, density, vnuc, len(r), alpha, qelec, nshell,
                   z, r, r, r, r)


def slater_exchange_alpha(Z, electron_count, *, exchange='slater', alpha_x=1.5,
                          fac_ihx=-1.0, fac_hxs=1.0, exchange_alpha_fn=None):
    """Resolve the Slater α passed into the JIT'd SCF.

    * ``exchange='slater'`` → return ``alpha_x`` unchanged.
    * ``exchange='gu'``     → compute Gu's α_xc(Z, N) from
      :func:`atomx.physics.local_exchange.get_hxs` using the
      requested ``fac_ihx``/``fac_hxs``, then convert to the
      Slater convention used inside :func:`_sfield`.

    **Units.**  FAC's ``ahx`` (what ``GetHXS`` returns) and
    the plain Slater ``α`` (what DHFS / this solver expects)
    are *different* coefficients of the same functional form:

    * FAC:     ``V_x = -ahx · (4π·ρ)^(1/3)``
    * Slater:  ``V_x = -α   · (3·ρ/π)^(1/3)``

    Matching the two gives

        α_Slater = ahx · (4π² / 3)^(1/3) ≈ ahx · 2.3609.

    Without this conversion FAC's default AHX ≈ 0.5147 (Ne)
    would land near α = 0.5, producing 2.4× too little exchange
    and a total binding energy ~2.5 eV / 10 eV too shallow at Z=10.
    With the conversion applied, atomx's shooting matches sfac
    to ≤ 1 × 10⁻⁴ relative across the Task 0 calibration panel.

    The value is held fixed across the SCF loop, mirroring FAC
    (``potential->ahx`` set once per ``OptimizeRadial``,
    ``radial.c:2348``).
    """
    if exchange_alpha_fn is not None:
        n_total = float(electron_count)
        return float(exchange_alpha_fn(float(Z), n_total))
    if exchange == 'gu':
        from atomx.physics.local_exchange import get_hxs
        n_total = float(electron_count)
        ahx = get_hxs(
            Z=float(Z), N=n_total,
            ihx=fac_ihx, hxs=fac_hxs,
        )
        # Convert FAC's ahx convention to the Slater α the
        # shooting _sfield uses.  See docstring for derivation.
        slater_conv = (4.0 * math.pi ** 2 / 3.0) ** (1.0 / 3.0)
        return ahx * slater_conv
    return float(alpha_x)
