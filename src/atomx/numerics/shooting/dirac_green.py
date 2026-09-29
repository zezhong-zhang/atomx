"""Local Dirac fundamental solutions and Green assembly.

Numerical kernels only: no occupations, CI density, SCF policy, correction
selection, or electron-projector choice belongs in this module. The legacy
HF tail convention is retained as an explicit green_solve option.
"""
import numpy as np
from numba import njit
from .constants import SL
from .dirac_integrator import dinw, doutw, dir_integrate
from .quadrature import slag6
from .exponential_quadrature import exponential_prefix_weights, apply_exponential_prefix

_EPS_ODE = 1.e-11
_BIG = 10**9


def homogeneous_pair(e, kappa, r, ind, np_, va, vb, vc, vd):
    """Regular/decaying homogeneous Dirac solutions of the local problem.

    Returns ``(p_reg, q_reg, p_irr, q_irr, w_const, ncut, iotp, nzero)``
    where the ``*_reg`` arrays (length ``ncut``) are the regular-at-origin
    solution integrated outward, the ``*_irr`` arrays (length ``np_``,
    nonzero on ``[1, ncut-1]``) decay at infinity, ``w_const`` is the
    r-independent Wronskian ``P_reg Q_irr - Q_reg P_irr`` (arbitrary
    overall scale — consistent with the returned arrays), ``ncut`` is one
    past the practical infinity, ``iotp`` the outer classical turning
    point and ``nzero`` the node count of the regular solution (Sturm
    index: the number of homogeneous eigenvalues below ``e``).
    """
    ak = float(kappa)
    l = -kappa - 1 if kappa < 0 else kappa
    fl1 = 0.5 * l * (l + 1)

    # Outer classical turning point (scan from outside in, as in dbound).
    iotp = 1
    for jj in range(2, np_ + 1):
        idx = np_ + 1 - jj
        rn = r[idx]
        if rn < 1.0e-30:
            continue
        j = ind[idx]
        rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
        if e - (rvn + fl1 / rn) / rn > 0.0:
            iotp = idx
            break
    iotp += 1
    iotp = max(2, min(iotp, np_ - 2))

    # Decaying solution: inward from practical infinity down to r[1].
    p_irr = np.zeros(np_)
    q_irr = np.zeros(np_)
    _, ilast = dinw(e, _EPS_ODE, ak, 1, 0,
                    r, ind, np_, va, vb, vc, vd, p_irr, q_irr)
    ncut = min(ilast + 1, np_)

    # Regular solution: outward from the origin to the practical
    # infinity.  nr=_BIG disables the node early-exit and iotp_in=_BIG
    # disables the past-turning-point blow-up stop — beyond the turning
    # point the growing solution is the numerically dominant one, so the
    # outward direction stays stable all the way to r_cut.
    p_reg = np.zeros(ncut)
    q_reg = np.zeros(ncut)
    nzero, _, _, _ = doutw(e, _EPS_ODE, ak, _BIG, _BIG, 0,
                           r[:ncut], ind[:ncut], ncut,
                           va, vb, vc, vd, p_reg, q_reg)

    # Wronskian: exactly constant in exact arithmetic; evaluate near the
    # turning point where both solutions are well conditioned (median
    # over a small window for robustness against local roundoff).
    lo = max(1, iotp - 3)
    hi = min(ncut - 1, iotp + 4)
    if hi <= lo:
        lo, hi = max(1, ncut // 2), max(2, ncut // 2 + 1)
    w_win = (p_reg[lo:hi] * q_irr[lo:hi] - q_reg[lo:hi] * p_irr[lo:hi])
    w_const = float(np.median(w_win))

    return p_reg, q_reg, p_irr, q_irr, w_const, ncut, iotp, nzero


def green_solve(chi_p, chi_q, p_reg, q_reg, p_irr, q_irr,
                 w_const, difr, ncut, *, stable_tail=False):
    """Particular solution of ``(h_D + V_loc - eps) phi = chi``.

    Variation-of-parameters assembly on the grid (see module docstring).
    ``chi_p/chi_q`` are full-length arrays; the result is returned on
    ``[0, ncut)`` and is zero at index 0 (P(0) = Q(0) = 0).
    """
    u = (p_reg * chi_p[:ncut] + q_reg * chi_q[:ncut]) * difr[:ncut]
    v = (p_irr[:ncut] * chi_p[:ncut] + q_irr[:ncut] * chi_q[:ncut]) \
        * difr[:ncut]
    cum_u = slag6(1.0, u, ncut)
    if stable_tail:
        # Direct inward integration avoids cancellation of two nearly equal
        # cumulative totals in the far tail. Used by constrained responses;
        # retain the legacy path for existing HF reproducibility controls.
        v_tail = slag6(1.0, v[::-1].copy(), ncut)[::-1]
    else:
        cum_v = slag6(1.0, v, ncut)
        v_tail = cum_v[ncut - 1] - cum_v
    fac = -1.0 / (SL * w_const)
    p = fac * (p_reg * v_tail + p_irr[:ncut] * cum_u)
    q = fac * (q_reg * v_tail + q_irr[:ncut] * cum_u)
    return p, q


@njit(cache=True)
def box_inward(energy, kappa, r, va, vb, vc, vd):
    """Inward homogeneous solution with P(R)=0, Q(R)=1, scaled as needed."""
    n = len(r)
    p, q = np.zeros(n), np.zeros(n)
    q[-1] = 1.
    for i in range(n-2, 0, -1):
        pp, qq, log_scale, _, _, _, _ = dir_integrate(
            energy, float(kappa), 1e-11, 0, p[i+1], q[i+1], r[i+1], r[i],
            va[i], vb[i], vc[i], vd[i])
        if log_scale > 0:
            factor = np.exp(-log_scale)
            p[i+1:] *= factor
            q[i+1:] *= factor
        p[i], q[i] = pp, qq
    return p, q


def box_pair(energy, kappa, radius, potential):
    n = len(radius)
    p, q = np.zeros(n), np.zeros(n)
    doutw(energy, 1e-11, float(kappa), 10**9, 10**9, 0,
           radius, np.arange(n, dtype=np.int64), n, *potential, p, q)
    pi, qi = box_inward(energy, kappa, radius, *potential)
    w = p[1:]*qi[1:]-q[1:]*pi[1:]
    if not np.all(np.isfinite(w)):
        raise FloatingPointError('non-finite box Dirac fundamental pair')
    wronskian = float(np.median(w[n//3:2*n//3]))
    return p, q, pi, qi, wronskian, n, n//2, 0




@njit(cache=True)
def scaled_box_pair(energy, kappa, r, va, vb, vc, vd):
    """Unit-norm fundamental spinors and log amplitudes on the full box.

    Per-step scaling prevents overflow/underflow of deep-core solutions.
    P(R)=0 supplies a finite-box boundary without cutting off nonlocal
    exchange sources at a homogeneous orbital's practical infinity.
    """
    n = len(r)
    regular, inward = np.zeros((2, n)), np.zeros((2, n))
    lr, li = np.zeros(n), np.zeros(n)
    nodes = 0
    for i in range(n-1):
        p, q, logscale, _, _, count, _ = dir_integrate(energy, float(kappa),
            1e-11, 0, regular[0, i], regular[1, i], r[i], r[i+1],
            va[i], vb[i], vc[i], vd[i])
        norm = np.hypot(p, q)
        regular[0, i+1], regular[1, i+1] = p/norm, q/norm
        lr[i+1] = lr[i]+logscale+np.log(norm)
        nodes += count
    inward[1, -1] = 1.
    for i in range(n-2, 0, -1):
        p, q, logscale, _, _, _, _ = dir_integrate(energy, float(kappa),
            1e-11, 0, inward[0, i+1], inward[1, i+1], r[i+1], r[i],
            va[i], vb[i], vc[i], vd[i])
        norm = np.hypot(p, q)
        inward[0, i], inward[1, i] = p/norm, q/norm
        li[i] = li[i+1]+logscale+np.log(norm)
    lr[0], li[0] = lr[1], li[1]
    # Local eigenvalues of the first-order radial Dirac system are ±sqrt(d).
    # Integrate the evanescent factor explicitly; in oscillatory regions d<0
    # and ordinary polynomial source integration is appropriate. This scale
    # does not use the artificial amplitude of a unit-normalized spinor.
    rates = np.zeros(n)
    for i in range(1, n):
        midpoint = .5*(r[i]+r[i-1])
        j = i-1
        potential = (va[j]+midpoint*(vb[j]+midpoint*(vc[j]+midpoint*vd[j])))/midpoint
        difference = energy-potential
        discriminant = (kappa/midpoint)**2-difference*(difference+2*SL**2)/SL**2
        rates[i] = np.sqrt(max(0., discriminant))*(r[i]-r[i-1])
    return regular, inward, lr, li, nodes, rates


@njit(cache=True)
def _scaled_slag6(values, logs):
    """exp(-logs[i]) * integral(values * exp(logs)), in linear time."""
    n = len(values)
    result = np.zeros(n)
    first = np.array([475., 1427., -798., 482., -173., 27.])/1440.
    second = np.array([-27., 637., 1022., -258., 77., -11.])/1440.
    center = np.array([11., -93., 802., 802., -93., 11.])/1440.
    for i in range(1, n):
        start, reverse = i-3, False
        weights = center
        if i < 3:
            start = 0
            weights = first if i == 1 else second
        elif i >= n-2:
            start, reverse = n-6, True
            weights = first if i == n-1 else second
        value = result[i-1]*np.exp(logs[i-1]-logs[i])
        for j in range(6):
            index = start+j
            value += weights[5-j if reverse else j]*values[index]*np.exp(logs[index]-logs[i])
        result[i] = value
    return result


def prepare_scaled_green(pair):
    """Reusable source-quadrature maps for a fixed fundamental pair."""
    rates = pair[5]
    inward_rates = np.concatenate((np.zeros(1), rates[:0:-1]))
    return (exponential_prefix_weights(pair[2], rates),
            exponential_prefix_weights(pair[3][::-1].copy(), inward_rates))


def scaled_green_solve(chi_p, chi_q, pair, jacobian, quadrature=None):
    """Full-box Green action, independent of fundamental-solution scale."""
    regular, inward, lr, li, _, _ = pair
    if quadrature is None:
        quadrature = prepare_scaled_green(pair)
    u = (regular[0]*chi_p+regular[1]*chi_q)*jacobian
    v = (inward[0]*chi_p+inward[1]*chi_q)*jacobian
    prefix = apply_exponential_prefix(u, *quadrature[0])
    suffix = apply_exponential_prefix(v[::-1].copy(), *quadrature[1])[::-1]
    wronskian = regular[0]*inward[1]-regular[1]*inward[0]
    if np.any(abs(wronskian[1:]) < 1e-20):
        raise FloatingPointError('singular scaled Dirac Green operator')
    result = np.zeros_like(regular)
    result[:, 1:] = -(regular[:, 1:]*suffix[1:] + inward[:, 1:]*prefix[1:])/(SL*wronskian[1:])
    return result[0], result[1]
