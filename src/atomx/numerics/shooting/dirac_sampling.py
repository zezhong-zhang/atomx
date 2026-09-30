"""Dense output from the same Dirac power series used for propagation.

The potential spline and regular origin boundary condition are retained.
Sampling a continuum on an observable's mesh must not replace its accurate
ODE solution by a piecewise-linear wave. One polynomial is reused for all
requested points in an integration interval. If a full interval cannot be
represented by one converged series, the existing adaptive integrator samples
that interval instead. No finite differences or new differential equation
are introduced here.
"""
import math

import numba as nb
import numpy as np

from .dirac_integrator import dir_integrate
from .dirac_power_series import dir0


@nb.njit(cache=True, nogil=True)
def _origin_seed(energy, kappa, tolerance, radius, p, q, a, b, c, d):
    """Resolve a regular seed and match its scale to the stored endpoint.

    A rapidly varying first potential interval may need multiple propagation
    steps. Normalization at an arbitrary shorter seed radius must be undone
    before sampling the physical wave; backward integration towards the
    singular origin would instead amplify the irregular solution.
    """
    seed = radius
    for _ in range(64):
        seed *= .5
        if seed <= 1e-10:
            break
        sp, sq, ca, cb, s, t, terms = dir0(
            energy, float(kappa), tolerance, 0., 0., 0., seed, a, b, c, d)
        if terms < 60:
            ep, eq, logscale, _, _, _, _ = dir_integrate(
                energy, float(kappa), tolerance, 0, sp, sq, seed, radius, a, b, c, d)
            scale = (ep*p+eq*q)/(ep*ep+eq*eq)*math.exp(-logscale)
            return seed, sp, sq, ca, cb, s, t, terms, scale
    raise RuntimeError('continuum regular origin seed cannot be resolved')


def validate_output_grid(grid, radius):
    """Reject extrapolation: a continuum wave does not have a constant tail."""
    target = np.asarray(grid, dtype=float)
    if (target.ndim != 1 or not np.all(np.isfinite(target))
            or np.any(np.diff(target) < 0)):
        raise ValueError('continuum output grid must be finite, one-dimensional and nondecreasing')
    if len(target) and (target[0] < radius[0] or target[-1] > radius[-1]):
        raise ValueError('continuum output grid lies outside the solved radial interval')
    return np.ascontiguousarray(target)


@nb.njit(cache=True, nogil=True)
def sample_dirac_solution(target, r, p, q, ind, va, vb, vc, vd, energy, kappa, tolerance):
    """Sample normalized P/Q without interpolation error from a linear wave.

    ``target`` must be sorted and lie inside ``r``; the Python API validates
    it once before threaded work. DIR0 coefficients use x=(r-r_left)/h;
    the first interval includes its regular Frobenius powers x**s, x**(s+t).
    """
    sampled_p = np.empty(len(target))
    sampled_q = np.empty(len(target))
    interval = -1
    terms = 0
    ca = np.empty(60)
    cb = np.empty(60)
    s = t = factor = 0.
    seed = sp = sq = origin_scale = 0.
    origin_split = False
    left = 0
    anchor = end = 0
    local_p = local_q = 0.
    for i in range(len(target)):
        radius = target[i]
        while left+1 < len(r) and r[left+1] <= radius:
            left += 1
        if radius == r[left]:
            sampled_p[i], sampled_q[i] = p[left], q[left]
            continue
        right = left+1
        index = ind[left]
        if interval != left:
            anchor, end = left, right
            factor = 1.
            local_p, local_q = p[left], q[left]
            if r[left] > 1e-10:
                # Homogeneous Dirac evolution is independent of amplitude.
                # Resolve its local shape at unit scale; normalized high-l
                # continuum waves can be subnormal or zero in the core.
                factor = max(abs(p[left]), abs(q[left]))
                right_scale = max(abs(p[right]), abs(q[right]))
                if factor*tolerance < np.finfo(np.float64).tiny and right_scale > factor:
                    # The right endpoint retains more significant digits
                    # when outward renormalization underflowed the left one.
                    anchor, end = right, left
                    factor = right_scale
                if factor != 0.:
                    local_p, local_q = p[anchor]/factor, q[anchor]/factor
            _, _, ca, cb, s, t, terms = dir0(
                energy, float(kappa), tolerance, local_p, local_q, r[anchor], r[end],
                va[index], vb[index], vc[index], vd[index])
            origin_split = r[left] <= 1e-10 and terms >= 60
            if origin_split:
                seed, sp, sq, ca, cb, s, t, terms, origin_scale = _origin_seed(
                    energy, kappa, tolerance, r[right], p[right], q[right],
                    va[index], vb[index], vc[index], vd[index])
                factor = origin_scale/abs(np.sum(ca[:terms]))
            elif r[left] <= 1e-10 and terms < 60:
                # DIR0 fixes the origin-series amplitude independently of
                # the energy-normalized wave. Recover its single scale.
                factor = p[right]/np.sum(ca[:terms])
            interval = left
        if origin_split and radius >= seed:
            pf, qf, logscale, _, _, _, _ = dir_integrate(
                energy, float(kappa), tolerance, 0, sp, sq, seed, radius,
                va[index], vb[index], vc[index], vd[index])
            sampled_p[i] = pf*origin_scale*math.exp(logscale)
            sampled_q[i] = qf*origin_scale*math.exp(logscale)
        elif terms >= 60:
            pf, qf, logscale, _, _, _, _ = dir_integrate(
                energy, float(kappa), tolerance, 0, local_p, local_q, r[anchor], radius,
                va[index], vb[index], vc[index], vd[index])
            factor_step = factor*math.exp(logscale)
            sampled_p[i], sampled_q[i] = pf*factor_step, qf*factor_step
        else:
            x = radius/seed if origin_split else (radius-r[anchor])/(r[end]-r[anchor])
            pv = qv = 0.
            for order in range(terms-1, -1, -1):
                pv = pv*x+ca[order]
                qv = qv*x+cb[order]
            sampled_p[i] = factor*x**s*pv
            sampled_q[i] = factor*x**(s+t)*qv
    if not np.all(np.isfinite(sampled_p)) or not np.all(np.isfinite(sampled_q)):
        raise FloatingPointError('nonfinite Dirac dense output')
    return sampled_p, sampled_q
