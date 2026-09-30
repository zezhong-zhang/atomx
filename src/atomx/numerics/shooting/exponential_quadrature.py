"""Exponentially fitted, scaled prefix integration of sampled sources.

On each unit-index cell, factor out a supplied exponential rate. Interpolate
the remaining source with six neighbouring nodes and integrate its product
with the exponential using analytic polynomial moments. Keeping sampled log
amplitudes in the residual source preserves cancellations with a normalized
spinor's direction at nodes and boundaries. This remains a sixth-order source
interpolation, not exact source integration; refinement is required.
"""
import numpy as np
from numba import njit


def _tables():
    coefficients = np.empty((5, 6, 6))
    for shift in range(1, 6):
        stencil = np.arange(6)-shift
        for j in range(6):
            others = np.delete(stencil, j)
            coefficients[shift-1, j] = (
                np.polynomial.polynomial.polyfromroots(others)
                / np.prod(stencil[j]-others))
    coefficients.setflags(write=False)
    return coefficients


_COEFFICIENTS = _tables()


@njit(cache=True)
def _exponential_moments(slope, moments):
    """Integrals of x**m exp(slope*x) on [-1, 0], m = 0,...,5.

    Upward recurrence cancels near zero. There, evaluate the highest moment
    with a positive series and recurse downward, adding only positive terms.
    Above eight the upward recurrence is well conditioned. No moment depends
    on the source, so the resulting Green map remains linear.
    """
    tail = np.exp(-slope)
    if slope < 8.:
        term = tail/6.
        value = term
        for k in range(1, 128):
            term *= slope/(6+k)
            value += term
            if term <= 1e-16*value:
                break
        moments[5] = value
        for m in range(5, 0, -1):
            moments[m-1] = (slope*moments[m]+tail)/m
    else:
        moments[0] = -np.expm1(-slope)/slope
        for m in range(1, 6):
            moments[m] = (m*moments[m-1]-tail)/slope
    for m in range(1, 6, 2):
        moments[m] = -moments[m]


@njit(cache=True)
def exponential_prefix_weights(logs, rates=None):
    """Prepare a linear source-integration map for one fixed log-amplitude grid.

    ``rates[i]`` is the nonnegative exponential scale on cell [i-1,i], in
    index units. The Dirac caller supplies local evanescent rates from its ODE.
    If omitted, use absolute adjacent log differences, useful for analytic
    exponential controls. A zero rate recovers ordinary six-node integration.
    """
    n = len(logs)
    if n < 6 or not np.all(np.isfinite(logs)):
        raise ValueError('scaled integration requires at least six finite log amplitudes')
    if rates is None:
        rates = np.zeros(n)
        rates[1:] = np.abs(np.diff(logs))
    if len(rates) != n or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError('scaled integration needs matching nonnegative finite rates')
    weights = np.zeros((n, 6))
    decay = np.ones(n)
    moments = np.empty(6)
    for i in range(1, n):
        step = logs[i]-logs[i-1]
        slope = rates[i]
        if abs(step) > 100 or slope > 100:
            raise ValueError('radial grid too coarse for scaled Green quadrature')
        decay[i] = np.exp(-step)
        start = max(0, min(i-3, n-6))
        shift = i-start-1
        _exponential_moments(slope, moments)
        for j in range(6):
            for m in range(6):
                weights[i, j] += _COEFFICIENTS[shift, j, m]*moments[m]
            weights[i, j] *= np.exp(logs[start+j]-logs[i]-slope*(start+j-i))
    if not np.all(np.isfinite(weights)):
        raise ValueError('non-finite scaled integration weights; refine radial grid')
    return decay, weights


@njit(cache=True)
def apply_exponential_prefix(values, decay, weights):
    """Apply prepared integration weights in O(N), without source-dependent fitting."""
    n = len(values)
    if len(decay) != n or weights.shape != (n, 6):
        raise ValueError('scaled integration source and weights have different sizes')
    result = np.zeros(n)
    for i in range(1, n):
        start = max(0, min(i-3, n-6))
        value = result[i-1]*decay[i]
        for j in range(6):
            value += weights[i, j]*values[start+j]
        result[i] = value
    return result
