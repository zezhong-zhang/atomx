"""Exponentially fitted, scaled prefix integration of sampled sources.

On each unit-index cell, factor out a supplied exponential rate. Interpolate
the remaining source with six neighbouring nodes and integrate its product
with the exponential using subdivided Gauss quadrature. Keeping sampled log
amplitudes in the residual source preserves cancellations with a normalized
spinor's direction at nodes and boundaries. This remains a sixth-order source
interpolation, not exact source integration; refinement is required.
"""
import numpy as np
from numba import njit


def _tables():
    nodes, weights = np.polynomial.legendre.leggauss(16)
    # The Green kernel currently permits at most 100 log units per cell.
    basis = np.zeros((14, 5, 13*16, 6))
    quadrature = np.zeros((14, 13*16))
    for panels in range(1, 14):
        points = (-1 + (np.arange(panels)[:, None] + .5 + .5*nodes)/panels).ravel()
        quadrature[panels, :len(points)] = np.tile(weights*.5/panels, panels)
        for shift in range(1, 6):
            stencil = np.arange(6)-shift
            for j in range(6):
                values = np.ones_like(points)
                for k in range(6):
                    if j != k:
                        values *= (points-stencil[k])/(stencil[j]-stencil[k])
                basis[panels, shift-1, :len(points), j] = values
    basis.setflags(write=False)
    quadrature.setflags(write=False)
    return basis, quadrature


_BASIS, _QUADRATURE = _tables()
_NODES = np.polynomial.legendre.leggauss(16)[0]
_NODES.setflags(write=False)


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
    for i in range(1, n):
        step = logs[i]-logs[i-1]
        slope = rates[i]
        if abs(step) > 100 or slope > 100:
            raise ValueError('radial grid too coarse for scaled Green quadrature')
        decay[i] = np.exp(-step)
        start = max(0, min(i-3, n-6))
        shift = i-start-1
        panels = max(1, int(np.ceil(slope/8)))
        for k in range(16*panels):
            point = -1+(k//16+.5+.5*_NODES[k % 16])/panels
            factor = _QUADRATURE[panels, k]*np.exp(slope*point)
            for j in range(6):
                weights[i, j] += factor*_BASIS[panels, shift, k, j]
        for j in range(6):
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
