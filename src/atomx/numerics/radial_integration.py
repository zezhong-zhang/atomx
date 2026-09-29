"""Cumulative radial integration using the representation's native rule."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _gauss_prefix_matrix(order):
    """Integrals of Gauss-node cardinal polynomials from -1 to each node."""
    from numpy.polynomial.legendre import leggauss, legint, legval, legvander
    nodes, _ = leggauss(order)
    coefficients = np.linalg.solve(legvander(nodes, order-1), np.eye(order))
    integral = legint(coefficients, axis=0)
    matrix = (legval(nodes, integral).T-legval(-1., integral))
    matrix.setflags(write=False)
    return matrix


def cumulative_radial_integral(values, workspace):
    """Integral to each native node; values have the workspace grid shape.

    B-spline Gauss cells use local polynomial partial integrals and full Gauss
    cell integrals. Shooting uses SLAG6 in uniform index with dr/ds. Legacy
    arbitrary sampled grids retain explicit trapezoidal integration.
    """
    radius = np.asarray(workspace['gr'])
    weights = np.asarray(workspace['grw'])
    values = np.asarray(values).reshape(radius.shape)
    if radius.ndim == 2:
        width_half = weights.sum(axis=1)/2
        full = np.sum(values*weights, axis=1)
        before = np.concatenate(([0.], np.cumsum(full)[:-1]))
        return before[:, None] + width_half[:, None]*(values @ _gauss_prefix_matrix(radius.shape[1]).T)
    if workspace.get('integration_rule') == 'slag6':
        from atomx.numerics.shooting.quadrature import slag6
        offset = workspace['origin_offset']
        jacobian = np.asarray(workspace['jacobian'])
        source = np.zeros_like(jacobian)
        source[offset:] = values
        return slag6(1., source*jacobian, len(source))[offset:]
    from scipy.integrate import cumulative_trapezoid
    return cumulative_trapezoid(values, radius, initial=0.)
