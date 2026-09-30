"""Regular scaled special functions for radial numerical kernels."""

import numpy as np
from scipy.special import factorial2, spherical_jn


def scaled_spherical_jn(order, x):
    """Return (2*order+1)!! j_order(x)/x**order, with value 1 at zero.

    This even function is evaluated by its regular series near zero,
    avoiding division of tiny quantities and retaining retardation terms.
    """
    if order < 0 or int(order) != order:
        raise ValueError('spherical Bessel order must be a nonnegative integer')
    x = np.abs(np.asarray(x, dtype=float))
    factor = np.empty_like(x)
    small = x < .1
    x2 = x[small]**2
    term = np.ones_like(x2)
    series = term.copy()
    for n in range(1, 7):
        term *= -x2 / (2 * n * (2 * order + 2 * n + 1))
        series += term
    factor[small] = series
    factor[~small] = (factorial2(2 * order + 1) * spherical_jn(order, x[~small])
                      / x[~small]**order)
    return factor
