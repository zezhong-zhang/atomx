"""SLAG6: 6-point Lagrange integration — radial.f lines 7456-7498.

Piecewise six-point Lagrange integration of a uniformly tabulated function.
For non-uniform grids, the caller multiplies by the Jacobian DIFR(I) first.
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def slag6(h, y, n):
    """Piecewise six-point Lagrange cumulative integration.

    Input:
        h:    grid spacing (1.0 for index-spaced data with Jacobian pre-applied)
        y:    array of function values y(1:n)
        n:    number of data points (must be >= 6)

    Returns:
        s:    cumulative integral s(i) = integral from x(1) to x(i)
    """
    s = np.zeros(n)
    if n < 6:
        # Fallback to trapezoidal for tiny arrays
        for i in range(1, n):
            s[i] = s[i - 1] + 0.5 * h * (y[i] + y[i - 1])
        return s

    hr = h / 1440.0
    y1 = 0.0
    y2 = y[0]
    y3 = y[1]
    y4 = y[2]

    s[0] = 0.0
    s[1] = hr * (475.0 * y2 + 1427.0 * y3 - 798.0 * y4
                 + 482.0 * y[3] - 173.0 * y[4] + 27.0 * y[5])
    s[2] = s[1] + hr * (-27.0 * y2 + 637.0 * y3 + 1022.0 * y4
                         - 258.0 * y[3] + 77.0 * y[4] - 11.0 * y[5])

    for i in range(3, n - 2):
        y1 = y2
        y2 = y3
        y3 = y4
        y4 = y[i]
        s[i] = s[i - 1] + hr * (11.0 * (y1 + y[i + 2])
                                  - 93.0 * (y2 + y[i + 1])
                                  + 802.0 * (y3 + y4))

    y5 = y[n - 2]
    y6 = y[n - 1]
    s[n - 2] = s[n - 3] + hr * (-27.0 * y6 + 637.0 * y5 + 1022.0 * y4
                                  - 258.0 * y3 + 77.0 * y2 - 11.0 * y1)
    s[n - 1] = s[n - 2] + hr * (475.0 * y6 + 1427.0 * y5 - 798.0 * y4
                                  + 482.0 * y3 - 173.0 * y2 + 27.0 * y1)
    return s
