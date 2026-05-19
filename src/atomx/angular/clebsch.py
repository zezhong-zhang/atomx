"""Clebsch-Gordan coefficients from Wigner 3j symbols.

<j1 m1; j2 m2 | j3 m3> = (-1)^(j1-j2+m3) * sqrt(2*j3+1) * (j1  j2  j3 )
                                                              (m1  m2 -m3)
"""

import numpy as np
from atomx.angular.wigner import wigner_3j
from atomx._numba import nb


@nb.njit(cache=True)
def clebsch_gordan(tj1, tm1, tj2, tm2, tj3, tm3):
    """Clebsch-Gordan coefficient <j1 m1; j2 m2 | j3 m3>.

    All arguments are TWICE the angular momentum values.

    Returns
    -------
    float
    """
    # Selection rule: m1 + m2 = m3
    if tm1 + tm2 != tm3:
        return 0.0

    # CG = (-1)^(j1-j2+m3) * sqrt(2*j3+1) * 3j(j1,j2,j3; m1,m2,-m3)
    w3j = wigner_3j(tj1, tj2, tj3, tm1, tm2, -tm3)

    phase_exp = (tj1 - tj2 + tm3) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    return phase * np.sqrt(tj3 + 1.0) * w3j
