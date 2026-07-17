"""Angular factors for one-body pseudoscalar operators."""

from __future__ import annotations

import math

from atomx.angular.coefficients import _l_from_kappa


def pseudoscalar_angular(kappa_a: int, kappa_r: int) -> float:
    """Reduced angular factor for a rank-0 odd-parity operator.

    This is the common angular structure for large-small cross
    pseudoscalar operators such as the spin-independent PNC weak
    Hamiltonian and the T,P-odd eEDM operator.  The operator has odd
    parity and rank zero, so it connects only opposite-parity states
    with the same total ``j``.
    """
    l_a = _l_from_kappa(kappa_a)
    l_r = _l_from_kappa(kappa_r)

    if (l_a + l_r) % 2 == 0:
        return 0.0

    tj_a = 2 * abs(kappa_a) - 1
    tj_r = 2 * abs(kappa_r) - 1
    if tj_a != tj_r:
        return 0.0

    phase_exp = (tj_r + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0
    return phase * (kappa_a - kappa_r) / math.sqrt(tj_a + 1.0)


__all__ = ["pseudoscalar_angular"]
