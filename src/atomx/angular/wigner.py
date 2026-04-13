"""Wigner 3j, 6j, and 9j symbols via Racah formula.

Uses log-factorial caching for numerical stability at high angular momenta.
All angular momentum arguments are twice the physical value (integer storage):
  j=1/2 → tj=1, j=1 → tj=2, j=3/2 → tj=3, etc.

The "t" prefix means "twice j" throughout this module.
"""

import numpy as np
import numba as nb

# Pre-compute log-factorials up to 300! (sufficient for most atomic physics)
_MAX_FACT = 500
_LOG_FACT = np.zeros(_MAX_FACT + 1)
for _i in range(2, _MAX_FACT + 1):
    _LOG_FACT[_i] = _LOG_FACT[_i - 1] + np.log(_i)


@nb.njit(cache=True)
def _log_factorial(n):
    """Log of n! using cached values."""
    if n < 0:
        return -1e30  # sentinel for invalid
    if n <= 500:
        # Inline the precomputed table
        result = 0.0
        for i in range(2, n + 1):
            result += np.log(i)
        return result
    # Stirling's approximation for very large n
    return n * np.log(n) - n + 0.5 * np.log(2.0 * np.pi * n)


@nb.njit(cache=True)
def _triangle_coefficient(ta, tb, tc):
    """Log of the triangle coefficient Delta(a,b,c).

    Delta(a,b,c)^2 = (a+b-c)!(a-b+c)!(-a+b+c)! / (a+b+c+1)!

    All arguments are twice the angular momentum (integers).
    Returns log(Delta^2) and whether the triangle is valid.
    """
    # Triangle condition: |a-b| <= c <= a+b, and a+b+c is even
    if (ta + tb + tc) % 2 != 0:
        return 0.0, False
    s = ta + tb + tc
    if s < 0:
        return 0.0, False
    n1 = (ta + tb - tc) // 2
    n2 = (ta - tb + tc) // 2
    n3 = (-ta + tb + tc) // 2
    n4 = (ta + tb + tc) // 2 + 1
    if n1 < 0 or n2 < 0 or n3 < 0:
        return 0.0, False
    log_delta_sq = _log_factorial(n1) + _log_factorial(n2) + _log_factorial(n3) - _log_factorial(n4)
    return log_delta_sq, True


@nb.njit(cache=True)
def wigner_3j(tj1, tj2, tj3, tm1, tm2, tm3):
    """Wigner 3j symbol using Racah's formula.

    ( j1  j2  j3 )
    ( m1  m2  m3 )

    All arguments are TWICE the angular momentum values (integers).
    E.g., j=1/2, m=1/2 → tj=1, tm=1.

    Returns
    -------
    float
        Value of the 3j symbol.
    """
    # Selection rules
    if tm1 + tm2 + tm3 != 0:
        return 0.0
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm3) > tj3:
        return 0.0
    if (tj1 + tm1) % 2 != 0 or (tj2 + tm2) % 2 != 0 or (tj3 + tm3) % 2 != 0:
        return 0.0

    log_delta_sq, valid = _triangle_coefficient(tj1, tj2, tj3)
    if not valid:
        return 0.0

    # Racah formula summation
    # Convert to half-integer arithmetic
    j1, j2, j3 = tj1 // 2, tj2 // 2, tj3 // 2
    m1, m2, m3 = tm1 // 2, tm2 // 2, tm3 // 2

    # For half-integer j, we work with twice-j directly
    # The sum index t runs over integer values satisfying:
    #   t >= max(0, j2-j3-m1, j1-j3+m2) (using twice-values)
    #   t <= min(j1+j2-j3, j1-m1, j2+m2)  (using twice-values)

    # Using twice-values throughout:
    s1 = (tj1 + tj2 - tj3) // 2
    s2 = (tj1 - tm1) // 2
    s3 = (tj2 + tm2) // 2
    s4 = (tj2 - tj3 - tm1) // 2  # can be negative
    s5 = (tj1 - tj3 + tm2) // 2  # can be negative

    t_min = max(0, max(-s4, -s5))
    # Clamp s4, s5 for t_min calculation:
    # t >= 0, t >= -(j2-j3) + m1/2 ... let's use the standard formula
    # t_min = max(0, j2-j3-m1, j1-j3+m2) — but these use half-integers
    # With twice-values:
    t_min_a = 0
    t_min_b = (tj2 - tj3 - tm1) // 2  # = j2 - j3 - m1 (can be half-int → but j+m is always int)
    t_min_c = (tj1 - tj3 + tm2) // 2
    # Hmm, these should all be integers since j±m is always integer
    t_min = max(t_min_a, max(t_min_b, t_min_c))

    t_max = min(s1, min(s2, s3))

    if t_min > t_max:
        return 0.0

    # Prefactor
    log_pre = 0.5 * log_delta_sq
    log_pre += 0.5 * (_log_factorial(s2) + _log_factorial((tj1 + tm1) // 2)
                       + _log_factorial(s3) + _log_factorial((tj2 - tm2) // 2)
                       + _log_factorial((tj3 + tm3) // 2) + _log_factorial((tj3 - tm3) // 2))

    # Sum over t: sign per term is (-1)^t only
    result = 0.0
    for t in range(t_min, t_max + 1):
        log_term = (_log_factorial(t) + _log_factorial(s1 - t)
                    + _log_factorial(s2 - t) + _log_factorial(s3 - t)
                    + _log_factorial(t - t_min_b) + _log_factorial(t - t_min_c))
        sign = 1.0 if t % 2 == 0 else -1.0
        result += sign * np.exp(log_pre - log_term)

    # Overall phase: (-1)^(j1-j2-m3)
    phase_exp = (tj1 - tj2 - tm3) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    return phase * result


@nb.njit(cache=True)
def wigner_6j(tj1, tj2, tj3, tl1, tl2, tl3):
    """Wigner 6j symbol using Racah formula.

    { j1  j2  j3 }
    { l1  l2  l3 }

    All arguments are TWICE the angular momentum values.

    Returns
    -------
    float
    """
    # Check four triangle conditions
    _, v1 = _triangle_coefficient(tj1, tj2, tj3)
    _, v2 = _triangle_coefficient(tj1, tl2, tl3)
    _, v3 = _triangle_coefficient(tl1, tj2, tl3)
    _, v4 = _triangle_coefficient(tl1, tl2, tj3)
    if not (v1 and v2 and v3 and v4):
        return 0.0

    ld1, _ = _triangle_coefficient(tj1, tj2, tj3)
    ld2, _ = _triangle_coefficient(tj1, tl2, tl3)
    ld3, _ = _triangle_coefficient(tl1, tj2, tl3)
    ld4, _ = _triangle_coefficient(tl1, tl2, tj3)
    log_delta = 0.5 * (ld1 + ld2 + ld3 + ld4)

    # Summation bounds
    s1 = (tj1 + tj2 + tj3) // 2
    s2 = (tj1 + tl2 + tl3) // 2
    s3 = (tl1 + tj2 + tl3) // 2
    s4 = (tl1 + tl2 + tj3) // 2
    t_min = max(s1, max(s2, max(s3, s4)))

    r1 = (tj1 + tj2 + tl1 + tl2) // 2
    r2 = (tj2 + tj3 + tl2 + tl3) // 2
    r3 = (tj3 + tj1 + tl3 + tl1) // 2
    t_max = min(r1, min(r2, r3))

    if t_min > t_max:
        return 0.0

    result = 0.0
    for t in range(t_min, t_max + 1):
        log_num = _log_factorial(t + 1)
        log_den = (_log_factorial(t - s1) + _log_factorial(t - s2)
                   + _log_factorial(t - s3) + _log_factorial(t - s4)
                   + _log_factorial(r1 - t) + _log_factorial(r2 - t)
                   + _log_factorial(r3 - t))
        sign = 1.0 if t % 2 == 0 else -1.0
        result += sign * np.exp(log_num - log_den + log_delta)

    return result


@nb.njit(cache=True)
def wigner_9j(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9):
    """Wigner 9j symbol as sum over 6j symbols.

    { j1  j2  j3 }
    { j4  j5  j6 }
    { j7  j8  j9 }

    All arguments are TWICE the angular momentum values.
    """
    # Sum over intermediate angular momentum t
    t_min = max(abs(tj1 - tj9), max(abs(tj4 - tj8), abs(tj2 - tj6)))
    t_max = min(tj1 + tj9, min(tj4 + tj8, tj2 + tj6))

    result = 0.0
    for tt in range(t_min, t_max + 1, 2):  # step by 2 (twice-values)
        val = (wigner_6j(tj1, tj4, tj7, tj8, tj9, tt)
               * wigner_6j(tj2, tj5, tj8, tj4, tt, tj6)
               * wigner_6j(tj3, tj6, tj9, tt, tj1, tj2))
        sign = 1.0 if tt % 2 == 0 else -1.0
        result += sign * (tt + 1) * val

    return result
