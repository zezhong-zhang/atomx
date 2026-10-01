"""Wigner 3j, 6j, and 9j symbols via Racah formula.

Short Racah sums use cached log-factorials. Longer alternating sums use
exact integer arithmetic before the final square root to avoid cancellation.
All angular momentum arguments are twice the physical value (integer storage):
  j=1/2 → tj=1, j=1 → tj=2, j=3/2 → tj=3, etc.

The "t" prefix means "twice j" throughout this module.
"""

import math
from functools import lru_cache

import numba as nb
import numpy as np

# Pre-compute log-factorials for the short-sum numerical path.
_MAX_FACT = 500
_LOG_FACT = np.zeros(_MAX_FACT + 1)
for _i in range(2, _MAX_FACT + 1):
    _LOG_FACT[_i] = _LOG_FACT[_i - 1] + np.log(_i)


@lru_cache(maxsize=4096)
def _factorial(n):
    return math.factorial(n)


def _triangle_ratio(a, b, c):
    """Exact numerator/denominator of Delta(a/2,b/2,c/2)^2."""
    return (_factorial((a+b-c)//2) * _factorial((a-b+c)//2)
            * _factorial((-a+b+c)//2), _factorial((a+b+c)//2+1))


def _sqrt_ratio(num, den):
    """Square root of a positive integer ratio without squaring-range limits."""
    exponent = (num.bit_length() - den.bit_length()) // 2
    # Normalize the squared value before float conversion, so a representable
    # tiny Wigner symbol survives even when its square would underflow.
    ratio = (num / (den << (2*exponent)) if exponent >= 0
             else (num << (-2*exponent)) / den)
    return math.ldexp(math.sqrt(ratio), exponent)


@lru_cache(maxsize=32768)
def _wigner_3j_exact(j1, j2, j3, m1, m2, m3):
    """Racah sum of integer binomial products; selection rules checked upstream."""
    s1, s2, s3 = (j1+j2-j3)//2, (j1-m1)//2, (j2+m2)//2
    b, c = (j2-j3-m1)//2, (j1-j3+m2)//2
    lo, hi = max(0, b, c), min(s1, s2, s3)
    # Pair the six denominator factorials into three binomial coefficients.
    total = sum((-1 if t % 2 else 1) * math.comb(s1, t)
                * math.comb(s2-b, s2-t) * math.comb(s3-c, s3-t)
                for t in range(lo, hi+1))
    if total == 0:
        return 0.0
    num, den = _triangle_ratio(j1, j2, j3)
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        num *= _factorial((j+m)//2) * _factorial((j-m)//2)
    common = _factorial(s1) * _factorial(s2-b) * _factorial(s3-c)
    # Combine BEFORE conversion: individual factorials may exceed float range.
    value = _sqrt_ratio(num * total * total, den * common * common)
    return (-value if total < 0 else value) * (-1 if ((j1-j2-m3)//2) % 2 else 1)


@lru_cache(maxsize=32768)
def _wigner_6j_exact(j1, j2, j3, l1, l2, l3):
    """Integer Racah terms times an exact rational square root."""
    triangles = ((j1, j2, j3), (j1, l2, l3), (l1, j2, l3), (l1, l2, j3))
    ss = [sum(tri)//2 for tri in triangles]
    rr = [(j1+j2+l1+l2)//2, (j2+j3+l2+l3)//2, (j3+j1+l3+l1)//2]
    total = 0
    for t in range(max(ss), min(rr)+1):
        den = math.prod(_factorial(t-s) for s in ss)
        den *= math.prod(_factorial(r-t) for r in rr)
        # The denominator indices sum to t, so each term is an integer
        # multinomial coefficient times (t+1).
        term = _factorial(t+1) // den
        total += -term if t % 2 else term
    if total == 0:
        return 0.0
    num, den = total * total, 1
    for tri in triangles:
        n, d = _triangle_ratio(*tri)
        num *= n
        den *= d
    value = _sqrt_ratio(num, den)
    return -value if total < 0 else value


@nb.njit(cache=True)
def _log_factorial(n):
    """Log of n! using cached values."""
    if n < 0:
        return -1e30  # sentinel for invalid
    if n <= 500:
        return _LOG_FACT[n]
    return math.lgamma(n + 1.0)


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

    # Racah bounds; each factorial argument is an integer after selection.
    s1 = (tj1 + tj2 - tj3) // 2
    s2 = (tj1 - tm1) // 2
    s3 = (tj2 + tm2) // 2
    t_min_b = (tj2 - tj3 - tm1) // 2
    t_min_c = (tj1 - tj3 + tm2) // 2
    t_min = max(0, max(t_min_b, t_min_c))

    t_max = min(s1, min(s2, s3))

    if t_min > t_max:
        return 0.0

    if t_max - t_min > 4 or tj1 + tj2 + tj3 > 2 * _MAX_FACT - 2:
        with nb.objmode(exact='float64'):
            exact = _wigner_3j_exact(tj1, tj2, tj3, tm1, tm2, tm3)
        return exact

    # Prefactor
    log_pre = 0.5 * log_delta_sq
    log_pre += 0.5 * (_log_factorial(s2) + _log_factorial((tj1 + tm1) // 2)
                       + _log_factorial(s3) + _log_factorial((tj2 - tm2) // 2)
                       + _log_factorial((tj3 + tm3) // 2) + _log_factorial((tj3 - tm3) // 2))

    # Sum over t: sign per term is (-1)^t only
    result, magnitude = 0.0, 0.0
    for t in range(t_min, t_max + 1):
        log_term = (_log_factorial(t) + _log_factorial(s1 - t)
                    + _log_factorial(s2 - t) + _log_factorial(s3 - t)
                    + _log_factorial(t - t_min_b) + _log_factorial(t - t_min_c))
        sign = 1.0 if t % 2 == 0 else -1.0
        term = np.exp(log_pre - log_term)
        result += sign * term
        magnitude += term

    if magnitude > 16.0 * abs(result):
        with nb.objmode(exact='float64'):
            exact = _wigner_3j_exact(tj1, tj2, tj3, tm1, tm2, tm3)
        return exact

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
    # Check and evaluate each triangle once.
    ld1, v1 = _triangle_coefficient(tj1, tj2, tj3)
    ld2, v2 = _triangle_coefficient(tj1, tl2, tl3)
    ld3, v3 = _triangle_coefficient(tl1, tj2, tl3)
    ld4, v4 = _triangle_coefficient(tl1, tl2, tj3)
    if not (v1 and v2 and v3 and v4):
        return 0.0

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

    if t_max - t_min > 4 or t_max + 1 > _MAX_FACT:
        with nb.objmode(exact='float64'):
            exact = _wigner_6j_exact(tj1, tj2, tj3, tl1, tl2, tl3)
        return exact

    result, magnitude = 0.0, 0.0
    for t in range(t_min, t_max + 1):
        log_num = _log_factorial(t + 1)
        log_den = (_log_factorial(t - s1) + _log_factorial(t - s2)
                   + _log_factorial(t - s3) + _log_factorial(t - s4)
                   + _log_factorial(r1 - t) + _log_factorial(r2 - t)
                   + _log_factorial(r3 - t))
        sign = 1.0 if t % 2 == 0 else -1.0
        term = np.exp(log_num - log_den + log_delta)
        result += sign * term
        magnitude += term

    if magnitude > 16.0 * abs(result):
        with nb.objmode(exact='float64'):
            exact = _wigner_6j_exact(tj1, tj2, tj3, tl1, tl2, tl3)
        return exact

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
