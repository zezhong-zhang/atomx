"""Estimates of unresolved tails of nonnegative series."""
import math


def geometric_tail_estimate(profile, last, *, ratio_cap=0.85):
    """Estimate the sum beyond ``last`` from its last two terms.

    A geometrically decaying series has tail ``q_last*r/(1-r)``.
    Cap the ratio to keep the estimate bounded outside that regime, and
    return ``False`` there. A bounded estimate is not an error bound or
    proof of convergence; consumers must retain the flag and refine the
    explicit series independently. The estimate stays continuous as the
    final terms change from decaying to rising.
    """
    if not 0 < ratio_cap < 1:
        raise ValueError('geometric tail ratio cap must lie strictly between zero and one')
    if last < 2:
        return 0.0, True
    q1, q0 = float(profile[last]), float(profile[last-1])
    if not (math.isfinite(q1) and math.isfinite(q0)) or q1 <= 0 or q0 <= 0:
        return 0.0, True
    ratio = q1/q0
    if ratio >= ratio_cap:
        return q1*ratio_cap/(1-ratio_cap), False
    return q1*ratio/(1-ratio), True
