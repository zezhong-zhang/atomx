"""Finite real rank-one CI response; independent of physical method policy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankOneCIResponse:
    """Normalized response pieces, excluding the trial radial increment.

    Original vectors are not changed. ``phase`` multiplies the entire
    reference column and its residual. Reported-root and retained-vector
    Rayleigh semantics are distinct; no lost eigenvalue precision is restored.
    """

    phase: int
    norm0: float
    norm1: float
    reported_terms: tuple
    rayleigh_terms: tuple
    reported_response: float
    rayleigh_response: float
    quotient_minus_reported: tuple
    residual_norms: tuple
    coefficient_change_norm: float
    working_dtype: str
    mantissa_fraction_bits: int


def real_rank_one_response(H0, H1, c0, c1, lambda0, lambda1, *, phase):
    """Retain complete finite norm/residual corrections at two real endpoints.

    This is pure-array algebra, not an eigenpair certificate. Physical callers
    must validate the same ordered basis, endpoint operators and target first.
    The radial term is ``c1.T (H1-H0) c1 / (c1.T c1)``; it is deliberately not
    obtained by subtracting two large total energies here.
    """
    if (isinstance(phase, (bool, np.bool_))
            or not isinstance(phase, (int, np.integer)) or phase not in (-1, 1)):
        raise ValueError("rank-one response phase must be whole-column +/-1")
    arrays = [np.asarray(value) for value in (H0, H1, c0, c1)]
    if any(np.iscomplexobj(value) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("rank-one response requires finite real arrays")
    n = arrays[2].size
    if (arrays[2].shape != (n,) or arrays[3].shape != (n,)
            or arrays[0].shape != (n, n) or arrays[1].shape != (n, n) or n == 0
            or any(not np.array_equal(h, h.T) for h in arrays[:2])):
        raise ValueError("rank-one response requires same complete symmetric matrices/vectors")
    if (np.ndim(lambda0) != 0 or np.ndim(lambda1) != 0
            or np.iscomplexobj(lambda0) or np.iscomplexobj(lambda1)
            or not np.isfinite(lambda0) or not np.isfinite(lambda1)):
        raise ValueError("rank-one response requires finite real reported roots")
    wide = np.longdouble
    h0, h1, a, b = [value.astype(wide) for value in arrays]
    l0, l1 = wide(lambda0), wide(lambda1)
    n0, n1 = a @ a, b @ b
    if not (n0 > 0 and n1 > 0 and np.isfinite(n0 + n1)):
        raise ValueError("rank-one response requires positive finite CI norms")
    x0, d = phase * a, b - phase * a
    shifted = h0.copy()
    shifted.flat[::n + 1] -= l0
    shifted1 = h1.copy()
    shifted1.flat[::n + 1] -= l1
    r0bar, r1 = shifted @ x0, shifted1 @ b
    reported = (d @ shifted @ d / n1, 2 * d @ r0bar / n1,
                x0 @ r0bar / n1, -b @ r1 / n1)
    # Quotients are centered before restoring the common origin. This avoids
    # rounding the deep common constant into an intermediate absolute q0.
    origin = h0[0, 0]
    centered = h0.copy()
    centered.flat[::n + 1] -= origin
    q0_centered = a @ centered @ a / n0
    centered.flat[::n + 1] -= q0_centered
    rq0bar = centered @ x0
    rayleigh = (d @ centered @ d / n1, 2 * d @ rq0bar / n1,
                x0 @ rq0bar / n1)
    values = (*reported, *rayleigh, x0 @ r0bar / n0, b @ r1 / n1)
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite rank-one CI response")
    return RankOneCIResponse(
        phase=int(phase), norm0=float(n0), norm1=float(n1),
        reported_terms=tuple(float(v) for v in reported),
        rayleigh_terms=tuple(float(v) for v in rayleigh),
        reported_response=float(sum(reported, wide(0))),
        rayleigh_response=float(sum(rayleigh, wide(0))),
        quotient_minus_reported=(float(x0 @ r0bar / n0), float(b @ r1 / n1)),
        residual_norms=(float(np.sqrt(r0bar @ r0bar / n0)),
                        float(np.sqrt(r1 @ r1 / n1))),
        coefficient_change_norm=float(np.sqrt(d @ d)),
        working_dtype=wide.__name__, mantissa_fraction_bits=int(np.finfo(wide).nmant),
    )
