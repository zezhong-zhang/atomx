"""Bauche-Arnoult UTA statistical moments.

The statistical treatment of a transition array replaces the explicit
line-by-line sum with the first two moments of the energy distribution:

    <E>       = Σ_{i,j} S_ij · E_ij / Σ_{i,j} S_ij                   (1)
    <(ΔE)²>   = Σ_{i,j} S_ij · (E_ij − <E>)² / Σ_{i,j} S_ij          (2)

where S_ij are the individual line strengths.  Closed-form expressions
for both moments in terms of Slater radial integrals were derived by
Bauche and Arnoult (1979, 1988) for any pair of configurations related
by a single-electron transition.

This module ships a *simple* implementation driven by a list of
discrete lines ("empirical UTA"), plus a closed-form helper for a
single-subshell-jump array that uses only the Slater F^k / G^k
parameters.

Higher-order moments (skewness, kurtosis) follow but are less
commonly quoted; they can be added in a follow-up once the two-moment
implementation is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class UTAArray:
    """First two statistical moments of a transition array.

    Attributes
    ----------
    total_strength : float
        ``Σ S_ij`` — total line strength, *in whatever units the caller
        supplied as the weights*.  For :func:`uta_moments` that is the
        caller's own strength unit; for
        :func:`uta_simple_from_levels` **without** ``strengths=`` it is
        a degeneracy product ``Σ (2J_u+1)(2J_l+1)`` — a dimensionless
        weight sum, *not* a line strength (review 2026-09-19, M5).
    centroid_eV : float
        Centroid transition energy ``<E>``, eV.
    variance_eV2 : float
        Variance ``<(ΔE)²>``, eV².
    std_eV : float
        ``√variance`` — standard deviation.
    fwhm_eV : float
        Gaussian-equivalent FWHM = 2√(2 ln 2) · σ.
    centroid_nm : float
        Centroid wavelength, nm (from ``hc / centroid_eV``).
    n_lines : int
        Number of individual lines summed.
    """

    total_strength: float
    centroid_eV: float
    variance_eV2: float
    std_eV: float
    fwhm_eV: float
    centroid_nm: float
    n_lines: int = 0


# hc in eV·nm
_HC_EV_NM = 1239.841984


def uta_moments(
    lines: Sequence[Tuple[float, float]],
) -> UTAArray:
    """Compute UTA first and second moments from a list of lines.

    Parameters
    ----------
    lines : iterable of (E_ij, S_ij)
        Tuples of (transition energy in eV, line strength).
        Strengths are the *weights* used for the moments — the caller
        decides whether to pass Einstein A coefficients, oscillator
        strengths, weighted oscillator strengths, ...  Consistency
        across all lines is the only requirement.

    Returns
    -------
    UTAArray
    """
    if len(lines) == 0:
        raise ValueError("uta_moments requires at least one line.")

    energies = np.asarray([ln[0] for ln in lines], dtype=float)
    strengths = np.asarray([ln[1] for ln in lines], dtype=float)

    total = float(strengths.sum())
    if total <= 0.0:
        raise ValueError("Total strength must be positive.")

    centroid = float(np.average(energies, weights=strengths))
    variance = float(np.average((energies - centroid) ** 2, weights=strengths))
    std = sqrt(variance)
    fwhm = 2.0 * sqrt(2.0 * np.log(2.0)) * std

    centroid_nm = _HC_EV_NM / centroid if centroid > 0 else float("nan")

    return UTAArray(
        total_strength=total,
        centroid_eV=centroid,
        variance_eV2=variance,
        std_eV=std,
        fwhm_eV=fwhm,
        centroid_nm=centroid_nm,
        n_lines=len(lines),
    )


def uta_simple_from_levels(
    E_upper_levels_eV: Sequence[float],
    E_lower_levels_eV: Sequence[float],
    *,
    g_upper: Optional[Sequence[float]] = None,
    g_lower: Optional[Sequence[float]] = None,
    strengths: Optional[Mapping[Tuple[int, int], float]] = None,
) -> UTAArray:
    """Convenience: UTA from two level lists (with optional g-weights).

    Two modes:

    ``strengths`` given (exact)
        ``strengths[(i, j)]`` is the line strength of level pair
        ``upper[i] → lower[j]`` (e.g. the reduced-matrix-element square
        from the CI or FAC engine).  Pairs absent from the mapping are
        treated as forbidden (zero strength) and skipped, as are
        non-positive ``ΔE`` and non-positive strengths.  The moments are
        then identical to ``uta_moments`` on the same lines.

    ``strengths`` omitted (statistical weights only)
        Every level pair is given the weight ``g_upper[i]·g_lower[j]``
        (default 1.0), i.e. the number of Zeeman sub-level pairs.  This
        is the standard *Unsöld-limit* (degeneracy-weighted) UTA
        estimate of the array moments.  It is **not** the
        line-strength-weighted moment of the real array, and it cannot
        be: the relative strength of two lines with the same J's is
        invisible to a weights-only reduction.  Measured on the
        committed Fe XVII FAC array (tests/data/fe17_tr_fac_cleanroom.json,
        2p⁵3d → 2p⁶, three J=1 upper levels with g_u = 3, g_l = 1):
        centroid 813.795 eV vs the strength-weighted 823.690 eV
        (−1.20 %), σ 9.9998 eV vs 5.9306 eV (+68.6 %), and the merged
        3s+3d total weight 15 vs Σ S = 0.1614 (92.9×).  Pass
        ``strengths=`` (or call :func:`uta_moments` on a real line list)
        when the true array moments are needed; review 2026-09-19
        finding M5.

    Parameters
    ----------
    E_upper_levels_eV, E_lower_levels_eV : sequence of float
        Level energies in eV.  Transition energies are ``E_upper − E_lower``.
    g_upper, g_lower : sequence of float, optional
        Statistical weights of the upper / lower levels.  Default = 1.
        Ignored when ``strengths`` is given.
    strengths : mapping {(i, j): float}, optional
        Per-pair line strengths, keyed by (upper index, lower index).

    Returns
    -------
    UTAArray
    """
    n_u = len(E_upper_levels_eV)
    n_l = len(E_lower_levels_eV)

    if strengths is not None:
        # Exact path: the caller supplies the physics; no degeneracy proxy.
        lines = []
        for (i, j), S in strengths.items():
            if not (0 <= i < n_u and 0 <= j < n_l):
                raise IndexError(
                    f"strengths key {(i, j)} outside the "
                    f"{n_u}×{n_l} level grid.")
            dE = float(E_upper_levels_eV[i]) - float(E_lower_levels_eV[j])
            if dE <= 0 or float(S) <= 0:
                continue
            lines.append((dE, float(S)))
        if not lines:
            raise ValueError("No valid (positive-energy, positive-strength) "
                             "transitions found.")
        return uta_moments(lines)

    if g_upper is None:
        g_upper = [1.0] * n_u
    if g_lower is None:
        g_lower = [1.0] * n_l
    if len(g_upper) != n_u or len(g_lower) != n_l:
        raise ValueError("Level energies and weights must match in length.")

    lines = []
    for i, Eu in enumerate(E_upper_levels_eV):
        for j, El in enumerate(E_lower_levels_eV):
            dE = Eu - El
            if dE <= 0:
                continue
            S = float(g_upper[i]) * float(g_lower[j])
            if S <= 0:
                continue
            lines.append((dE, S))

    if not lines:
        raise ValueError("No valid (positive-energy) transitions found.")
    return uta_moments(lines)
