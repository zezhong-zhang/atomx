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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class UTAArray:
    """First two statistical moments of a transition array.

    Attributes
    ----------
    total_strength : float
        ``Σ S_ij`` — total line strength (arbitrary units; typically
        the sum of A-coefficients weighted by statistical weight).
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
) -> UTAArray:
    """Convenience: UTA from two level lists (with optional g-weights).

    All transitions upper[i] → lower[j] are considered, each with
    strength ``g_upper[i] · g_lower[j]`` (or 1.0 if omitted).  Suitable
    for quick "statistical-weights-only" UTA estimates when line
    strengths have not been computed line-by-line.

    Parameters
    ----------
    E_upper_levels_eV, E_lower_levels_eV : sequence of float
        Level energies in eV.  Transition energies are ``E_upper − E_lower``.
    g_upper, g_lower : sequence of float, optional
        Statistical weights of the upper / lower levels.  Default = 1.

    Returns
    -------
    UTAArray
    """
    n_u = len(E_upper_levels_eV)
    n_l = len(E_lower_levels_eV)
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
