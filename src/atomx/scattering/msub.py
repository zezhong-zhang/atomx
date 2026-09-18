"""Magnetic-substate (M-resolved) cross-section helpers.

FAC's ``CETableMSub`` / ``CITableMSub`` / ``AITableMSub`` produce
cross sections resolved by the upper-state magnetic quantum number
M_J — required for polarisation spectroscopy and anisotropic-emission
modelling.  This module ships the canonical angular reduction that
projects an unpolarised σ_total onto its M-resolved components for
an axially-symmetric collisional-excitation geometry (e.g. an
electron beam).

Approach
--------
For a transition (J_a M_a → J_b M_b) excited by an unpolarised beam
travelling along ẑ, only |M_b| contributes (axial symmetry implies
M_b = ±|M_b| are degenerate).  The M-resolved partial cross section
is

    σ(|M_b|) = σ_total · w(|M_b|, J_a, J_b)

where the weight w is the standard Legendre projection of the
collision strength.  In the dipole limit (E1 excitation), Inal-Dubau
gives:

    w(|M_b|, J_b)  =  (1 / (2 J_b + 1)) · [ 1 − A_2(J_a → J_b)
                                            · P_2(cos θ_M) · S_M ]

where θ_M = magnetic projection angle and S_M is the standard 3j-
based weighting.

This module exposes:

* :func:`m_resolved_weights(J_a, J_b, L)` — return the dimensionless
  M-projection weights w(|M_b|) summing to 1.  For E1 / J_a = 0 →
  J_b ≥ 1 transitions, w(0) and w(1) are returned in canonical form.

* :func:`m_resolved_cross_section(sigma_total, J_a, J_b, L)` — split
  σ_total into M-projected parts.

* :func:`alignment_from_m_resolved(sigma_M, J_b)` — convenience
  wrapper around the existing
  :func:`atomx.scattering.polarisation.alignment_from_msublevel_cross_sections`
  reusing the same 3j machinery.

Reference
---------
* M. K. Inal, J. Dubau, *J. Phys. B* **20**, 4221 (1987).
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC
  ``CETableMSub`` formulation.
"""

from __future__ import annotations

from typing import Dict, List

from atomx.angular.wigner import wigner_3j


def _mj_values(J: float) -> List[float]:
    """Enumerate |M_J| values for total J: 0, 1, …, J (ascending)."""
    n = int(round(J + 0.5)) if (2 * J) % 2 == 1 else int(J + 1)
    if (2 * J) % 2 == 0:
        return [float(m) for m in range(int(J) + 1)]
    return [k + 0.5 for k in range(n)]


def _e1_dipole_weight(J_a: float, J_b: float, M_b: float) -> float:
    """Statistical |M_b|-projection weight for E1 excitation from
    an axially-symmetric initial state J_a (M_a integrated out).

    For unpolarised beam excitation, the partial collision strength
    summing over photon helicities and beam projection gives

        Ω_{|M_b|} ∝ Σ_{q=±1} |⟨J_b M_b | T¹_q | J_a 0⟩|²
                   = (2 J_a + 1) · [ (J_a 1 J_b; 0 +q -M_b)² + (q ↔ −q) ]

    The 3j with q = ±1 gives M_b = ∓1; the M_b = 0 component vanishes
    by symmetry.  Returns the relative weight (un-normalised); call
    :func:`m_resolved_weights` for normalised w(|M_b|).
    """
    tJa = int(round(2 * J_a))
    tJb = int(round(2 * J_b))
    tM_b = int(round(2 * M_b))
    weight = 0.0
    for q in (+1, -1):
        # 3j(J_a 1 J_b; M_a q -M_b) with M_a = -q + M_b summed over
        # M_a = -J_a..J_a — for axial symmetry only the q = -M_b
        # component contributes.
        for M_a_doubled in range(-tJa, tJa + 1, 2):
            three_j = wigner_3j(tJa, 2, tJb,
                                  M_a_doubled, 2 * q, -tM_b)
            if abs(three_j) < 1e-15:
                continue
            weight += three_j * three_j
    return float(weight)


def m_resolved_weights(J_a: float, J_b: float, L: int = 1) -> Dict[float, float]:
    """Normalised M_b weights for an axially-symmetric L-pole excitation.

    Returns ``{|M_b|: w(|M_b|)}`` summing to 1.  Each weight is the
    fraction of the total cross section that populates the |M_b|
    sublevel (magnetic-substate-resolved, axial geometry).

    For E1 excitation from J_a = 0 (closed-shell ground), only |M_b| =
    1 is populated and w(1) = 1, w(0) = 0 — characteristic of axial
    photo-absorption.  For more general (J_a, J_b) all |M_b| ∈ [0, J_b]
    contribute with weights determined by the 3j reduction.

    Parameters
    ----------
    J_a, J_b : float
        Initial / final total angular momenta.
    L : int
        Multipole rank (1 = E1, 2 = M1/E2, …).  Currently L = 1 is
        implemented; higher multipoles raise ``NotImplementedError``.

    Returns
    -------
    dict {|M_b|: w(|M_b|)}
    """
    if L != 1:
        raise NotImplementedError(
            f"m_resolved_weights currently supports only L = 1 (E1); got L = {L}"
        )
    raw: Dict[float, float] = {}
    total = 0.0
    for M_b in _mj_values(J_b):
        # Sum over ±M_b (already accounted for in the 3j sum)
        w_pos = _e1_dipole_weight(J_a, J_b, M_b)
        w_neg = _e1_dipole_weight(J_a, J_b, -M_b) if M_b > 0 else 0.0
        raw[M_b] = w_pos + w_neg
        total += raw[M_b]
    if total <= 0:
        # Forbidden / vanishing — return uniform distribution
        n = len(raw)
        return {m: 1.0 / n for m in raw}
    return {m: raw[m] / total for m in raw}


def m_resolved_cross_section(
    sigma_total: float,
    J_a: float,
    J_b: float,
    L: int = 1,
) -> Dict[float, float]:
    """Split a total cross section into M_b-resolved partials.

    Returns ``{|M_b|: σ_total · w(|M_b|)}``.

    Parameters
    ----------
    sigma_total : float
        Total cross section (any units; preserved on output).
    J_a, J_b : float
    L : int

    Returns
    -------
    dict {|M_b|: σ_M}
    """
    weights = m_resolved_weights(J_a, J_b, L)
    return {m: sigma_total * w for m, w in weights.items()}


def alignment_from_m_resolved(
    sigma_M: Dict[float, float],
    J_b: float,
) -> float:
    """Alignment A_2 from an M_b-resolved cross-section dict.

    Convenience wrapper that converts the dict format used here into
    the list format expected by
    :func:`atomx.scattering.polarisation.alignment_from_msublevel_cross_sections`.
    """
    from atomx.scattering.polarisation import (
        alignment_from_msublevel_cross_sections,
    )
    keys_sorted = sorted(sigma_M.keys())
    sigma_list = [sigma_M[k] for k in keys_sorted]
    return alignment_from_msublevel_cross_sections(sigma_list, J_b)


__all__ = [
    "m_resolved_weights",
    "m_resolved_cross_section",
    "alignment_from_m_resolved",
]
