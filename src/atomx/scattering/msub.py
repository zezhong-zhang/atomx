"""Magnetic-substate helpers with explicit excitation physics.

Electron-impact sublevel populations cannot be recovered from a scalar
cross section and J values. Supply calculated sublevel weights, or explicitly
select the unpolarized transverse photoabsorption model. The latter sums
photon helicities q=±1 and is not a FAC CETableMSub calculation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from atomx.angular.wigner import wigner_3j


def _mj_values(J: float) -> List[float]:
    """Enumerate |M_J| values for total J: 0, 1, …, J (ascending)."""
    n = int(round(J + 0.5)) if (2 * J) % 2 == 1 else int(J + 1)
    if (2 * J) % 2 == 0:
        return [float(m) for m in range(int(J) + 1)]
    return [k + 0.5 for k in range(n)]


def _e1_dipole_weight(J_a: float, J_b: float, M_b: float) -> float:
    """E1 photoabsorption weight for one signed final M_b.

    Sum squared 3j amplitudes over equally populated initial M_a and
    transverse photon helicities q=±1. M_b=0 is generally allowed when
    J_a>0. This is an optical angular model, not a collision strength.
    """
    tJa = int(round(2 * J_a))
    tJb = int(round(2 * J_b))
    tM_b = int(round(2 * M_b))
    weight = 0.0
    for q in (+1, -1):
        # 3j(J_a 1 J_b; M_a q -M_b), with M_a + q = M_b.
        for M_a_doubled in range(-tJa, tJa + 1, 2):
            three_j = wigner_3j(tJa, 2, tJb,
                                  M_a_doubled, 2 * q, -tM_b)
            if abs(three_j) < 1e-15:
                continue
            weight += three_j * three_j
    return float(weight)


def m_resolved_weights(J_a: float, J_b: float, L: int = 1, *,
                       geometry: str = "electron-impact", sublevel_weights=None) -> Dict[float, float]:
    """Normalize supplied collision weights or compute explicit photoabsorption.

    ``sublevel_weights`` maps |M_b| to combined ±M populations from a
    collision calculation. ``geometry='photoabsorption'`` instead uses
    transverse unpolarized E1 photons along the quantization axis.
    """
    for J in (J_a, J_b):
        if not np.isfinite(J) or J < 0 or abs(2*J-round(2*J)) > 1e-10:
            raise ValueError("J must be a nonnegative integer or half integer")
    if geometry not in {"electron-impact", "photoabsorption"}:
        raise ValueError("geometry must be electron-impact or photoabsorption")
    if sublevel_weights is not None:
        if geometry != "electron-impact":
            raise ValueError("collision weights require electron-impact geometry")
        if set(sublevel_weights) != set(_mj_values(J_b)):
            raise ValueError("sublevel_weights must supply every |M_b|")
        values = np.asarray(list(sublevel_weights.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("sublevel weights must be nonnegative with positive sum")
        return {m: float(v/values.sum()) for m, v in sublevel_weights.items()}
    if geometry == "electron-impact":
        raise ValueError("Electron-impact excitation requires calculated sublevel_weights; "
                         "a scalar total does not determine magnetic populations")
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
        raise ValueError("Forbidden E1 photoabsorption angular momentum triangle")
    return {m: raw[m] / total for m in raw}


def m_resolved_cross_section(
    sigma_total: float,
    J_a: float,
    J_b: float,
    L: int = 1,
    *, geometry: str = "electron-impact", sublevel_weights=None,
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
    if not np.isfinite(sigma_total) or sigma_total < 0:
        raise ValueError("sigma_total must be finite and nonnegative")
    weights = m_resolved_weights(J_a, J_b, L, geometry=geometry,
                                sublevel_weights=sublevel_weights)
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
