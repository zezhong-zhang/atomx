"""Weak-field Zeeman shift helpers.

For an external magnetic field **B** and a state with well-defined
``|J, m_J⟩`` quantum numbers,

    ΔE_Zeeman = g_J · μ_B · B · m_J.                                  (1)

This module ships a pure-Python helper :func:`weak_field_shift_MHz`
returning the shift in megahertz given ``g_J``, ``m_J`` and ``B`` in
tesla, plus :func:`zeeman_shift_table` that enumerates the full
``m_J = −j … +j`` ladder for one orbital.

Strong-field (Paschen–Back) and quadratic (diamagnetic) Zeeman effects
are *not* covered here; they require diagonalising the full
hyperfine + Zeeman Hamiltonian in the F-coupled basis and will appear
in a later release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from atomx.properties.zeeman.lande import BOHR_MAGNETON_MHZ_PER_TESLA, LandeFactor


@dataclass
class ZeemanShift:
    """Weak-field Zeeman splitting of one ``|j, m_j⟩`` manifold.

    Attributes
    ----------
    orbital : str
        Label of the orbital, e.g. ``"1s1/2"``.
    gJ : float
        Bound-electron Landé factor used.
    j : float
        Total angular momentum.
    B_Tesla : float
        External field magnitude, tesla.
    shifts_MHz : dict
        ``{m_J: ΔE_MHz}`` for every ``m_J`` in ``{−j, −j+1, …, +j}``.
    splitting_MHz : float
        ``ΔE(m_J = +j) − ΔE(m_J = −j)`` — full span of the manifold.
    """

    orbital: str
    gJ: float
    j: float
    B_Tesla: float
    shifts_MHz: Dict[float, float] = field(default_factory=dict)
    splitting_MHz: float = 0.0


def weak_field_shift_MHz(gJ: float, m_j: float, B_Tesla: float) -> float:
    """Single-level weak-field Zeeman shift ``gJ · μ_B · B · m_J`` in MHz."""
    return gJ * m_j * B_Tesla * BOHR_MAGNETON_MHZ_PER_TESLA


def zeeman_shift_table(
    lande: LandeFactor,
    B_Tesla: float,
    *,
    g_source: str = "bound",
) -> ZeemanShift:
    """Enumerate m_J shifts across an orbital's Zeeman manifold.

    Parameters
    ----------
    lande : LandeFactor
        Output of :func:`atomx.properties.zeeman.lande.lande_factor`.
    B_Tesla : float
        External magnetic-field magnitude in tesla.
    g_source : str
        Which refinement of g to use: ``"nr"``, ``"dirac"``, ``"bound"``
        (default).

    Returns
    -------
    ZeemanShift
    """
    if g_source not in ("nr", "dirac", "bound"):
        raise ValueError(
            f"g_source must be 'nr', 'dirac', or 'bound'; got {g_source!r}."
        )
    g = getattr(lande, g_source)
    j = lande.j

    shifts: Dict[float, float] = {}
    # Enumerate m_J = −j, −j+1, …, +j
    m_vals: List[float] = []
    m = -j
    while m <= j + 1e-9:
        m_vals.append(m)
        m += 1.0
    for m in m_vals:
        shifts[m] = weak_field_shift_MHz(g, m, B_Tesla)

    span = shifts[m_vals[-1]] - shifts[m_vals[0]]

    return ZeemanShift(
        orbital=lande.label,
        gJ=g,
        j=j,
        B_Tesla=B_Tesla,
        shifts_MHz=shifts,
        splitting_MHz=span,
    )
