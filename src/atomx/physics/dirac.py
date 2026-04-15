"""Relativistic Dirac equation — physics-level invariants + Sommerfeld formula.

The coupled radial Dirac equation for the large (P) and small (Q)
components reads::

    c (dP/dr − κ P/r)        =  (E + 2 c² − V) Q,
   −c (dQ/dr + κ Q/r)        =  (E − V)     P

in Hartree units with c = 1/α.  The κ = −(l+1) for j = l+½ and
κ = +l for j = l−½.  The closed-form point-nucleus eigenvalues (the
*Sommerfeld fine-structure formula*) serve as validation anchors for
every numerical Dirac solver in atomx.

This module owns the equation *object* plus the analytic
hydrogenic-Dirac energy expressions.  B-spline matrix assembly lives
in :mod:`atomx.numerics.bspline.dirac_assembly`; hydrogenic radial
wavefunctions live in :mod:`atomx.numerics.analytic.hydrogenic_dirac`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomx.constants import ALPHA, C_SQ


# ---------------------------------------------------------------------------
# Equation object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dirac:
    """Two-component relativistic radial equation."""

    name: str = "dirac"
    is_relativistic: bool = True
    n_components: int = 2

    @property
    def has_small_component(self) -> bool:
        return True

    def eigenvalue_floor(self) -> float:
        """Dirac-sea / positive-energy boundary, ``−m_e c² = −c²`` in Hartree."""
        return -C_SQ


# ---------------------------------------------------------------------------
# Sommerfeld fine-structure formula
# ---------------------------------------------------------------------------


def hydrogenic_dirac_energy(n: int, kappa: int, Z: int) -> float:
    """Exact Dirac energy eigenvalue for a hydrogenic ion.

    ``E_{n,κ} = c² · {[1 + (αZ)² / (n_r + γ)²]^{−1/2} − 1}``

    with  ``γ = √(κ² − (αZ)²)`` and  ``n_r = n − |κ|``.

    Parameters
    ----------
    n : int
        Principal quantum number (``n ≥ 1``).
    kappa : int
        Dirac quantum number (``κ ≠ 0``): ``κ = −(l+1)`` for ``j = l+½``,
        ``κ = +l`` for ``j = l−½``.
    Z : int
        Nuclear charge.

    Returns
    -------
    E : float
        Binding energy in Hartree (negative for bound states).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if kappa == 0:
        raise ValueError("kappa cannot be zero")

    aZ = ALPHA * Z
    gamma = np.sqrt(kappa * kappa - aZ * aZ)
    n_r = n - abs(kappa)

    if n_r < 0:
        raise ValueError(
            f"Invalid quantum numbers: n={n}, kappa={kappa} gives n_r = {n_r} < 0"
        )

    denom = n_r + gamma
    return C_SQ * (1.0 / np.sqrt(1.0 + (aZ / denom) ** 2) - 1.0)


def list_hydrogenic_states(n_max: int, Z: int):
    """Enumerate hydrogenic Dirac bound states up to ``n_max``.

    Returns a list of ``{n, l, j, kappa, energy_hartree}`` dicts.
    """
    states = []
    for n in range(1, n_max + 1):
        for l in range(n):
            # j = l + 1/2  →  κ = −(l+1)
            kappa_minus = -(l + 1)
            states.append({
                "n": n, "l": l, "j": l + 0.5, "kappa": kappa_minus,
                "energy_hartree": hydrogenic_dirac_energy(n, kappa_minus, Z),
            })
            # j = l − 1/2  →  κ = +l   (only l ≥ 1)
            if l >= 1:
                kappa_plus = l
                states.append({
                    "n": n, "l": l, "j": l - 0.5, "kappa": kappa_plus,
                    "energy_hartree": hydrogenic_dirac_energy(n, kappa_plus, Z),
                })
    return states


__all__ = [
    "Dirac",
    "hydrogenic_dirac_energy",
    "list_hydrogenic_states",
]
