"""Stark shift / external-electric-field perturbations.

Companion to :mod:`atomx.properties.zeeman` for the magnetic field.  When
both fields are present, use :func:`stark_zeeman_table` to combine
them into a single (M_J, F, B) shift table — equivalent to FAC's
``StructureEB`` knob.

This module ships closed-form perturbative expressions valid for
weak fields:

* :func:`linear_stark_shift_hydrogenic` — first-order Stark splitting
  of hydrogenic Rydberg manifolds (parabolic quantum numbers).
* :func:`quadratic_stark_shift` — second-order shift from the static
  dipole polarizability α₀ (works for any non-degenerate level).
* :func:`stark_zeeman_table` — combine ΔE_Zeeman(gJ, m_J, B) and
  ΔE_Stark(α, F) into a single per-(m_J, F, B) shift in MHz.

Hartree-atomic-units convention: electric field F in V/m converts to
atomic units via 1 a.u. = 5.142 × 10¹¹ V/m.

References
----------
* H. A. Bethe, E. E. Salpeter, *Quantum Mechanics of One- and Two-
  Electron Atoms* (1957), §59 (Stark effect).
* Foot, *Atomic Physics* (2005), §7.3.
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC `StructureEB`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from atomx.constants import HARTREE_EV


# Atomic-units conversion: 1 a.u. of electric field = 5.14220674763 × 10¹¹ V/m
_AU_FIELD_VPM = 5.14220674763e11


@dataclass
class StarkZeemanShift:
    """Combined Stark + Zeeman shift for a single (m_J, F, B) point.

    Attributes
    ----------
    m_J : float
        Magnetic quantum number.
    B_Tesla : float
        Magnetic field in tesla.
    F_Vpm : float
        Electric field in V/m.
    delta_E_zeeman_MHz : float
    delta_E_stark_MHz : float
    delta_E_total_MHz : float
    """

    m_J: float
    B_Tesla: float
    F_Vpm: float
    delta_E_zeeman_MHz: float
    delta_E_stark_MHz: float
    delta_E_total_MHz: float


def linear_stark_shift_hydrogenic(
    n: int, k: int, F_Vpm: float, *, Z: int = 1,
) -> float:
    """Linear (first-order) Stark splitting of a hydrogenic n-manifold.

    For a degenerate hydrogenic level the n-fold degeneracy is broken
    in an electric field by the linear Stark term

        ΔE = (3/2) · n · k · F · a₀ / Z

    in atomic units, where k = n_1 − n_2 is the parabolic Stark
    quantum number ranging over k ∈ {−(n−1), −(n−3), …, +(n−1)}.

    Parameters
    ----------
    n : int
        Principal quantum number.
    k : int
        Parabolic Stark index k = n_1 − n_2.
    F_Vpm : float
        Electric field strength in V/m.
    Z : int
        Nuclear charge (default 1 for hydrogen).

    Returns
    -------
    float
        Stark shift in MHz.
    """
    if n < 1:
        raise ValueError("linear_stark_shift_hydrogenic: n must be >= 1")
    if abs(k) >= n:
        raise ValueError(
            f"linear_stark_shift_hydrogenic: |k| < n required, got k={k}, n={n}"
        )
    F_au = F_Vpm / _AU_FIELD_VPM
    deltaE_au = 1.5 * n * k * F_au / Z
    deltaE_MHz = deltaE_au * HARTREE_EV * 2.41798934e8  # eV → MHz
    return deltaE_MHz


def quadratic_stark_shift(alpha_a0_cubed: float, F_Vpm: float) -> float:
    """Quadratic Stark shift from a static dipole polarizability.

    For any non-degenerate level the leading-order shift in a uniform
    field F is

        ΔE = − (1/2) · α₀ · F²

    where α₀ is the static dipole polarizability (a₀³).  Returns the
    shift in MHz.

    Parameters
    ----------
    alpha_a0_cubed : float
        Static dipole polarizability in atomic units (a₀³).
    F_Vpm : float
        Electric field in V/m.

    Returns
    -------
    float
        Quadratic Stark shift in MHz (typically negative for the
        ground state — fields lower the energy).
    """
    F_au = F_Vpm / _AU_FIELD_VPM
    deltaE_au = -0.5 * alpha_a0_cubed * F_au * F_au
    deltaE_MHz = deltaE_au * HARTREE_EV * 2.41798934e8
    return deltaE_MHz


def stark_zeeman_table(
    gJ: float,
    J: float,
    alpha_a0_cubed: float,
    B_Tesla_grid: np.ndarray,
    F_Vpm_grid: np.ndarray,
) -> List[StarkZeemanShift]:
    """Tabulate Zeeman + quadratic Stark shifts for every (m_J, B, F).

    Equivalent to FAC's ``StructureEB`` for non-degenerate levels —
    the linear Stark term is zero unless the level is degenerate
    (use :func:`linear_stark_shift_hydrogenic` for hydrogenic n).

    Parameters
    ----------
    gJ : float
        Landé g-factor of the level.
    J : float
        Total angular momentum.
    alpha_a0_cubed : float
        Static dipole polarizability in atomic units (a₀³).
    B_Tesla_grid : array_like
    F_Vpm_grid : array_like

    Returns
    -------
    list[StarkZeemanShift]
        One entry per (m_J, B, F) point.
    """
    from atomx.properties.zeeman.shift import weak_field_shift_MHz
    out: List[StarkZeemanShift] = []
    # Build m_J range: −J, −J+1, …, +J
    n_m = int(round(2 * J + 1))
    m_vals = [-J + k for k in range(n_m)]
    for B in np.atleast_1d(np.asarray(B_Tesla_grid, dtype=float)):
        for F in np.atleast_1d(np.asarray(F_Vpm_grid, dtype=float)):
            for m in m_vals:
                dE_z = weak_field_shift_MHz(gJ, m, float(B))
                dE_s = quadratic_stark_shift(alpha_a0_cubed, float(F))
                out.append(StarkZeemanShift(
                    m_J=float(m), B_Tesla=float(B), F_Vpm=float(F),
                    delta_E_zeeman_MHz=dE_z,
                    delta_E_stark_MHz=dE_s,
                    delta_E_total_MHz=dE_z + dE_s,
                ))
    return out


__all__ = [
    "StarkZeemanShift",
    "linear_stark_shift_hydrogenic",
    "quadratic_stark_shift",
    "stark_zeeman_table",
]
