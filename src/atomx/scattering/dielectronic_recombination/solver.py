"""Dielectronic recombination (DR) rate coefficients.

Dielectronic recombination is a two-step resonant process:

  1. A free electron is *captured* into a doubly-excited resonance
     (inverse Auger, rate ∝ detailed-balance ∝ Γ_Auger).
  2. The resonance decays *radiatively* to a lower singly-bound level,
     stabilising the recombined ion.

At plasma temperature T_e, the rate coefficient per resonance d is

    α_d^DR(T)
        =  (h² / (2π m_e k_B T_e))^{3/2}   ←   de-Broglie prefactor
           · (g_d / (2 g_i))                ←   statistical weight ratio
           · (Γ_Auger · Γ_radiative / Γ_total)   ←   branching ratio
           · exp(−E_d / k_B T_e)                   ←   Boltzmann tail   (1)

The total DR rate coefficient sums over all resonances.  In
convenient atomic units:

    α^DR(T_e) [cm³/s]
        = 6.603 × 10⁻²⁴  ·  T_e[eV]^{−3/2}
                         ·  Σ_d  ω_d · ⟨Γ⟩_d  ·  exp(−E_d/T_e)        (2)

with  ω_d = g_d / (2 g_i)  and  ⟨Γ⟩_d = Γ_A · Γ_r / (Γ_A + Γ_r).

References
----------
A. Burgess, *Astrophys. J.* **139**, 776 (1965).
D. W. Savin, *Astrophys. J.* **562**, 1259 (2001) — Fe XVI benchmark.
M. F. Gu, *Astrophys. J.* **590**, 1131 (2003) — FAC DR calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi, sqrt
from typing import Callable, Dict, List, Sequence

import numpy as np

from atomx.scattering.autoionization.auger import AutoionizationState
from atomx.scattering.dielectronic_recombination.enumerate import (
    DRResonanceSpec,
    build_autoionization_states,
    enumerate_dr_resonances,
    resonance_strength_rows,
    _initial_statistical_weight,
)


# --- Universal prefactor for eq. (2) -------------------------------------
#
# Expression: (h² / (2π m_e k_B))^{3/2}  in units of cm³ · eV^{3/2} · s.
# CODATA 2018:
#   h  = 6.62607015e-34  J·s
#   m_e = 9.1093837015e-31 kg
#   k_B = 1.380649e-23  J/K
#   1 eV = 1.602176634e-19 J
# Gives (numerically, with T in eV rather than K):
#   Prefactor = 6.60305e-24  cm³/s · eV^{3/2}
#
# See Burgess 1965 eq. (6), or Osterbrock "Astrophysics of Gaseous
# Nebulae" §5.7.
_DR_PREFACTOR_CM3_S_EV_32 = 6.60305e-24


@dataclass
class DRRateCoefficient:
    """Dielectronic-recombination rate coefficient, tabulated on T_e.

    Attributes
    ----------
    T_eV : np.ndarray
        Temperature grid in eV.
    alpha_cm3_s : np.ndarray
        α^DR(T_e) in cm³/s at each temperature.
    per_resonance : dict, optional
        ``{resonance_label: α_cm3_s}`` — contribution of each resonance
        summed over T.  Useful for identifying dominant resonances.
    initial_state_label : str, optional
        Label of the initial ion ground state.
    initial_state_g : float
        Statistical weight of the initial ion ground state (``2J_i + 1``).
    """

    T_eV: np.ndarray
    alpha_cm3_s: np.ndarray
    per_resonance: Dict[str, np.ndarray] = field(default_factory=dict)
    initial_state_label: str = ""
    initial_state_g: float = 1.0
    resonances: List[AutoionizationState] = field(default_factory=list)

    def at_T(self, T_eV: float) -> float:
        """Linear-interpolate α^DR at an arbitrary temperature."""
        return float(np.interp(T_eV, self.T_eV, self.alpha_cm3_s))


def dr_rate_coefficient(
    resonances: Sequence[AutoionizationState],
    T_eV,
    *,
    initial_state_g: float = 1.0,
    initial_state_label: str = "",
) -> DRRateCoefficient:
    """Burgess-formula DR rate coefficient from a list of resonances.

    Implements eq. (2) above.  Each resonance contributes a
    ``T_e^{−3/2} · exp(−E_d / T_e)`` term weighted by its statistical
    branching ratio.

    Parameters
    ----------
    resonances : sequence of :class:`AutoionizationState`
        Doubly-excited resonances with populated ``E_resonance_eV``,
        ``J``, ``gamma_auger_per_s``, ``a_radiative_per_s``.
    T_eV : float or array_like
        Electron temperatures in eV.  Output is vectorised over this.
    initial_state_g : float
        ``2 J_i + 1`` statistical weight of the initial ion ground
        state (the one the free electron captures onto).
    initial_state_label : str, optional
        Informational only.

    Returns
    -------
    :class:`DRRateCoefficient`
    """
    T_eV = np.atleast_1d(np.asarray(T_eV, dtype=float))
    if np.any(T_eV <= 0):
        raise ValueError("Temperatures must be positive (eV).")

    alpha_total = np.zeros_like(T_eV, dtype=float)
    per = {}

    for d in resonances:
        w = d.total_width_per_s
        if w <= 0.0:
            continue
        # Branching ratio: Γ_A Γ_r / (Γ_A + Γ_r)  in s⁻¹
        branch = d.gamma_auger_per_s * d.a_radiative_per_s / w
        if branch <= 0.0:
            continue
        omega = d.g / (2.0 * initial_state_g)

        # Boltzmann factor  exp(-E_d / T_e)
        boltz = np.exp(-d.E_resonance_eV / T_eV)
        contribution = (
            _DR_PREFACTOR_CM3_S_EV_32
            * T_eV ** (-1.5)
            * omega
            * branch
            * boltz
        )
        alpha_total += contribution
        per[d.label] = contribution

    return DRRateCoefficient(
        T_eV=T_eV,
        alpha_cm3_s=alpha_total,
        per_resonance=per,
        initial_state_label=initial_state_label,
        initial_state_g=initial_state_g,
        resonances=list(resonances),
    )


def dr_strengths(
    resonances: Sequence[AutoionizationState] | DRRateCoefficient,
    T_eV=None,
    *,
    initial_state_g: float | None = None,
) -> List[dict]:
    """Return per-resonance DR contribution rows for diagnostics.

    Parameters
    ----------
    resonances
        Either a resonance sequence or the :class:`DRRateCoefficient`
        returned by :func:`dr_calculate`.
    T_eV
        Temperature grid in eV.  Optional when ``resonances`` is a
        :class:`DRRateCoefficient`; its stored grid is used by default.
    initial_state_g
        Statistical weight of the recombining ion ground state.  Defaults
        to the value stored on ``DRRateCoefficient`` or ``1``.
    """

    if isinstance(resonances, DRRateCoefficient):
        rc = resonances
        seq = rc.resonances
        T = rc.T_eV if T_eV is None else T_eV
        g_i = rc.initial_state_g if initial_state_g is None else initial_state_g
    else:
        if T_eV is None:
            raise ValueError("T_eV is required when passing raw resonances")
        seq = resonances
        T = T_eV
        g_i = 1.0 if initial_state_g is None else initial_state_g
    return resonance_strength_rows(seq, T, initial_state_g=g_i)


def dr_calculate(
    initial_atom,
    target_iso,
    *,
    n_max: int = 10,
    l_max: int | None = None,
    T_eV=1000.0,
    initial_state_g: float | None = None,
    initial_state_label: str = "",
    min_resonance_eV: float = 1e-9,
    auger_rate_fn: Callable[[DRResonanceSpec, float], float] | None = None,
    radiative_rate_fn: Callable[
        [DRResonanceSpec, float],
        float | tuple[float, dict[str, float]],
    ] | None = None,
) -> DRRateCoefficient:
    """End-to-end dielectronic-recombination rate driver.

    ``initial_atom`` is the recombining ion and ``target_iso`` is the
    recombined ion ground configuration, either as an ``Atom``-like object or
    a configuration string.  The driver enumerates doubly-excited resonances,
    builds :class:`AutoionizationState` objects, and calls the existing
    :func:`dr_rate_coefficient` kernel unchanged.

    The present public Auger API exposes the low-level two-electron radial
    integral but not a complete configuration-to-channel summation.  Until the
    J-block matrix-element layer is available, default Auger/radiative widths
    are screened hydrogenic estimates.  Production callers can pass
    ``auger_rate_fn`` and ``radiative_rate_fn`` hooks to supply ab-initio
    matrix-element sums without changing this driver API.
    """

    if hasattr(initial_atom, "solve") and not getattr(initial_atom, "converged", False):
        initial_atom.solve()

    g_i = (
        _initial_statistical_weight(initial_atom)
        if initial_state_g is None else float(initial_state_g)
    )
    label = initial_state_label or getattr(initial_atom, "_config_strings", [""])[0]
    specs = enumerate_dr_resonances(
        initial_atom,
        target_iso,
        n_max=n_max,
        l_max=l_max,
    )
    resonances = build_autoionization_states(
        specs,
        initial_atom,
        min_resonance_eV=min_resonance_eV,
        auger_rate_fn=auger_rate_fn,
        radiative_rate_fn=radiative_rate_fn,
    )
    return dr_rate_coefficient(
        resonances,
        T_eV,
        initial_state_g=g_i,
        initial_state_label=label,
    )
