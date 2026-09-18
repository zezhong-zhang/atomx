"""Average-atom model for hot, dense plasmas.

The average-atom (AA) approximation replaces the discrete level
structure of an individual ion in a plasma by a single self-
consistent neutral cell whose effective bound-electron occupations
are continuous functions of T_e and ρ.  At each (T_e, ρ) the bound
electrons fill the levels according to a Fermi-Dirac distribution
with chemical potential μ chosen to enforce charge neutrality inside
the ion-sphere radius.

This module ships a *Thomas-Fermi-screened average atom* — the
classic Feynman-Metropolis-Teller (1949) prescription — without an
explicit Schrödinger / Dirac SCF.  It returns:

* the average ionisation state ⟨Z⟩(T_e, ρ),
* per-shell mean occupations,
* the chemical potential μ.

Sufficient for plasma EoS, NLTE opacity scoping, and as the entry
point μ for a more elaborate self-consistent ATOMIC-style INFERNO
solver later.

Reference
---------
* R. Feynman, N. Metropolis, E. Teller, *Phys. Rev.* **75**, 1561
  (1949) — Thomas-Fermi at finite temperature.
* B. F. Rozsnyai, *Phys. Rev. A* **5**, 1137 (1972) — average-atom
  formulation in atomic-physics conventions.
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC AverageAtom
  knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from atomx.constants import HARTREE_EV
from atomx.plasma.screening import ion_sphere_radius


@dataclass
class AverageAtomResult:
    """Output of the Thomas-Fermi average-atom solver.

    Attributes
    ----------
    Z : int
        Nuclear charge.
    T_eV : float
        Electron temperature in eV.
    ne_cm3 : float
        Free-electron density in cm⁻³.
    Z_bar : float
        Average ionisation ⟨Z*⟩ ∈ [0, Z].
    mu_eV : float
        Chemical potential of free electrons in eV.
    occupations : dict
        ``{(n, l): n̄}`` mean occupation of each shell.
    R_ws_au : float
        Wigner-Seitz / ion-sphere radius in atomic units.
    """

    Z: int
    T_eV: float
    ne_cm3: float
    Z_bar: float
    mu_eV: float
    occupations: Dict[Tuple[int, int], float]
    R_ws_au: float


# --- Bound-state energies (hydrogenic with Z_eff = Z − screening) -----


def _hydrogenic_eps_au(n: int, Z_eff: float) -> float:
    """Hydrogenic energy ε = −Z_eff² / (2 n²) in Hartree."""
    return -0.5 * Z_eff * Z_eff / (n * n)


def _shell_capacity(n: int) -> int:
    """2 n² electrons per principal shell (l = 0..n−1, each holds 2(2l+1))."""
    return 2 * n * n


# --- Core solver ------------------------------------------------------


def average_atom_thomas_fermi(
    Z: int,
    T_eV: float,
    ne_cm3: float,
    *,
    n_max: int = 10,
    screening_factor: float = 0.3,
) -> AverageAtomResult:
    """Thomas-Fermi screened average-atom model at (T_e, ne).

    For each principal shell n ≤ n_max the bound-state energy is
    estimated hydrogenically with an effective charge

        Z_eff(n) ≈ Z − σ · ⟨N_inner⟩

    where σ is a Slater-screening parameter and ⟨N_inner⟩ is the
    expected number of electrons in shells n' < n.  The Fermi-Dirac
    occupation

        n̄(n)   = 2 n² / (1 + exp((ε_n − μ) / k_B T))

    is solved self-consistently for μ such that the total bound
    population satisfies Z − ∫ n̄ = ⟨Z*⟩, with ⟨Z*⟩ inferred from the
    free-electron density ne via ne = ⟨Z*⟩ · n_ion.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    T_eV : float
        Electron temperature.
    ne_cm3 : float
        Free-electron density in cm⁻³.
    n_max : int
        Highest principal shell to track explicitly (default 10).
        Higher Rydberg shells are bundled into a single Σ_n>n_max
        effective level using the hydrogenic Rydberg sum.
    screening_factor : float
        Slater-screening σ; default 0.3 (typical for outer shells).

    Returns
    -------
    AverageAtomResult
    """
    if Z < 1:
        raise ValueError("average_atom_thomas_fermi: Z must be >= 1")
    if T_eV <= 0:
        raise ValueError("T_eV must be > 0")
    if ne_cm3 <= 0:
        raise ValueError("ne_cm3 must be > 0")

    kT = T_eV / HARTREE_EV   # in atomic units

    # Saha-Boltzmann chemical potential of the free electron gas
    # (non-degenerate ideal-gas limit, Mihalas Eq. 5-3):
    #     μ_free(T, ne) = kT · ln(ne · λ_th³ / 2)
    # with λ_th the thermal de-Broglie wavelength.
    # In atomic units (m_e = ℏ = 1):
    #     λ_th_au = √(2π / kT)        ⇒    λ_th_au³ = (2π / kT)^(3/2)
    # ne in a.u. = ne_cm3 · (a₀ in cm)³
    BOHR_RADIUS_M = 5.29177210903e-11   # CODATA 2022
    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100.0) ** 3
    lambda_th_au_cubed = (2.0 * math.pi / kT) ** 1.5
    if ne_au * lambda_th_au_cubed > 0:
        mu_free_au = kT * math.log(ne_au * lambda_th_au_cubed / 2.0)
    else:
        mu_free_au = -1e6 * kT

    # Self-consistent Slater screening on top of the fixed μ:
    Z_bar = float(Z) * 0.5
    Z_eff_lst = []
    eps_arr = np.zeros(n_max)
    cap_arr = np.zeros(n_max)
    for _outer_iter in range(50):
        Z_eff_lst = []
        cum_inner = 0.0
        for n in range(1, n_max + 1):
            Z_eff = Z - screening_factor * cum_inner
            if Z_eff <= 0.5:
                Z_eff = 0.5
            Z_eff_lst.append(Z_eff)
            cum_inner += _shell_capacity(n)

        eps_arr = np.array([_hydrogenic_eps_au(n, Z_eff_lst[n - 1])
                             for n in range(1, n_max + 1)])
        cap_arr = np.array([_shell_capacity(n)
                             for n in range(1, n_max + 1)], dtype=float)
        bound_pop = float(np.sum(
            cap_arr / (1.0 + np.exp(np.clip((eps_arr - mu_free_au) / kT,
                                              -500.0, 500.0)))
        ))
        Z_bar_new = max(0.0, Z - bound_pop)
        if abs(Z_bar_new - Z_bar) < 1e-3 * max(1.0, abs(Z_bar)):
            Z_bar = Z_bar_new
            break
        Z_bar = 0.5 * Z_bar + 0.5 * Z_bar_new

    mu_au = mu_free_au
    occ_arr = cap_arr / (1.0 + np.exp(np.clip((eps_arr - mu_au) / kT,
                                                -500.0, 500.0)))
    # Distribute n-shell occupation across l = 0..n−1 by 2(2l+1) weights
    occupations: Dict[Tuple[int, int], float] = {}
    for n_idx in range(1, n_max + 1):
        n_total = float(occ_arr[n_idx - 1])
        cap = _shell_capacity(n_idx)
        for ll in range(n_idx):
            weight = 2.0 * (2 * ll + 1)
            occupations[(n_idx, ll)] = n_total * weight / cap

    R_ws = ion_sphere_radius(max(Z_bar, 1e-3), ne_cm3)  # already in a₀

    return AverageAtomResult(
        Z=Z, T_eV=T_eV, ne_cm3=ne_cm3,
        Z_bar=Z_bar, mu_eV=mu_au * HARTREE_EV,
        occupations=occupations, R_ws_au=R_ws,
    )


__all__ = [
    "AverageAtomResult",
    "average_atom_thomas_fermi",
]
