"""Radiative recombination (RR) cross sections and rate coefficients.

The inverse of photoionisation:

    A^{q+} + e⁻  →  A^{(q-1)+}(n l j) + γ

is related to photoionisation by Milne detailed balance (Milne 1924):

    σ_RR(E_e; n) = (g_n / (2 g_{q+}))
                   · (ε² / (m_e c² · E_e))
                   · σ_PI(ε; n)

with ε = E_e + I_n the photon energy, ``g`` statistical weights,
``E_e`` the free-electron kinetic energy, and ``σ_PI`` the
photoionisation cross section for the bound orbital ``n``.

In atomic units (m_e c² = c² = α⁻², where α ≈ 1/137):

    σ_RR(E_e; n) [a₀²] = (g_n / (2 g_{q+}))
                         · α² · ε² / E_e
                         · σ_PI(ε; n) [a₀²].

This module:

- :func:`rr_cross_section` — total RR cross section summed over
  capture into specified bound orbitals of the (N-electron) atom,
  using AtomX's existing :func:`photoionization_cross_section`.
- :func:`rr_rate_coefficient` — Maxwellian-averaged α_RR(T_e).
- Returns are in atomic units (cross sections in a₀²) by default;
  optional cgs unit converters provided.

Limitations
-----------
- Single-orbital approximation: assumes the ``(N-1)``-electron core
  is identical in PI and RR (no shake-up).
- Length-gauge dipole only.
- Includes only E1; M1/E2 are typically <1 % at energies below
  Z·keV.

References
----------
- E. A. Milne, *Phil. Mag.* **47**, 209 (1924).
- W. J. Karzas, R. Latter, *ApJS* **6**, 167 (1961) — hydrogenic
  benchmark.
- D. A. Verner, G. J. Ferland, *ApJS* **103**, 467 (1996) —
  closed-form RR fits used in spectral synthesis.
- M. F. Gu, *Astrophys. J.* **589**, 1085 (2003) — FAC RR algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from atomx.constants import ALPHA, BOHR_RADIUS_M, HARTREE_EV
from atomx.numerics.bspline.continuum import photoionization_cross_section
from atomx.prop.transitions.angular import line_strength_factor
from atomx.scattering.models import ContinuumModel


BOHR_CM2 = (BOHR_RADIUS_M * 100.0) ** 2  # a₀² in cm²


def _final_e1_kappas(bound_kappa: int) -> list[int]:
    l_bound = bound_kappa if bound_kappa > 0 else -bound_kappa - 1
    out = []
    for l_f in (l_bound - 1, l_bound + 1):
        if l_f < 0:
            continue
        out.append(-(l_f + 1))
        if l_f > 0:
            out.append(l_f)
    return out


def _shooting_photoionization_cross_section(
    scf_result,
    sh,
    P_bound,
    Q_bound,
    photon_au: np.ndarray,
    continuum_model: ContinuumModel,
) -> np.ndarray:
    """PI cross section using direct shooting continuum from a model."""
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_shooting

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("shooting PI requires B-spline data for bound orbitals")
    gr = bsp_data["gr"].ravel()
    sort_idx = np.argsort(gr)
    gr_sorted = gr[sort_idx]
    Pb_grid = eval_P(P_bound, bsp_data).ravel()[sort_idx]
    Qb_grid = eval_Q(Q_bound, bsp_data).ravel()[sort_idx]

    potential_model = continuum_model.potential
    if potential_model is None:
        from atomx.scattering.models import PotentialModel
        potential_model = PotentialModel.bare()
    V_func = potential_model.build_callable(
        scf_result,
        Z=float(scf_result.Z),
        exclude_label=sh.label,
    )
    n_electrons = sum(float(s.occupation) for s in scf_result.subshells)
    z_residual = max(float(scf_result.Z) - n_electrons + 1.0, 1.0)

    sigma = np.zeros_like(photon_au, dtype=float)
    for idx, omega in enumerate(photon_au):
        electron_energy = float(omega + scf_result.orbital_energies[sh.label])
        if electron_energy <= 0.0:
            continue
        for kf in _final_e1_kappas(sh.kappa):
            angular_factor = line_strength_factor(sh.kappa, kf, 1, "E") / abs(sh.kappa)
            if angular_factor == 0.0:
                continue
            r, P_cont, Q_cont, _delta = solve_dirac_continuum_shooting(
                electron_energy,
                kf,
                V_func,
                n_grid=continuum_model.shooting_n_grid,
                Z_residual=z_residual,
            )
            Pb = np.interp(r, gr_sorted, Pb_grid, left=0.0, right=0.0)
            Qb = np.interp(r, gr_sorted, Qb_grid, left=0.0, right=0.0)
            dme = np.trapezoid((P_cont * Pb + Q_cont * Qb) * r, r)
            sigma[idx] += (
                4.0 * np.pi ** 2 * ALPHA * float(omega) / 3.0
                * dme ** 2 * angular_factor
            )
    return sigma


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RRResult:
    """Outcome of an RR cross-section calculation.

    Attributes
    ----------
    energies_eV
        Free-electron kinetic energies, eV.
    sigma_total_a0_sq
        Σ_n σ_RR(E_e; n) in atomic units (a₀²).
    sigma_by_orbital
        Per-orbital partial cross sections (a₀²).
    sigma_total_cm2
        Same as ``sigma_total_a0_sq`` but in cm² (convenience).
    provenance
        Free-form notes on which orbitals contributed.
    """

    energies_eV: np.ndarray
    sigma_total_a0_sq: np.ndarray
    sigma_by_orbital: Dict[str, np.ndarray] = field(default_factory=dict)
    sigma_total_cm2: Optional[np.ndarray] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.sigma_total_cm2 is None:
            self.sigma_total_cm2 = self.sigma_total_a0_sq * BOHR_CM2


# ---------------------------------------------------------------------------
# Cross section
# ---------------------------------------------------------------------------


def rr_cross_section(
    scf_result,
    energies_eV: np.ndarray,
    *,
    capture_orbitals: Optional[List[str]] = None,
    g_initial: Optional[int] = None,
    g_final: Optional[int] = None,
    continuum_model: Any = None,
) -> RRResult:
    """Total radiative-recombination cross section via Milne detailed balance.

    Parameters
    ----------
    scf_result
        SCF result of the *recombined* (N-electron) atom.  Each
        occupied orbital is treated as a possible capture target.
    energies_eV
        Free-electron kinetic energies, eV.
    capture_orbitals
        Optional whitelist of subshell labels (e.g. ``["1s1/2"]``).
        Default: every occupied subshell.
    g_initial
        Statistical weight ``2J_ion + 1`` of the recombining ion,
        excluding the free electron.  The free-electron spin factor
        is included explicitly in the Milne relation.  Default: 1 for
        a closed-shell or hydrogenic bare ion core.
    g_final
        Statistical weight ``g_n`` of the recombined bound state.
        Default: ``2j_n + 1`` of the captured electron's subshell —
        appropriate for closed-shell (N-1)-cores.
    continuum_model
        Optional compatibility hook for benchmark callers.  The current
        implementation uses the continuum data already attached to
        ``scf_result``.

    Returns
    -------
    RRResult
    """
    energies = np.asarray(energies_eV, dtype=float)
    energies_au = energies / HARTREE_EV
    if np.any(energies_au <= 0):
        raise ValueError("energies_eV must all be > 0.")

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("SCFResult is missing _bsp_data — re-run solve().")
    Z = scf_result.Z

    # Default statistical weights
    if g_initial is None:
        g_initial = 1  # closed-shell-core ion: J=0 → 2J+1 = 1

    sigma_total = np.zeros_like(energies)
    by_orbital: Dict[str, np.ndarray] = {}
    prov_per_orb: Dict[str, Dict[str, float]] = {}

    for sh in scf_result.subshells:
        if capture_orbitals is not None and sh.label not in capture_orbitals:
            continue
        eps_n = scf_result.orbital_energies.get(sh.label)
        if eps_n is None or eps_n >= 0.0:
            continue
        I_n_eV = -eps_n * HARTREE_EV  # binding energy
        I_n_au = -eps_n

        # σ_PI(ε) for this subshell — solve continuum on demand
        # AtomX takes photon energy in atomic units (hartree).
        photon_au = energies_au + I_n_au   # ε = E_e + I_n

        orb_list = scf_result.orbitals_raw.get(sh.kappa, [])
        kappa_subshells = [s for s in scf_result.subshells if s.kappa == sh.kappa]
        idx_in_kappa = kappa_subshells.index(sh)
        if idx_in_kappa >= len(orb_list):
            continue
        P_b, Q_b, _ = orb_list[idx_in_kappa]

        if (
            isinstance(continuum_model, ContinuumModel)
            and continuum_model.method.lower() == "shooting"
        ):
            sigma_PI_a0sq = _shooting_photoionization_cross_section(
                scf_result,
                sh,
                P_b,
                Q_b,
                photon_au,
                continuum_model,
            )
        else:
            sigma_PI_a0sq = photoionization_cross_section(
                P_b, Q_b, sh.kappa, eps_n, photon_au, Z, bsp_data,
            )

        # Statistical-weight ratio
        g_n = g_final if g_final is not None else (2 * abs(sh.kappa))
        # Milne factor in atomic units:
        # σ_RR/σ_PI = (g_n/(2 g_ion)) · α²·ω²/(2 E_e).
        # The 2 in the degeneracy ratio is the free-electron spin.
        milne = (ALPHA ** 2) * (photon_au ** 2) / (2.0 * energies_au)
        sigma_RR = (g_n / (2.0 * g_initial)) * milne * sigma_PI_a0sq

        by_orbital[sh.label] = sigma_RR
        sigma_total += sigma_RR
        prov_per_orb[sh.label] = {
            "I_n_eV": I_n_eV,
            "g_n": g_n,
            "occupation": float(sh.occupation),
        }

    return RRResult(
        energies_eV=energies,
        sigma_total_a0_sq=sigma_total,
        sigma_by_orbital=by_orbital,
        provenance={
            "Z": Z,
            "g_initial": g_initial,
            "continuum_model": continuum_model,
            "per_orbital": prov_per_orb,
        },
    )


# ---------------------------------------------------------------------------
# Rate coefficient
# ---------------------------------------------------------------------------


def rr_rate_coefficient(
    rr_result: RRResult,
    temperature_eV: np.ndarray,
) -> np.ndarray:
    """Maxwellian-averaged RR rate coefficient α_RR(T_e), cm³/s."""
    from atomx.plasma.rates import maxwellian_rate
    T_arr = np.atleast_1d(np.asarray(temperature_eV, dtype=float))
    sigma_cm2 = rr_result.sigma_total_cm2
    if sigma_cm2 is None:
        sigma_cm2 = rr_result.sigma_total_a0_sq * BOHR_CM2
    out = np.array([
        maxwellian_rate(sigma_cm2, rr_result.energies_eV, float(T))
        for T in T_arr
    ])
    return out if out.shape != (1,) else float(out[0])


def rr_cross_section_rydberg(
    scf_result,
    energies_eV: np.ndarray,
    *,
    n_max_per_kappa: int = 8,
    kappa_max: int = 4,
    g_initial: int = 1,
) -> RRResult:
    """First-principles RR including capture into the full Rydberg
    series.

    Unlike :func:`rr_cross_section` (which sums only over occupied
    subshells of the SCF result), this function builds the V^{N-1}
    bound-state basis for every κ in scope and includes capture into
    every bound Rydberg state — required to recover the hydrogenic
    α_RR ≈ Σ_n α_n / n^≈3 baseline.

    Parameters
    ----------
    scf_result
        SCF result for the recombined system (the (N-1)-electron
        ion's potential is reconstructed from the converged orbitals).
    energies_eV
        Free-electron kinetic energies, eV.
    n_max_per_kappa
        Cap on the number of bound Rydberg states per κ included
        (lowest-energy retained).
    kappa_max
        Largest |κ| in the partial-wave sum.
    g_initial
        2J+1 of the recombining ion (default 1 for closed-shell core).

    Returns
    -------
    RRResult
    """
    from atomx.methods.mbpt import _build_virtual_basis, _l_from_kappa
    from atomx.numerics.bspline.convol import ConvolCache

    energies = np.asarray(energies_eV, dtype=float)
    energies_au = energies / HARTREE_EV
    if np.any(energies_au <= 0):
        raise ValueError("energies_eV must all be > 0.")

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("SCFResult is missing _bsp_data — re-run solve().")
    Z = scf_result.Z
    cache = ConvolCache(bsp_data)

    # Enumerate κ list
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if 0 < abs(k) and _l_from_kappa(k) <= kappa_max]

    sigma_total = np.zeros_like(energies)
    by_state: Dict[str, np.ndarray] = {}
    prov: Dict[str, Dict[str, float]] = {}

    for kappa in kappa_list:
        occ, virt = _build_virtual_basis(scf_result, kappa, cache=cache,
                                          potential_mode="VN-1")
        # Each bound state at energy < 0 is a capture target
        bound_states = [(P, Q, eps) for (P, Q, eps) in (list(occ) + list(virt))
                         if eps < 0.0]
        bound_states.sort(key=lambda t: t[2])
        bound_states = bound_states[:n_max_per_kappa]

        l = _l_from_kappa(kappa)
        j2 = 2 * abs(kappa) - 1
        for n_idx, (P_n, Q_n, eps_n) in enumerate(bound_states):
            n_principal = l + 1 + n_idx  # rough mapping (shifts can occur)
            label = f"{n_principal}{'spdfghi'[l] if l<7 else f'l{l}'}{j2}/2"

            I_n_au = -eps_n
            photon_au = energies_au + I_n_au
            sigma_PI = photoionization_cross_section(
                P_n, Q_n, kappa, eps_n, photon_au, Z, bsp_data,
            )
            g_n = j2 + 1
            milne = (ALPHA ** 2) * (photon_au ** 2) / (2.0 * energies_au)
            sigma_RR = (g_n / (2.0 * g_initial)) * milne * sigma_PI

            by_state[label] = sigma_RR
            sigma_total += sigma_RR
            prov[label] = {
                "I_n_eV": I_n_au * HARTREE_EV,
                "kappa": kappa,
                "n_principal_estimate": n_principal,
                "eps_n_au": float(eps_n),
            }

    return RRResult(
        energies_eV=energies,
        sigma_total_a0_sq=sigma_total,
        sigma_by_orbital=by_state,
        provenance={
            "Z": Z,
            "g_initial": g_initial,
            "method": "Rydberg sum over V^{N-1} bound states",
            "n_max_per_kappa": n_max_per_kappa,
            "kappa_max": kappa_max,
            "per_state": prov,
        },
    )


__all__ = [
    "RRResult",
    "rr_cross_section",
    "rr_cross_section_rydberg",
    "rr_rate_coefficient",
]
