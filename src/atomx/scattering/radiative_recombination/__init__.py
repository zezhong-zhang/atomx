"""Radiative recombination (RR) cross sections and rate coefficients.

The inverse of photoionisation:

    A^{q+} + e⁻  →  A^{(q-1)+}(n l j) + γ

is related to photoionisation by Milne detailed balance (Milne 1924):

    σ_RR(E_e; n) = (g_n / g_{q+})
                   · (ε² / (2 m_e c² · E_e))
                   · σ_PI(ε; n)

with ε = E_e + I_n the photon energy, ``g`` statistical weights,
``E_e`` the free-electron kinetic energy, and ``σ_PI`` the
photoionisation cross section for the bound orbital ``n``.

In atomic units (m_e c² = c² = α⁻², where α ≈ 1/137):

    σ_RR(E_e; n) [a₀²] = (g_n / g_{q+})
                         · α² · ε² / (2 E_e)
                         · σ_PI(ε; n) [a₀²].

In detailed balance the free-electron spin degeneracy (2) cancels the
photon's two polarisations, so no factor 2 multiplies the ion
degeneracy ``g_{q+}``; the remaining 1/2 comes from
``E_e = p² / (2 m_e)``.  This is the standard Milne form (Gu 2003,
ApJ 589, 1085) and is anchored absolutely by the H 1s RR rate at
1e4 K (Storey & Hummer 1995; test_hydrogenic_cross_sections).

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
- Single-active-electron bound-free matrix elements in the long-wavelength
  approximation.
- Direct shooting continuum currently supports E1 only; B-spline continuum
  supports E1, M1, and E2.

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
from atomx.numerics.bspline.continuum import solve_continuum
from atomx.scattering.photoionization.direct import photoionization_cross_section
from atomx.properties.transitions.angular import (
    allowed_final_kappas,
    line_strength_factor,
)
from atomx.properties.transitions.multipole import radial_eL, radial_mL
from atomx.scattering.models import ContinuumModel

BOHR_CM2 = (BOHR_RADIUS_M * 100.0) ** 2  # a₀² in cm²
AU_VELOCITY_CM_S = ALPHA * 299_792_458.0 * 100.0

DEFAULT_MULTIPOLES = (("E", 1), ("M", 1), ("E", 2))


def _final_e1_kappas(bound_kappa: int) -> list[int]:
    return allowed_final_kappas(bound_kappa, 1, "E")


def _final_m1_kappas(bound_kappa: int) -> list[int]:
    return allowed_final_kappas(bound_kappa, 1, "M")


def _final_e2_kappas(bound_kappa: int) -> list[int]:
    return allowed_final_kappas(bound_kappa, 2, "E")


def _multipole_label(multipole_type: str, L: int) -> str:
    return f"{multipole_type.upper()}{int(L)}"


def _multipole_prefactor_scale(multipole_type: str, L: int, omega: float) -> float:
    """Scale the existing E1 PI prefactor to an EL/ML long-wave prefactor."""
    if L < 1:
        raise ValueError("multipole rank L must be >= 1")
    if omega <= 0.0:
        return 0.0
    # (2L+1)!! = 1*3*5*...*(2L+1); equals
    # properties.transitions.rates._double_factorial(2*L+1) — kept inline here
    # to avoid a scattering→properties import. Keep the two in sync.
    dbl_fact = 1
    for n in range(1, 2 * L + 2, 2):
        dbl_fact *= n
    pref_L = (
        2.0 * ALPHA ** (2 * L + 1) * omega ** (2 * L + 1)
        * (L + 1) * (2 * L + 1)
        / (L * dbl_fact ** 2)
    )
    pref_e1 = (4.0 / 3.0) * ALPHA ** 3 * omega ** 3
    return pref_L / pref_e1


def _normalise_multipoles(multipoles) -> tuple[tuple[str, int], ...]:
    if multipoles is None:
        return DEFAULT_MULTIPOLES
    out = []
    for item in multipoles:
        if isinstance(item, str):
            kind = item[0].upper()
            rank = int(item[1:])
        else:
            kind = str(item[0]).upper()
            rank = int(item[1])
        if kind not in {"E", "M"}:
            raise ValueError(f"unknown multipole type {kind!r}")
        if rank < 1:
            raise ValueError("multipole rank must be >= 1")
        pair = (kind, rank)
        if pair not in out:
            out.append(pair)
    if not out:
        raise ValueError("at least one multipole must be requested")
    return tuple(out)


def _radial_multipole(P_bound, Q_bound, P_cont, Q_cont, bsp_data, kind: str, L: int):
    if kind == "M":
        return radial_mL(P_bound, Q_bound, P_cont, Q_cont, bsp_data, L=L)
    return radial_eL(P_bound, Q_bound, P_cont, Q_cont, bsp_data, L=L)


def _residual_hartree_potential(result, capture_label):
    """Frozen V^(N-1) direct field on the bound-state quadrature grid.

    Removing one captured electron preserves the core screening and its
    Coulomb tail. This local channel approximation omits core relaxation
    and nonlocal exchange; it is recorded in the result provenance.
    """
    from atomx.numerics.bspline.evaluate import compute_yk, eval_P, eval_Q
    data = result._bsp_data
    radius = data["gr"]
    density = np.zeros_like(radius)
    number = 0.0
    if hasattr(result, "orbitals_by_subshell"):
        orbitals = result.orbitals_by_subshell()
    else:
        orbitals = {(sh.n, sh.kappa): (*result.orbitals_dict[sh.n, sh.kappa], sh.occupation)
                    for sh in result.subshells}
    for key, (p, q, occupation) in orbitals.items():
        sh = next(sh for sh in result.subshells if (sh.n, sh.kappa) == key)
        weight = max(0.0, occupation - (1.0 if sh.label == capture_label else 0.0))
        if weight:
            density += weight * (eval_P(p, data) ** 2 + eval_Q(q, data) ** 2)
            number += weight
    if hasattr(result, "one_body_density_matrix"):
        keys, one_body = result.one_body_density_matrix()
        # The diagonal contributions above already include the hole.
        # Retain configuration interference in the frozen residual density.
        sampled = {key: (eval_P(orbitals[key][0], data), eval_Q(orbitals[key][1], data))
                   for key in keys}
        for i, a in enumerate(keys):
            for j in range(i):
                b = keys[j]
                if a[1] == b[1]:
                    pa, qa = sampled[a]
                    pb, qb = sampled[b]
                    density += 2 * one_body[i, j] * (pa * pb + qa * qb)
    hartree = compute_yk(density, radius, data["grw"], data["nv"], radius.shape[1], 0)
    nuclear = getattr(result, "_nuclear_potential", None)

    def potential(r):
        r = np.asarray(r)
        direct = np.interp(r, radius.ravel(), hartree.ravel(), left=float(hartree.flat[0]), right=0.0)
        direct = np.where(r > radius.flat[-1], number / r, direct)
        return (-result.Z / r if nuclear is None else np.vectorize(nuclear, otypes=[float])(r)) + direct

    return potential, float(result.Z) - number


def _rr_potential(result, sh, model):
    if model is None or model.potential is None:
        return _residual_hartree_potential(result, sh.label)
    potential = model.potential.build_callable(result, exclude_label=sh.label)
    residual = (float(result.Z) if model.potential.kind == "bare" else
                float(result.Z) - sum(s.occupation for s in result.subshells) + min(1.0, sh.occupation))
    return potential, residual


def _rr_potential_matrices(result, sh, model):
    from atomx.numerics.bspline.integrals import _assemble_integral
    data = result._bsp_data
    potential, _ = _rr_potential(result, sh, model)
    # solve_continuum already contains the point nuclear Hamiltonian.
    extra = potential(data["gr"]) + result.Z / data["gr"]
    weights = data["grw"] * extra
    matrices = []
    for component in ("p", "q"):
        order = data.get("ks" + component, data["ks"])
        basis = data["bsp_" + component]
        matrices.append(_assemble_integral(data["nv"], order, order, data["ns"], basis, basis, weights))
    return tuple(matrices)


def _bound_free_multipole_cross_sections(
    bound_P,
    bound_Q,
    bound_kappa: int,
    bound_energy: float,
    photon_energies: np.ndarray,
    Z: int,
    bsp_data,
    multipoles,
    V_scf_PP=None, V_scf_QQ=None,
) -> Dict[str, np.ndarray]:
    """Photoionization partial cross sections for E/M multipoles."""
    partials = {
        _multipole_label(kind, L): np.zeros_like(photon_energies, dtype=float)
        for kind, L in multipoles
    }

    for idx, omega in enumerate(photon_energies):
        electron_energy = float(omega + bound_energy)
        if electron_energy <= 0.0:
            continue

        continuum_cache = {}
        for kind, L in multipoles:
            label = _multipole_label(kind, L)
            for kf in allowed_final_kappas(bound_kappa, L, kind):
                angular_factor = (
                    line_strength_factor(bound_kappa, kf, L, kind)
                    / (2.0 * abs(bound_kappa))
                )
                if angular_factor == 0.0:
                    continue
                if kf not in continuum_cache:
                    Pf, Qf, _E_actual = solve_continuum(
                        kf, electron_energy, Z, bsp_data, V_scf_PP, V_scf_QQ,
                        energy_normalized=True,
                    )
                    continuum_cache[kf] = (Pf, Qf)
                Pf, Qf = continuum_cache[kf]
                rme = _radial_multipole(
                    bound_P, bound_Q, Pf, Qf, bsp_data, kind, L
                )
                prefactor = (
                    4.0 * np.pi ** 2 * ALPHA * float(omega) / 3.0
                    * _multipole_prefactor_scale(kind, L, float(omega))
                )
                partials[label][idx] += prefactor * rme ** 2 * angular_factor

    return partials


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
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_many

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("shooting PI requires B-spline data for bound orbitals")
    gr = bsp_data["gr"].ravel()
    sort_idx = np.argsort(gr)
    gr_sorted = gr[sort_idx]
    Pb_grid = eval_P(P_bound, bsp_data).ravel()[sort_idx]
    Qb_grid = eval_Q(Q_bound, bsp_data).ravel()[sort_idx]

    V_func, z_residual = _rr_potential(scf_result, sh, continuum_model)

    sigma = np.zeros_like(photon_au, dtype=float)
    channels = [(kf, line_strength_factor(sh.kappa, kf, 1, "E") / (2 * abs(sh.kappa)))
                for kf in _final_e1_kappas(sh.kappa)]
    channels = [(kf, factor) for kf, factor in channels if factor != 0.0]
    active = [(idx, float(omega), float(omega + scf_result.orbital_energies[sh.label]))
              for idx, omega in enumerate(photon_au)
              if omega + scf_result.orbital_energies[sh.label] > 0.0]
    if not active or not channels:
        return sigma

    # All energies and partial waves share one potential spline and radial
    # grid.  Prepare once, solve independent waves in parallel, and sample
    # the bound orbital only once on that common grid.
    requests = [(kf, energy) for _idx, _omega, energy in active
                for kf, _factor in channels]
    r, waves = solve_dirac_continuum_many(
        requests, V_func,
        n_grid=continuum_model.shooting_n_grid,
        r_max=continuum_model.shooting_r_max or 50.0,
        Z_residual=z_residual,
    )
    Pb = np.interp(r, gr_sorted, Pb_grid, left=0.0, right=0.0)
    Qb = np.interp(r, gr_sorted, Qb_grid, left=0.0, right=0.0)
    radial_weight_P = Pb * r
    radial_weight_Q = Qb * r
    for idx, omega, electron_energy in active:
        for kf, angular_factor in channels:
            wave = waves[(kf, electron_energy)]
            if wave is None:
                raise RuntimeError(
                    f"shooting PI continuum failed at E={electron_energy} Ha, kappa={kf}")
            P_cont, Q_cont = wave
            dme = np.trapezoid(P_cont * radial_weight_P + Q_cont * radial_weight_Q, r)
            sigma[idx] += (
                4.0 * np.pi ** 2 * ALPHA * omega / 3.0
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
    sigma_by_multipole
        Per-multipole partial cross sections (a₀²), keyed like ``"E1"``.
    sigma_total_cm2
        Same as ``sigma_total_a0_sq`` but in cm² (convenience).
    provenance
        Free-form notes on which orbitals contributed.
    """

    energies_eV: np.ndarray
    sigma_total_a0_sq: np.ndarray
    sigma_by_orbital: Dict[str, np.ndarray] = field(default_factory=dict)
    sigma_by_multipole: Dict[str, np.ndarray] = field(default_factory=dict)
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
    multipoles=None,
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
        excluding the free electron (whose spin degeneracy cancels
        the photon polarisations in the Milne relation).  Default: 1
        for a closed-shell or hydrogenic bare ion core.
    g_final
        Statistical weight ``g_n`` of the recombined bound state.
        Default: ``2j_n + 1`` of the captured electron's subshell —
        appropriate for closed-shell (N-1)-cores.
    continuum_model
        Optional explicit local potential model. By default, construct the
        frozen residual Hartree core after removing one captured electron.
        Nonlocal exchange and relaxation of the residual core are omitted
        and this approximation is recorded in the result provenance.
    multipoles
        Iterable of multipoles to include, e.g. ``("E1", "M1", "E2")``
        or ``(("E", 1), ("M", 1))``.  Default includes E1, M1, and E2.

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

    multipoles = _normalise_multipoles(multipoles)
    sigma_total = np.zeros_like(energies)
    by_orbital: Dict[str, np.ndarray] = {}
    by_multipole: Dict[str, np.ndarray] = {
        _multipole_label(kind, L): np.zeros_like(energies)
        for kind, L in multipoles
    }
    prov_per_orb: Dict[str, Dict[str, float]] = {}

    for sh in scf_result.subshells:
        if capture_orbitals is not None and sh.label not in capture_orbitals:
            continue
        if sh.occupation <= 0:
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

        if continuum_model is not None and not isinstance(continuum_model, ContinuumModel):
            raise TypeError("continuum_model must be a ContinuumModel")
        if continuum_model is not None and continuum_model.method.lower() not in {"shooting", "bspline"}:
            raise ValueError("unknown continuum numerical method")
        potentials = (None, None)
        if continuum_model is None or continuum_model.method.lower() == "bspline":
            potentials = _rr_potential_matrices(scf_result, sh, continuum_model)
        if (
            isinstance(continuum_model, ContinuumModel)
            and continuum_model.method.lower() == "shooting"
        ):
            if multipoles != (("E", 1),):
                raise NotImplementedError(
                    "shooting RR currently supports only the E1 multipole"
                )
            sigma_PI_partials = {
                "E1": _shooting_photoionization_cross_section(
                    scf_result,
                    sh,
                    P_b,
                    Q_b,
                    photon_au,
                    continuum_model,
                )
            }
        elif multipoles == (("E", 1),):
            sigma_PI_partials = {
                "E1": photoionization_cross_section(
                    P_b, Q_b, sh.kappa, eps_n, photon_au, Z, bsp_data, *potentials,
                )
            }
        else:
            sigma_PI_partials = _bound_free_multipole_cross_sections(
                P_b, Q_b, sh.kappa, eps_n, photon_au, Z, bsp_data, multipoles, *potentials
            )

        # Statistical-weight ratio
        g_n = g_final if g_final is not None else (2 * abs(sh.kappa))
        # Milne factor in atomic units:
        # σ_RR/σ_PI = (g_n/g_ion) · α²·ω²/p_e²,
        # p_e² = E_e (2 + α² E_e), including relativistic phase space.
        # The free-electron spin degeneracy cancels the two photon
        # polarisations in detailed balance; the 1/2 in ω²/(2 E_e)
        # comes from E_e = p²/(2 m_e) (Gu 2003, ApJ 589, 1085, eq. 1).
        milne = (ALPHA ** 2) * (photon_au ** 2) / (energies_au * (2.0 + ALPHA ** 2 * energies_au))
        sigma_RR_partials = {
            label: (g_n / g_initial) * milne * sigma_PI
            for label, sigma_PI in sigma_PI_partials.items()
        }
        sigma_RR = np.sum(list(sigma_RR_partials.values()), axis=0)

        by_orbital[sh.label] = sigma_RR
        sigma_total += sigma_RR
        for label, sigma_part in sigma_RR_partials.items():
            by_multipole.setdefault(label, np.zeros_like(energies))
            by_multipole[label] += sigma_part
        prov_per_orb[sh.label] = {
            "I_n_eV": I_n_eV,
            "g_n": g_n,
            "occupation": float(sh.occupation),
            "multipoles": list(sigma_RR_partials),
        }

    return RRResult(
        energies_eV=energies,
        sigma_total_a0_sq=sigma_total,
        sigma_by_orbital=by_orbital,
        sigma_by_multipole=by_multipole,
        provenance={
            "Z": Z,
            "g_initial": g_initial,
            "continuum_model": continuum_model,
            "continuum_potential": ("frozen V^(N-1) Hartree; no nonlocal exchange or core relaxation"
                                    if continuum_model is None or continuum_model.potential is None
                                    else continuum_model.potential.kind),
            "multipoles": [_multipole_label(kind, L) for kind, L in multipoles],
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


def rrc_spectrum(
    scf_result,
    T_e_eV: float,
    energies_eV: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Radiative recombination continuum emissivity on a photon-energy grid.

    Returns photon emissivity coefficient in photons cm³ s⁻¹ eV⁻¹.  For each
    capture orbital, the electron kinetic energy is ``E_gamma - I_n`` and the
    differential coefficient is ``v(E) * sigma_RR(E) * f_M(E, T)``.
    """
    T = float(T_e_eV)
    if T <= 0.0:
        raise ValueError("T_e_eV must be > 0")
    photon_energies = np.asarray(energies_eV, dtype=float)
    if np.any(photon_energies <= 0.0):
        raise ValueError("energies_eV must all be > 0")

    spectrum = np.zeros_like(photon_energies, dtype=float)
    capture_orbitals = kwargs.get("capture_orbitals")

    for sh in scf_result.subshells:
        if capture_orbitals is not None and sh.label not in capture_orbitals:
            continue
        eps_n = scf_result.orbital_energies.get(sh.label)
        if eps_n is None or eps_n >= 0.0:
            continue
        binding_eV = -eps_n * HARTREE_EV
        electron_eV = photon_energies - binding_eV
        mask = electron_eV > 0.0
        if not np.any(mask):
            continue

        rr = rr_cross_section(
            scf_result,
            electron_eV[mask],
            capture_orbitals=[sh.label],
            g_initial=kwargs.get("g_initial"),
            g_final=kwargs.get("g_final"),
            continuum_model=kwargs.get("continuum_model"),
            multipoles=kwargs.get("multipoles"),
        )
        pdf_per_eV = (
            2.0 / np.sqrt(np.pi)
            * np.sqrt(electron_eV[mask])
            / T ** 1.5
            * np.exp(-electron_eV[mask] / T)
        )
        velocity_cm_s = AU_VELOCITY_CM_S * np.sqrt(
            2.0 * electron_eV[mask] / HARTREE_EV
        )
        spectrum[mask] += velocity_cm_s * rr.sigma_total_cm2 * pdf_per_eV

    return spectrum


def rr_cross_section_rydberg(
    scf_result,
    energies_eV: np.ndarray,
    *,
    n_max_per_kappa: int = 8,
    kappa_max: int = 4,
    g_initial: int = 1,
) -> RRResult:
    """Independent-particle RR into a truncated Rydberg series.

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
    from atomx.methods.mbpt import _l_from_kappa
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac

    energies = np.asarray(energies_eV, dtype=float)
    energies_au = energies / HARTREE_EV
    if np.any(energies_au <= 0):
        raise ValueError("energies_eV must all be > 0.")

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("SCFResult is missing _bsp_data — re-run solve().")
    Z = scf_result.Z
    occupied = [sh for sh in scf_result.subshells if sh.occupation > 0]
    if not occupied:
        raise ValueError("the recombined system must contain a captured electron")
    outer = max(occupied, key=lambda sh: (sh.n, _l_from_kappa(sh.kappa), -abs(sh.kappa)))
    potential_pp, potential_qq = _rr_potential_matrices(scf_result, outer, None)
    core_occupations = {(sh.n, sh.kappa): max(0., sh.occupation - (sh is outer))
                        for sh in occupied}

    # Enumerate κ list
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if abs(k) > 0 and _l_from_kappa(k) <= kappa_max]

    sigma_total = np.zeros_like(energies)
    by_state: Dict[str, np.ndarray] = {}
    prov: Dict[str, Dict[str, float]] = {}

    for kappa in kappa_list:
        # Use one local residual-core Hamiltonian for bound and continuum
        # states. Mixing HF bound states with bare-nuclear continuum waves
        # is not a consistent independent-particle recombination model.
        ns = bsp_data["ns"]
        h, s = build_dirac_matrices(ns, kappa, Z, bsp_data)
        h[:ns, :ns] += potential_pp
        h[ns:, ns:] += potential_qq
        basis = solve_dirac(h, s, ns, kappa, nsp=bsp_data.get("nsp"), nsq=bsp_data.get("nsq"))
        # Each bound state at energy < 0 is a capture target
        has_kb = bsp_data.get("ksp", bsp_data["ks"]) != bsp_data.get("ksq", bsp_data["ks"])
        first_physical = int(kappa > 0 and not has_kb)
        bound_states = [basis.get_orbital(i) for i, eps in enumerate(basis.eigenvalues)
                        if eps < 0.0 and i >= first_physical]
        bound_states.sort(key=lambda t: t[2])
        bound_states = bound_states[:n_max_per_kappa]

        l = _l_from_kappa(kappa)
        j2 = 2 * abs(kappa) - 1
        for n_idx, (P_n, Q_n, eps_n) in enumerate(bound_states):
            n_principal = l + 1 + n_idx  # rough mapping (shifts can occur)
            vacancies = max(0., j2 + 1 - core_occupations.get((n_principal, kappa), 0.))
            if vacancies < 1e-12:
                continue
            label = f"{n_principal}{'spdfghi'[l] if l<7 else f'l{l}'}{j2}/2"

            I_n_au = -eps_n
            photon_au = energies_au + I_n_au
            sigma_PI = photoionization_cross_section(
                P_n, Q_n, kappa, eps_n, photon_au, Z, bsp_data,
                V_scf_PP=potential_pp, V_scf_QQ=potential_qq,
            )
            g_n = vacancies
            milne = (ALPHA ** 2) * (photon_au ** 2) / (energies_au * (2.0 + ALPHA ** 2 * energies_au))
            sigma_RR = (g_n / g_initial) * milne * sigma_PI

            by_state[label] = sigma_RR
            sigma_total += sigma_RR
            prov[label] = {
                "I_n_eV": I_n_au * HARTREE_EV,
                "kappa": kappa,
                "n_principal_estimate": n_principal,
                "eps_n_au": float(eps_n),
                "vacancies": vacancies,
            }

    return RRResult(
        energies_eV=energies,
        sigma_total_a0_sq=sigma_total,
        sigma_by_orbital=by_state,
        provenance={
            "Z": Z,
            "g_initial": g_initial,
            "method": "Rydberg sum in a common frozen residual Hartree potential",
            "approximation": "local independent particles; statistical Pauli vacancies; no nonlocal exchange/core relaxation",
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
    "rrc_spectrum",
]
