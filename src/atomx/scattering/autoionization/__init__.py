"""Autoionization, Auger rates, and dielectronic recombination (DR).

This module implements the three companion processes that govern the
population and decay of doubly-excited (autoionizing) atomic states:

1. **Auger / autoionization** — radiationless decay of a doubly-
   excited state via ejection of one electron to the continuum.  The
   rate is a two-electron Coulomb matrix element with one bound and
   one continuum orbital.

2. **Dielectronic recombination** (DR) — the inverse of Auger, where
   a free electron is captured into a doubly-excited state that
   subsequently decays radiatively to a singly-bound configuration.
   The DR rate coefficient combines Auger and radiative widths via
   Burgess' (1965) detailed-balance formula.

3. **Resonant photoionization** — photoabsorption with a resonance
   structure from autoionizing intermediate states (handled in
   :mod:`atomx.scattering.photoionization`).

Public API
----------
:class:`AutoionizationState`
    Dataclass holding ``(E_resonance, J, Γ_Auger, A_radiative, ...)``
    for a single resonance.

:func:`auger_rate_two_electron`
    Compute Γ_Auger from a pair of bound orbitals + one continuum
    orbital via the two-electron Coulomb matrix element (Slater
    integrals) and the appropriate angular coupling.

:func:`dr_rate_coefficient`
    Detailed-balance DR rate coefficient α^DR(T_e) from a list of
    :class:`AutoionizationState` resonances.

Reference
---------
M. F. Gu, *Astrophys. J. Suppl.* **153**, 389 (2004) — FAC's approach.
A. Burgess, *Astrophys. J.* **139**, 776 (1965) — first DR theory.
D. W. Savin, *Astrophys. J.* **562**, 1259 (2001) — benchmark Fe XVI DR.
"""

from atomx.scattering.autoionization.auger import (
    AutoionizationState,
    auger_rate_two_electron,
)
from atomx.scattering.dielectronic_recombination.solver import (
    DRRateCoefficient,
    dr_rate_coefficient,
)


def _orbital_key_from_label(result, label: str):
    for n, kappa in getattr(result, "orbitals_dict", {}):
        l = kappa if kappa > 0 else -kappa - 1
        j2 = 2 * abs(kappa) - 1
        l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
        if f"{n}{l_char}{j2}/2" == label:
            return (n, kappa)
    return None


def _state_occupation(result, state_index: int | None, key) -> float:
    if key is None:
        return 0.0
    csfs = list(getattr(result, "csfs", []))
    mixing = getattr(result, "mixing_coefficients", None)
    if state_index is not None and mixing is not None and csfs:
        occ = 0.0
        for i, csf in enumerate(csfs):
            coeff2 = float(abs(mixing[i, int(state_index)]) ** 2)
            for sh in getattr(csf, "subshells", []):
                if (sh.n, sh.kappa) == key:
                    occ += coeff2 * float(sh.occupation)
                    break
        return occ
    occs = getattr(result, "occupations", {})
    if key in occs:
        return float(occs[key])
    for sh in getattr(result, "subshells", []):
        if (sh.n, sh.kappa) == key:
            return float(sh.occupation)
    return 0.0


def _channel_population_factor(result, state_index, source_key, source2_key, hole_key):
    q1 = _state_occupation(result, state_index, source_key)
    q2 = _state_occupation(result, state_index, source2_key)
    qh = _state_occupation(result, state_index, hole_key)
    g1 = 2 * abs(source_key[1])
    g2 = 2 * abs(source2_key[1])
    gh = 2 * abs(hole_key[1])
    vacancy_h = max(0.0, gh - qh)
    if source_key == source2_key:
        pair = max(0.0, q1 * max(q1 - 1.0, 0.0) / max(g1 * max(g1 - 1, 1), 1))
    else:
        pair = max(0.0, q1 * q2 / max(g1 * g2, 1))
    return float(pair * vacancy_h / max(gh, 1))


def _coarse_auger_rate(
    result,
    *,
    hole_label: str,
    source_label: str,
    source2_label: str,
    resonance_energy_eV: float | None,
    state_index: int | None,
    kappa_max: int = 4,
    exchange_sign: float = 1.0,
    shooting_n_grid: int | None = None,
    **_kwargs,
):
    import numpy as np

    from atomx.angular.coefficients import exchange_k_range, exchange_lambda
    from atomx.constants import AU_RATE_PER_S, HARTREE_EV
    from atomx.numerics.bspline.continuum import solve_continuum

    source_key = _orbital_key_from_label(result, source_label)
    source2_key = _orbital_key_from_label(result, source2_label)
    hole_key = _orbital_key_from_label(result, hole_label)
    if source_key is None or source2_key is None or hole_key is None:
        rate = 0.0
    else:
        orbitals = getattr(result, "orbitals_dict", {})
        bsp_data = getattr(result, "bsp_data", None) or getattr(result, "_bsp_data", None)
        if not orbitals or not bsp_data:
            rate = 0.0
        else:
            P_a, Q_a = orbitals[source_key]
            P_b, Q_b = orbitals[source2_key]
            P_c, Q_c = orbitals[hole_key]
            E_eV = float(resonance_energy_eV or 0.0)
            E_ha = E_eV / HARTREE_EV
            amplitude = 0.0
            if E_ha > 0.0:
                if shooting_n_grid:
                    from atomx.numerics.bspline.evaluate import compute_yk, eval_P, eval_Q
                    from atomx.numerics.shooting.dirac_continuum import (
                        solve_dirac_continuum_shooting,
                    )

                    gr = bsp_data["gr"]
                    grw = bsp_data["grw"]
                    Z = float(getattr(result, "Z", 0))

                    def _bare_potential(r):
                        return -Z / np.maximum(r, 1e-30)

                    def _eval_bound(P, Q):
                        return eval_P(P, bsp_data), eval_Q(Q, bsp_data)

                    Pa_g, Qa_g = _eval_bound(P_a, Q_a)
                    Pb_g, Qb_g = _eval_bound(P_b, Q_b)
                    Pc_g, Qc_g = _eval_bound(P_c, Q_c)

                    def _grid_slater(P1, Q1, P2, Q2, P3, Q3, P4, Q4, k):
                        rho13 = P1 * P3 + Q1 * Q3
                        rho24 = P2 * P4 + Q2 * Q4
                        y24 = compute_yk(
                            rho24, gr, grw, bsp_data["nv"], gr.shape[1], k
                        )
                        return float(np.sum(grw * rho13 * y24))

                for kappa_eps in range(-int(kappa_max), int(kappa_max) + 1):
                    if kappa_eps == 0:
                        continue
                    if shooting_n_grid:
                        try:
                            r_eps, P_eps_r, Q_eps_r, _ = solve_dirac_continuum_shooting(
                                E_ha,
                                kappa_eps,
                                _bare_potential,
                                r_max=float(bsp_data["t"][-1]),
                                n_grid=int(shooting_n_grid),
                                Z_residual=float(getattr(result, "Z", 0)),
                            )
                            P_eps_g = np.interp(gr.ravel(), r_eps, P_eps_r).reshape(gr.shape)
                            Q_eps_g = np.interp(gr.ravel(), r_eps, Q_eps_r).reshape(gr.shape)
                        except Exception:
                            continue
                    else:
                        try:
                            P_eps, Q_eps, _ = solve_continuum(
                                kappa_eps, E_ha, int(getattr(result, "Z", 0)), bsp_data,
                                energy_normalized=True,
                            )
                        except Exception:
                            continue
                    direct_ks = set(exchange_k_range(source_key[1], hole_key[1]))
                    direct_ks &= set(exchange_k_range(source2_key[1], kappa_eps))
                    exchange_ks = set(exchange_k_range(source2_key[1], hole_key[1]))
                    exchange_ks &= set(exchange_k_range(source_key[1], kappa_eps))
                    for k in sorted(direct_ks):
                        wa = exchange_lambda(source_key[1], hole_key[1], k)
                        wb = exchange_lambda(source2_key[1], kappa_eps, k)
                        if wa <= 0.0 or wb <= 0.0:
                            continue
                        angular = float(np.sqrt(wa * wb))
                        if shooting_n_grid:
                            Rk = _grid_slater(
                                Pa_g, Qa_g, Pb_g, Qb_g,
                                Pc_g, Qc_g, P_eps_g, Q_eps_g, k,
                            )
                            amplitude += 2.0 * np.pi * angular ** 2 * Rk ** 2
                        else:
                            amplitude += auger_rate_two_electron(
                                P_a, Q_a, P_b, Q_b, P_c, Q_c, P_eps, Q_eps,
                                angular_W=angular, k=k, bsp_data=bsp_data,
                            )
                    for k in sorted(exchange_ks):
                        wa = exchange_lambda(source2_key[1], hole_key[1], k)
                        wb = exchange_lambda(source_key[1], kappa_eps, k)
                        if wa <= 0.0 or wb <= 0.0:
                            continue
                        angular = float(np.sqrt(wa * wb))
                        if shooting_n_grid:
                            Rk = _grid_slater(
                                Pb_g, Qb_g, Pa_g, Qa_g,
                                Pc_g, Qc_g, P_eps_g, Q_eps_g, k,
                            )
                            amplitude += exchange_sign * 2.0 * np.pi * angular ** 2 * Rk ** 2
                        else:
                            amplitude += exchange_sign * auger_rate_two_electron(
                                P_b, Q_b, P_a, Q_a, P_c, Q_c, P_eps, Q_eps,
                                angular_W=angular, k=k, bsp_data=bsp_data,
                            )
                pop = _channel_population_factor(
                    result, state_index, source_key, source2_key, hole_key
                )
                rate = max(0.0, float(amplitude) * pop * AU_RATE_PER_S)
            else:
                rate = 0.0

    return {
        "Gamma_per_s": float(rate),
        "energy_eV": float(resonance_energy_eV or 0.0),
        "state_index": "avg" if state_index is None else int(state_index),
        "hole_label": hole_label,
        "source_label": source_label,
        "source2_label": source2_label,
    }


def auger_rate_csf_average(
    result,
    *,
    hole_label: str,
    source_label: str,
    source2_label: str,
    resonance_energy_eV: float | None = None,
    **kwargs,
):
    """Coarse CSF-averaged Auger rate for a named two-electron channel."""
    return _coarse_auger_rate(
        result,
        hole_label=hole_label,
        source_label=source_label,
        source2_label=source2_label,
        resonance_energy_eV=resonance_energy_eV,
        state_index=None,
        **kwargs,
    )


def auger_rate_level_resolved(
    result,
    state_index: int,
    *,
    hole_label: str,
    source_label: str,
    source2_label: str,
    resonance_energy_eV: float | None = None,
    **kwargs,
):
    """Coarse level-resolved Auger rate for a named two-electron channel."""
    energies = getattr(result, "state_energies", None)
    energy_eV = 0.0
    if resonance_energy_eV is not None:
        energy_eV = float(resonance_energy_eV)
    elif energies is not None and len(energies) > int(state_index):
        from atomx.constants import HARTREE_EV
        energy_eV = float((energies[int(state_index)] - energies[0]) * HARTREE_EV)
    return _coarse_auger_rate(
        result,
        hole_label=hole_label,
        source_label=source_label,
        source2_label=source2_label,
        resonance_energy_eV=energy_eV,
        state_index=int(state_index),
        **kwargs,
    )

__all__ = [
    "AutoionizationState",
    "DRRateCoefficient",
    "auger_rate_csf_average",
    "auger_rate_level_resolved",
    "auger_rate_two_electron",
    "dr_rate_coefficient",
]
