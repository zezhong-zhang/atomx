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
from atomx.scattering.autoionization.molecular import (
    ActiveMolecularOrbital,
    AugerStick,
    AugerTransitionAmplitude,
    ContinuumChannel,
    CoreHoleState,
    DiscreteMOContinuumChannel,
    LocalizedValenceActiveSpace,
    SpectrumResult,
    TwoHoleCSF,
    TwoHoleFinalState,
    compute_molecular_auger,
    enumerate_two_hole_csfs,
    local_valence_active_space,
    molecular_auger_final_states,
    molecular_auger_initial_state,
    molecular_auger_spectrum,
    molecular_auger_transition_amplitude,
)

_DR_EXPORTS = {
    "DRRateCoefficient",
    "dr_calculate",
    "dr_rate_coefficient",
    "dr_strengths",
}

_MANIFOLD_EXPORTS = {
    "ManifoldLevel",
    "ResonanceManifold",
    "auger_levels",
    "manifold_from_configs",
}

# Set to a list to capture the per-(l4, K) Slater integrals the LS chain
# consumes (channel diagnostics, e.g. against XATOM's printed R_K).
_LS_TRACE = None


def __getattr__(name):
    if name in _DR_EXPORTS:
        from atomx.scattering.dielectronic_recombination import solver

        value = getattr(solver, name)
        globals()[name] = value
        return value
    if name in _MANIFOLD_EXPORTS:
        from atomx.scattering.autoionization import mscheme_ci

        value = getattr(mscheme_ci, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _orbital_key_from_label(result, label: str):
    for n, kappa in getattr(result, "orbitals_dict", {}):
        orbital_l = kappa if kappa > 0 else -kappa - 1
        j2 = 2 * abs(kappa) - 1
        l_char = "spdfghiklmno"[orbital_l] if orbital_l < 12 else f"l{orbital_l}"
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
    """Per-resonance source-pair / hole-vacancy population factor.

    For ``state_index is None`` (the ``auger_rate_csf_average`` entry point)
    the caller is asking for the *per-resonance* Auger rate of one source
    pair decaying into one hole.  The average-of-configuration angular
    factor ``exchange_lambda`` already folds in the per-subshell averaging,
    so the population prefactor is just 1.0.  Reading the bound-orbital
    occupations off a solved-atom result is the wrong thing to do here:
    captured-electron orbitals (3s, 3p, 3d, …) are essentially unoccupied
    in the ground-state solve, which collapsed the prefactor to ~1e-9 and
    drove the per-pair rate down to ~1e5 s⁻¹ (the HYP-530 magnitude bug).

    For an explicit ``state_index`` (the level-resolved entry point) we
    still honour the CSF + mixing-coefficient occupation that
    ``_state_occupation`` extracts, so callers that genuinely want the
    configuration-average rate over a closed-shell CSF can keep using
    ``auger_rate_level_resolved`` with a populated result.
    """
    if state_index is None:
        return 1.0
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
    kappa_max: int = 3,
    exchange_sign: float = 1.0,
    shooting_n_grid: int | None = None,
    population_factor: float | None = None,
    J_res: float | None = None,
    J_final: float = 0.0,
    slater_method: str = "poisson",
    angular_mode: str = "ls",
    **_kwargs,
):
    """Coarse Auger rate with a selectable angular treatment.

    ``angular_mode`` selects the channel angular chain (both share the
    same radials, continuum and pair normalisation, so the difference
    isolates the angular/relativistic treatment):

    * ``"ls"`` (default) — the exact non-relativistic LS-coupled chain
      of XATOM (decay.f90 ``Auger_group_rate_extended``; coherent K-sum
      per (l4, L), exchange interference, (2L+1)/(2l3+1) weights).
    * ``"jj"`` — the FAC-faithful relativistic jj chain (faclib
      SlaterTotal ``sd``/``se`` transcription, commit 3cd7eae wiring):
      sd = R^k reduced-C products without 6j, se = full t-sum with
      6j{j0 j2 k; j1 j3 t}, phase (-1)^(t+k), per-k |W R|^2 terms.
      CAVEAT: the per-k accumulation is incoherent — on Ne KLL it
      over-counts the coherent LS chain by 2.7-5.6x per channel, so
      JJ/LS from this mode is NOT a relativistic measure; a coherent
      J-summed jj channel chain is still to be wired for that.

    XATOM is a Schrodinger (non-relativistic) code, FAC a Dirac
    (relativistic) one; keeping both chains in atomx against the same
    radials quantifies the relativistic contribution to the rate.
    ``J_res is not None`` overrides both with the Cowan-Grant
    J-resolved coefficient.
    """
    import numpy as np

    from atomx.angular.coefficients import _twoj_from_kappa as _twoj
    from atomx.angular.coefficients import (
        cowan_grant_auger_B,
        exchange_k_range,
        reduced_cl_j2,
        reduced_matrix_ls,
    )
    from atomx.angular.wigner import wigner_6j
    from atomx.constants import AU_RATE_PER_S, HARTREE_EV
    from atomx.numerics.bspline.continuum import solve_continuum

    source_key = _orbital_key_from_label(result, source_label)
    source2_key = _orbital_key_from_label(result, source2_label)
    hole_key = _orbital_key_from_label(result, hole_label)
    if source_key is None or source2_key is None or hole_key is None:
        rate = 0.0
    else:
        orbitals = getattr(result, "orbitals_dict", {})
        bsp_data = getattr(result, "bsp_data", None) or getattr(
            result, "_bsp_data", None
        )
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
                    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
                    from atomx.numerics.shooting.dirac_continuum import (
                        solve_dirac_continuum_shooting,
                    )
                    from atomx.numerics.yk import compute_yk

                    gr = bsp_data["gr"]
                    grw = bsp_data["grw"]
                    Z = float(getattr(result, "Z", 0))

                    # CONTINUUM POTENTIAL (2026-10-02 fix): the bare
                    # -Z/r field over-binds the Auger electron (measured
                    # R^0 inflated 3.6x on Ne KLL).  Physical field: the
                    # FINAL state (two source electrons out, one fills
                    # the hole; tail -Z_res/r, Z_res = Z - N + 1):
                    #   V_eps = -Z/r + J[rho_init] - Y0[rho_a+rho_b-rho_c]
                    # mirroring XATOM's final-configuration field without
                    # a second SCF solve.
                    from atomx.numerics.quadrature import yk_on_grid as _yk0

                    def _eval_bound(P, Q):
                        return eval_P(P, bsp_data), eval_Q(Q, bsp_data)

                    Pa_g, Qa_g = _eval_bound(P_a, Q_a)
                    Pb_g, Qb_g = _eval_bound(P_b, Q_b)
                    Pc_g, Qc_g = _eval_bound(P_c, Q_c)

                    _r_flat = gr.ravel()
                    _w_flat = grw.ravel()
                    _rho_full = np.zeros_like(_r_flat)
                    # Full initial-state density.  The former
                    # orbitals_by_subshell probe silently returned {}
                    # on MCDHFResult, dropping the +J[rho_full] term and
                    # over-binding the Auger continuum by J[rho_N];
                    # restoring it moved R^1(2p2p;1s eps_d) from 0.67x
                    # to 0.94x of XATOM (2026-10-02).
                    if getattr(result, "occupations", None):
                        _occ_items = [
                            (_k, float(_ok))
                            for _k, _ok in result.occupations.items()
                        ]
                    else:
                        _occ_items = [
                            ((_sh.n, _sh.kappa), float(_sh.occupation))
                            for _sh in getattr(result, "subshells", [])
                        ]
                    for _k, _ok in _occ_items:
                        _pair = orbitals.get(_k)
                        if _pair is None:
                            continue
                        _rho_full += _ok * (
                            eval_P(_pair[0], bsp_data).ravel() ** 2
                            + eval_Q(_pair[1], bsp_data).ravel() ** 2)
                    _rho_delta = (Pa_g.ravel() ** 2 + Qa_g.ravel() ** 2
                                  + Pb_g.ravel() ** 2 + Qb_g.ravel() ** 2
                                  - Pc_g.ravel() ** 2 - Qc_g.ravel() ** 2)
                    _V0 = (-Z / _r_flat
                           + _yk0(_rho_full, _r_flat, 0)
                           - _yk0(_rho_delta, _r_flat, 0))

                    def _bare_potential(r):
                        return np.interp(r, _r_flat, _V0)



                    # FINE GRID (aliasing fix, 2026-10-02): the B-spline
                    # Gauss grid reaches dr ~ 2.8 a0 in the outer region
                    # while an 800 eV continuum has wavelength ~0.8 a0;
                    # integrating the oscillatory interpolated continuum
                    # on it ALIASES (measured R^0 inflated ~2x on Ne
                    # KLL).  Mirror mscheme_ci/jj_level: make_fine_grid
                    # sized by the continuum momentum + the fused
                    # grid_slater_rk kernel.
                    from atomx.numerics.quadrature import grid_slater_rk as _gsr
                    from atomx.numerics.quadrature import make_fine_grid
                    _p_max = np.sqrt(2.0 * max(E_ha, 1e-3))
                    fg = make_fine_grid(float(bsp_data["t"][-1]),
                                        _p_max, z_inner=max(Z, 1.0))

                    def _to_fg(G):
                        return np.interp(fg, gr.ravel(), G.ravel())

                    Pa_f, Qa_f = _to_fg(Pa_g), _to_fg(Qa_g)
                    Pb_f, Qb_f = _to_fg(Pb_g), _to_fg(Qb_g)
                    Pc_f, Qc_f = _to_fg(Pc_g), _to_fg(Qc_g)

                    def _grid_slater(P1, Q1, P2, Q2, P3, Q3, P4, Q4, k):
                        P4f = np.interp(fg, gr.ravel(), P4.ravel())
                        Q4f = np.interp(fg, gr.ravel(), Q4.ravel())
                        return _gsr(P1, Q1, P2, Q2, P3, Q3, P4f, Q4f,
                                    fg, k)

                _l1 = source_key[1] if source_key[1] > 0 else -source_key[1] - 1
                _l2 = (source2_key[1] if source2_key[1] > 0
                       else -source2_key[1] - 1)
                _l3 = hole_key[1] if hole_key[1] > 0 else -hole_key[1] - 1

                def _continuum(kappa_eps):
                    """Solve the Auger continuum for one kappa.

                    Returns fine-grid arrays in shooting mode (the caller
                    feeds them to ``_grid_slater``) or B-spline expansion
                    coefficients otherwise.
                    """
                    if shooting_n_grid:
                        try:
                            r_eps, P_eps_r, Q_eps_r, _ = (
                                solve_dirac_continuum_shooting(
                                    E_ha,
                                    kappa_eps,
                                    _bare_potential,
                                    r_max=float(bsp_data["t"][-1]),
                                    n_grid=int(shooting_n_grid),
                                    Z_residual=float(getattr(result, "Z", 0)),
                                    match_phase=False,
                                )
                            )
                            return (
                                np.interp(
                                    gr.ravel(), r_eps, P_eps_r
                                ).reshape(gr.shape),
                                np.interp(
                                    gr.ravel(), r_eps, Q_eps_r
                                ).reshape(gr.shape),
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"Auger continuum failed: kappa={kappa_eps}, "
                                f"E={E_ha} Ha") from exc
                    try:
                        return solve_continuum(
                            kappa_eps,
                            E_ha,
                            int(getattr(result, "Z", 0)),
                            bsp_data,
                            energy_normalized=True,
                        )[:2]
                    except Exception as exc:
                        raise RuntimeError(
                            f"Auger continuum failed: kappa={kappa_eps}, "
                            f"E={E_ha} Ha") from exc

                def _radials(kappa_eps):
                    """Continuum-solve once and return a cached R(K) accessor.

                    The returned callable maps a multipole rank K to the
                    Slater pair (R12, R21) with the XATOM pairing: R12
                    couples (source-1, hole) x (source-2, continuum),
                    R21 the exchanged (source-2, hole) x (source-1,
                    continuum).
                    """
                    cache = {}
                    if shooting_n_grid:
                        P_eps_g, Q_eps_g = _continuum(kappa_eps)

                        def radial(K):
                            if K not in cache:
                                cache[K] = (
                                    _grid_slater(
                                        Pa_f, Qa_f, Pb_f, Qb_f,
                                        Pc_f, Qc_f, P_eps_g, Q_eps_g, K,
                                    ),
                                    _grid_slater(
                                        Pb_f, Qb_f, Pa_f, Qa_f,
                                        Pc_f, Qc_f, P_eps_g, Q_eps_g, K,
                                    ),
                                )
                            return cache[K]
                    else:
                        P_eps, Q_eps = _continuum(kappa_eps)
                        from atomx.scattering.autoionization.auger import (
                            _slater_integral,
                        )

                        def radial(K):
                            if K not in cache:
                                cache[K] = (
                                    _slater_integral(
                                        P_a, Q_a, P_b, Q_b, P_c, Q_c,
                                        P_eps, Q_eps, k=K,
                                        bsp_data=bsp_data,
                                        method=slater_method,
                                    ),
                                    _slater_integral(
                                        P_b, Q_b, P_a, Q_a, P_c, Q_c,
                                        P_eps, Q_eps, k=K,
                                        bsp_data=bsp_data,
                                        method=slater_method,
                                    ),
                                )
                            return cache[K]

                    return radial

                if J_res is None and str(angular_mode).lower() == "jj":
                    # FAC-FAITHFUL RELATIVISTIC jj chain (faclib
                    # SlaterTotal sd/se transcription, the 3cd7eae
                    # wiring): sd = R^k(ab;cd) ReducedCL products with
                    # NO 6j; se = full t-sum (t parity-stepped over the
                    # triad range) with 6j{j0 j2 k; j1 j3 t}, phase
                    # (-1)^(t+k), swapped radial R^t(a b; d c).  Terms
                    # accumulate per k as 2pi (W R)^2 (incoherent across
                    # k).  Kept as the relativistic counterpart of the
                    # LS chain below so atomx can measure the Dirac vs
                    # Schrodinger angular contribution on identical
                    # radials and normalisation.  A faithful
                    # AutoionizeRateUTA0 transcription (J-summed
                    # m-average, recombination.c:1673) is being
                    # calibrated against the Ne KLL anchors — see the
                    # atomx trace 2026-10-02; until it validates it
                    # stays out of the tree.
                    for kappa_eps in range(-int(kappa_max), int(kappa_max) + 1):
                        if kappa_eps == 0:
                            continue
                        if shooting_n_grid:
                            P_eps_g, Q_eps_g = _continuum(kappa_eps)
                        else:
                            P_eps, Q_eps = _continuum(kappa_eps)
                        direct_ks = set(
                            exchange_k_range(source_key[1], hole_key[1]))
                        direct_ks &= set(
                            exchange_k_range(source2_key[1], kappa_eps))
                        for k in sorted(direct_ks):
                            ca = reduced_cl_j2(
                                _twoj(source_key[1]), 2 * k,
                                _twoj(hole_key[1]))
                            cb = reduced_cl_j2(
                                _twoj(source2_key[1]), 2 * k,
                                _twoj(kappa_eps))
                            if ca == 0.0 or cb == 0.0:
                                continue
                            angular = float(ca * cb)
                            if shooting_n_grid:
                                Rk = _grid_slater(
                                    Pa_f, Qa_f, Pb_f, Qb_f,
                                    Pc_f, Qc_f, P_eps_g, Q_eps_g, k,
                                )
                                amplitude += 2.0 * np.pi * angular**2 * Rk**2
                            else:
                                amplitude += auger_rate_two_electron(
                                    P_a, Q_a, P_b, Q_b, P_c, Q_c,
                                    P_eps, Q_eps,
                                    angular_W=angular,
                                    k=k,
                                    bsp_data=bsp_data,
                                    method=slater_method,
                                )
                        # se: t-sum over the exchange triad range
                        t_min = max(abs(_twoj(source_key[1])
                                       - _twoj(kappa_eps)),
                                     abs(_twoj(source2_key[1])
                                         - _twoj(hole_key[1]))) // 2
                        t_max = min(_twoj(source_key[1]) + _twoj(kappa_eps),
                                    _twoj(source2_key[1])
                                    + _twoj(hole_key[1])) // 2
                        for t2 in range(2 * t_min, 2 * t_max + 1, 2):
                            k = t2 // 2
                            ca_e = reduced_cl_j2(
                                _twoj(source_key[1]), t2,
                                _twoj(kappa_eps))
                            cb_e = reduced_cl_j2(
                                _twoj(source2_key[1]), t2,
                                _twoj(hole_key[1]))
                            if ca_e == 0.0 or cb_e == 0.0:
                                continue
                            sixj_e = wigner_6j(
                                _twoj(source_key[1]),
                                _twoj(hole_key[1]),
                                2 * k,
                                _twoj(source2_key[1]),
                                _twoj(kappa_eps), t2)
                            if abs(sixj_e) < 1e-15:
                                continue
                            phase_e = -1.0 if ((t2 // 2 + k) % 2) else 1.0
                            angular = float(
                                phase_e * (2.0 * k + 1.0) * sixj_e
                                * ca_e * cb_e)
                            if shooting_n_grid:
                                Rk = _grid_slater(
                                    Pb_f, Qb_f, Pa_f, Qa_f,
                                    Pc_f, Qc_f, P_eps_g, Q_eps_g, k,
                                )
                                amplitude += (
                                    exchange_sign * 2.0 * np.pi
                                    * angular**2 * Rk**2
                                )
                            else:
                                amplitude += exchange_sign * (
                                    auger_rate_two_electron(
                                        P_b, Q_b, P_a, Q_a, P_c, Q_c,
                                        P_eps, Q_eps,
                                        angular_W=angular,
                                        k=k,
                                        bsp_data=bsp_data,
                                        method=slater_method,
                                    )
                                )
                elif J_res is None:
                    # EXACT XATOM LS chain (decay.f90
                    # Auger_group_rate_extended, transcribed and validated
                    # 1.0000 on all three Ne KLL channels against the
                    # vendored xatom binary, 2026-10-02).  The K-sum is
                    # COHERENT within each (l4, L) before squaring — the
                    # fourth-attempt jj sd/se path squared per k and
                    # missed the 2*R12*R21 interference, which is why Ne
                    # 2s2p/2p2p sat at 4.4x/3.0x while 2s2s (a single-K
                    # channel where the interference is the tau^2
                    # renormalisation itself) closed at 1.0x.
                    for l4 in range(_l1 + _l2 + _l3 + 1):
                        # one j = l4 + 1/2 representative kappa: the LS
                        # chain is spin-free, the Dirac j-splitting of the
                        # continuum enters only through the radial
                        kappa_eps = -(l4 + 1)
                        R_K = _radials(kappa_eps)
                        R12c, R21c = {}, {}
                        for L in range(abs(_l1 - _l2), _l1 + _l2 + 1):
                            M0 = 0.0  # S=0 amplitude, SYS PRA 83 Eq.(9)
                            M1 = 0.0  # S=1 amplitude
                            for K in range(_l1 + _l2 + _l3 + 1):
                                T12 = (
                                    reduced_matrix_ls(K, _l3, _l1)
                                    * reduced_matrix_ls(K, l4, _l2)
                                    * wigner_6j(
                                        2 * _l3, 2 * l4, 2 * L,
                                        2 * _l2, 2 * _l1, 2 * K)
                                )
                                if T12 != 0.0:
                                    if K not in R12c:
                                        R12c[K] = R_K(K)[0]
                                    M0 += R12c[K] * T12
                                    M1 += R12c[K] * T12
                                T21 = (
                                    reduced_matrix_ls(K, _l3, _l2)
                                    * reduced_matrix_ls(K, l4, _l1)
                                    * wigner_6j(
                                        2 * _l3, 2 * l4, 2 * L,
                                        2 * _l1, 2 * _l2, 2 * K)
                                )
                                if T21 != 0.0:
                                    if K not in R21c:
                                        R21c[K] = R_K(K)[1]
                                    v = R21c[K] * T21
                                    if L % 2 == 0:  # (-1)^(L+S)
                                        M0 += v
                                        M1 -= v
                                    else:
                                        M0 -= v
                                        M1 += v
                            if M0 != 0.0 or M1 != 0.0:
                                # (-1)^(L+l1+l4) phase and tau^2 applied
                                # via _tau2 at the rate level below
                                sgn = -1.0 if (L + _l1 + l4) % 2 else 1.0
                                amplitude += (2 * L + 1) * (sgn * M0) ** 2
                                amplitude += 3 * (2 * L + 1) * (sgn * M1) ** 2
                        if _LS_TRACE is not None and (R12c or R21c):
                            _LS_TRACE.append({
                                "channel": (source_label, source2_label,
                                            hole_label),
                                "l4": l4,
                                "R12": dict(R12c),
                                "R21": dict(R21c),
                            })
                    amplitude *= np.pi / (2 * _l3 + 1)
                else:
                    for kappa_eps in range(-int(kappa_max), int(kappa_max) + 1):
                        if kappa_eps == 0:
                            continue
                        if shooting_n_grid:
                            P_eps_g, Q_eps_g = _continuum(kappa_eps)
                        else:
                            P_eps, Q_eps = _continuum(kappa_eps)
                        direct_ks = set(
                            exchange_k_range(source_key[1], hole_key[1]))
                        direct_ks &= set(
                            exchange_k_range(source2_key[1], kappa_eps))
                        for k in sorted(direct_ks):
                            angular = cowan_grant_auger_B(
                                source_key[1],
                                source2_key[1],
                                hole_key[1],
                                kappa_eps,
                                k,
                                float(J_res),
                                float(J_final),
                            )
                            if abs(angular) < 1e-15:
                                continue
                            if shooting_n_grid:
                                Rk = _grid_slater(
                                    Pa_f, Qa_f, Pb_f, Qb_f,
                                    Pc_f, Qc_f, P_eps_g, Q_eps_g, k,
                                )
                                amplitude += 2.0 * np.pi * angular**2 * Rk**2
                            else:
                                amplitude += auger_rate_two_electron(
                                    P_a, Q_a, P_b, Q_b, P_c, Q_c,
                                    P_eps, Q_eps,
                                    angular_W=angular,
                                    k=k,
                                    bsp_data=bsp_data,
                                    method=slater_method,
                                )
                        # exchange t-sum over the triad range (source -> eps,
                        # source2 -> hole), Cowan-Grant B with swapped
                        # sources — unchanged from the pre-LS-chain wiring
                        t_min = max(abs(_twoj(source_key[1])
                                       - _twoj(kappa_eps)),
                                     abs(_twoj(source2_key[1])
                                         - _twoj(hole_key[1]))) // 2
                        t_max = min(_twoj(source_key[1]) + _twoj(kappa_eps),
                                    _twoj(source2_key[1])
                                    + _twoj(hole_key[1])) // 2
                        for t2 in range(2 * t_min, 2 * t_max + 1, 2):
                            k = t2 // 2
                            angular = cowan_grant_auger_B(
                                source2_key[1],
                                source_key[1],
                                hole_key[1],
                                kappa_eps,
                                k,
                                float(J_res),
                                float(J_final),
                            )
                            if abs(angular) < 1e-15:
                                continue
                            if shooting_n_grid:
                                Rk = _grid_slater(
                                    Pb_f, Qb_f, Pa_f, Qa_f,
                                    Pc_f, Qc_f, P_eps_g, Q_eps_g, k,
                                )
                                amplitude += (
                                    exchange_sign * 2.0 * np.pi
                                    * angular**2 * Rk**2
                                )
                            else:
                                amplitude += exchange_sign * (
                                    auger_rate_two_electron(
                                        P_b, Q_b, P_a, Q_a, P_c, Q_c,
                                        P_eps, Q_eps,
                                        angular_W=angular,
                                        k=k,
                                        bsp_data=bsp_data,
                                        method=slater_method,
                                    )
                                )
                if population_factor is None:
                    pop = _channel_population_factor(
                        result, state_index, source_key, source2_key, hole_key
                    )
                else:
                    pop = float(population_factor)
                # XATOM pair normalisation (Bhalla PRA 8, 649 Eq.(3) /
                # SYS PRA 83, 033402 Eq.(8)): N12 counts source pairs per
                # LS SUBSHELL — occupation summed over the j-split
                # kappa pair, degeneracy 4l+2 — and tau^2 = 1/2 applies
                # to same-(n,l) pairs: the exchange-symmetric pair-state
                # renormalisation that the LS-chain (-1)^(L+S) structure
                # requires (XATOM's i1 == i2 rule).  The same-kappa rule
                # used before the LS-chain wiring mishandled j-split
                # pairs of one LS subshell.
                _ls_pair = (source_key[0] == source2_key[0] and _l1 == _l2)

                def _ls_occupation(key):
                    n, kap = key
                    lq = kap if kap > 0 else -kap - 1
                    q = _state_occupation(result, None, (n, -(lq + 1)))
                    if lq > 0:
                        q += _state_occupation(result, None, (n, lq))
                    return q

                if state_index is None:
                    _g1 = 2 * (2 * _l1 + 1)
                    _g2 = 2 * (2 * _l2 + 1)
                    _q1 = _ls_occupation(source_key)
                    _q2 = _ls_occupation(source2_key)
                    if min(_q1, _q2) < 0.5:
                        # PER-RESONANCE caller (DR captured pairs: 3s/3p
                        # unoccupied in the ground solve) — a definite
                        # pair decays, no pair normalisation (HYP-530
                        # semantics); tau^2 still renormalises the
                        # equivalent-pair state
                        _n12 = 1.0
                    elif _ls_pair:
                        _n12 = max(_q1 * max(_q1 - 1.0, 0.0),
                                   0.0) / max(_g1 * (_g1 - 1), 1)
                    else:
                        _n12 = _q1 * _q2 / max(_g1 * _g2, 1)
                else:
                    _n12 = pop  # level-resolved callers keep their factor
                _tau2 = 0.5 if _ls_pair else 1.0
                rate = max(0.0, float(amplitude) * _tau2 * _n12
                           * AU_RATE_PER_S)
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
    shooting_n_grid: int | None = None,
    **kwargs,
):
    """Coarse CSF-averaged Auger rate for a named two-electron channel.

    Returns the per-resonance Auger rate for one source-pair decaying into
    one hole (units s⁻¹).  The average-of-configuration angular factor
    ``exchange_lambda`` already includes the per-subshell averaging, so no
    extra population prefactor is applied.  Defaults to the
    δ(E)-normalised Dirac-continuum shooting solver for the Slater
    integral — the B-spline pseudo-state + ``sqrt(2ρ/π)`` scaling in
    ``solve_continuum`` mis-estimates the inner-profile amplitude by an
    order of magnitude for KLL Auger energies; pass ``shooting_n_grid=0``
    to fall back to the B-spline Slater path if needed.
    """
    if shooting_n_grid is None:
        shooting_n_grid = 400
    return _coarse_auger_rate(
        result,
        hole_label=hole_label,
        source_label=source_label,
        source2_label=source2_label,
        resonance_energy_eV=resonance_energy_eV,
        state_index=None,
        shooting_n_grid=shooting_n_grid,
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
    shooting_n_grid: int | None = None,
    J_res: float | None = None,
    J_final: float = 0.0,
    **kwargs,
):
    """Coarse level-resolved Auger rate for a named two-electron channel.

    Passing ``J_res`` switches the angular factor from the configuration-
    averaged exchange proxy to the Cowan-Grant J-resolved coefficient.
    """
    if shooting_n_grid is None:
        shooting_n_grid = 400
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
        shooting_n_grid=shooting_n_grid,
        J_res=J_res,
        J_final=J_final,
        **kwargs,
    )


def auger_rate_j_resolved(
    result,
    *,
    J_res: float,
    hole_label: str,
    source_label: str,
    source2_label: str,
    resonance_energy_eV: float | None = None,
    state_index: int | None = None,
    J_final: float = 0.0,
    **kwargs,
):
    """J-resolved Auger rate for a named two-electron channel."""
    return _coarse_auger_rate(
        result,
        hole_label=hole_label,
        source_label=source_label,
        source2_label=source2_label,
        resonance_energy_eV=resonance_energy_eV,
        state_index=state_index,
        J_res=J_res,
        J_final=J_final,
        **kwargs,
    )


__all__ = [
    "AutoionizationState",
    "DRRateCoefficient",
    "ManifoldLevel",
    "ResonanceManifold",
    "auger_levels",
    "auger_rate_csf_average",
    "auger_rate_j_resolved",
    "auger_rate_level_resolved",
    "auger_rate_two_electron",
    "dr_calculate",
    "dr_rate_coefficient",
    "dr_strengths",
    "manifold_from_configs",
]
