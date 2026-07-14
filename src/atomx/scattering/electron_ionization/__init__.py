"""Electron-impact ionisation (EII) cross sections and rates.

Two methods are provided:

1. **Binary-Encounter-Bethe / Binary-Encounter-Dipole (BEB / BED)** —
   Kim & Rudd 1994.  Single-orbital semi-empirical model that needs
   only the orbital binding energy U, mean kinetic energy ⟨T⟩, and an
   optional dipole-Bethe parameter Q.  Accuracy ~30 % for valence
   shells, faster degradation for inner shells; widely used in
   plasma-physics databases (NIFS, NIST EII).
2. **Distorted-wave (DW) EII** — First-principles DW Born approximation
   (Younger 1981 / Pindzola-Griffin / Gu FAC 2008).  Frozen-core
   V^{N-1} static-Hartree distorted waves, J-coupled antisymmetric
   Slater integrals, and Coulomb-Bethe high-ℓ top-up.  Validated
   against Younger 1981 for He⁺ (≤15 %) and NIST Argon near
   threshold.  Access via ``method="dw"`` in :func:`eii_cross_section`.

Maxwellian-averaged rate coefficients are produced by
:func:`eii_rate_coefficient`; non-Maxwellian (e.g. κ-distribution)
support is on the to-do list.

References
----------
- Y.-K. Kim, M. E. Rudd, *Phys. Rev. A* **50**, 3954 (1994) —
  original BED/BEB formulation.
- Y.-K. Kim, K. K. Irikura, M. A. Ali, *J. Res. NIST* **105**, 285
  (2000) — BEB extension to molecular ionisation.
- M. F. Gu, *Can. J. Phys.* **86**, 675 (2008) — FAC EII algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from atomx.constants import HARTREE_EV, BOHR_RADIUS_M, C_SQ
from atomx.scattering._jit_kernels import eii_dw_inner_sum, trapezoidal_integrate


# π·a₀² in cm² (atomic-physics cross-section unit)
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100.0) ** 2  # ≈ 8.797e-17 cm²
# Rydberg energy in eV
RYD_EV = HARTREE_EV / 2.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EIIResult:
    """Outcome of an electron-impact ionisation calculation.

    Attributes
    ----------
    energies_eV
        Incident electron energies (eV).
    sigma_total_cm2
        Total ionisation cross section ∑_a σ_a(E) in cm².
    sigma_by_orbital
        ``{label: σ_a(E)}`` — per-orbital contributions.
    method
        ``"BEB"`` or ``"BED"`` or ``"DW"``.
    provenance
        Free-form notes (Kim-Rudd parameters used per orbital).
    """

    energies_eV: np.ndarray
    sigma_total_cm2: np.ndarray
    sigma_by_orbital: Dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "BEB"
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Kim-Rudd BEB single-orbital cross section
# ---------------------------------------------------------------------------


def beb_cross_section(
    T_eV: np.ndarray,
    U_eV: float,
    avgT_eV: float,
    N_electrons: float,
    *,
    n_principal: int = 1,
    Q: float = 1.0,
) -> np.ndarray:
    """Binary-Encounter-Bethe (BEB) cross section for a single subshell.

    Delegates to the numba-JIT kernel in :mod:`atomx.scattering.electron_ionization.beb`.
    """
    from atomx.scattering.electron_ionization.beb import beb_cross_section as _beb_jit
    return _beb_jit(T_eV, U_eV, avgT_eV, N_electrons,
                    n_principal=n_principal, Q=Q)


# ---------------------------------------------------------------------------
# Top-level entry: total EII over all subshells
# ---------------------------------------------------------------------------


def eii_cross_section(
    scf_result,
    energies_eV: np.ndarray,
    *,
    method: str = "auto",
    Q_per_orbital: Optional[Dict[str, float]] = None,
    avgT_per_orbital_eV: Optional[Dict[str, float]] = None,
    kappa_max: int = 5,
    n_outgoing_energies: int = 6,
    dw_energy_cutoff_per_I: float = 5.0,
    continuum_model: Any = None,
) -> EIIResult:
    """Total electron-impact ionisation cross section.

    Parameters
    ----------
    scf_result
        Converged ``SCFResult``.
    energies_eV
        Incident electron energy grid in eV.
    method
        - ``"BEB"`` (default — Kim-Rudd 1994 semi-empirical, ~30 %
          accurate for valence ionisation; matches NIST Argon σ(100 eV)
          to 25 %).
        - ``"BED"`` — per-orbital Q from ``Q_per_orbital``.
        - ``"DW"`` — first-principles distorted-wave Born with
          delta-energy-normalised continuum waves, partial-wave-summed
          J-coupled Slater integrals.  **Status: validated near
          threshold for hydrogenic ions** (σ(He⁺, 100 eV) = 2.5e-18 cm²
          vs Younger 1981 ~3e-18, 13 % gap from basis truncation).
          **Underestimates badly for many-electron neutrals** because
          the projectile/outgoing waves use the bare-nuclear potential
          rather than the SCF mean field; **and at high E** because the
          B-spline pseudo-continuum DOS estimate degrades.  Use BEB for
          production routine use until DW is upgraded with a Bethe-
          Born high-E patch and SCF-distorted projectile waves.
    Q_per_orbital, avgT_per_orbital_eV
        Only used by BEB/BED; see below.
    kappa_max
        DW-only: largest |κ| in the partial-wave expansion of the
        projectile and outgoing electrons.  Default 4 covers s..g.
    n_outgoing_energies
        DW-only: number of E_1 quadrature nodes for the integral
        over the outgoing-electron energy share.  Default 8.

    Returns
    -------
    EIIResult
    """
    energies_arr = np.asarray(energies_eV, dtype=float)
    method_u = method.upper()
    if method_u == "DW":
        return _eii_dw_total(
            scf_result, energies_arr,
            kappa_max=kappa_max,
            n_outgoing_energies=n_outgoing_energies,
            continuum_model=continuum_model,
        )
    if method_u == "AUTO":
        # Hybrid: DW for E < cutoff·I_outermost (where partial-wave
        # convergence is feasible), BEB above (Kim-Rudd extends across
        # the full energy range with O(30 %) accuracy).  This matches
        # what FAC effectively does with its DW + Bethe-Born blend.
        # OUTERMOST = least-bound = ε closest to zero from below.
        # max() of negative numbers picks the largest (closest to 0).
        eps_outer = max(
            (e for e in scf_result.orbital_energies.values() if e < 0.0),
            default=-1.0,
        )
        I_outer_eV = -eps_outer * HARTREE_EV
        cutoff_eV = dw_energy_cutoff_per_I * I_outer_eV
        dw_mask = energies_arr <= cutoff_eV
        sigma_total = np.zeros_like(energies_arr)
        sigma_by_orbital: Dict[str, np.ndarray] = {}
        prov: Dict[str, Any] = {"hybrid": True, "DW_E_max_eV": float(cutoff_eV)}
        if np.any(dw_mask):
            r_dw = _eii_dw_total(
                scf_result, energies_arr[dw_mask],
                kappa_max=kappa_max,
                n_outgoing_energies=n_outgoing_energies,
                continuum_model=continuum_model,
            )
            sigma_total[dw_mask] = r_dw.sigma_total_cm2
            for k, v in r_dw.sigma_by_orbital.items():
                sigma_by_orbital.setdefault(k,
                    np.zeros_like(energies_arr))
                sigma_by_orbital[k][dw_mask] = v
            prov["DW_provenance"] = r_dw.provenance
        if not np.all(dw_mask):
            r_beb = eii_cross_section(
                scf_result, energies_arr[~dw_mask], method="BEB",
                Q_per_orbital=Q_per_orbital,
                avgT_per_orbital_eV=avgT_per_orbital_eV,
            )
            sigma_total[~dw_mask] = r_beb.sigma_total_cm2
            for k, v in r_beb.sigma_by_orbital.items():
                sigma_by_orbital.setdefault(k,
                    np.zeros_like(energies_arr))
                sigma_by_orbital[k][~dw_mask] = v
            prov["BEB_provenance"] = r_beb.provenance
        return EIIResult(
            energies_eV=energies_arr,
            sigma_total_cm2=sigma_total,
            sigma_by_orbital=sigma_by_orbital,
            method="auto (DW+BEB)",
            provenance=prov,
        )
    if method_u not in ("BEB", "BED"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'auto', 'DW', "
            f"'BEB', or 'BED'."
        )

    energies = np.asarray(energies_eV, dtype=float)
    sigma_total = np.zeros_like(energies)
    by_orbital: Dict[str, np.ndarray] = {}
    prov_per_orb: Dict[str, Dict[str, float]] = {}

    for sh in scf_result.subshells:
        eps = scf_result.orbital_energies.get(sh.label)
        if eps is None or eps >= 0.0:
            continue
        U_eV = -eps * HARTREE_EV
        if avgT_per_orbital_eV and sh.label in avgT_per_orbital_eV:
            avgT_eV = avgT_per_orbital_eV[sh.label]
        else:
            # Virial approx: ⟨T⟩ ≈ |ε_a|.  Exact for pure hydrogenic;
            # within ~10 % for many atomic shells.  Override via
            # ``avgT_per_orbital_eV`` for accuracy.
            avgT_eV = U_eV
        Q = 1.0
        if method.upper() == "BED":
            if Q_per_orbital is None or sh.label not in Q_per_orbital:
                raise ValueError(
                    f"BED method requires Q_per_orbital for {sh.label!r}."
                )
            Q = Q_per_orbital[sh.label]

        sigma_a = beb_cross_section(
            energies, U_eV=U_eV, avgT_eV=avgT_eV,
            N_electrons=float(sh.occupation),
            n_principal=sh.n, Q=Q,
        )
        by_orbital[sh.label] = sigma_a
        sigma_total += sigma_a
        prov_per_orb[sh.label] = dict(U_eV=U_eV, avgT_eV=avgT_eV,
                                       N=float(sh.occupation), Q=Q,
                                       n=sh.n)

    return EIIResult(
        energies_eV=energies,
        sigma_total_cm2=sigma_total,
        sigma_by_orbital=by_orbital,
        method=method.upper(),
        provenance=dict(per_orbital=prov_per_orb,
                         Z=scf_result.Z),
    )


# ---------------------------------------------------------------------------
# Distorted-wave Born EII (first-principles default)
# ---------------------------------------------------------------------------
#
# Younger (1981) / Pindzola-Griffin (1987) / Gu (FAC 2008) formulation:
#
#   σ_DW(E_i ; orbital a) = (4π / k_i²) · (q_a/(2j_a+1))
#                           · Σ_{κ_in, κ_1, κ_2} ∫_0^{T_max/2} dE_1
#                             · Σ_J (2J+1) |amp_J|²
#
# with k_i² = 2 E_i (atomic units), T_max = E_i − I_a, and
#
#   amp_J(κ_in, κ_1, κ_2)
#     = M_J^direct  −  (−1)^{j_1+j_2−J} M_J^exchange
#
#   M_J^direct = Σ_k (−1)^{j_in+j_1+J}
#                · {j_a j_in J ; j_2 j_1 k}
#                · ⟨κ_a‖C^k‖κ_1⟩ ⟨κ_in‖C^k‖κ_2⟩
#                · R^k(a, ψ_in, ψ_1, ψ_2)
#
# (exchange swaps the labels of the two outgoing electrons).
#
# The radial Slater integral R^k is between the bound orbital `a`
# and three continuum waves at energies E_i (incoming), E_1 and
# E_2 = T_max − E_1 (outgoing); evaluated via the existing
# `compute_slater_rk_cell` cell algorithm.


def _orbital_rms_radius(P, Q, bsp_data) -> float:
    """⟨r²⟩^{1/2} of a Dirac orbital on the B-spline grid."""
    import numpy as _np
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    P_g = eval_P(P, bsp_data)
    Q_g = eval_Q(Q, bsp_data)
    r_sq = float(_np.sum(grw * (P_g ** 2 + Q_g ** 2) * gr ** 2))
    norm = float(_np.sum(grw * (P_g ** 2 + Q_g ** 2)))
    return (r_sq / max(norm, 1e-30)) ** 0.5


def _bound_optical_strength(P_a, Q_a, kappa_a, eps_a, scf_result,
                             cache, static_potentials, *,
                             occupation: float = 1.0):
    """Total per-electron optical (dipole) oscillator strength out of
    bound orbital `a` to the continuum.

        f_a = (2/3) · q_a/(2j_a+1) · Σ_{κ_f}  (E_f − E_a) · |⟨a‖r·C¹‖f⟩|²

    The q_a/(2j_a+1) occupation factor reduces the closed-shell
    Wigner-Eckart sum to its open-shell value (matches the same
    correction applied in dipole_polarizability).  For a closed
    subshell q_a = 2j_a+1, factor = 1, recovering the standard
    closed-shell sum-rule total of q_a per orbital — Σ_n f_n = q_a.

    The result is in atomic units (dimensionless).
    """
    import numpy as _np
    from atomx.numerics.bspline.dirac_assembly import (
        build_dirac_matrices, solve_dirac,
    )
    from atomx.properties.transitions.angular import angular_eL, selection_rules
    from atomx.properties.transitions.multipole import radial_eL

    bsp_data = scf_result._bsp_data
    Z = scf_result.Z
    ns_full = bsp_data["ns"]
    nsp = bsp_data.get("nsp", ns_full)
    nsq = bsp_data.get("nsq", ns_full)

    # Allowed E1-final κ values: ℓ_f = ℓ_a ± 1, j_f = ℓ_f ± 1/2.
    l_a = kappa_a if kappa_a > 0 else -kappa_a - 1
    candidate_kappas: list[int] = []
    for l_f in (l_a - 1, l_a + 1):
        if l_f < 0:
            continue
        candidate_kappas.append(-(l_f + 1))    # j_f = l_f + 1/2
        if l_f > 0:
            candidate_kappas.append(l_f)        # j_f = l_f − 1/2
    if not candidate_kappas:
        return 0.0

    # Static-Hartree potential for V^{N-1} continuum (use the one
    # cached for the orbital being ionised).
    label = next(
        (s.label for s in scf_result.subshells if s.kappa == kappa_a),
        None,
    )
    if label not in static_potentials:
        static_potentials[label] = _build_static_hartree(
            scf_result, label, cache,
        )
    V_PP, V_QQ = static_potentials[label]

    f_a_total = 0.0
    for kappa_f in candidate_kappas:
        if not selection_rules(kappa_a, kappa_f, 1, "E"):
            continue
        ang = angular_eL(kappa_a, kappa_f, 1)
        if abs(ang) < 1e-15:
            continue
        H, S = build_dirac_matrices(ns_full, kappa_f, Z, bsp_data)
        n_pp = V_PP.shape[0]; n_qq = V_QQ.shape[0]
        H[:n_pp, :n_pp] += V_PP
        H[ns_full:ns_full + n_qq, ns_full:ns_full + n_qq] += V_QQ
        res_e = solve_dirac(H, S, ns_full, kappa_f, nsp=nsp, nsq=nsq)
        evals = res_e.eigenvalues

        for i, eps_f in enumerate(evals):
            # Sum over all dipole-allowed transitions from `a` to
            # both bound and continuum eigenstates of κ_f.  Restrict
            # to physical (above Dirac sea) and ABOVE eps_a so the
            # transition energy ω = ε_f − ε_a is positive.
            if eps_f <= eps_a + 1e-9:
                continue
            if eps_f < -1.5 * C_SQ:  # below Dirac sea
                continue

            vec = res_e.coefficients[:, i]
            P_f = _np.zeros(ns_full); Q_f = _np.zeros(ns_full)
            P_f[:nsp] = vec[:nsp]
            Q_f[:nsq] = vec[ns_full:ns_full + nsq]

            rad = radial_eL(P_a, Q_a, P_f, Q_f, bsp_data, L=1)
            me_sq = (ang * rad) ** 2
            f_a_total += (2.0 / 3.0) * (eps_f - eps_a) * me_sq

    # Occupation reweighting: closed-shell Wigner-Eckart sum →
    # q_a/(2j_a+1) for fractional occupancy.
    tj_a = 2 * abs(kappa_a) - 1
    return float(f_a_total) * occupation / float(tj_a + 1)


def _build_static_hartree(scf_result, exclude_label, cache):
    """Build the static (direct) Hartree potential from converged DHF
    orbitals: V_H = Σ_b q_b' V_d^0(b), where q_b' = q_b − δ(b == exclude).

    Returns (V_PP, V_QQ) matrices to add to the bare-nuclear Dirac
    Hamiltonian for projectile/outgoing distorted waves.

    For closed-shell N-electron atom + ionisation of orbital `a`:
    setting exclude_label = a.label gives the V^{N-1} potential
    seen by the outgoing electron and (in this implementation) by
    the projectile too — the standard "frozen-core" DW prescription
    in FAC and Pindzola-Griffin.
    """
    import numpy as _np
    from atomx.numerics.bspline.convol import compute_direct_potential_cell

    bsp_data = scf_result._bsp_data
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    V_PP = _np.zeros((nsp, nsp))
    V_QQ = _np.zeros((nsq, nsq))

    for sh in scf_result.subshells:
        q = float(sh.occupation)
        if exclude_label is not None and sh.label == exclude_label:
            q -= 1.0
        if q <= 0.0:
            continue
        orb_list = scf_result.orbitals_raw.get(sh.kappa, [])
        kappa_subs = [s for s in scf_result.subshells if s.kappa == sh.kappa]
        idx = kappa_subs.index(sh)
        if idx >= len(orb_list):
            continue
        Pb, Qb, _ = orb_list[idx]
        V_d_PP, V_d_QQ = compute_direct_potential_cell(
            Pb, Qb, bsp_data, 0, cache=cache
        )
        n_pp = min(V_d_PP.shape[0], nsp)
        n_qq = min(V_d_QQ.shape[0], nsq)
        V_PP[:n_pp, :n_pp] += q * V_d_PP[:n_pp, :n_pp]
        V_QQ[:n_qq, :n_qq] += q * V_d_QQ[:n_qq, :n_qq]
    return V_PP, V_QQ


def _eii_dw_total(
    scf_result,
    energies_eV: np.ndarray,
    *,
    kappa_max: int,
    n_outgoing_energies: int,
    continuum_model: Any = None,
) -> "EIIResult":
    """Relativistic distorted-wave EII on the FAC-exact CE machinery.

    Ionization is treated as excitation into the continuum: for each
    ejected-electron energy E1 the two-electron T-matrix

        T_J = <(eps1 kap1, eps2 kap2) J | 1/r12 | (a, eps_in kap_in) J>_A

    is assembled exactly as in
    :func:`atomx.scattering.distorted_wave.solver.dw_excitation_cross_section`
    (exact m-summation angular coefficients, direct minus exchange,
    delta(E)-normalized shooting continuum waves in the FAC-convention
    (N-1)/N mean potential, fine-grid Slater integrals), and

        sigma(E_i) = pi / p_i^2 * q_a / (2 j_a + 1)
                     * Int_0^{Tmax/2} dE1  2 pi^2
                       Sum_{J, kap_in, kap_1, kap_2} (2J+1) |T_J|^2

    The E1 integral runs to Tmax/2 (identical outgoing electrons; the
    antisymmetrized |D - E|^2 covers the exchange).  A geometric high-l
    tail on the incident partial-wave profile tops up l > kappa_max.
    """
    from atomx.constants import HARTREE_EV as _HEV
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.scattering.distorted_wave.solver import (
        _allowed_k,
        _cc_direct,
        _cc_exchange,
        _continuum_on_grid,
        _distorting_potential_fn,
        _geometric_tail,
        _l_of,
        _make_fine_grid,
        _p_rel,
        _shooting_points,
        _tj_of,
        _yk_fine,
    )

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("SCFResult is missing _bsp_data — re-run solve().")
    Z = scf_result.Z

    energies = np.asarray(energies_eV, dtype=float)
    sigma_total = np.zeros_like(energies)
    by_orbital: Dict[str, np.ndarray] = {}
    prov_per_orb: Dict[str, Dict[str, float]] = {}

    # FAC-convention mean potential of the target; continuum waves for the
    # projectile and both outgoing electrons are solved in it.
    V_func = _distorting_potential_fn(scf_result, bsp_data, Z, screening="fac")
    N_bound = sum(float(sh.occupation) for sh in scf_result.subshells)
    Z_res = max(float(Z) - N_bound + 1.0, 1.0)
    r_max_box = float(bsp_data["t"][-1])

    E_max_au = float(np.max(energies)) / _HEV
    p_max = _p_rel(max(E_max_au, 1.0))
    r_fine = _make_fine_grid(bsp_data, p_max)
    dr_fine = np.gradient(r_fine)
    n_shoot = _shooting_points(r_max_box, p_max)

    gr_flat = bsp_data["gr"].ravel()
    _o = np.argsort(gr_flat)
    gr_sorted = gr_flat[_o]

    def _bound_to_fine(coef_P, coef_Q):
        Pg = eval_P(coef_P, bsp_data).ravel()[_o]
        Qg = eval_Q(coef_Q, bsp_data).ravel()[_o]
        return (np.interp(r_fine, gr_sorted, Pg),
                np.interp(r_fine, gr_sorted, Qg))

    cont_cache: Dict[tuple, Any] = {}

    def _wave(kappa: int, energy_au: float):
        key = (int(kappa), round(float(energy_au), 10))
        if key not in cont_cache:
            try:
                cont_cache[key] = _continuum_on_grid(
                    int(kappa), float(energy_au), V_func, Z_res,
                    r_fine, r_max_box, n_shoot)
            except Exception:
                cont_cache[key] = None
        return cont_cache[key]

    kappa_list = [k for k in range(-kappa_max - 1, kappa_max + 1)
                  if k != 0 and _l_of(k) <= kappa_max]

    for sh in scf_result.subshells:
        eps_a = scf_result.orbital_energies.get(sh.label)
        if eps_a is None or eps_a >= 0.0:
            continue
        I_a_au = -eps_a
        kappa_a = sh.kappa
        tj_a = _tj_of(kappa_a)
        l_a = _l_of(kappa_a)
        q_a = float(sh.occupation)
        orb_list = scf_result.orbitals_raw.get(kappa_a, [])
        kappa_subs = [s for s in scf_result.subshells if s.kappa == kappa_a]
        idx_in_kappa = kappa_subs.index(sh)
        if idx_in_kappa >= len(orb_list):
            continue
        P_a_c, Q_a_c, _q = orb_list[idx_in_kappa]
        P_a, Q_a = _bound_to_fine(P_a_c, Q_a_c)

        sigma_a = np.zeros_like(energies)

        for E_i_idx, E_i_eV in enumerate(energies):
            E_i_au = E_i_eV / _HEV
            T_max = E_i_au - I_a_au
            if T_max <= 0.0:
                continue
            p_i_sq = 2.0 * E_i_au + (E_i_au * E_i_au) / C_SQ

            E1_grid = np.linspace(0.05 * T_max, 0.49 * T_max,
                                  n_outgoing_energies)
            integrand = np.zeros_like(E1_grid)

            for j_E1, E_1_au in enumerate(E1_grid):
                E_2_au = T_max - E_1_au
                prof_l = np.zeros(kappa_max + 1)
                # Y^k caches shared across the third kappa loop.
                yd_cache: Dict[tuple, np.ndarray] = {}
                ye_cache: Dict[tuple, np.ndarray] = {}

                for kappa_in in kappa_list:
                    w_in = _wave(kappa_in, E_i_au)
                    if w_in is None:
                        continue
                    tj_in = _tj_of(kappa_in)
                    l_in = _l_of(kappa_in)
                    for kappa_1 in kappa_list:
                        w_1 = _wave(kappa_1, E_1_au)
                        if w_1 is None:
                            continue
                        tj_1 = _tj_of(kappa_1)
                        l_1 = _l_of(kappa_1)
                        rho_a1 = P_a * w_1[0] + Q_a * w_1[1]
                        for kappa_2 in kappa_list:
                            tj_2 = _tj_of(kappa_2)
                            l_2 = _l_of(kappa_2)
                            if (l_a + l_in + l_1 + l_2) % 2 != 0:
                                continue
                            tJ_lo = max(abs(tj_a - tj_in), abs(tj_1 - tj_2))
                            tJ_hi = min(tj_a + tj_in, tj_1 + tj_2)
                            if tJ_lo > tJ_hi:
                                continue
                            ks = [k for k in _allowed_k(kappa_a, kappa_1)
                                  if k in _allowed_k(kappa_in, kappa_2)]
                            ts = [t for t in _allowed_k(kappa_a, kappa_2)
                                  if t in _allowed_k(kappa_in, kappa_1)]
                            if not ks and not ts:
                                continue
                            w_2 = _wave(kappa_2, E_2_au)
                            if w_2 is None:
                                continue
                            # Direct radial R^k(a,in;1,2):
                            #   rho13 = a*1, rho24 = in*2 (Y cached).
                            Rd = {}
                            for k in ks:
                                ck = (kappa_in, kappa_2, k)
                                y = yd_cache.get(ck)
                                if y is None:
                                    rho_in2 = (w_in[0] * w_2[0]
                                               + w_in[1] * w_2[1])
                                    y = _yk_fine(rho_in2, r_fine, dr_fine, k)
                                    yd_cache[ck] = y
                                Rd[k] = float(np.sum(rho_a1 * y * dr_fine))
                            # Exchange radial R^t(a,in;2,1):
                            #   rho13 = a*2, rho24 = in*1 (Y cached).
                            Re = {}
                            if ts:
                                rho_a2 = P_a * w_2[0] + Q_a * w_2[1]
                                for t in ts:
                                    ct = (kappa_in, kappa_1, t)
                                    y = ye_cache.get(ct)
                                    if y is None:
                                        rho_in1 = (w_in[0] * w_1[0]
                                                   + w_in[1] * w_1[1])
                                        y = _yk_fine(rho_in1, r_fine,
                                                     dr_fine, t)
                                        ye_cache[ct] = y
                                    Re[t] = float(np.sum(rho_a2 * y
                                                         * dr_fine))
                            acc = 0.0
                            for tJ in range(tJ_lo, tJ_hi + 1, 2):
                                D = 0.0
                                for k in ks:
                                    ang = _cc_direct(kappa_1, kappa_2,
                                                     kappa_a, kappa_in,
                                                     tJ, k)
                                    if ang != 0.0:
                                        D += Rd[k] * ang
                                E_x = 0.0
                                for t in ts:
                                    ang = _cc_exchange(kappa_1, kappa_2,
                                                       kappa_a, kappa_in,
                                                       tJ, t)
                                    if ang != 0.0:
                                        E_x += Re[t] * ang
                                T = D - E_x
                                if T != 0.0:
                                    acc += (tJ + 1) * T * T
                            prof_l[l_in] += acc

                om = float(np.sum(prof_l))
                # Geometric high-l tail on the incident partial wave.
                om += _geometric_tail(prof_l, kappa_max)
                integrand[j_E1] = 2.0 * np.pi ** 2 * om

            integral = float(np.trapezoid(integrand, E1_grid))
            sigma_a_au = (np.pi / p_i_sq) * (q_a / (tj_a + 1.0)) * integral
            sigma_a[E_i_idx] = sigma_a_au * BOHR_RADIUS_M ** 2 * 1e4

        by_orbital[sh.label] = sigma_a
        sigma_total += sigma_a
        prov_per_orb[sh.label] = {
            "I_eV": I_a_au * _HEV, "occupation": q_a,
            "kappa_max": kappa_max,
            "n_E1_quadrature": n_outgoing_energies,
            "continuum": "shooting delta(E), FAC (N-1)/N mean potential",
        }

    include_local_exchange = False
    if continuum_model is not None:
        potential = getattr(continuum_model, "potential", None)
        include_local_exchange = bool(getattr(potential, "include_exchange", False))

    return EIIResult(
        energies_eV=energies,
        sigma_total_cm2=sigma_total,
        sigma_by_orbital=by_orbital,
        method="DW",
        provenance={
            "per_orbital": prov_per_orb,
            "Z": scf_result.Z,
            "continuum_model": continuum_model,
            "include_local_exchange": include_local_exchange,
        },
    )



# ---------------------------------------------------------------------------
# Maxwellian-averaged rate coefficient
# ---------------------------------------------------------------------------


def eii_rate_coefficient(
    eii_result: EIIResult,
    temperature_eV: np.ndarray,
) -> np.ndarray:
    """Maxwellian rate coefficient ⟨σv⟩ from a precomputed EII cross section.

    α(T) = ∫₀^∞ σ(E) v(E) f_M(E; T) dE
         = (8/(π m_e (kT)³))^{1/2} ∫₀^∞ σ(E) E e^{-E/kT} dE

    Parameters
    ----------
    eii_result
        Output of :func:`eii_cross_section` on a sufficiently dense
        energy grid (extending to ≥ 10× T_max).
    temperature_eV
        Maxwellian temperature(s) in eV.

    Returns
    -------
    np.ndarray
        Rate coefficient in cm³/s on the same shape as
        ``temperature_eV``.
    """
    # Use the existing Maxwellian-rate helper from atomx.plasma.rates
    # to avoid duplicating the integrator.
    from atomx.plasma.rates import maxwellian_rate
    T_arr = np.atleast_1d(np.asarray(temperature_eV, dtype=float))
    out = np.array([
        maxwellian_rate(eii_result.sigma_total_cm2,
                         eii_result.energies_eV, float(T))
        for T in T_arr
    ])
    return out if out.shape != (1,) else float(out[0])


__all__ = [
    "EIIResult",
    "eii_cross_section",
    "eii_rate_coefficient",
    "beb_cross_section",
]
