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
    e1_grid_frac: Any = None,
    _capture: Any = None,
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

    The E1 integral runs over the FAC-equivalent ejected-energy window
    [~1.5e-3 T, T/2] with log-spaced low nodes (FAC ionization.c
    CIRadialQkIntegratedTable, YEG0 = 5e-4): dsigma/dE1 is dominated by
    soft ejection, and a window starting at 0.05 T loses 30-45% of the
    total at high E.  Identical outgoing electrons: the antisymmetrized
    |D - E|^2 covers the exchange (Slater-Condon, no extra recoupling
    phase - the exact m-summation absorbs it).  A geometric high-l tail
    on the incident partial-wave profile tops up l > kappa_max.
    """
    from atomx.constants import HARTREE_EV as _HEV
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.numerics.shooting.dirac_continuum import (
        solve_dirac_continuum_many,
    )
    from atomx.numerics.quadrature import (
        make_fine_grid, p_rel, yk_pair_dot_batch_on_grid)
    from atomx.scattering._jit_kernels import eii_dw_channel_jsum
    from atomx.scattering.distorted_wave.solver import (
        _allowed_k,
        _cc_direct,
        _cc_exchange,
        _continuum_on_grid,
        _distorting_potential_fn,
        _geometric_tail,
        _l_of,
        _shooting_points,
        _tj_of,
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
    p_max = p_rel(max(E_max_au, 1.0))
    r_fine = make_fine_grid(r_max_box, p_max, z_inner=float(Z))
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
    n_kappa = len(kappa_list)
    n_fine = r_fine.shape[0]

    def _stack_waves(energy_au):
        """All partial waves at one energy as (P, Q) row stacks + a
        per-kappa availability mask (missing waves leave zero rows;
        the channel kernel masks them out)."""
        P = np.zeros((n_kappa, n_fine))
        Q = np.zeros((n_kappa, n_fine))
        ok = np.zeros(n_kappa, dtype=np.bool_)
        for i, kap in enumerate(kappa_list):
            w = _wave(kap, energy_au)
            if w is not None:
                P[i] = w[0]
                Q[i] = w[1]
                ok[i] = True
        return P, Q, ok

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
        # support of the ionized orbital: every radial integral of this
        # shell is weighted by a density containing P_a — truncate the
        # Y functions and dots there (|rho| < 1e-16*max contributes ~1e-13
        # relative, far below quadrature accuracy).
        _amax = float(np.max(np.abs(P_a)))
        _sup = np.nonzero(np.abs(P_a) > 1.0e-16 * _amax)[0]
        i_sup = int(_sup[-1]) if len(_sup) else len(P_a) - 1

        sigma_a = np.zeros_like(energies)

        # Energy-independent channel table for this shell: parity/triangle
        # selection, multipole lists, and the m-summed angular coefficients
        # per total J.  Previously recomputed for every ejected-energy node
        # of every incident energy — it only depends on the kappas.
        channels = []
        for kappa_in in kappa_list:
            tj_in = _tj_of(kappa_in)
            l_in = _l_of(kappa_in)
            for kappa_1 in kappa_list:
                tj_1 = _tj_of(kappa_1)
                l_1 = _l_of(kappa_1)
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
                    jrows = []
                    for tJ in range(tJ_lo, tJ_hi + 1, 2):
                        ccd = [(k, _cc_direct(kappa_1, kappa_2,
                                              kappa_a, kappa_in, tJ, k))
                               for k in ks]
                        ccd = [(k, a) for (k, a) in ccd if a != 0.0]
                        cce = [(t, _cc_exchange(kappa_1, kappa_2,
                                                kappa_a, kappa_in, tJ, t))
                               for t in ts]
                        cce = [(t, a) for (t, a) in cce if a != 0.0]
                        if ccd or cce:
                            jrows.append((tJ, ccd, cce))
                    if jrows:
                        channels.append((kappa_in, kappa_1, kappa_2,
                                         l_in, ks, ts, jrows))

        # Flat packs of the channel table for the numba node kernels —
        # energy-independent, built once per shell.  Y^k builds are
        # deduplicated exactly like the old per-node yd/ye caches
        # (direct: (kappa_in, kappa_2, k); exchange: (kappa_in,
        # kappa_1, t)), Slater values like the old per-node Rd/Re dots
        # (one value slot per (kappa_1|kappa_2, Y build)); only
        # multipoles referenced by a nonzero angular coefficient are
        # kept.
        kidx = {kap: i for i, kap in enumerate(kappa_list)}
        d_map: Dict[tuple, int] = {}
        e_map: Dict[tuple, int] = {}
        d_val: Dict[tuple, int] = {}
        e_val: Dict[tuple, int] = {}
        ch_in = []; ch_1 = []; ch_2 = []; ch_lin = []
        ch_jr0 = []; ch_jr1 = []
        jw = []; jd0 = []; jd1 = []; je0 = []; je1 = []
        d_slot = []; d_ang = []
        e_slot = []; e_ang = []
        for (kappa_in, kappa_1, kappa_2, l_in, ks, ts, jrows) in channels:
            ch_in.append(kidx[kappa_in]); ch_1.append(kidx[kappa_1])
            ch_2.append(kidx[kappa_2]); ch_lin.append(l_in)
            ch_jr0.append(len(jw))
            for (tJ, ccd, cce) in jrows:
                jw.append(float(tJ + 1))
                jd0.append(len(d_slot))
                for (k, ang) in ccd:
                    row = d_map.setdefault(
                        (kidx[kappa_in], kidx[kappa_2], k), len(d_map))
                    d_slot.append(d_val.setdefault(
                        (kidx[kappa_1], row), len(d_val)))
                    d_ang.append(ang)
                jd1.append(len(d_slot))
                je0.append(len(e_slot))
                for (t, ang) in cce:
                    row = e_map.setdefault(
                        (kidx[kappa_in], kidx[kappa_1], t), len(e_map))
                    e_slot.append(e_val.setdefault(
                        (kidx[kappa_2], row), len(e_val)))
                    e_ang.append(ang)
                je1.append(len(e_slot))
            ch_jr1.append(len(jw))

        def _i64(v):
            return np.asarray(v, dtype=np.int64)

        def _csr_slots(val_map, n_rows):
            """Group value slots by Y row (CSR) for the fused dot
            kernel; returns (ptr, rho row per slot, old->new slot map)."""
            rows = _i64([key[1] for key in val_map])
            rhos = _i64([key[0] for key in val_map])
            order = np.argsort(rows, kind="stable")
            ptr = np.zeros(n_rows + 1, dtype=np.int64)
            ptr[1:] = np.cumsum(np.bincount(rows, minlength=n_rows))
            remap = np.empty(order.shape[0], dtype=np.int64)
            remap[order] = np.arange(order.shape[0])
            return ptr, rhos[order], remap

        ch_in = _i64(ch_in); ch_1 = _i64(ch_1); ch_2 = _i64(ch_2)
        ch_lin = _i64(ch_lin); ch_jr0 = _i64(ch_jr0); ch_jr1 = _i64(ch_jr1)
        jw = np.asarray(jw, dtype=np.float64)
        jd0 = _i64(jd0); jd1 = _i64(jd1); je0 = _i64(je0); je1 = _i64(je1)
        d_ang = np.asarray(d_ang, dtype=np.float64)
        e_ang = np.asarray(e_ang, dtype=np.float64)
        # Unique Y^k build rows (wave-stack indices + multipole order)
        # and their consumer slots.
        db_a = _i64([key[0] for key in d_map])
        db_b = _i64([key[1] for key in d_map])
        db_k = _i64([key[2] for key in d_map])
        eb_a = _i64([key[0] for key in e_map])
        eb_b = _i64([key[1] for key in e_map])
        eb_k = _i64([key[2] for key in e_map])
        d_ptr, d_cons_rho, d_remap = _csr_slots(d_val, len(d_map))
        e_ptr, e_cons_rho, e_remap = _csr_slots(e_val, len(e_map))
        d_slot = d_remap[_i64(d_slot)] if d_slot else _i64([])
        e_slot = e_remap[_i64(e_slot)] if e_slot else _i64([])

        P_a_s = P_a[:i_sup + 1]
        Q_a_s = Q_a[:i_sup + 1]
        dr_s = dr_fine[:i_sup + 1]

        for E_i_idx, E_i_eV in enumerate(energies):
            E_i_au = E_i_eV / _HEV
            T_max = E_i_au - I_a_au
            if T_max <= 0.0:
                continue
            p_i_sq = 2.0 * E_i_au + (E_i_au * E_i_au) / C_SQ

            if e1_grid_frac is not None:
                E1_grid = np.asarray(e1_grid_frac, dtype=float) * T_max
            else:
                # FAC-equivalent ejected-energy window (ionization.c
                # CIRadialQkIntegratedTable: y in [YEG0*I/T, 0.5],
                # log-spaced): soft ejection dominates dsigma/dE1, so the
                # low-E1 nodes are log-spaced down to ~1e-3 of T; the old
                # 0.05T..0.49T linear window missed ~30-45% of the
                # integral at high E.
                n_lo = max(n_outgoing_energies // 2, 3)
                n_hi = max(n_outgoing_energies - n_lo, 3)
                E1_grid = np.concatenate([
                    np.geomspace(1.5e-3, 0.08, n_lo),
                    np.linspace(0.13, 0.499, n_hi),
                ]) * T_max
            integrand = np.zeros_like(E1_grid)

            # Thread-parallel prefill of every continuum wave used by this
            # incident energy: projectile + all ejected/scattered nodes.
            energies_needed = {round(E_i_au, 10)}
            for E_1_au in E1_grid:
                energies_needed.add(round(float(E_1_au), 10))
                energies_needed.add(round(float(T_max - E_1_au), 10))
            try:
                reqs = [(k, E) for k in kappa_list
                        for E in energies_needed]
                _r_s, raw = solve_dirac_continuum_many(
                    reqs, V_func, r_max=r_max_box, n_grid=n_shoot,
                    Z_residual=Z_res, interp_grid=r_fine)
                for (kap, E), pq in raw.items():
                    key = (int(kap), round(float(E), 10))
                    if key not in cont_cache:
                        cont_cache[key] = pq
            except Exception:
                pass

            WinP, WinQ, ok_in = _stack_waves(E_i_au)

            for j_E1, E_1_au in enumerate(E1_grid):
                E_2_au = T_max - E_1_au
                W1P, W1Q, ok_1 = _stack_waves(E_1_au)
                W2P, W2Q, ok_2 = _stack_waves(E_2_au)
                # Bound-continuum density rows (trapezoid weights folded
                # in), then all direct/exchange Slater values of the
                # node in two prange batches: Y^k builds parallel over
                # ROWS with the consumer dots fused in (each row is the
                # serial fused kernel — a batched 2D-cumsum variant was
                # tried and lost: the reversed-slice cumsums force
                # full-matrix copies in this memory-bound regime).
                RhoA1 = (P_a_s * W1P[:, :i_sup + 1]
                         + Q_a_s * W1Q[:, :i_sup + 1]) * dr_s
                RhoA2 = (P_a_s * W2P[:, :i_sup + 1]
                         + Q_a_s * W2Q[:, :i_sup + 1]) * dr_s
                Rd = yk_pair_dot_batch_on_grid(
                    WinP, WinQ, W2P, W2Q, db_a, db_b, db_k,
                    ok_in[db_a] & ok_2[db_b], RhoA1, d_ptr, d_cons_rho,
                    r_fine, i_cut=i_sup)
                Re = yk_pair_dot_batch_on_grid(
                    WinP, WinQ, W1P, W1Q, eb_a, eb_b, eb_k,
                    ok_in[eb_a] & ok_1[eb_b], RhoA2, e_ptr, e_cons_rho,
                    r_fine, i_cut=i_sup)
                # Slater-Condon for the antisymmetrized outgoing pair:
                # T = D - E exactly; the exact m-summation in
                # _cc_exchange absorbs all recoupling phases (verified:
                # a J-dependent (-1)^{j1+j2-J} variant degrades
                # agreement with FAC's exact-grid values).
                prof_l = eii_dw_channel_jsum(
                    Rd, Re, ok_in, ok_1, ok_2,
                    ch_in, ch_1, ch_2, ch_lin, ch_jr0, ch_jr1,
                    jw, jd0, jd1, je0, je1,
                    d_slot, d_ang, e_slot, e_ang,
                    kappa_max + 1)

                om = float(np.sum(prof_l))
                # Geometric high-l tail on the incident partial wave.
                om += _geometric_tail(prof_l, kappa_max)
                integrand[j_E1] = 2.0 * np.pi ** 2 * om
            if _capture is not None:
                _capture[(sh.label, float(E_i_eV))] = (
                    E1_grid.copy(), integrand.copy(), T_max)
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


def excitation_autoionization_cross_section(
    atom_ground,
    excited_atoms,
    energies_eV,
    *,
    branching=None,
    L_max: int = 6,
    kappa_max: int = 8,
):
    """Excitation-autoionization (EA) contribution to total ionization.

    Indirect ionization: collisional excitation of an inner-shell
    electron into an autoionizing configuration, followed by Auger
    decay,

        sigma_EA(E) = sum_j sigma_exc(g -> j; E) * B_a(j),
        B_a = Gamma_A / (Gamma_A + Gamma_r),

    which dominates total EII for many M-shell ions (e.g. Na-like:
    2p -> 3l excitation above the 2p threshold).  The excitation cross
    sections come from the FAC-verified DW machinery
    (:func:`atomx.scattering.distorted_wave.solver.dw_excitation_cross_section`)
    and the branching ratios from the determinant-CI autoionization
    machinery (Auger widths 1.005-1.023 of clean-room FAC) or user
    input.

    Parameters
    ----------
    atom_ground
        Solved ground-state :class:`~atomx.atom.Atom`.
    excited_atoms
        Sequence of solved atoms, one per autoionizing excited
        configuration (each must lie above the ionization limit).
    energies_eV
        Incident-electron grid.
    branching
        Autoionization branching ratio per excited configuration.
        Either a sequence aligned with ``excited_atoms``, or ``None``
        for B_a = 1 (valid when Gamma_A >> Gamma_r, the usual case for
        low-Z / strong-Auger configurations; for high Z compute B_a
        from :class:`ResonanceManifold` widths).

    Returns
    -------
    (sigma_EA_cm2, per_config)
        Total EA cross section and ``{index: sigma}`` per configuration.
    """
    from atomx.scattering.distorted_wave.solver import (
        dw_excitation_cross_section)

    energies = np.asarray(energies_eV, dtype=float)
    if branching is None:
        branching = [1.0] * len(excited_atoms)
    if len(branching) != len(excited_atoms):
        raise ValueError("branching must align with excited_atoms")
    total = np.zeros_like(energies)
    per = {}
    for i, (exc, B) in enumerate(zip(excited_atoms, branching)):
        B = float(B)
        if not (0.0 <= B <= 1.0):
            raise ValueError(f"branching[{i}] = {B} outside [0, 1]")
        sigma, _om = dw_excitation_cross_section(
            atom_ground, exc, energies, L_max=L_max, kappa_max=kappa_max)
        per[i] = sigma * B
        total = total + per[i]
    return total, per


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
    "excitation_autoionization_cross_section",
    "EIIResult",
    "eii_cross_section",
    "eii_rate_coefficient",
    "beb_cross_section",
]
