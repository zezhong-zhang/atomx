"""Electron-impact ionisation (EII) cross sections and rates.

Two methods are provided:

1. **Binary-Encounter-Bethe / Binary-Encounter-Dipole (BEB / BED)** —
   Kim & Rudd 1994.  Single-orbital semi-empirical model that needs
   only the orbital binding energy U, mean kinetic energy ⟨T⟩, and an
   optional dipole-Bethe parameter Q.  Accuracy ~30 % for valence
   shells, faster degradation for inner shells; widely used in
   plasma-physics databases (NIFS, NIST EII).
2. **Distorted-wave (DW) EII** — TODO.  FAC's primary EII method
   (``ionization.c``).  Plug-in point left in :func:`eii_cross_section`
   for ``method="dw"``.

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

from atomx.constants import HARTREE_EV, BOHR_RADIUS_M


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

    σ_BEB(T) = (S / (t + (u + 1)/n)) · [
                  (Q · ln t / 2) · (1 - 1/t²)
                + (2 - Q) · (1 - 1/t - ln t / (t + 1))
              ]

    with ``t = T/U``, ``u = ⟨T⟩/U``, ``S = 4π a₀² N · (R/U)²``,
    where ``R`` is the Rydberg energy and ``N`` the orbital
    occupation.  Below threshold (T < U) the cross section is zero.

    Parameters
    ----------
    T_eV
        Incident electron energy(s), eV.
    U_eV
        Orbital binding energy = -ε_a, eV.
    avgT_eV
        Mean kinetic energy of the bound orbital, eV.  Use
        |ε_a| (virial) if no explicit KE is available.
    N_electrons
        Orbital occupation (q_a = 2j_a+1 for a closed subshell).
    n_principal
        Principal quantum number n (denominator correction;
        Kim-Rudd recommend n=1 unless using BEB-vM).
    Q
        Dipole-Bethe parameter.  ``Q = 1`` is the standard BEB; set
        ``Q = 2 M_dipole²`` if dipole oscillator-strength data are
        available (BED model).

    Returns
    -------
    np.ndarray
        Cross section in cm² on the same energy grid as ``T_eV``.
    """
    T = np.asarray(T_eV, dtype=float)
    out = np.zeros_like(T)
    above = T > U_eV
    if not np.any(above):
        return out

    t = T[above] / U_eV
    u = avgT_eV / U_eV
    S = 4.0 * PI_A0_SQ_CM2 * N_electrons * (RYD_EV / U_eV) ** 2

    log_t = np.log(t)
    bracket = (
        0.5 * Q * log_t * (1.0 - 1.0 / t ** 2)
        + (2.0 - Q) * (1.0 - 1.0 / t - log_t / (t + 1.0))
    )
    out[above] = S / (t + (u + 1.0) / n_principal) * bracket
    # Guard against tiny negative numerical noise just above threshold
    out = np.where(out < 0.0, 0.0, out)
    return out


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
    from atomx.prop.transitions.angular import angular_eL, selection_rules
    from atomx.prop.transitions.multipole import radial_eL

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
            if eps_f < -1.5 * (137.036 ** 2):  # below Dirac sea
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
    """Distorted-wave Born approximation for total EII cross section.

    Implementation:
    - Frozen-core V^{N-1} static-Hartree distorted waves for both the
      projectile and the two outgoing electrons (Pindzola-Griffin /
      FAC prescription) — proper screening for neutrals.
    - Asymptotic free-particle box DOS √(R/(π·k)) for delta-energy
      normalisation, robust at high energy.
    - J-coupled antisymmetric Slater integrals between bound + 3
      continuum waves; partial-wave sum to ``kappa_max``.
    """
    from atomx.constants import HARTREE_EV as _HEV
    from atomx.numerics.bspline.convol import (
        ConvolCache, compute_slater_rk_cell,
    )
    from atomx.angular.wigner import wigner_6j

    # Local helpers — duplicate the small ones from the MBPT module to
    # avoid a circular import.
    def _l_of(k):
        return k if k > 0 else -k - 1
    def _tj_of(k):
        return 2 * abs(k) - 1
    def _allowed_k(k_a, k_b):
        ja, jb = abs(k_a) - 0.5, abs(k_b) - 0.5
        kmin, kmax = int(abs(ja - jb)), int(ja + jb)
        la, lb = _l_of(k_a), _l_of(k_b)
        return [k for k in range(kmin, kmax + 1) if (k + la + lb) % 2 == 0]

    import math

    def _reduced_C(k_a, k_b, k):
        la, lb = _l_of(k_a), _l_of(k_b)
        if (la + k + lb) % 2 != 0:
            return 0.0
        from atomx.angular.wigner import wigner_3j
        tja, tjb = _tj_of(k_a), _tj_of(k_b)
        w = wigner_3j(tja, 2 * k, tjb, -1, 0, 1)
        if abs(w) < 1e-15:
            return 0.0
        phase = (-1) ** ((tja + 1) // 2)
        return phase * math.sqrt((tja + 1) * (tjb + 1)) * w

    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError("SCFResult is missing _bsp_data — re-run solve().")
    Z = scf_result.Z
    cache = ConvolCache(bsp_data)

    energies = np.asarray(energies_eV, dtype=float)
    sigma_total = np.zeros_like(energies)
    by_orbital: Dict[str, np.ndarray] = {}
    prov_per_orb: Dict[str, Dict[str, float]] = {}

    # Continuum-wave cache.  solve_continuum returns BOX-normalised
    # states (∫|ψ|² = 1); for cross sections we need DELTA-ENERGY
    # normalisation (⟨ψ_E|ψ_E'⟩ = δ(E−E')).  The conversion factor
    # is √(2 ρ(E)/π) where ρ(E) is the local eigenvalue density-of-
    # states.  We estimate ρ(E) ≈ 1/ΔE_local from the spacing of
    # neighbouring B-spline eigenvalues and fold it into ψ here.
    # Build the V^{N-1} static-Hartree potential once per ionisation
    # target — outside the per-energy loop.
    cont_cache: Dict[tuple, tuple] = {}
    static_potentials: Dict[str, tuple] = {}

    # Box radius for asymptotic free-particle DOS (R / (π k))
    R_box = float(bsp_data["t"][-1])

    def _continuum(kappa: int, energy_au: float, exclude_label: str):
        key = (kappa, round(energy_au, 5), exclude_label)
        cached = cont_cache.get(key)
        if cached is not None:
            return cached
        # Get cached V^{N-1} static-Hartree for this exclude_label
        if exclude_label not in static_potentials:
            static_potentials[exclude_label] = _build_static_hartree(
                scf_result, exclude_label, cache,
            )
        V_PP, V_QQ = static_potentials[exclude_label]

        from atomx.numerics.bspline.dirac_assembly import (
            build_dirac_matrices, solve_dirac,
        )
        ns_full = bsp_data["ns"]
        nsp = bsp_data.get("nsp", ns_full)
        nsq = bsp_data.get("nsq", ns_full)
        H, S = build_dirac_matrices(ns_full, kappa, Z, bsp_data)
        # Add static Hartree to bare-nuclear H
        n_pp = V_PP.shape[0]; n_qq = V_QQ.shape[0]
        H[:n_pp, :n_pp] += V_PP
        H[ns_full:ns_full + n_qq, ns_full:ns_full + n_qq] += V_QQ
        res_e = solve_dirac(H, S, ns_full, kappa, nsp=nsp, nsq=nsq)

        evals = res_e.eigenvalues
        # Restrict to physical positive-energy continuum
        mask = evals > 0.0
        idx_phys = np.where(mask)[0]
        evals_phys = evals[idx_phys]
        order = np.argsort(evals_phys)
        evals_phys = evals_phys[order]
        idx_phys = idx_phys[order]

        if len(evals_phys) == 0:
            P = np.zeros(ns_full); Q = np.zeros(ns_full)
            cont_cache[key] = (P, Q)
            return P, Q

        nearest = int(np.argmin(np.abs(evals_phys - energy_au)))
        # Asymptotic free-particle DOS in box: ρ(E) = R/(π·k) where
        # k = √(2 E).  This is robust at high energy where the
        # eigenvalue spectrum gets sparse — much more reliable than
        # the local-spacing estimate.  For positive E (continuum) this
        # is exact for hard-walled box (no Coulomb tail) and
        # leading-order correct for screened/Coulomb tail.
        k_loc = math.sqrt(max(2.0 * energy_au, 1e-12))
        rho = R_box / (math.pi * k_loc)

        # Extract the eigenvector and split P/Q
        i_full = idx_phys[nearest]
        vec = res_e.coefficients[:, i_full]
        P = np.zeros(ns_full); Q = np.zeros(ns_full)
        P[:nsp] = vec[:nsp]
        Q[:nsq] = vec[ns_full:ns_full + nsq]

        # Delta-E normalisation factor √(2 ρ(E)/π)
        norm = math.sqrt(2.0 * rho / math.pi)
        P *= norm
        Q *= norm
        cont_cache[key] = (P, Q)
        return P, Q

    # Build kappa list: ±1, ±2, … with l ≤ kappa_max
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if 0 < abs(k) and _l_of(k) <= kappa_max]

    for sh in scf_result.subshells:
        eps_a = scf_result.orbital_energies.get(sh.label)
        if eps_a is None or eps_a >= 0.0:
            continue
        I_a_au = -eps_a
        kappa_a = sh.kappa
        tj_a = _tj_of(kappa_a)
        q_a = float(sh.occupation)
        # Bound orbital coefficients
        orb_list = scf_result.orbitals_raw.get(kappa_a, [])
        kappa_subs = [s for s in scf_result.subshells if s.kappa == kappa_a]
        idx_in_kappa = kappa_subs.index(sh)
        if idx_in_kappa >= len(orb_list):
            continue
        P_a, Q_a, _ = orb_list[idx_in_kappa]

        sigma_a = np.zeros_like(energies)

        # Cache the optical oscillator strength + orbital RMS radius
        # once per shell — they depend only on the bound orbital, not
        # on the projectile energy.  Used by the Coulomb-Bethe
        # high-ℓ top-up below.
        f_a_cached: Optional[float] = None
        r_a_rms_cached: Optional[float] = None

        for E_i_idx, E_i_eV in enumerate(energies):
            E_i_au = E_i_eV / _HEV
            T_max = E_i_au - I_a_au
            if T_max <= 0.0:
                continue
            k_i_sq = 2.0 * E_i_au

            # E_1 quadrature on [0, T_max/2] — skip endpoints to avoid
            # threshold singularities.
            E1_grid = np.linspace(0.05 * T_max, 0.49 * T_max, n_outgoing_energies)
            integrand = np.zeros_like(E1_grid)

            for j_E1, E_1_au in enumerate(E1_grid):
                E_2_au = T_max - E_1_au
                # Triple partial-wave sum
                acc = 0.0
                for kappa_in in kappa_list:
                    P_in, Q_in = _continuum(kappa_in, E_i_au, sh.label)
                    tj_in = _tj_of(kappa_in)

                    for kappa_1 in kappa_list:
                        # Direct allowed multipoles k_d on (a,1):(in,2)
                        kd = _allowed_k(kappa_a, kappa_1)
                        if not kd:
                            continue
                        P_1, Q_1 = _continuum(kappa_1, E_1_au, sh.label)
                        tj_1 = _tj_of(kappa_1)

                        for kappa_2 in kappa_list:
                            tj_2 = _tj_of(kappa_2)
                            # Total-J range: J ∈ |j_a-j_in| .. j_a+j_in,
                            # AND ∈ |j_1-j_2| .. j_1+j_2.
                            tJ_lo = max(abs(tj_a - tj_in), abs(tj_1 - tj_2))
                            tJ_hi = min(tj_a + tj_in, tj_1 + tj_2)
                            if tJ_lo > tJ_hi:
                                continue
                            l_a = _l_of(kappa_a); l_in = _l_of(kappa_in)
                            l_1 = _l_of(kappa_1); l_2 = _l_of(kappa_2)
                            if (l_a + l_in + l_1 + l_2) % 2 != 0:
                                continue

                            kd_pair = [k for k in kd if k in _allowed_k(kappa_in, kappa_2)]
                            ke_pair = [k for k in _allowed_k(kappa_a, kappa_2)
                                        if k in _allowed_k(kappa_in, kappa_1)]
                            if not kd_pair and not ke_pair:
                                continue

                            P_2, Q_2 = _continuum(kappa_2, E_2_au, sh.label)

                            # Reduced C^k pieces
                            C_a1 = {k: _reduced_C(kappa_a, kappa_1, k) for k in kd_pair}
                            C_in2 = {k: _reduced_C(kappa_in, kappa_2, k) for k in kd_pair}
                            C_a2 = {k: _reduced_C(kappa_a, kappa_2, k) for k in ke_pair}
                            C_in1 = {k: _reduced_C(kappa_in, kappa_1, k) for k in ke_pair}

                            # Radial Slater integrals
                            Rd: Dict[int, float] = {}
                            for k in kd_pair:
                                if abs(C_a1[k]) < 1e-15 or abs(C_in2[k]) < 1e-15:
                                    continue
                                Rd[k] = compute_slater_rk_cell(
                                    P_a, Q_a, P_in, Q_in, P_1, Q_1, P_2, Q_2,
                                    bsp_data, k, cache=cache,
                                )
                            Re: Dict[int, float] = {}
                            for k in ke_pair:
                                if abs(C_a2[k]) < 1e-15 or abs(C_in1[k]) < 1e-15:
                                    continue
                                Re[k] = compute_slater_rk_cell(
                                    P_a, Q_a, P_in, Q_in, P_2, Q_2, P_1, Q_1,
                                    bsp_data, k, cache=cache,
                                )
                            if not Rd and not Re:
                                continue

                            # Sum over total J
                            J_sum = 0.0
                            for tJ in range(tJ_lo, tJ_hi + 1, 2):
                                # M_J^dir = Σ_k (-1)^{j_in+j_1+J}·{j_a j_in J; j_2 j_1 k}·X_d^k
                                MJd = 0.0
                                for k, Rk in Rd.items():
                                    sj = wigner_6j(tj_a, tj_in, tJ,
                                                    tj_2, tj_1, 2 * k)
                                    if abs(sj) < 1e-15:
                                        continue
                                    Xk = C_a1[k] * C_in2[k] * Rk
                                    MJd += sj * Xk
                                MJd *= (-1) ** ((tj_in + tj_1 + tJ) // 2)

                                MJe = 0.0
                                for k, Rk in Re.items():
                                    sj = wigner_6j(tj_a, tj_in, tJ,
                                                    tj_1, tj_2, 2 * k)
                                    if abs(sj) < 1e-15:
                                        continue
                                    Xk = C_a2[k] * C_in1[k] * Rk
                                    MJe += sj * Xk
                                MJe *= (-1) ** ((tj_in + tj_2 + tJ) // 2)

                                ph_swap = (-1) ** ((tj_1 + tj_2 - tJ) // 2)
                                amp = MJd - ph_swap * MJe
                                J_sum += (tJ + 1) * amp * amp

                            acc += J_sum

                integrand[j_E1] = acc

            # Trapezoidal integral over E_1
            integral = float(np.trapezoid(integrand, E1_grid)) \
                if hasattr(np, "trapezoid") else float(np.trapz(integrand, E1_grid))

            # Standard DW Born prefactor (Bransden-Joachain Eq. 12.137,
            # Gu FAC 2008): σ = (4π/k_i²)·(q_a/(2j_a+1))·∫dE_1 |T|²
            sigma_a_au = (4.0 * np.pi / k_i_sq) * (q_a / (tj_a + 1.0)) * integral

            # ── Coulomb-Bethe high-ℓ top-up (dipole-channel tail) ──
            # The DW partial-wave sum is truncated at κ_max.  For the
            # dipole-allowed E1 channel the projectile partial-wave
            # tail (L_in > κ_max) decays slowly as 1/L².  The Bethe
            # asymptote with proper L-tail damping (Burgess & Sheorey
            # 1974):
            #   σ_CB = (4π·a₀²/E_i)·M_a²·ln(α·E_i/I_a) · η(κ_max)
            # with η(L_max) ≈ 1/(L_max+1)² for the geometric-series
            # tail, and M_a² = f_a/(2·I_a) is the dimensionless
            # generalised oscillator strength at threshold.  This adds
            # only the tail BEYOND κ_max, not the full dipole.
            if E_i_au > 1.5 * I_a_au:
                if f_a_cached is None:
                    f_a_cached = _bound_optical_strength(
                        P_a, Q_a, kappa_a, eps_a, scf_result, cache,
                        static_potentials,
                        occupation=q_a,
                    )
                # M_a² ≈ f_a / (2·I_a) — the threshold value of the
                # generalised oscillator strength integrated over the
                # continuum.
                M_sq = f_a_cached / max(2.0 * I_a_au, 1e-6)
                # Compactness gate: skip C-B when the orbital is so
                # compact that the explicit DW partial-wave sum at
                # κ_max already captures all relevant L (typical of
                # hydrogenic ions).  Threshold ⟨r_a⟩_rms ≥ 1 a₀ marks
                # "diffuse valence" where the top-up is needed.
                if r_a_rms_cached is None:
                    r_a_rms_cached = _orbital_rms_radius(
                        P_a, Q_a, scf_result._bsp_data
                    )
                if r_a_rms_cached >= 1.0:
                    # Dipole-channel L-tail decays as 1/L; fraction of
                    # contribution missing from κ_max-truncated sum is
                    # ≈ 1/(κ_max+1).
                    eta_tail = 1.0 / (kappa_max + 1.0)
                    sigma_cb_au = (
                        (4.0 * np.pi / E_i_au) * M_sq *
                        math.log(max(E_i_au / I_a_au, 1.0)) *
                        eta_tail
                    )
                    sigma_a_au += sigma_cb_au

            # Convert from a₀² to cm²
            sigma_a[E_i_idx] = sigma_a_au * BOHR_RADIUS_M ** 2 * 1e4

        by_orbital[sh.label] = sigma_a
        sigma_total += sigma_a
        prov_per_orb[sh.label] = {
            "I_eV": I_a_au * _HEV, "occupation": q_a,
            "kappa_max": kappa_max,
            "n_E1_quadrature": n_outgoing_energies,
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
