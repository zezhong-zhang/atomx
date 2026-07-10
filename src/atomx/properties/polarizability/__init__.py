"""Static and dynamic dipole polarisability α(ω).

Closed-shell DHF + V^{N-1} virtuals — sum-over-states form

    α(ω) = (2/3) Σ_{a∈occ} Σ_{r∈virt}
              ω_{ra} · |⟨κ_a‖r·C^1‖κ_r⟩|² / (ω_{ra}² − ω²)

with reduced matrix element

    ⟨κ_a‖r·C^1‖κ_r⟩ = ⟨κ_a‖C^1‖κ_r⟩ · ∫ (P_a r P_r + Q_a r Q_r) dr.

The (2j_a+1) factor for the closed-shell sum is built into the
reduced matrix element via Wigner-Eckart.

Length gauge ("Babushkin") is used by default; velocity gauge can
be selected for gauge-invariance checks.

Outputs
-------
- ``alpha_static`` — α(0) in atomic units (a₀³).
- ``alpha_dynamic`` — α(ω) on a user-supplied frequency grid.
- ``black_body_shift`` — Stark shift in a 300 K black-body radiation
  field (uses the Mitroy-Safronova-Clark Eq. 7).
- ``per_orbital`` — partial sums broken down by occupied subshell.

References
----------
- J. Mitroy, M. S. Safronova, C. W. Clark, *J. Phys. B* **43**,
  202001 (2010) — review of α(ω) calculations and BBR shift.
- W. R. Johnson, *Atomic Structure Theory* (Springer, 2007), §9.11.
- A. Derevianko, S. G. Porsev, K. Beloy, *Phys. Rev. A* **82**,
  030901(R) (2010).
- M. S. Safronova, U. I. Safronova, M. G. Kozlov, *Phys. Rev. A*
  **97**, 022507 (2018).

Note
----
The single-determinant (uncorrelated) sum-over-states gives the
standard "frozen-core HF" polarisability.  RPA / TDHF screening is
not yet folded in (planned via :mod:`atomx.methods.rpa`); for
heavy alkali clock atoms RPA can change α by 10-30 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from atomx.angular.coefficients import _l_from_kappa
from atomx.numerics.bspline.convol import ConvolCache
from atomx.properties.transitions.angular import angular_eL, selection_rules
from atomx.properties.transitions.multipole import radial_eL


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PolarizabilityResult:
    """Outcome of a dipole polarisability calculation.

    Attributes
    ----------
    alpha_static
        α(0) in atomic units (a₀³).
    omega_grid
        Frequency grid (hartree) on which the dynamic polarisability
        was evaluated; ``None`` for static-only.
    alpha_dynamic
        α(ω) on ``omega_grid`` in a₀³.
    per_orbital
        ``{label: α_a}`` — partial static polarisability from each
        occupied subshell.
    black_body_shift_K
        BBR Stark shift at 300 K in hartree (Eq. 7 of Mitroy et al.
        2010).  Computed only if ``include_bbr=True``.
    cpu_seconds
        Wall time for the SOS sum.
    provenance
        Free-form notes on basis size, virtual cutoff, gauge.
    """

    alpha_static: float = 0.0
    omega_grid: Optional[np.ndarray] = None
    alpha_dynamic: Optional[np.ndarray] = None
    per_orbital: Dict[str, float] = field(default_factory=dict)
    black_body_shift_K: Optional[float] = None
    cpu_seconds: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def dipole_polarizability(
    scf_result,
    *,
    omega: Optional[np.ndarray] = None,
    kappa_max: int = 4,
    n_virt_max: Optional[int] = None,
    include_bbr: bool = False,
    bbr_temperature_K: float = 300.0,
    gauge: str = "length",
    e_virt_max: Optional[float] = None,
    potential_mode: str = "VN-1",
) -> PolarizabilityResult:
    """Compute the dipole polarisability α(ω) of a closed-shell atom.

    Parameters
    ----------
    scf_result
        Converged closed-shell ``SCFResult``.
    omega
        Optional 1D array of photon energies (hartree).  If ``None``,
        only the static value α(0) is computed.
    kappa_max
        Largest |κ| in the virtual partial-wave basis.  Default 4
        covers the dipole transitions s↔p, p↔d, d↔f.
    n_virt_max
        Cap on virtuals per κ (lowest-energy retained).
    include_bbr
        If True, evaluate the 300-K (or ``bbr_temperature_K``)
        black-body Stark shift.
    bbr_temperature_K
        BBR temperature in kelvin.
    gauge
        ``"length"`` (default, Babushkin) or ``"velocity"``
        (Coulomb).  Only "length" is fully implemented; "velocity"
        falls back to length with a warning.
    e_virt_max
        Optional energy ceiling on virtual orbitals.

    Returns
    -------
    PolarizabilityResult
    """
    import time
    t0 = time.time()

    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")
    # Allow open subshells; the SOS form below sums explicit
    # occupations weight q_a / (2j_a+1) to handle hydrogenic and
    # alkali-like cases too.

    if gauge != "length":
        import warnings
        warnings.warn(
            "Velocity gauge for SOS polarisability not yet implemented; "
            "falling back to length (Babushkin) gauge.",
            stacklevel=2,
        )
        gauge = "length"

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)

    # ── Build virtual basis for every κ in scope ────────────────────
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if 0 < abs(k) and _l_from_kappa(k) <= kappa_max]

    occ_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]] = {}
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

    # Deferred import: _build_virtual_basis is a mean-field (methods-layer)
    # service; importing it lazily keeps properties → methods off the
    # module-load path (avoids the layer inversion).
    from atomx.methods.mbpt import _build_virtual_basis

    for kappa in kappa_list:
        occ, virt = _build_virtual_basis(scf_result, kappa, cache=cache,
                                          e_max=e_virt_max,
                                          potential_mode=potential_mode)
        kappa_subshells = [s for s in scf_result.subshells if s.kappa == kappa]
        occ_with_labels: List[Tuple[np.ndarray, np.ndarray, float, str]] = []
        for i, (P, Q, _eps_v_n1) in enumerate(occ):
            if i < len(kappa_subshells):
                sh = kappa_subshells[i]
                lab = sh.label
                eps_use = scf_result.orbital_energies.get(lab, _eps_v_n1)
            else:
                lab = f"occ-{kappa}-{i}"
                eps_use = _eps_v_n1
            occ_with_labels.append((P, Q, eps_use, lab))
        if n_virt_max is not None:
            virt = virt[:n_virt_max]
        occ_basis[kappa] = occ_with_labels
        virt_basis[kappa] = virt

    # ── Sum over states: α(ω) = (2/3) Σ_a Σ_r ω·|⟨a‖r‖r⟩|²/(ω²−ω_ph²) ──
    omega_arr = np.asarray(omega, dtype=float) if omega is not None else None
    alpha_static = 0.0
    alpha_dyn = (np.zeros_like(omega_arr) if omega_arr is not None else None)
    per_orbital: Dict[str, float] = {}

    # Map labels → occupations for the q_a/(2j_a+1) weighting.
    sh_by_label = {s.label: s for s in scf_result.subshells}

    for kappa_a, occ_list in occ_basis.items():
        for (P_a, Q_a, eps_a, lab_a) in occ_list:
            sh_a = sh_by_label.get(lab_a)
            if sh_a is None:
                continue
            tj_a = 2 * abs(kappa_a) - 1   # twice j_a
            # Occupation weight q_a/(2j_a+1).  Reduces to 1 for closed
            # subshells; gives 1/2 for half-filled (e.g. hydrogenic
            # 1s¹) so the SOS reproduces the Dalgarno closed-form
            # α(H) = 9/2 a₀³ instead of double-counting.
            occ_weight = float(sh_a.occupation) / float(tj_a + 1)

            alpha_a = 0.0
            for kappa_r in kappa_list:
                if not selection_rules(kappa_a, kappa_r, 1, "E"):
                    continue
                ang = angular_eL(kappa_a, kappa_r, 1)  # ⟨κ_a‖C¹‖κ_r⟩
                if abs(ang) < 1e-15:
                    continue
                for (P_r, Q_r, eps_r) in virt_basis[kappa_r]:
                    omega_ra = eps_r - eps_a
                    if omega_ra <= 1e-10:
                        continue
                    rad = radial_eL(P_a, Q_a, P_r, Q_r, bsp_data, L=1)
                    me_sq = (ang * rad) ** 2  # |⟨a‖r·C¹‖r⟩|²
                    # Static
                    contrib_stat = (2.0 / 3.0) * occ_weight * me_sq / omega_ra
                    alpha_a += contrib_stat
                    # Dynamic
                    if alpha_dyn is not None and omega_arr is not None:
                        denom = omega_ra * omega_ra - omega_arr * omega_arr
                        with np.errstate(divide="ignore", invalid="ignore"):
                            alpha_dyn += (2.0 / 3.0) * occ_weight * omega_ra * me_sq / np.where(
                                np.abs(denom) > 1e-12, denom, np.nan
                            )
            per_orbital[lab_a] = alpha_a
            alpha_static += alpha_a

    # ── Black-body shift (Mitroy-Safronova-Clark Eq. 7) ─────────────
    bbr_shift = None
    if include_bbr and omega_arr is not None and alpha_dyn is not None:
        bbr_shift = _bbr_shift_from_alpha(omega_arr, alpha_dyn, bbr_temperature_K)
    elif include_bbr:
        # Static-limit BBR Stark shift (Mitroy-Safronova-Clark Eq. 4):
        #   Δ_BBR ≈ -(1/2) ⟨E²⟩ α(0),
        # with ⟨E²⟩ = (8.319 V/cm)² · (T/300 K)⁴.
        # 1 a.u. of E-field = 5.14220674763e9 V/cm.
        E_field_au = (8.319 / 5.14220674763e9) * (bbr_temperature_K / 300.0) ** 2
        bbr_shift = -0.5 * E_field_au * E_field_au * alpha_static

    elapsed = time.time() - t0

    return PolarizabilityResult(
        alpha_static=alpha_static,
        omega_grid=omega_arr,
        alpha_dynamic=alpha_dyn,
        per_orbital=per_orbital,
        black_body_shift_K=bbr_shift,
        cpu_seconds=elapsed,
        provenance={
            "kappa_max": kappa_max,
            "n_virt_max": n_virt_max,
            "gauge": gauge,
            "n_occupied_subshells": len(scf_result.subshells),
            "n_virtuals": {k: len(v) for k, v in virt_basis.items()},
        },
    )


def _bbr_shift_from_alpha(
    omega_grid: np.ndarray,
    alpha_omega: np.ndarray,
    T_K: float,
) -> float:
    """Mitroy-Safronova-Clark Eq. 7 BBR Stark shift.

    Δ_BBR = -(1/2) · (8/15π³) · (αT)⁴ · ∫₀^∞ x³ α_R(xT) / (e^x − 1) dx

    Simplified low-frequency form for ω_resonance >> kT:

    Δ_BBR ≈ -(1/2) ⟨E²⟩ α(0),  ⟨E²⟩ = (8.319 V/cm)²(T/300)⁴

    For high accuracy at moderate T, integrate the Planck-weighted
    α_R(ω).  Returns the shift in hartree.
    """
    from atomx.constants import HARTREE_EV
    # Boltzmann constant in hartree/K
    kT_au = (T_K / 11604.5) / HARTREE_EV  # = k_B T in hartree

    # Filter physical (real) part of α_omega
    real_alpha = np.real(alpha_omega)
    finite = np.isfinite(real_alpha)
    if not np.any(finite):
        return 0.0
    omega = omega_grid[finite]
    alpha = real_alpha[finite]

    # Planck weight function: x³ / (e^x - 1) where x = ω/kT.
    # Average alpha over the Planck spectrum, then reuse the static
    # black-body field formula. This guarantees the constant-alpha
    # limit matches the static branch above.
    x = omega / kT_au
    weight = np.where(x > 700, 0.0, np.where(x < 1e-6, x ** 2, x ** 3 / np.expm1(x)))
    integrand = weight * alpha

    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    denom = float(_trap(weight, x))
    if abs(denom) < 1e-30:
        return 0.0
    alpha_planck = float(_trap(integrand, x) / denom)

    E_field_au = (8.319 / 5.14220674763e9) * (T_K / 300.0) ** 2
    return -0.5 * E_field_au * E_field_au * alpha_planck


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------


def compute(scf_result=None, **kwargs) -> PolarizabilityResult:
    """Backwards-compat — call :func:`dipole_polarizability` directly."""
    if scf_result is None:
        raise NotImplementedError(
            "polarizability compute() requires a converged SCFResult; "
            "call dipole_polarizability(scf_result, ...)."
        )
    return dipole_polarizability(scf_result, **kwargs)


__all__ = ["PolarizabilityResult", "dipole_polarizability", "compute"]
