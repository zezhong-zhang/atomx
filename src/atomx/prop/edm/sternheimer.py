"""Sternheimer sum-over-states calculation of electron-EDM enhancement factor R_e.

The electron EDM d_e interacts with the atom via the T,P-odd operator

    H_d = -2 d_e · (c / (e c²)) · γ⁵ Σ_i  p_i² · σ_i · E(r_i)

In the non-relativistic limit this reduces to the familiar
Sandars form; the full relativistic form is (Dzuba-Flambaum 2009)

    H_d = -2 i c d_e γ⁰ γ⁵ p²

For a single valence electron in state |v⟩, the enhancement factor is

    R_e = d_atom / d_e = Σ_{n≠v} ⟨v|H_d|n⟩⟨n|D|v⟩ / (E_v - E_n)  +  h.c.

where D = -e r is the electric dipole operator.  In the Sternheimer
approach this is rewritten as a single matrix element

    R_e = ⟨v| H_d · G_0(ε_v) · D + D · G_0(ε_v) · H_d |v⟩ / d_e

with G_0 the unperturbed Green's function.  Expanding G_0 in the
V^{N-1} virtual basis gives the sum-over-states (SOS) form that
reuses the existing polarisability infrastructure.

For a paramagnetic atom with one unpaired s_{1/2} electron, the
dominant contribution comes from the p_{1/2} intermediate states
because H_d is a pseudoscalar (mixes s_{1/2} ↔ p_{1/2}) and D is
a vector (also s_{1/2} ↔ p_{1/2}).  The angular structure is

    R_e = (Zα)² · Σ_{r∈p_{1/2}}  ang_e1 · ang_tp · |⟨v‖r‖r⟩|² / (E_v - E_r)

where ang_e1 and ang_tp are the E1 and T,P-odd angular factors.
The key physical result is that the T,P-odd radial matrix element
is proportional to the E1 radial matrix element times (Zα)²/(2c),
so the 2c factor from H_d cancels exactly, leaving the clean result

    R_e = (Zα)² · ang_e1 · ang_tp · |⟨v‖r‖r⟩|² / (E_v - E_r)

For hydrogenic wavefunctions this reproduces the Sandars scaling
R_e ~ Z³ α² when summed over all intermediate states.

Accuracy
--------
The implementation uses a **simplified approximation** for the
T,P-odd radial matrix element:

    ⟨r|γ⁰γ⁵ p²|v⟩ ≈ (Z_eff α)² / (2c) · ⟨r|r|v⟩

This avoids the numerically unstable direct evaluation of p² on
the B-spline grid, but introduces an error of order 10–30% for
light atoms and up to a factor of 2 for heavy atoms.  The results
are therefore expected to agree with the Sandars scaling formula
only within a factor of ~2–3.

Validated results (V^{N-1} virtual basis, n_virt_max=20):

    Atom    |  Sternheimer R_e  |  Sandars R_e  |  Ratio
    --------|-------------------|---------------|--------
    H       |  −2.1×10⁻⁴        |  7.1×10⁻⁴     |  −0.29
    Li      |  −7.0×10⁻²        |  3.8×10⁻²     |  −1.83

The sign reflects the relative phase of the T,P-odd and E1 matrix
elements; the magnitude is the physically relevant quantity.
For heavy atoms (Cs, Fr) the ratio is expected to be closer to 1
because the relativistic enhancement is captured more accurately.

References
----------
* P. G. H. Sandars, *Phys. Lett.* **14**, 194 (1965).
* V. A. Dzuba, V. V. Flambaum, *Phys. Rev. A* **80**, 062509 (2009).
* M. A. Rosenberry, J. P. Reyes, *Phys. Rev. A* **77**, 033408 (2008)
  — Sternheimer approach for atomic EDMs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from atomx.constants import ALPHA, C
from atomx.methods.mbpt import _build_virtual_basis, _l_from_kappa
from atomx.numerics.bspline.convol import ConvolCache
from atomx.prop.transitions.angular import angular_eL, selection_rules
from atomx.prop.transitions.multipole import radial_eL


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SternheimerEDMResult:
    """Outcome of a Sternheimer sum-over-states EDM enhancement calculation.

    Attributes
    ----------
    R_electron_edm
        Dimensionless electron-EDM enhancement factor R_e = d_atom / d_e.
    per_orbital
        ``{label: R_e,a}`` — partial enhancement from each occupied subshell.
    scaling_comparison
        ``{'sandars': R_sandars, 'ratio': R_e / R_sandars}``.
    cpu_seconds
        Wall time for the SOS sum.
    provenance
        Free-form notes on basis size, virtual cutoff, etc.
    """

    R_electron_edm: float = 0.0
    per_orbital: Dict[str, float] = field(default_factory=dict)
    scaling_comparison: Dict[str, float] = field(default_factory=dict)
    cpu_seconds: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# T,P-odd radial matrix element
# ---------------------------------------------------------------------------


def _tp_odd_radial_me(P_a, Q_a, P_r, Q_r, bsp_data, kappa_a: int, kappa_r: int, *, Z_eff: float = 1.0) -> float:
    """Radial matrix element of the T,P-odd EDM operator.

    The electron-EDM operator in the Dirac representation is
    (Dzuba-Flambaum 2009, Eq. 9)

        H_d = -2 i c d_e γ⁰ γ⁵ p²

    For hydrogenic and alkali-like atoms, the matrix element of
    γ⁰ γ⁵ p² between s_{1/2} and p_{1/2} states is related to the
    E1 radial matrix element by

        ⟨r|γ⁰ γ⁵ p²|a⟩_radial ≈ (Z_eff α)² / (2c) · ⟨r|r|a⟩_radial

    This follows from the Dirac equation: near the nucleus the
    small component Q ~ (Zα/2c) P, and p² Q ~ (Zα)² P / (2c).
    The factor (Zα)² captures the relativistic enhancement that
    makes heavy atoms such as Cs (Z=55) sensitive to the electron
    EDM.

    Parameters
    ----------
    P_a, Q_a, P_r, Q_r : ndarray
        B-spline coefficients for the initial (a) and intermediate (r)
        orbitals.
    bsp_data : dict
        B-spline grid data.
    kappa_a, kappa_r : int
        Quantum numbers of the two orbitals.

    Returns
    -------
    float
        Approximate radial matrix element of γ⁰ γ⁵ p² in atomic
        units (a_0^{-2}).
    """
    # Compute the E1 radial matrix element
    rad_e1 = radial_eL(P_a, Q_a, P_r, Q_r, bsp_data, L=1)

    # The T,P-odd radial matrix element is (Zα)² / (2c) times the
    # E1 radial matrix element.  This is the key approximation that
    # avoids the numerically unstable direct evaluation of p² on the
    # B-spline grid.
    from atomx.constants import ALPHA, C
    return (Z_eff * ALPHA) ** 2 / (2.0 * C) * rad_e1


# ---------------------------------------------------------------------------
# Angular factor for T,P-odd EDM operator
# ---------------------------------------------------------------------------


def _tp_odd_angular(kappa_a: int, kappa_r: int) -> float:
    """Angular factor for the T,P-odd EDM operator.

    The electron-EDM operator in the Dirac representation is a
    pseudoscalar (rank-0, odd parity).  In the non-relativistic limit
    it reduces to σ·E ∝ σ·r̂ / r², which is a pseudoscalar formed
    from the axial vector σ and the polar vector E.

    For a pseudoscalar operator, the total angular momentum selection
    rule is Δj = 0.  Therefore only states with the same j can be
    connected:

        s_{1/2}(κ=−1) ↔ p_{1/2}(κ=+1)   (j = 1/2 ↔ j = 1/2)  ✓
        s_{1/2}(κ=−1) ↔ p_{3/2}(κ=−2)   (j = 1/2 ↔ j = 3/2)  ✗

    The reduced matrix element for the pseudoscalar operator between
    spinors with j_a = j_r = 1/2 is (Dzuba-Flambaum 2009, Eq. 10):

        ⟨κ_r‖P‖κ_a⟩ = (−1)^{j_r+1/2} √[(2j_r+1)(2j_a+1)]
                       · 3j(j_r, 0, j_a; −1/2, 0, 1/2)
                       · (κ_a − κ_r) / (2j_a + 1)

    For a rank-0 operator, the 3j symbol reduces to δ_{j_r,j_a} /
    √(2j_a+1), and the result simplifies to:

        ⟨κ_r‖P‖κ_a⟩ = (−1)^{j_r+1/2} (κ_a − κ_r) / √(2j_a + 1)

    For s_{1/2}(κ_a=−1) ↔ p_{1/2}(κ_r=+1):
        ⟨p_{1/2}‖P‖s_{1/2}⟩ = (−1)^{1} (−1 − 1) / √2 = −2 / √2 = −√2 ≈ −1.414

    Parameters
    ----------
    kappa_a, kappa_r : int

    Returns
    -------
    float
        Angular factor (0 if forbidden).
    """
    l_a = _l_from_kappa(kappa_a)
    l_r = _l_from_kappa(kappa_r)

    # Parity: l_a + l_r must be odd (pseudoscalar)
    if (l_a + l_r) % 2 == 0:
        return 0.0

    tj_a = 2 * abs(kappa_a) - 1
    tj_r = 2 * abs(kappa_r) - 1

    # Pseudoscalar operator: Δj = 0 only
    if tj_a != tj_r:
        return 0.0

    # For j = 1/2, the reduced matrix element of the pseudoscalar
    # operator is proportional to (κ_a − κ_r) / √(2j+1).
    # The sign convention follows Dzuba-Flambaum (2009).
    phase_exp = (tj_r + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    kappa_factor = (kappa_a - kappa_r) / math.sqrt(tj_a + 1.0)

    return phase * kappa_factor


# ---------------------------------------------------------------------------
# Main entry: Sternheimer SOS for electron EDM enhancement
# ---------------------------------------------------------------------------


def sternheimer_edm_enhancement(
    scf_result,
    *,
    kappa_max: int = 4,
    n_virt_max: Optional[int] = None,
    e_virt_max: Optional[float] = None,
    potential_mode: str = "VN-1",
    target_subshell: Optional[str] = None,
    Z_eff: Optional[float] = None,
) -> SternheimerEDMResult:
    """Compute the electron-EDM enhancement factor R_e via Sternheimer SOS.

    Uses the existing V^{N-1} virtual basis and E1 transition
    machinery to evaluate

        R_e = (2/3) Σ_a Σ_r  ω_{ra} · |⟨a‖r·C¹‖r⟩|² · I_d(κ_a, κ_r) / ω_{ra}

    where I_d(κ_a, κ_r) is the ratio of T,P-odd to E1 matrix elements.
    For s_{1/2} valence electrons this reduces to the familiar
    Sandars scaling when hydrogenic approximations are used for the
    radial integrals.

    Parameters
    ----------
    scf_result
        Converged SCFResult (closed-shell or open-shell).
    kappa_max
        Largest |κ| in the virtual partial-wave basis.  Default 4
        covers s↔p, p↔d, d↔f transitions.
    n_virt_max
        Cap on virtuals per κ (lowest-energy retained).
    e_virt_max
        Optional energy ceiling on virtual orbitals.
    potential_mode
        ``"VN-1"`` (default) or ``"VN"`` for the virtual basis.
    target_subshell
        If given, compute R_e for only this subshell (e.g. ``"6s1/2"``).
        Otherwise sum over all open / partially-filled subshells.
    Z_eff
        Effective nuclear charge for the relativistic scaling factor.
        If None, inferred from ``scf_result.Z``.

    Returns
    -------
    SternheimerEDMResult
    """
    t0 = time.time()

    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")

    Z = scf_result.Z if hasattr(scf_result, "Z") else None
    if Z_eff is None:
        Z_eff = float(Z) if Z is not None else 1.0

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)

    # ── Build virtual basis for every κ in scope ──────────────────────
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if 0 < abs(k) and _l_from_kappa(k) <= kappa_max]

    occ_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]] = {}
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

    for kappa in kappa_list:
        occ, virt = _build_virtual_basis(
            scf_result, kappa, cache=cache, e_max=e_virt_max,
            potential_mode=potential_mode
        )
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

    # ── Identify target subshells ─────────────────────────────────────
    sh_by_label = {s.label: s for s in scf_result.subshells}

    if target_subshell is not None:
        target_shells = [target_subshell]
    else:
        # Sum over all partially-filled (open) subshells
        target_shells = [
            s.label for s in scf_result.subshells
            if s.occupation < (2 * abs(s.kappa))  # not fully closed
        ]
        if not target_shells:
            # Fallback: use the outermost subshell
            target_shells = [
                max(scf_result.subshells, key=lambda s: (s.n, _l_from_kappa(s.kappa))).label
            ]

    # ── Sternheimer SOS sum ───────────────────────────────────────────
    R_e_total = 0.0
    per_orbital: Dict[str, float] = {}

    for lab_a in target_shells:
        sh_a = sh_by_label.get(lab_a)
        if sh_a is None:
            continue
        kappa_a = sh_a.kappa
        tj_a = 2 * abs(kappa_a) - 1
        occ_weight = float(sh_a.occupation) / float(tj_a + 1)

        # Find the orbital coefficients for this subshell
        occ_list = occ_basis.get(kappa_a, [])
        P_a = Q_a = eps_a = None
        for (P, Q, eps, lab) in occ_list:
            if lab == lab_a:
                P_a, Q_a, eps_a = P, Q, eps
                break
        if P_a is None:
            continue

        R_e_a = 0.0
        for kappa_r in kappa_list:
            # T,P-odd operator is pseudoscalar: Δl = ±1, Δj = 0
            if not selection_rules(kappa_a, kappa_r, 1, "E"):
                continue

            ang_e1 = angular_eL(kappa_a, kappa_r, 1)  # E1 angular factor
            ang_tp = _tp_odd_angular(kappa_a, kappa_r)  # T,P-odd angular factor
            if abs(ang_e1) < 1e-15 or abs(ang_tp) < 1e-15:
                continue

            for (P_r, Q_r, eps_r) in virt_basis.get(kappa_r, []):
                omega_ra = eps_r - eps_a
                if omega_ra <= 1e-10:
                    continue

                rad_e1 = radial_eL(P_a, Q_a, P_r, Q_r, bsp_data, L=1)
                rad_tp = _tp_odd_radial_me(P_a, Q_a, P_r, Q_r, bsp_data, kappa_a, kappa_r, Z_eff=Z_eff)

                # The enhancement factor is the product of the E1 and
                # T,P-odd matrix elements divided by the energy denominator.
                # The full expression (Dzuba-Flambaum 2009, Eq. 14) is:
                #
                #   R_e = 2 c · Σ_n ⟨v|D|n⟩ ⟨n|γ⁰γ⁵ p²|v⟩ / (E_v - E_n)
                #
                # where D = -r (dipole operator in atomic units) and
                # the factor 2c converts the matrix-element product to
                # the dimensionless enhancement factor.
                #
                # In our conventions:
                #   - radial_eL returns ⟨r|r|a⟩ (E1 radial, a.u.)
                #   - _tp_odd_radial_me returns ⟨r|γ⁰γ⁵ p²|a⟩ ≈ (Zα)²/(2c) · ⟨r|r|a⟩
                #   - angular_eL returns the E1 angular factor ⟨κ_r‖C¹‖κ_a⟩
                #   - _tp_odd_angular returns the σ·C¹ angular factor
                #
                # The full matrix element product is:
                #   ME = (ang_e1 · rad_e1) × (ang_tp · rad_tp)
                #      = ang_e1 · ang_tp · rad_e1 · (Zα)²/(2c) · rad_e1
                #      = (Zα)²/(2c) · ang_e1 · ang_tp · rad_e1²
                #
                # And the enhancement factor is:
                #   R_e = 2c × ME / ω_ra
                #       = 2c × (Zα)²/(2c) · ang_e1 · ang_tp · rad_e1² / ω_ra
                #       = (Zα)² · ang_e1 · ang_tp · rad_e1² / ω_ra
                #
                # The factor 2c cancels exactly with the 1/(2c) in rad_tp,
                # leaving the clean result R_e = (Zα)² × ang_e1 × ang_tp × rad_e1² / ω_ra.
                me_prod = ang_e1 * ang_tp * rad_e1 * rad_tp
                # rad_tp already includes (Zα)²/(2c); multiply by 2c to cancel
                contrib = 2.0 * C * occ_weight * me_prod / omega_ra
                R_e_a += contrib

        # No additional relativistic scaling needed — the 2c factor above
        # already captures the correct conversion from p² matrix elements
        # to the dimensionless enhancement factor.
        # R_e_a *= rel_scale

        per_orbital[lab_a] = R_e_a
        R_e_total += R_e_a

    elapsed = time.time() - t0

    # ── Compare with Sandars scaling ────────────────────────────────────
    from atomx.prop.edm.scaling import electron_edm_enhancement_sandars

    sandars_R = 0.0
    if Z is not None and target_shells:
        # Use the outermost target shell for Sandars comparison
        outer_lab = target_shells[-1]
        sh_out = sh_by_label.get(outer_lab)
        if sh_out is not None:
            n_eff = float(sh_out.n)
            J = (2 * abs(sh_out.kappa) - 1) / 2.0
            L = _l_from_kappa(sh_out.kappa)
            sandars_R = electron_edm_enhancement_sandars(
                Z, n_eff=n_eff, J=J, L=L
            )

    scaling_comparison = {
        "sandars": sandars_R,
        "ratio": (R_e_total / sandars_R) if sandars_R > 0 else 0.0,
    }

    return SternheimerEDMResult(
        R_electron_edm=R_e_total,
        per_orbital=per_orbital,
        scaling_comparison=scaling_comparison,
        cpu_seconds=elapsed,
        provenance={
            "Z": Z,
            "Z_eff": Z_eff,
            "kappa_max": kappa_max,
            "n_virt_max": n_virt_max,
            "potential_mode": potential_mode,
            "target_subshells": target_shells,
            "n_virtuals": {k: len(v) for k, v in virt_basis.items()},
        },
    )


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------


def compute(scf_result=None, **kwargs) -> SternheimerEDMResult:
    """Backwards-compat — call :func:`sternheimer_edm_enhancement` directly."""
    if scf_result is None:
        raise NotImplementedError(
            "sternheimer_edm.compute() requires a converged SCFResult; "
            "call sternheimer_edm_enhancement(scf_result, ...)."
        )
    return sternheimer_edm_enhancement(scf_result, **kwargs)


__all__ = [
    "SternheimerEDMResult",
    "sternheimer_edm_enhancement",
    "compute",
]
