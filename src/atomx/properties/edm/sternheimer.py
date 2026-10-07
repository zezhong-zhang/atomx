"""Sternheimer sum-over-states calculation of electron-EDM enhancement factor R_e.

The electron EDM ``d_e`` couples to the nuclear electric field through the
T,P-odd one-body operator (Sandars; Khriplovich & Lamoreaux; Dzuba &
Flambaum, Phys. Rev. A 80, 062509 (2009))

    H_d = -d_e · β γ⁵ Σ·E(r),

with ``E(r) = -∇V`` the electrostatic field at the electron and ``Σ`` the
4×4 spin operator.  In the Dirac representation ``β γ⁵`` is purely
off-diagonal — it mixes the large and small spinor components — so, exactly
like the PNC weak operator and the magnetic multipoles, its radial matrix
element is a **large–small cross integral**

    ⟨r| H_d |a⟩ / d_e  =  ang_tp · ∫₀^∞ E(r) · [P_r(r) Q_a(r) − Q_r(r) P_a(r)] dr,

rather than a large–large (E1-like) integral.  This is the decisive physical
point: the enhancement is dominated by the near-nuclear region where the
small component ``Q ~ (Zα)/(2c) · P`` carries the relativistic boost that
makes heavy atoms (Cs, Tl, Fr) orders of magnitude more sensitive to ``d_e``
than light atoms — Sandars' central result.  Approximating the T,P-odd matrix
element as proportional to the E1 matrix element discards this boost and
underestimates ``R_e`` for heavy atoms by orders of magnitude.

For a single valence electron in state ``|v⟩`` the dimensionless
enhancement factor is the second-order perturbative sum

    R_e ≡ d_atom / d_e
        = 2 · occ_w · Σ_{r≠v}  ⟨v‖D‖r⟩ · ⟨r‖h_d‖v⟩ / (E_v − E_r),

where ``D = r C¹`` is the E1 operator, ``h_d = H_d / d_e``, and ``occ_w`` is
the subshell occupation weight.  Expanding the sum over the ``V^{N-1}``
virtual basis reuses the existing polarisability infrastructure
(:func:`atomx.methods.mbpt.build_virtual_basis`,
:func:`~atomx.properties.transitions.multipole.radial_eL`,
:func:`~atomx.properties.transitions.angular.angular_eL`).

Near the nucleus the electron sees the full unscreened nuclear charge, so
the field kernel is taken as ``E(r) = Z/r²`` from the bare Coulomb potential
``V = −Z/r`` — the leading (and dominant) term.  Electronic Hartree
screening of the field and finite-nuclear-size regularisation are not yet
folded in; together with the single-configuration (Dirac–Fock) reference
this places the calculation at the usual ~factor-of-2 accuracy of the
scaling formulas in :mod:`atomx.properties.edm.scaling` **for the heavy
atoms those formulas were calibrated on**.  It does *not* hold for light
atoms: for H 1s the SOS gives ``R_e = 0.0304260`` against the closed form's
``7.10018e-4``, a ratio of 42.85 (measured at ``kappa_max=4,
n_virt_max=20``; see
``tests/test_sternheimer_edm.py::test_sandars_comparison_present``).  The
closed form is a heavy-atom ``Z³`` scaling law with no light-atom validity,
so the two numbers are not expected to agree at Z=1.  ``R_e`` is returned
unsigned-magnitude-correct; the overall sign is convention/phase dependent
(see Sandars scaling, which is also unsigned).

References
----------
* P. G. H. Sandars, *Phys. Lett.* **14**, 194 (1965) — relativistic
  enhancement of the electron-EDM signal.
* I. B. Khriplovich, S. K. Lamoreaux, *CP Violation Without Strangeness*
  (Springer, 1997), §3 — ``β γ⁵ Σ·E`` operator form.
* V. A. Dzuba, V. V. Flambaum, *Phys. Rev. A* **80**, 062509 (2009).
* J. S. M. Ginges, V. V. Flambaum, *Phys. Rep.* **397**, 63 (2004) —
  atomic EDM theory review.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from atomx.angular.recoupling import l_of_kappa
from atomx.numerics.bspline.convol import ConvolCache
from atomx.numerics.bspline.evaluate import eval_P as _eval_P
from atomx.numerics.bspline.evaluate import eval_Q as _eval_Q
from atomx.properties._sos_basis import build_sos_virtual_basis
from atomx.properties.transitions.angular import angular_eL, selection_rules
from atomx.properties.transitions.multipole import radial_eL

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


def _tp_odd_radial_me(P_a, Q_a, P_r, Q_r, bsp_data, *, Z: float) -> float:
    """Radial matrix element of the T,P-odd electron-EDM operator ``h_d``.

    With ``H_d = -d_e β γ⁵ Σ·E`` and the electric field ``E(r) = -∇V``, the
    ``β γ⁵`` block (off-diagonal in the Dirac representation) mixes the large
    and small spinor components, giving a large–small cross integral — the
    same structure as the PNC weak matrix element
    (:func:`atomx.properties.pnc.weak_hamiltonian_radial_me`) and the magnetic
    multipoles.  Per unit ``d_e`` (the ``d_e`` cancels in the enhancement
    factor ``R_e = d_atom/d_e``):

        ⟨r|h_d|a⟩_radial = ∫₀^∞ E(r) · [P_r(r) Q_a(r) − Q_r(r) P_a(r)] dr.

    The kernel is the electric field from the **bare nuclear** Coulomb
    potential, ``E(r) = Z/r²`` (``V = −Z/r``): the eEDM signal is dominated
    by the near-nuclear region, where electronic screening vanishes and the
    electron sees the full nuclear charge ``Z``.  The ``Q ~ (Zα)/(2c)·P``
    scaling of the small component supplies the relativistic enhancement,
    so this integral reproduces Sandars' heavy-atom boost without any
    ad-hoc ``(Zα)²`` factor.

    Parameters
    ----------
    P_a, Q_a, P_r, Q_r : ndarray
        B-spline coefficients for the valence (a) and intermediate (r)
        orbitals.
    bsp_data : dict
        B-spline grid data (uses the ``gr`` / ``grw`` quadrature grid).
    Z : float
        Full nuclear charge (the near-nuclear field is unscreened).

    Returns
    -------
    float
        Radial matrix element of ``h_d`` in atomic units (a.u. of
        electric field).
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    P_a_r = _eval_P(P_a, bsp_data)
    Q_a_r = _eval_Q(Q_a, bsp_data)
    P_r_r = _eval_P(P_r, bsp_data)
    Q_r_r = _eval_Q(Q_r, bsp_data)
    # Electric-field magnitude from V = -Z/r:  |E| = |dV/dr| = Z / r^2.
    E_field = Z / (gr ** 2)
    # beta gamma^5 mixes large <-> small components with the relative
    # sign encoded in (P_r Q_a - Q_r P_a); cf. the velocity-gauge E1
    # and M1 cross integrals in transitions.multipole.
    integrand = E_field * (P_r_r * Q_a_r - Q_r_r * P_a_r)
    return float(np.sum(grw * integrand))


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
    l_a = l_of_kappa(kappa_a)
    l_r = l_of_kappa(kappa_r)

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
) -> SternheimerEDMResult:
    """Compute the electron-EDM enhancement factor R_e via Sternheimer SOS.

    Uses the existing V^{N-1} virtual basis and E1 transition machinery
    to evaluate the second-order perturbative sum

        R_e = 2 · occ_w · Σ_{a,r} ⟨a‖r·C¹‖r⟩ · ⟨r‖h_d‖a⟩ / (E_a − E_r),

    where ``h_d`` is the T,P-odd electron-EDM operator
    ``H_d / d_e = -β γ⁵ Σ·E`` evaluated as a large–small cross integral
    (see :func:`_tp_odd_radial_me`).  The ``Q ~ (Zα)/(2c) P`` scaling of
    the small component supplies the relativistic enhancement that makes
    heavy atoms (Cs, Tl, Fr) sensitive to ``d_e``.

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
        If given, compute R_e for only this subshell (e.g. ``"6p1/2"``).
        Otherwise sum over all open / partially-filled subshells.

    Notes
    -----
    The eEDM kernel uses the full (unscreened) nuclear charge
    ``scf_result.Z``, which is the physically correct field in the
    near-nuclear region that dominates the signal.

    Returns
    -------
    SternheimerEDMResult
    """
    t0 = time.time()

    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")

    Z = scf_result.Z if hasattr(scf_result, "Z") else None
    if Z is None:
        raise ValueError(
            "sternheimer_edm_enhancement requires scf_result.Z "
            "(nuclear charge) for the electron-EDM operator kernel."
        )

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)

    # ── Build virtual basis for every κ in scope ──────────────────────
    # ── Validate target_subshell early (before expensive basis construction) ──
    sh_by_label = {s.label: s for s in scf_result.subshells}

    if target_subshell is not None:
        if target_subshell not in sh_by_label:
            raise ValueError(
                f"Unknown target_subshell '{target_subshell}'. "
                f"Available: {list(sh_by_label.keys())}")
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
                max(scf_result.subshells, key=lambda s: (s.n, l_of_kappa(s.kappa))).label
            ]

    kappa_list, occ_basis, virt_basis = build_sos_virtual_basis(
        scf_result, cache=cache, kappa_max=kappa_max,
        n_virt_max=n_virt_max, e_virt_max=e_virt_max,
        potential_mode=potential_mode,
    )

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
                rad_tp = _tp_odd_radial_me(P_a, Q_a, P_r, Q_r, bsp_data, Z=float(Z))

                # R_e = 2 . occ_w . Sum_r <a||D||r> <r||h_d||a> / (E_a - E_r):
                #   <a||D||r>   = ang_e1 . rad_e1     (E1 reduced ME),
                #   <r||h_d||a> = ang_tp . rad_tp     (eEDM reduced ME; d_e
                #                                    already divided out in
                #                                    _tp_odd_radial_me).
                # The factor 2 symmetrises the second-order EDM (h.c. term);
                # there is no 1/3 isotropic average because h_d is a
                # pseudoscalar (rank 0) that fixes the dipole along the spin
                # axis -- cf. the (2/3) in the polarisability SOS.
                me_prod = ang_e1 * ang_tp * rad_e1 * rad_tp
                contrib = 2.0 * occ_weight * me_prod / omega_ra
                R_e_a += contrib

        per_orbital[lab_a] = R_e_a
        R_e_total += R_e_a

    elapsed = time.time() - t0

    # ── Compare with Sandars scaling ────────────────────────────────────
    from atomx.properties.edm.scaling import electron_edm_enhancement_sandars

    sandars_R = 0.0
    if Z is not None and target_shells:
        # Use the outermost target shell for Sandars comparison
        outer_lab = target_shells[-1]
        sh_out = sh_by_label.get(outer_lab)
        if sh_out is not None:
            n_eff = float(sh_out.n)
            J = (2 * abs(sh_out.kappa) - 1) / 2.0
            L = l_of_kappa(sh_out.kappa)
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
            "kappa_max": kappa_max,
            "n_virt_max": n_virt_max,
            "potential_mode": potential_mode,
            "target_subshells": target_shells,
            "kernel": "dirac_beta_gamma5_E",
            "n_virtuals": {k: len(v) for k, v in virt_basis.items()},
        },
    )



__all__ = [
    "SternheimerEDMResult",
    "sternheimer_edm_enhancement",
]
