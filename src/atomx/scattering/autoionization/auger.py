"""Auger (autoionization) rate calculator.

For a doubly-excited state |d⟩ = |a, b⟩ (two electrons above the core)
at energy E_d above the singly-ionized ground state, the Auger rate
into channel |f; εκ_f⟩ = |bound final, continuum electron⟩ is

    Γ_Auger(d → f; εκ)  =  2π |⟨f; εκ | V_ee | d⟩|²                  (1)

with continuum orbital normalised per unit energy.  In jj-coupling the
matrix element expands over Slater integrals R^k(a,b,c,d) between two
bound orbitals and one continuum orbital:

    ⟨f; εκ | V_ee | a, b⟩
        =  Σ_k  W_k(κ_a, κ_b, κ_f, κ_ε)  ·  R^k(a, b, f, εκ)          (2)

where W_k is the angular-coupling coefficient and

    R^k(a, b, c, d)
        =  ∫∫ dr₁ dr₂  [P_a(r₁)P_c(r₁) + Q_a(r₁)Q_c(r₁)]
                        · r_<^k / r_>^(k+1)
                        · [P_b(r₂)P_d(r₂) + Q_b(r₂)Q_d(r₂)]           (3)

is the Slater radial integral.

Simplification for this release
-------------------------------
The full angular-momentum coupling is involved (see Grant 2007 §8.1).
As a **calibrated practical helper** this module ships

    auger_rate_two_electron(...)
        Accepts the orbitals and a precomputed angular coefficient
        (``W_k`` or its sum) and returns Γ_Auger in atomic units.

For most production uses the angular coefficients are the same as for
two-electron Coulomb matrix elements in MCDHF (see
:mod:`atomx.methods.mcdhf.matrix`); users can compute them there and pass them
in.  Validated against expected scalings; absolute cross-checks vs. FAC
are planned follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
from atomx.constants import ALPHA


@dataclass
class AutoionizationState:
    """A single autoionizing (doubly-excited) resonance.

    Attributes
    ----------
    label : str
        Identifier, e.g. ``"1s 2s 2p"`` or ``"2p^5 3d^2"``.
    E_resonance_eV : float
        Energy of the resonance *above* the singly-ionized ground
        state (i.e. the energy a free electron would need to excite
        into this level via dielectronic capture).
    J : float
        Total angular momentum of the resonance.
    gamma_auger_per_s : float
        Total Auger width (all ionization channels), in s⁻¹.
    a_radiative_per_s : float
        Total radiative decay width (all photon channels), in s⁻¹.
    channels : dict, optional
        Per-channel breakdown ``{final_state_label: Γ_per_s}``.
    """

    label: str
    E_resonance_eV: float
    J: float
    gamma_auger_per_s: float
    a_radiative_per_s: float
    channels: Dict[str, float] = field(default_factory=dict)

    @property
    def g(self) -> float:
        """Statistical weight ``2J + 1``."""
        return 2.0 * self.J + 1.0

    @property
    def total_width_per_s(self) -> float:
        """Γ_total = Γ_Auger + A_rad."""
        return self.gamma_auger_per_s + self.a_radiative_per_s

    @property
    def fluorescence_yield(self) -> float:
        """``Γ_r / (Γ_r + Γ_a)`` — probability of stabilisation via photon."""
        w = self.total_width_per_s
        if w <= 0.0:
            return 0.0
        return self.a_radiative_per_s / w


# ---------------------------------------------------------------------------
# Auger rate from a single Slater integral
# ---------------------------------------------------------------------------

def _slater_integral(
    P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d, k: int, bsp_data,
) -> float:
    """Compute the Slater radial integral R^k(a, b, c, d) on the B-spline grid.

    Direct O(N²) evaluation on the Gauss grid.  Not the most efficient
    method (the Poisson-solver convolution in atomx.methods.mcdhf.matrix is
    preferred for large-scale work), but transparent and sufficient
    for single-Auger-rate evaluation.
    """
    nv = bsp_data["nv"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    def _ev(coef, small=False):
        if small:
            return eval_orbital_on_grid(coef, bsp_q, nv, ksq, nsq)
        return eval_orbital_on_grid(coef, bsp_p, nv, ksp, nsp)

    fPa, fQa = _ev(P_a), _ev(Q_a, small=True)
    fPb, fQb = _ev(P_b), _ev(Q_b, small=True)
    fPc, fQc = _ev(P_c), _ev(Q_c, small=True)
    fPd, fQd = _ev(P_d), _ev(Q_d, small=True)

    rho13 = fPa * fPc + fQa * fQc     # density (a,c) at r₁
    rho24 = fPb * fPd + fQb * fQd     # density (b,d) at r₂

    # Flatten (nv, n_gauss) → 1-D for double integration
    r = gr.reshape(-1)
    w = grw.reshape(-1)
    rho13 = rho13.reshape(-1)
    rho24 = rho24.reshape(-1)

    # R^k = ∫ dr₁ dr₂ ρ13(r₁) ρ24(r₂) r_<^k / r_>^(k+1)
    # Vectorised: broadcast r₁ and r₂.
    r1 = r[:, None]
    r2 = r[None, :]
    r_less = np.minimum(r1, r2)
    r_greater = np.maximum(r1, r2)
    kernel = r_less ** k / r_greater ** (k + 1)

    integrand = rho13[:, None] * rho24[None, :] * kernel
    return float(np.einsum("i,j,ij->", w, w, integrand))


def auger_rate_two_electron(
    P_a, Q_a, P_b, Q_b,
    P_c, Q_c,
    P_eps, Q_eps,
    *,
    angular_W: float,
    k: int,
    bsp_data,
) -> float:
    """Auger rate for the single-channel two-electron process

        |a, b⟩   →   |c; ε_continuum⟩                                 (4)

    in atomic units of rate (Hartree/ℏ = 4.1341e16 s⁻¹).

    The user supplies the angular-coupling coefficient ``angular_W``
    and multipole rank ``k``; the radial Slater integral ``R^k`` is
    computed here from the four radial wavefunctions.

        Γ_Auger  =  2π · (angular_W)² · |R^k(a,b,c,ε)|²              (5)

    Parameters
    ----------
    P_a, Q_a, P_b, Q_b : ndarray
        Large/small components of the two initial bound orbitals.
    P_c, Q_c : ndarray
        Final bound orbital (c becomes the residual singly-bound state).
    P_eps, Q_eps : ndarray
        Continuum orbital (energy ε, kappa κ_ε; normalised per unit
        energy).  Obtain via :func:`atomx.numerics.bspline.continuum.solve_continuum`.
    angular_W : float
        Pre-computed angular coefficient ``W_k(κ_a, κ_b, κ_c, κ_ε)``.
    k : int
        Slater multipole rank.
    bsp_data : dict
        B-spline workspace.

    Returns
    -------
    float
        Auger rate in atomic units (convert to s⁻¹ with 4.1341e16).
    """
    R_k = _slater_integral(
        P_a, Q_a, P_b, Q_b,
        P_c, Q_c, P_eps, Q_eps,
        k=k, bsp_data=bsp_data,
    )
    return 2.0 * np.pi * (angular_W ** 2) * (R_k ** 2)
