"""Auger (autoionization) rate calculator.

For a doubly-excited state |d> = |a, b> (two electrons above the core)
at energy E_d above the singly-ionized ground state, the Auger rate
into channel |f; eps*kappa_f> = |bound final, continuum electron> is

    Gamma_Auger(d -> f; eps*kappa)  =  2pi |<f; eps*kappa | V_ee | d>|^2  (1)

with continuum orbital normalised per unit energy.  In jj-coupling the
matrix element expands over Slater integrals R^k(a,b,c,d) between two
bound orbitals and one continuum orbital:

    <f; eps*kappa | V_ee | a, b>
        =  Sum_k  W_k(kappa_a, kappa_b, kappa_f, kappa_eps)  ·  R^k(a, b, f, eps*kappa)  (2)

where W_k is the angular-coupling coefficient and

    R^k(a, b, c, d)
        =  int int dr1 dr2  [P_a(r1)P_c(r1) + Q_a(r1)Q_c(r1)]
                        · r_<^k / r_>^(k+1)
                        · [P_b(r2)P_d(r2) + Q_b(r2)Q_d(r2)]           (3)

is the Slater radial integral.

Simplification for this release
-------------------------------
The full angular-momentum coupling is involved (see Grant 2007 sec.8.1).
As a **calibrated practical helper** this module ships

    auger_rate_two_electron(...)
        Accepts the orbitals and a precomputed angular coefficient
        (``W_k`` or its sum) and returns Gamma_Auger in atomic units.

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
import numba as nb

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
        Total Auger width (all ionization channels), in s^-1.
    a_radiative_per_s : float
        Total radiative decay width (all photon channels), in s^-1.
    channels : dict, optional
        Per-channel breakdown ``{final_state_label: Gamma_per_s}``.
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
        """Gamma_total = Gamma_Auger + A_rad."""
        return self.gamma_auger_per_s + self.a_radiative_per_s

    @property
    def fluorescence_yield(self) -> float:
        """``Gamma_r / (Gamma_r + Gamma_a)`` — probability of stabilisation via photon."""
        w = self.total_width_per_s
        if w <= 0.0:
            return 0.0
        return self.a_radiative_per_s / w


# ---------------------------------------------------------------------------
# Numba-compiled Slater integral kernel
# ---------------------------------------------------------------------------


@nb.njit(cache=True, fastmath=True)
def _slater_double_integral(
    rho13: np.ndarray, rho24: np.ndarray,
    r: np.ndarray, w: np.ndarray, k: int,
) -> float:
    """Numba-compiled O(N^2) double integral for R^k Slater.

    Computes  sum_{i,j} w[i] * w[j] * rho13[i] * rho24[j] * kernel[i,j]
    where kernel[i,j] = r_min^k / r_max^(k+1).
    """
    n = r.shape[0]
    total = 0.0
    for i in range(n):
        ri = r[i]
        rho13_i = rho13[i]
        wi = w[i]
        for j in range(n):
            rj = r[j]
            if ri <= rj:
                rk_val = ri ** k / rj ** (k + 1)
            else:
                rk_val = rj ** k / ri ** (k + 1)
            total += wi * w[j] * rho13_i * rho24[j] * rk_val
    return total


# ---------------------------------------------------------------------------
# Auger rate from a single Slater integral
# ---------------------------------------------------------------------------

def _slater_integral(
    P_a, Q_a, P_b, Q_b, P_c, Q_c, P_d, Q_d, k: int, bsp_data,
) -> float:
    """Compute the Slater radial integral R^k(a, b, c, d) on the B-spline grid.

    Direct O(N^2) evaluation on the Gauss grid using the numba-compiled
    double-integral kernel for the inner loop.
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

    rho13 = (fPa * fPc + fQa * fQc).reshape(-1)
    rho24 = (fPb * fPd + fQb * fQd).reshape(-1)

    r = gr.reshape(-1)
    w = grw.reshape(-1)

    return _slater_double_integral(rho13, rho24, r, w, k)


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

        |a, b>   ->   |c; eps_continuum>                                 (4)

    in atomic units of rate (Hartree/hbar = 4.1341e16 s^-1).

    The user supplies the angular-coupling coefficient ``angular_W``
    and multipole rank ``k``; the radial Slater integral ``R^k`` is
    computed here from the four radial wavefunctions.

        Gamma_Auger  =  2pi * (angular_W)^2 * |R^k(a,b,c,eps)|^2        (5)

    Parameters
    ----------
    P_a, Q_a, P_b, Q_b : ndarray
        Large/small components of the two initial bound orbitals.
    P_c, Q_c : ndarray
        Final bound orbital (c becomes the residual singly-bound state).
    P_eps, Q_eps : ndarray
        Continuum orbital (energy eps, kappa kappa_eps; normalised per unit
        energy).  Obtain via :func:`atomx.numerics.bspline.continuum.solve_continuum`.
    angular_W : float
        Pre-computed angular coefficient ``W_k(kappa_a, kappa_b, kappa_c, kappa_eps)``.
    k : int
        Slater multipole rank.
    bsp_data : dict
        B-spline workspace.

    Returns
    -------
    float
        Auger rate in atomic units (convert to s^-1 with 4.1341e16).
    """
    R_k = _slater_integral(
        P_a, Q_a, P_b, Q_b,
        P_c, Q_c, P_eps, Q_eps,
        k=k, bsp_data=bsp_data,
    )
    return 2.0 * np.pi * (angular_W ** 2) * (R_k ** 2)
