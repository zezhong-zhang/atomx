"""Born-approximation electron-impact excitation cross section.

Closed-form helpers covering the high-energy plane-wave Born limit
of inelastic electron scattering, equivalent to FAC's ``SetCEBorn``
mode.  Replaces the more expensive distorted-wave partial-wave sum
when E ≫ ΔE — typically accurate to 10–20 % above 5 × threshold.

Approach
--------
The high-E Bethe-Born collision strength for a dipole-allowed
transition (a → b) is (Bethe 1930; van Regemorter 1962):

    Ω_dipole(E) ≈ (8 / √3) · g_a · f_ab · ⟨ḡ⟩(E/ΔE)

where f_ab is the absorption oscillator strength of the line,
ḡ ≈ 0.2–0.3 is the effective Gaunt factor, and g_a is the lower-state
weight 2 J_a + 1.  The cross section follows from the standard relation

    σ(E) = π / k_a²  ·  Ω(E) / g_a

with k_a² = 2 E in atomic units (non-relativistic).

For higher-multipole or forbidden lines, the Bethe limit reverts to
σ(E) = const·1/E with no log-term; we expose that via
:func:`born_excitation_forbidden`.

Reference
---------
* H. Bethe, *Ann. Phys.* **5**, 325 (1930).
* H. van Regemorter, *Astrophys. J.* **136**, 906 (1962).
* M. F. Gu, *Astrophys. J.* **579**, 1077 (2002) — FAC ``SetCEBorn``.
"""

from __future__ import annotations

import math

import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV

_A0_CM2 = (BOHR_RADIUS_M * 100.0) ** 2  # a₀² in cm²


def born_excitation_dipole(
    energies_eV: np.ndarray,
    f_ab: float,
    delta_E_eV: float,
    *,
    g_lower: int = 1,
    gaunt_factor: float = 0.2,
) -> np.ndarray:
    """Plane-wave Born σ for a dipole-allowed E1 excitation.

    σ(E) = (8 π² · a₀² · f_ab / (√3 · ΔE)) · ḡ · ln(E / E_th)
                                        / (E / ΔE)             (E > E_th)

    with E_th = ΔE the threshold energy and ḡ the effective Gaunt
    factor (van Regemorter 1962 quotes ḡ ≈ 0.2 for ions, 0.3 for
    neutrals).

    Parameters
    ----------
    energies_eV : array_like
        Incident electron energies in eV.
    f_ab : float
        Absorption oscillator strength (dimensionless, computed from
        :func:`atomx.properties.transitions.rates.oscillator_strength`).
    delta_E_eV : float
        Excitation energy E_ab in eV.
    g_lower : int
        Statistical weight 2 J_a + 1 of the lower state. Default 1.
    gaunt_factor : float
        Effective ḡ factor; 0.2 for ions, 0.3 for neutrals.

    Returns
    -------
    σ_cm2 : np.ndarray
        Cross section in cm².
    """
    if f_ab < 0:
        raise ValueError("born_excitation_dipole: f_ab must be >= 0")
    if delta_E_eV <= 0:
        raise ValueError("born_excitation_dipole: ΔE must be > 0")
    E = np.asarray(energies_eV, dtype=float)
    out = np.zeros_like(E)
    above = delta_E_eV < E
    if not np.any(above):
        return out
    Ee = E[above]
    # van Regemorter ALN form: σ = 8π²·a₀² · f_ab · ḡ · ln(E/ΔE) / (√3 · E)
    sigma = (8.0 * math.pi ** 2 * _A0_CM2 * f_ab * gaunt_factor
             * np.log(Ee / delta_E_eV)
             / (math.sqrt(3.0) * (Ee / HARTREE_EV)))
    sigma = np.maximum(sigma, 0.0)
    out[above] = sigma
    return out


def born_excitation_forbidden(
    energies_eV: np.ndarray,
    line_strength_au: float,
    delta_E_eV: float,
    L: int = 2,
    *,
    g_lower: int = 1,
) -> np.ndarray:
    """Plane-wave Born σ for an L-pole forbidden excitation.

    σ(E) = (4 π · a₀² / E) · (S / (2 L + 1) · (ΔE)^(2L−2))   (very high E,
                                                              non-rel)

    More accurate at E ≫ ΔE for M1, E2, M2.  Falls off as 1/E without
    the log term that dipole-allowed transitions enjoy.

    Parameters
    ----------
    energies_eV : array_like
    line_strength_au : float
        Squared reduced matrix element S = |⟨f‖r^L C^L‖i⟩|² in atomic
        units (a₀^{2L}).
    delta_E_eV : float
    L : int
        Multipole rank ≥ 2 (E2 / M1 / M2 / E3 / …).
    g_lower : int
        Statistical weight of the lower state.

    Returns
    -------
    σ_cm2 : np.ndarray
    """
    if L < 1:
        raise ValueError("born_excitation_forbidden: L must be >= 1")
    if line_strength_au < 0:
        raise ValueError("line_strength_au must be >= 0")
    if delta_E_eV <= 0:
        raise ValueError("delta_E_eV must be > 0")
    E = np.asarray(energies_eV, dtype=float)
    out = np.zeros_like(E)
    above = delta_E_eV < E
    if not np.any(above):
        return out
    Ee_au = E[above] / HARTREE_EV
    dE_au = delta_E_eV / HARTREE_EV
    sigma_a0 = (4.0 * math.pi / Ee_au
                * (line_strength_au / (2 * L + 1))
                * dE_au ** (2 * L - 2))
    out[above] = sigma_a0 * _A0_CM2
    return out


# ---------------------------------------------------------------------------
# Exact relativistic plane-wave Born from generalized-oscillator-strength
# moments (FAC excitation.c:CERadialQkBorn + radial.c:GeneralizedMoments).
# Unlike the Gaunt-factor helpers above, this uses the actual SCF orbitals
# and is exact in the Born limit — it anchors the high-energy completion of
# the distorted-wave sum.
# ---------------------------------------------------------------------------

from scipy.special import spherical_jn  # noqa: E402

ALPHA2 = 1.0 / 137.0359990836 ** 2

_N_Q_MASTER = 256
_N_Q_INT = 128


def gos_moment_table(rho, r, dr, k, q_grid, overlap=None):
    """M_k(q) = [Int rho(r) j_k(q r) dr - delta_{k0} S_ab] / q.

    ``rho = P_a P_b + Q_a Q_b`` is the bound transition density on grid
    ``r`` with weights ``dr``; ``overlap`` (the k = 0 subtraction,
    FAC radial.c:7683) defaults to Int rho dr, which removes the spurious
    monopole when the two SCF solves are not exactly orthogonal.  Only the
    support of rho is integrated.
    """
    m = float(np.max(np.abs(rho)))
    if m == 0.0:
        return np.zeros_like(np.asarray(q_grid, dtype=float))
    idx = np.nonzero(np.abs(rho) > 1e-16 * m)[0]
    i_sup = int(idx[-1]) + 1
    rs = r[:i_sup]
    w = (rho * dr)[:i_sup]
    s0 = (float(np.sum(w)) if overlap is None else float(overlap)) if k == 0 else 0.0
    q_arr = np.asarray(q_grid, dtype=float)
    out = np.empty(len(q_arr))
    for i, q in enumerate(q_arr):
        jk = spherical_jn(k, q * rs)
        out[i] = (float(np.dot(w, jk)) - s0) / q
    return out


class BornTransition:
    """Precomputed GOS tables for one CE transition; evaluates Omega^B(E).

    The collision strength of a one-electron jj transition set in the first
    Born approximation, derived from sigma = (8 pi / k_i^2) Int dq/q^3
    |eps(q)|^2 with the plane-wave expansion and CG orthogonality killing
    the multipole cross terms:

        Omega^B(E) = 8 sum_pairs pop sum_k (2k+1) <j_a||C^k||j_b>^2
                     Int_{c0-c1}^{c0+c1} dln q  M_k(q)^2  R_rel(q)

    with relativistic momenta c_i = sqrt(2 E_i (1 + alpha^2 E_i / 2)) and
    R_rel the free-electron kinematic factor of CERadialQkBorn (-> 1 as
    alpha -> 0).  The dipole term reproduces the exact Bethe limit
    Omega -> (8/3) S ln[(k_i+k_f)/(k_i-k_f)] with line strength
    S = <j_a||C^1||j_b>^2 D^2 (tested).  ``pop`` is the same
    average-configuration factor as the DW assembly
    (``_active_transition_pairs``: q_a (g_b - q_b)/g_b).

    ``pair_data`` is the fine-grid list of
    ``(kappa_a, kappa_b, pop, P_a, Q_a, P_b, Q_b)`` built by
    ``dw_excitation_cross_section``.
    """

    def __init__(self, pair_data, r_fine, dr_fine, dE_au, E_max_au,
                 L_max=None):
        from atomx.angular.coefficients import reduced_C
        from atomx.scattering.distorted_wave.solver import _allowed_k

        self.dE_au = float(dE_au)
        E_top = max(float(E_max_au), 2.0 * self.dE_au)
        p_top = math.sqrt(2.0 * E_top * (1.0 + 0.5 * ALPHA2 * E_top))
        q_lo = 1e-3 * math.sqrt(2.0 * self.dE_au)
        q_hi = 50.0 * (2.0 * p_top)
        self._lnq = np.linspace(math.log(q_lo), math.log(q_hi), _N_Q_MASTER)
        q_master = np.exp(self._lnq)

        self._channels = []
        for (ka, kb, pop, Pa, Qa, Pb, Qb) in pair_data:
            rho = Pa * Pb + Qa * Qb
            ks = _allowed_k(ka, kb)
            for k in (ks if L_max is None
                      else [k for k in ks if k <= L_max]):
                cl = reduced_C(ka, kb, k)
                if cl == 0.0:
                    continue
                mk = gos_moment_table(rho, r_fine, dr_fine, k, q_master)
                wgt = 8.0 * float(pop) * (2.0 * k + 1.0) * cl * cl
                self._channels.append((wgt, mk))

    def omega(self, E_inc_au: float, dE_au: float = None) -> float:
        """Omega^B at incident energy ``E_inc_au`` (0 below threshold)."""
        dE = self.dE_au if dE_au is None else float(dE_au)
        E1 = float(E_inc_au) - dE
        if E1 <= 0.0 or not self._channels:
            return 0.0
        E0 = float(E_inc_au)
        a0 = 1.0 + 0.5 * ALPHA2 * E0
        a1 = 1.0 + 0.5 * ALPHA2 * E1
        c0 = math.sqrt(2.0 * E0 * a0)
        c1 = math.sqrt(2.0 * E1 * a1)
        lnq = np.linspace(math.log(max(c0 - c1, 1e-30)),
                          math.log(c0 + c1), _N_Q_INT)
        q = np.exp(lnq)
        # relativistic kinematic factor (CERadialQkBorn, m <= 0 branch)
        a = a0 * a1
        d = 0.25 * ALPHA2 / a
        c = 0.5 * (c0 * c0 + c1 * c1 - q * q)
        h = (1.0 - c / (c0 * c1)) * (c0 * c1 * d) ** 2
        rel = a * ((1.0 + d * c) ** 2 + h)
        total = 0.0
        for wgt, mk in self._channels:
            m_q = np.interp(lnq, self._lnq, mk)
            total += wgt * float(np.trapezoid(m_q * m_q * rel, lnq))
        return total


def born_blend_weights(omega_dw, tails, p_born=0.5):
    """FAC's _topup = 2 Born blend weight per energy.

    w = (tail / Omega)^p_born, clamped monotone non-decreasing with
    energy by processing energies in descending order and capping each
    weight at the value found for the next-higher energy
    (excitation.c:1331-1333, with the descending-ie loop of
    CERadialQkTable).  Arrays must be sorted by ascending energy.
    """
    omega_dw = np.asarray(omega_dw, dtype=float)
    tails = np.asarray(tails, dtype=float)
    n = len(omega_dw)
    w = np.zeros(n)
    cap = None
    for i in range(n - 1, -1, -1):
        if omega_dw[i] > 0.0 and tails[i] > 0.0:
            s = min(abs(tails[i] / omega_dw[i]) ** p_born, 1.0)
        else:
            s = 0.0
        if cap is not None and s > cap:
            s = cap
        else:
            cap = s
        w[i] = s
    return w


__all__ = [
    "born_excitation_dipole",
    "born_excitation_forbidden",
    "BornTransition",
    "born_blend_weights",
    "gos_moment_table",
]
