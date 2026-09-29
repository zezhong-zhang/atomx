"""Flambaum–Ginges local radiative potential.

Reference
---------
V. V. Flambaum and J. S. M. Ginges,
    "The radiative potential method for calculations of QED radiative
    corrections to energy levels and electromagnetic amplitudes in
    many-electron atoms",
    Phys. Rev. A 72, 052115 (2005).  arXiv:physics/0507067.

Equations referenced in the code below correspond to the numbering in
the arXiv preprint.

Components implemented here
---------------------------
The full FG radiative potential (eq. 11) is

    Φ_rad(r) = Φ_U(r) + Φ_g(r) + Φ_f(r) + Φ_l(r) + (2/3) Φ_WC^simple(r)

This backend evaluates the **electric** pieces of the self-energy:

    V_SE_FG(r) = Φ_f(r) + Φ_l(r)

The Uehling piece Φ_U already lives in
:func:`atomx.corrections.qed.uehling_potential`.  The simplified
Wichmann–Kroll piece Φ_WC is a higher-order vacuum-polarization
correction and is exposed separately via
:func:`atomx.corrections.qedmod.flambaum_ginges.V_wichmann_kroll` so
that it can be added on the VP side rather than the SE side.

The **magnetic form factor** Φ_g (eq. 7) involves a non-local ``γ·∇``
operator; it contributes at the ~10 % level for high-n states but up to
~70 % for 1s at Z ≈ 10.  Its implementation is deferred — see the
comments at the bottom of this file for the derivation sketch.

Practical note: the :class:`~atomx.corrections.qedmod.shabaev.ShabaevLocalBackend`
(``"shabaev_local"``) provides a **different** decomposition that does
not require Φ_g separately — it implements the Shabaev–Tupitsyn–Yerokhin
model operator (local part), calibrated against Mohr's F(αZ), which
closes the 1s gap to ~3 % for Ne without needing the γ·∇ reduction.
Users wanting 1s precision should prefer ``"shabaev_local"`` over
``"flambaum_ginges"``.

All formulas below are written in atomic Hartree units (ℏ = m_e = e = 1,
c = 1/α).  Conversions from the paper's natural units (ℏ = c = 1,
m = 1/r_c):

    mc² (paper)  ->  1/α²  (Hartree)
    r_c (paper)  ->  α     (Bohr)
    Φ(r) = Ze/r  ->  Z/r   (Hartree)

Sign convention
---------------
The paper writes the *external scalar potential* ``Φ(r) = Z/r > 0`` and
defines the radiative potential ``Φ_rad`` as an additive correction to
that scalar potential.  atomx, by contrast, uses the electron's
*potential energy* convention ``V_nuc(r) = −Z/r < 0`` (directly what
enters the Dirac Hamiltonian).  Translation therefore involves a global
sign flip:

    V_atomx(r)  =  −Φ_paper(r)

Empirically verified against atomx's validated Uehling backend and
Mohr's tabulated SE.  Below every FG formula is written in atomx's
sign convention — i.e. already sign-flipped relative to the paper.

Accuracy
--------
Paper Table I: for H-like ions with Z = 10 … 110, the radiative
potential SE values for 5s, 5p₁/₂, 5p₃/₂ agree with the Mohr–Kim
tabulation to 0.1–3 %.  For 1s the paper warns only ~10 % accuracy
because the potential approximation overestimates short-range weight.
atomx therefore keeps Mohr as the default for 1s/2s/2p and recommends
FG for higher-n orbitals.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from atomx.constants import ALPHA, C
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.data.flambaum_ginges import (
    A_CUTOFF_COEFF,
    A_HIGH_FREQ_COEFFS,
    A_HIGH_FREQ_Z_REF,
    B_LOW_FREQ_CONST,
    B_LOW_FREQ_LINEAR,
    WK_DENOM_SCALE,
    WK_LEADING,
)

# Minimum peak amplitude of the SCF large-component P(r) below which the
# orbital is treated as numerically null (skips the FG evaluation).
_SCF_ORBITAL_NULL_THRESHOLD = 1e-12


# ---------------------------------------------------------------------------
# Coefficients and local building blocks
# ---------------------------------------------------------------------------

def _B_low(Z: int) -> float:
    """Low-frequency SE coefficient (FG eq. 9): B(Z) = b0 + b1 · Zα."""
    return B_LOW_FREQ_CONST + B_LOW_FREQ_LINEAR * ALPHA * Z


def _A_scale_high(Z: int) -> float:
    """High-frequency electric form-factor scale coefficient (FG eq. 10).

    ``A(Z,r) = A_scale(Z) · r / (r + A_CUTOFF_COEFF · Z² α²)``.

    ``A_scale`` is a polynomial in ``x = (Z − Z_ref) α`` whose
    coefficients live in :mod:`atomx.data.flambaum_ginges`.
    """
    a0, a2, a3, a4 = A_HIGH_FREQ_COEFFS
    x = (Z - A_HIGH_FREQ_Z_REF) * ALPHA
    return a0 + a2 * x * x + a3 * x ** 3 + a4 * x ** 4


# ---------------------------------------------------------------------------
# Radial potentials
# ---------------------------------------------------------------------------

def V_low_freq(r, Z: int) -> np.ndarray:
    """FG eq. 9 — low-frequency contribution  V_l(r) in Hartree (atomx sign).

        V_l(r) = +B(Z) · Z⁴ α³ · e^{−Z r}          (atomic units)

    (The paper writes −B(Z)... because it works in the unsigned-potential
    convention; atomx flips the sign as documented in the module docstring.)
    """
    r = np.asarray(r, dtype=float)
    return +_B_low(Z) * (Z ** 4) * (ALPHA ** 3) * np.exp(-Z * r)


def V_wichmann_kroll(r, Z: int) -> np.ndarray:
    """FG eq. 12 (with the (2/3) prefactor of eq. 11) — simplified
    Wichmann–Kroll higher-order vacuum polarization (atomx sign convention):

        V_WK(r) = +(4α/9π) · (Z/r) · 0.092 Z²α² / (1 + (1.62·c·r)⁴)

    WK is *opposite* sign to the Uehling VP (atomx Uehling is negative;
    WK is a small repulsive correction), in agreement with the
    known literature that WK partially cancels Uehling at short range
    for heavy elements.  The contribution is ~1 % of Uehling for Z > 80
    and negligible for light atoms.
    """
    r = np.asarray(r, dtype=float)
    cr = C * r
    numer = WK_LEADING * Z * Z * ALPHA * ALPHA
    denom = 1.0 + (WK_DENOM_SCALE * cr) ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        V = +(4.0 * ALPHA / (9.0 * np.pi)) * (Z / r) * numer / denom
    V = np.where(r > 0, V, 0.0)
    return V


def _high_freq_integrand(t: float, two_cr: float, log_arg: float) -> float:
    """Integrand of FG eq. 10 (without the −A(Z,r)·(α/π)·(Z/r) prefactor).

    ``I(t; 2cr, log_arg) = (1/√(t²−1)) ·
        [(1 − 1/(2t²))·(ln(t²−1) + 4·ln(1/(Zα) + 0.5)) − 3/2 + 1/t²] ·
        exp(−2 c r t)``.
    """
    if t <= 1.0:
        return 0.0
    t2 = t * t
    t2m1 = t2 - 1.0
    ln_t2m1 = np.log(t2m1)
    bracket = (1.0 - 0.5 / t2) * (ln_t2m1 + log_arg) - 1.5 + 1.0 / t2
    return bracket / np.sqrt(t2m1) * np.exp(-two_cr * t)


def V_high_freq(r, Z: int) -> np.ndarray:
    """FG eq. 10 — high-frequency electric form-factor contribution V_f(r).

    Evaluated via :func:`scipy.integrate.quad` for each radial point.
    Vectorised over any array shape (1-D, 2-D grid, …).
    """
    r_arr = np.asarray(r, dtype=float)
    out = np.zeros_like(r_arr)

    aZ = ALPHA * Z
    log_arg = 4.0 * np.log(1.0 / aZ + 0.5)   # constant in r
    A_scale = _A_scale_high(Z)
    cutoff = A_CUTOFF_COEFF * Z * Z * ALPHA * ALPHA    # A(Z,r) short-range cutoff

    flat_r = r_arr.reshape(-1)
    flat_V = out.reshape(-1)

    for i, ri in enumerate(flat_r):
        if ri <= 0.0:
            continue
        two_cr = 2.0 * C * ri
        A_of_r = A_scale * ri / (ri + cutoff)
        integral, _ = quad(
            _high_freq_integrand, 1.0, np.inf,
            args=(two_cr, log_arg),
            limit=100, epsabs=1e-12, epsrel=1e-8,
        )
        flat_V[i] = +A_of_r * (ALPHA / np.pi) * (Z / ri) * integral

    return out


def V_flambaum_ginges_electric(r, Z: int) -> np.ndarray:
    """Sum of electric FG self-energy pieces: ``V_l + V_f``."""
    return V_low_freq(r, Z) + V_high_freq(r, Z)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@register_backend
class FlambaumGingesBackend(SEBackend):
    """Flambaum–Ginges (PRA 72, 052115, 2005) local radiative potential.

    Evaluates ⟨a | V_l + V_f | a⟩ over the occupied orbitals.  The
    simplified Wichmann–Kroll piece is a *vacuum-polarization*
    contribution and is exposed separately.  The magnetic form-factor
    Φ_g (non-local γ·∇ operator, ~10 % effect on p-states) is deferred.
    """

    name = "flambaum_ginges"
    description = (
        "Flambaum-Ginges (PRA 72, 052115, 2005) local radiative "
        "potential: V_l (low-frequency) + V_f (high-frequency electric). "
        "Magnetic form-factor V_g not yet included."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        if bsp_data is None:
            raise ValueError(
                "Flambaum-Ginges backend requires bsp_data to evaluate "
                "⟨a|V_FG|a⟩ on the radial grid."
            )

        from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays

        gr, grw = workspace_arrays(bsp_data)

        # Build the local FG potential once on the grid — same for every
        # orbital, so we pay the quad cost only once.
        V_grid = V_flambaum_ginges_electric(gr, Z)

        per_orbital = {}
        E_SE = 0.0
        for P, Q, occ, _kappa, label in orbitals:
            if np.max(np.abs(P)) < _SCF_ORBITAL_NULL_THRESHOLD:
                per_orbital[label] = 0.0
                continue

            fP, fQ = eval_orbital_pair(P, Q, bsp_data)
            rho = fP * fP + fQ * fQ

            E_a = float(np.sum(grw * rho * V_grid))
            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)


# ---------------------------------------------------------------------------
# Deferred: magnetic form-factor Φ_g  (FG eq. 7)
# ---------------------------------------------------------------------------
# Φ_g(r) = (α/(4π m)) · i γ · ∇[ Φ(r) · ( ∫₁^∞ dt / (t² √(t² − 1)) · e^{−2trm} − 1 ) ]
#
# In the radial Dirac basis the operator i γ·∇ acting on a scalar V(r)
# couples the large and small components via the kappa quantum number.
# The diagonal radial matrix element reduces to
#
#     ⟨a | Φ_g | a⟩ = (α/(4π)) · ∫₀^∞ P_a(r) Q_a(r) · g_κ(r, Z) dr
#
# with g_κ containing the derivative of the Compton-length kernel and a
# κ-dependent prefactor.  To be added alongside the hyperfine B-constant
# calculation, which reuses the same radial machinery.
