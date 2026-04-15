"""Electric quadrupole hyperfine constant B.

For a single-electron Dirac orbital (n, κ, j = |κ| − 1/2) interacting
with a nucleus of spectroscopic quadrupole moment Q_nuc (barns), the
electric-quadrupole hyperfine constant is

    B = −Q_nuc · [ (2j − 1) / (2j + 2) ] · ⟨a | r⁻³ | a⟩_Dirac        (1)

with the Dirac radial integral

    ⟨a | r⁻³ | a⟩_Dirac = ∫₀^∞ [ P(r)² + Q(r)² ] / r³ dr              (2)

Reference: W. R. Johnson, *Atomic Structure Theory*, §8.8 (Springer,
2007); G. K. Woodgate, *Elementary Atomic Structure*, §9.6.  The
``(2j − 1)/(2j + 2)`` prefactor is the ``⟨j m=j | (3m_j² − j(j+1)) |
j m=j⟩`` Landé-like factor evaluated for the stretched state.

Selection rule
--------------
B vanishes identically for j = 1/2 orbitals (the ``(2j − 1)/(2j + 2)``
factor is zero), so ``hyperfine_B`` returns 0 for κ = ±1 regardless of
the orbital.  Non-trivial values arise only for p₃/₂, d₃/₂, d₅/₂, …

Sign and unit conventions
-------------------------
Q_nuc is supplied in barns (1 b = 10⁻²⁴ cm²); internally converted to
Bohr² via ``BARN_BOHR2``.  The integral ⟨r⁻³⟩ is in atomic units
(Bohr⁻³).  With the sign in eq. (1), a positive (oblate) nuclear Q
with a positive ⟨r⁻³⟩ gives a negative B, in agreement with the common
convention where B > 0 corresponds to a prolate nucleus probed by a
cigar-shaped electron density.

Accuracy notes
--------------
A single-configuration Dirac–Fock calculation typically underpredicts
experimental B values for alkali valence p-orbitals by 20-40 %, the
same core-polarization deficit that affects magnetic-dipole A.
Users can apply a ``correlation_factor`` on the hyperfine ladder to
account for this phenomenologically — see
:meth:`atomx.methods.dirac_fock.SCFResult.hyperfine_constants`.
"""

from __future__ import annotations

import numpy as np

from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
from atomx.constants import BOHR_RADIUS_M
from atomx.prop.hyperfine.dipole import HARTREE_MHZ


# 1 barn = 1e-24 cm² in Bohr² — exact with CODATA Bohr radius
# 1 Bohr = BOHR_RADIUS_M  in metres;  Bohr² = BOHR_RADIUS_M² m²
# so  1 barn  =  1e-28 m² / BOHR_RADIUS_M² m²  Bohr²
_BOHR_CM = BOHR_RADIUS_M * 100.0     # Bohr in cm  ≈ 5.29177e-9 cm
#: Conversion barns → Bohr².  ``Q_Bohr2 = Q_barns × BARN_BOHR2``.
BARN_BOHR2 = 1.0e-24 / (_BOHR_CM * _BOHR_CM)   # ≈ 3.572e-8 Bohr²/barn


# ---------------------------------------------------------------------------
# Radial integral ⟨r⁻³⟩
# ---------------------------------------------------------------------------

def radial_r_inv_cubed(P, Q, bsp_data) -> float:
    """Compute ⟨r⁻³⟩ = ∫ [P² + Q²] / r³ dr for one Dirac orbital.

    Parameters
    ----------
    P, Q : ndarray
        B-spline coefficient vectors of the large / small Dirac
        components.
    bsp_data : dict
        B-spline workspace (same as used throughout atomx).

    Returns
    -------
    float
        ``⟨r⁻³⟩`` in atomic units (Bohr⁻³).
    """
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    nv = bsp_data["nv"]

    fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
    fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
    rho = fP * fP + fQ * fQ

    return float(np.sum(grw * rho / (gr * gr * gr)))


# ---------------------------------------------------------------------------
# Electric quadrupole hyperfine constant B
# ---------------------------------------------------------------------------

def _j_factor(kappa: int) -> float:
    """Return (2j − 1) / (2j + 2) for Dirac quantum number κ.

    Zero for j = 1/2 (κ = ±1).
    """
    j = abs(kappa) - 0.5
    if j < 1.0 - 1e-12:
        return 0.0
    return (2.0 * j - 1.0) / (2.0 * j + 2.0)


def hyperfine_B(
    P,
    Q,
    bsp_data,
    kappa: int,
    Q_nuc_barns: float,
) -> float:
    """Electric quadrupole hyperfine constant B in Hartree atomic units.

    Implements eq. (1) above.  For κ = ±1 orbitals (j = 1/2) the
    angular factor vanishes and the function returns exactly zero,
    regardless of the radial integral.

    Parameters
    ----------
    P, Q : ndarray
        B-spline coefficients of the large / small Dirac components.
    bsp_data : dict
        B-spline workspace.
    kappa : int
        Dirac quantum number.
    Q_nuc_barns : float
        Nuclear electric quadrupole moment in barns (10⁻²⁴ cm²).  Pass
        the signed spectroscopic value.

    Returns
    -------
    float
        ``B`` in Hartree.  Multiply by :data:`~atomx.prop.hyperfine.dipole.HARTREE_MHZ`
        (imported here for convenience) to get MHz.
    """
    if kappa == 0:
        raise ValueError("Dirac quantum number κ cannot be zero.")

    ang = _j_factor(kappa)
    if ang == 0.0:
        return 0.0

    r_m3 = radial_r_inv_cubed(P, Q, bsp_data)
    Q_au = Q_nuc_barns * BARN_BOHR2

    return -Q_au * ang * r_m3


def hyperfine_B_MHz(
    P,
    Q,
    bsp_data,
    kappa: int,
    Q_nuc_barns: float,
) -> float:
    """Same as :func:`hyperfine_B`, converted to MHz (≡ B/h)."""
    return hyperfine_B(P, Q, bsp_data, kappa, Q_nuc_barns) * HARTREE_MHZ
