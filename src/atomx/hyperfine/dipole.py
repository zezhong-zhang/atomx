"""Magnetic dipole hyperfine constant A.

For a single-electron Dirac orbital with quantum numbers (n, κ, j), the
magnetic dipole hyperfine interaction with a point nuclear magnetic moment
is

    ⟨H_HF⟩ = A · I · J,

where the hyperfine constant A (in Hartree atomic units) reads

    A = α · g_I · (m_e / m_p) · [ κ / (j(j+1)) ] · ⟨P Q / r²⟩          (1)

with the radial integral

    ⟨P Q / r²⟩ = ∫₀^∞ P(r) · Q(r) / r² dr                             (2)

taken over the large (P) and small (Q) Dirac components.  The derivation
is standard — see, e.g.

    W. R. Johnson, *Atomic Structure Theory*, §8.7 (Springer, 2007).
    I. Lindgren & A. Rosén, Case Studies in Atomic Physics 4, 93 (1974).

Conventions
-----------
* The electron g-factor is fixed to the Dirac value g_s = 2 — anomalous
  magnetic-moment corrections (g_s/2 − 1 ≈ α/(2π)) are of order 0.1 % and
  are deferred to a later iteration.
* ``g_I = μ_I / I`` in units of the nuclear magneton μ_N; :mod:`atomx.
  hyperfine.nuclear_moments` tabulates μ_I for common isotopes.
* Point magnetic moment.  Finite-nuclear-magnetization (Bohr–Weisskopf)
  corrections are at the sub-percent level for light elements and are not
  yet included.

Sign & unit sanity check
------------------------
For a ground-state hydrogen atom (Z=1, 1s₁/₂, κ=−1, j=1/2, g_I=+5.5857)
the integral ⟨P Q / r²⟩ is negative (with the standard phase convention
Q ∝ −α · P × (sign factor)), so κ · ⟨P Q / r²⟩ > 0 and A > 0.  The
expected numerical value is ≈ 1420 MHz — this is the 21 cm hyperfine
splitting that gives rise to the HI line in astrophysics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from atomx.bspline.evaluate import eval_orbital_on_grid
from atomx.constants import (
    ALPHA,
    HARTREE_J,
    PROTON_ELECTRON_MASS_RATIO,
)


# Unit conversions ---------------------------------------------------------

# Planck constant in SI (J·s) — needed to convert Hartree → frequency units.
_H_PLANCK_J_S = 6.62607015e-34

#: Conversion factor from Hartree to MHz, i.e. ``E_Ha × HARTREE_MHZ = f_MHz``.
#: ``f = E/h`` with E in J, h in J·s.
HARTREE_MHZ = HARTREE_J / _H_PLANCK_J_S / 1.0e6


# ---------------------------------------------------------------------------
# Radial integral  ∫ P(r) Q(r) / r^2 dr
# ---------------------------------------------------------------------------

def hyperfine_radial_integral(P, Q, bsp_data) -> float:
    """Hyperfine radial integral ⟨P Q / r²⟩.

    Parameters
    ----------
    P, Q : ndarray
        B-spline coefficient vectors of the large/small Dirac components
        of a single orbital.
    bsp_data : dict
        B-spline workspace, as supplied by :class:`DFSolver`.

    Returns
    -------
    float
        ``∫₀^∞ P(r) Q(r) / r² dr`` evaluated on the quadrature grid.
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

    return float(np.sum(grw * fP * fQ / (gr * gr)))


# ---------------------------------------------------------------------------
# Magnetic dipole hyperfine constant A
# ---------------------------------------------------------------------------

def hyperfine_A(
    P,
    Q,
    bsp_data,
    kappa: int,
    g_I: float,
) -> float:
    """Magnetic dipole hyperfine constant A in Hartree atomic units.

    Implements eq. (1) above:

        A = α g_I (m_e/m_p) · [ κ / (j(j+1)) ] · ⟨P Q / r²⟩

    with ``j = |κ| − 1/2``.  The prefactor follows Johnson, *Atomic
    Structure Theory*, eq. 8.67 (Springer, 2007).

    Parameters
    ----------
    P, Q : ndarray
        B-spline coefficients of the large/small Dirac components.
    bsp_data : dict
        B-spline workspace.
    kappa : int
        Dirac quantum number of the orbital.
    g_I : float
        Nuclear g-factor in nuclear-magneton units (``μ_I / I``).

    Returns
    -------
    float
        ``A`` in Hartree.  Multiply by :data:`HARTREE_MHZ` to obtain MHz.
    """
    if kappa == 0:
        raise ValueError("Dirac quantum number κ cannot be zero.")

    j = abs(kappa) - 0.5
    j_fac = kappa / (j * (j + 1))

    rad = hyperfine_radial_integral(P, Q, bsp_data)

    return ALPHA * g_I * (1.0 / PROTON_ELECTRON_MASS_RATIO) * j_fac * rad


def hyperfine_A_MHz(
    P,
    Q,
    bsp_data,
    kappa: int,
    g_I: float,
) -> float:
    """Same as :func:`hyperfine_A`, converted to MHz (≡ A/h, ``h`` Planck)."""
    return hyperfine_A(P, Q, bsp_data, kappa, g_I) * HARTREE_MHZ
