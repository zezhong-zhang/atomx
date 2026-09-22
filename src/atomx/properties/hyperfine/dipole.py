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
* By default this module returns the bare Dirac result with ``g_s = 2``.
  The electron anomalous magnetic moment ``a_e = (g_s − 2)/2`` can be
  enabled via ``include_anomalous_moment=True``; it multiplies the
  result by ``(1 + a_e)`` with ``a_e = 1.15965218128e-3`` (CODATA).

  **Important — applying only a_e is usually incomplete.**
  The electron-anomaly correction is the first rung of a QED ladder
  whose other rungs (reduced mass, Breit/relativistic binding
  corrections, recoil, radiative corrections to the hyperfine operator
  itself, nuclear structure via Bohr–Weisskopf) tend to *cancel* its
  contribution for ground-state s-shells.  The canonical H 1s case is
  explicit (percentages relative to the 1421.27 MHz baseline):

      Dirac-Fock baseline (point nucleus, g_s=2, M=∞)  : 1421.27 MHz
      + reduced mass  (1 + m_e/m_p)^−3   (−0.163 %)    : 1418.95 MHz
      +  Schwinger a_e                   (+0.116 %)    : 1420.60 MHz
      +  higher-order QED + Breit        (≈ −0.010 %)  : 1420.45 MHz
      +  proton structure (Zemach/Bohr–Weisskopf −0.004 %) : 1420.40 MHz
                (experiment 1420.405751768 MHz; residual ≈ −6 ppm,
                 inside the ≈0.1 % accuracy of the DHF baseline)

  The **reduced-mass term is the dominant omission** in this module's
  number: the Dirac–Fock baseline here uses the *free* electron mass and
  an infinitely heavy point nucleus, while the physical HFS scales as
  ``(1 + m_e/M)^−3`` (Fermi energy ∝ m_r³, the electron magnetic moment
  ∝ 1/m_e) — −1632 ppm, i.e. −2.3 MHz for H 1s.  With that single term
  the 1421.27 MHz baseline lands at 1418.95 MHz, and the observed
  1420.405751768 MHz is recovered only after the anomaly (above) and the
  ≈ −40 ppm nuclear-structure term.  Adding ``include_anomalous_moment``
  *without* the reduced-mass term therefore overshoots by ≈ +0.12 %;
  atomx defaults to ``include_anomalous_moment=False`` so the bare
  Dirac-Fock result is reported, with individual rungs opt-in.  The
  caller-level ladder that *does* apply it is
  ``SCFResult.hyperfine_constants(include_finite_nuclear_mass=True)``
  (rung ``(M/(M+m_e))³``, methods/scf_observables.py).
  See Antognini, Lin & Meißner, arXiv:2208.04025, Eqs. (2) and (7)
  (E_F(H) = 1418.840082(9) kHz, defined with the reduced mass m_r³);
  review 2026-09-19 finding M9.
  A future release will bundle the full package as a single flag
  ``include_qed_hyperfine=True``.
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

import numpy as np

from atomx.constants import (
    ALPHA,
    HARTREE_J,
    PROTON_ELECTRON_MASS_RATIO,
)
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

# Unit conversions ---------------------------------------------------------

# Planck constant in SI (J·s) — needed to convert Hartree → frequency units.
_H_PLANCK_J_S = 6.62607015e-34

#: Conversion factor from Hartree to MHz, i.e. ``E_Ha × HARTREE_MHZ = f_MHz``.
#: ``f = E/h`` with E in J, h in J·s.
HARTREE_MHZ = HARTREE_J / _H_PLANCK_J_S / 1.0e6

#: Electron anomalous magnetic moment ``a_e = (g_s - 2)/2``.
#:
#: CODATA 2018 recommended value.  Leading-order = α/(2π) ≈ 1.16140e-3
#: (Schwinger 1948); higher QED loops bring it to 1.15965218128e-3.
#: Multiplies the Dirac hyperfine constant by ``(1 + a_e)``.
ELECTRON_ANOMALOUS_MOMENT = 1.15965218128e-3


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
    *,
    include_anomalous_moment: bool = False,
) -> float:
    """Magnetic dipole hyperfine constant A in Hartree atomic units.

    Implements eq. (1) above, with optional electron g-factor anomaly:

        A = (1 + a_e) · α · g_I · (m_e/m_p) · [ κ / (j(j+1)) ] · ⟨P Q / r²⟩

    with ``j = |κ| − 1/2`` and ``a_e = 1.15965218e-3`` (CODATA).  The
    Dirac prefactor without ``(1 + a_e)`` follows Johnson, *Atomic
    Structure Theory*, eq. 8.67 (Springer, 2007); the anomaly is the
    Schwinger 1948 correction plus higher QED loops.

    Error budget (review 2026-09-19 finding M9): this function carries
    **no reduced-mass term** — the radial integral comes from an
    electron-mass, infinitely heavy point-nucleus wavefunction, whereas
    the physical HFS scales as ``(1 + m_e/M)^−3`` (−1632 ppm for H,
    −2.3 MHz; the dominant omitted term).  For H 1s the returned
    1421.272852 MHz therefore sits +610 ppm above the measured
    1420.405751768 MHz; see the module docstring for the full ladder
    (reduced mass −0.163 %, then a_e +0.116 %, then ≈ −0.04 % QED +
    nuclear structure).

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
    include_anomalous_moment : bool, keyword-only
        When ``True`` multiply by ``(1 + a_e)``.  Default ``False`` so
        that the return value is the pure Dirac-Fock baseline — see the
        module docstring for why isolated application of a_e is
        typically incomplete.

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

    A = ALPHA * g_I * (1.0 / PROTON_ELECTRON_MASS_RATIO) * j_fac * rad
    if include_anomalous_moment:
        A *= (1.0 + ELECTRON_ANOMALOUS_MOMENT)
    return A


def hyperfine_A_MHz(
    P,
    Q,
    bsp_data,
    kappa: int,
    g_I: float,
    *,
    include_anomalous_moment: bool = False,
) -> float:
    """Same as :func:`hyperfine_A`, converted to MHz (≡ A/h, ``h`` Planck)."""
    return hyperfine_A(
        P, Q, bsp_data, kappa, g_I,
        include_anomalous_moment=include_anomalous_moment,
    ) * HARTREE_MHZ
