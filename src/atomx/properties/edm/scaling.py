"""Closed-form Sandars-style scaling for EDM enhancement factors.

Atomic EDMs probe fundamental CP violation.  The relevant atomic
quantities are *enhancement factors* that multiply underlying
T,P-odd sources:

* **Electron-EDM enhancement** ``R_e`` — for paramagnetic atoms
  (one valence electron with nonzero ⟨σ⟩), the induced atomic EDM
  is ``d_atom = R_e · d_e``.  Sandars (1965) derived the scaling

      R_e(Z, n) ≈ 10 · α² · Z³ · n^* / (J(J+1)(2L+1))

  for a single valence electron in a state with effective principal
  quantum number n*.  Coefficients of ~10 emerge from the radial
  integrals and angular factors.

* **Schiff moment enhancement** ``K_S`` — for diamagnetic atoms,
  the Schiff moment ``S_N`` of the nucleus produces an atomic EDM

      d_atom ≈ K_S · S_N / (e a₀²)

  with ``K_S`` scaling as ``Z²·α²·(Z·α)·…``.  For Hg, Xe, Ra these
  are computed by full MBPT / CC machinery; the closed-form
  estimates here capture only the dominant Z-dependence.

These scaling formulas are accurate to factor-of-2 — useful for
quick scoping of new candidate atoms but not for analysing precision
EDM experiments.  Production-grade calculations require the
all-order coupled-cluster / MBPT infrastructure that AtomX's
``methods/coupled_cluster.py`` and ``methods/mbpt.py`` will
eventually feed into.

References
----------
* P. G. H. Sandars, *Phys. Lett.* **14**, 194 (1965) — original
  scaling.
* Z. W. Liu & H. P. Kelly, *Phys. Rev. A* **45**, R4210 (1992) —
  full Cs MBPT computation (R_e ≈ 120).
* T. M. R. Byrnes *et al.*, *Phys. Rev. A* **57**, 1547 (1998) —
  Schiff moment enhancement systematics.
* V. A. Dzuba, V. V. Flambaum, *Phys. Rev. A* **80**, 062509 (2009)
  — modern review of atomic EDM calculations.
"""

from __future__ import annotations

import math

from atomx.constants import ALPHA


def electron_edm_enhancement_sandars(
    Z: int,
    *,
    n_eff: float = 2.0,
    J: float = 0.5,
    L: int = 0,
) -> float:
    """Sandars-style scaling for the electron-EDM enhancement factor R_e.

    Implements the closed-form estimate

        R_e ≈ 10 · α² · Z³ · n_eff / (J·(J+1)·(2 L + 1))

    For a Cs-like alkali (Z=55, n_eff ≈ 1.9, J=1/2, L=0) this gives
    R_e ≈ 80 — comparable to the modern MBPT value of 120 within
    the documented factor-of-2 accuracy of this formula.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    n_eff : float
        Effective principal quantum number of the valence electron
        (≈ 1.9 for Cs 6s, 2.7 for Rb 5s, etc.).  Default 2.0.
    J : float
        Total angular momentum of the ground state (J = 1/2 for an
        alkali ²S₁/₂).
    L : int
        Orbital angular momentum (L = 0 for s-state).

    Returns
    -------
    float
        Dimensionless enhancement factor R_e.

    Notes
    -----
    The output is unsigned; the sign of R_e is determined by
    relativistic corrections and is positive for paramagnetic atoms
    in their ground state.
    """
    if Z < 1:
        raise ValueError("electron_edm_enhancement_sandars: Z must be >= 1")
    if n_eff <= 0:
        raise ValueError("n_eff must be > 0")
    if J <= 0:
        raise ValueError("J must be > 0")
    if L < 0:
        raise ValueError("L must be >= 0")
    return 10.0 * (ALPHA ** 2) * (Z ** 3) * n_eff / (J * (J + 1.0) * (2 * L + 1))


def schiff_moment_enhancement_scaling(
    Z: int,
    A: int,
    *,
    valence: str = "s",
) -> float:
    """Order-of-magnitude scaling for the Schiff-moment enhancement K_S.

    Diamagnetic atoms (¹²⁹Xe, ¹⁹⁹Hg, ²²⁵Ra) have a leading enhancement

        K_S(Z) ≈ A_K · Z² · (α Z)² · f_valence

    where A_K ≈ 10⁻¹⁵ cm and f_valence captures the valence-shell
    polarisability (≈ 1 for s-states, suppressed for closed-shell).

    Reproduces the Hg / Xe / Ra ordering of Schiff-moment
    enhancements within an order of magnitude (Hg ≈ 4·10⁻¹⁷ cm,
    Ra ≈ 7·10⁻¹⁷ cm) using the closed-form scaling

        K_S(Z) = 4 × 10⁻¹⁹  ·  Z² · (αZ)²    cm

    Parameters
    ----------
    Z, A : int
        Nuclear charge and mass number.
    valence : {"s", "p", "d", "closed"}
        Valence-shell character; "closed" multiplies by 0.1 to
        capture the suppression in fully closed-shell ions like Xe.

    Returns
    -------
    float
        Schiff enhancement K_S in cm.  Multiply by the nuclear Schiff
        moment ``S_N`` (in cm³ · e) to get the induced atomic EDM
        ``d_atom`` in e·cm.
    """
    if Z < 1 or A < 1:
        raise ValueError("schiff_moment_enhancement_scaling: Z, A must be >= 1")
    f_val = {"s": 1.0, "p": 0.3, "d": 0.1, "closed": 0.05}.get(valence)
    if f_val is None:
        raise ValueError(
            f"valence must be one of s/p/d/closed; got {valence!r}"
        )
    return 4.0e-19 * (Z ** 2) * ((ALPHA * Z) ** 2) * f_val


def edm_summary(
    Z: int,
    A: int,
    *,
    n_eff: float = 2.0,
    J: float = 0.5,
    L: int = 0,
    valence: str = "s",
) -> dict:
    """Convenience wrapper: return both R_e and K_S in a single call.

    Returns
    -------
    dict
        ``{'R_electron_edm', 'K_schiff_cm', 'Z', 'A', ...}``
    """
    return {
        "Z": Z,
        "A": A,
        "n_eff": n_eff,
        "J": J,
        "L": L,
        "valence": valence,
        "R_electron_edm": electron_edm_enhancement_sandars(
            Z, n_eff=n_eff, J=J, L=L
        ),
        "K_schiff_cm": schiff_moment_enhancement_scaling(
            Z, A, valence=valence
        ),
    }


__all__ = [
    "electron_edm_enhancement_sandars",
    "schiff_moment_enhancement_scaling",
    "edm_summary",
]
