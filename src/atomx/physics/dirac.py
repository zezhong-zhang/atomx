"""Relativistic Dirac equation — physics-level invariants + Sommerfeld formula.

The coupled radial Dirac equation for the large (P) and small (Q)
components reads::

    c (dP/dr + κ P/r)        =  (E + 2 c² − V) Q,
    c (−dQ/dr + κ Q/r)       =  (E − V)     P

in Hartree units with c = 1/α.  The κ = −(l+1) for j = l+½ and
κ = +l for j = l−½.  The closed-form point-nucleus eigenvalues (the
*Sommerfeld fine-structure formula*) serve as validation anchors for
every numerical Dirac solver in atomx.

This module owns the equation *object* plus the analytic
hydrogenic-Dirac energy expressions.  B-spline matrix assembly lives
in :mod:`atomx.numerics.bspline.dirac_assembly`; hydrogenic radial
wavefunctions live in :mod:`atomx.methods.hydrogenic_reference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from atomx.constants import ALPHA, C_SQ, C

# ---------------------------------------------------------------------------
# Equation object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dirac:
    """Two-component relativistic radial equation."""

    name: str = "dirac"
    is_relativistic: bool = True
    n_components: int = 2

    @property
    def has_small_component(self) -> bool:
        return True

    def eigenvalue_floor(self) -> float:
        """Dirac-sea / positive-energy boundary, ``−m_e c² = −c²`` in Hartree."""
        return -C_SQ

    def radial_action(self, P, Q, dP, dQ, r, potential, kappa, *, dilation_angle=0.0):
        """Rest-mass-subtracted h_D action on supplied radial values.

        The representation owns the derivatives and quadrature; this
        module owns the Dirac signs, kappa convention and rest energy.
        Returns (hP, hQ) in Hartree times the radial amplitude. It includes
        no exchange, electron Coulomb operator or boundary Bloch term.
        Complex wave amplitudes are retained. For uniform complex scaling,
        derivatives and 1/r couplings acquire exp(-i*theta); ``potential``
        must already be V(r*exp(i*theta)). The rest mass is never scaled.
        This action alone does not construct a no-pair projector or certify
        a many-electron resonance.
        """
        arrays = tuple(np.asarray(v) for v in (P, Q, dP, dQ, r, potential))
        P, Q, dP, dQ, r, potential = arrays
        if (
            any(v.shape != P.shape or not np.isfinite(v).all() for v in arrays)
            or np.iscomplexobj(r)
            or np.any(r <= 0)
            or not isinstance(kappa, (int, np.integer))
            or kappa == 0
        ):
            raise ValueError(
                "Dirac action requires aligned finite radial arrays and nonzero kappa"
            )
        if (np.iscomplexobj(dilation_angle) or not np.isscalar(dilation_angle)
                or not np.isfinite(dilation_angle) or abs(dilation_angle) >= np.pi / 2):
            raise ValueError("dilation_angle must be real, finite and |theta| < pi/2")
        scale = 1.0 if dilation_angle == 0 else np.exp(-1j * dilation_angle)
        return (
            potential * P + scale * C * (-dQ + kappa * Q / r),
            scale * C * (dP + kappa * P / r) + (potential - 2 * C_SQ) * Q,
        )

    def continuum_wave_number(self, energy_au: float) -> float:
        """Relativistic dispersion ``k = √(2E(1 + α²E/2))`` (Hartree, ``E > 0``).

        From ``(ck)² = E(E + 2c²)`` with ``E`` the kinetic energy — the
        wave number the two-component continuum spinors oscillate with
        asymptotically.  Reduces to the Schrödinger ``√(2E)`` as
        ``E/c² → 0``; at 200 keV the difference is ~9% in ``k``.
        """
        if energy_au <= 0.0:
            raise ValueError(
                f"continuum_wave_number requires E > 0, got {energy_au}"
            )
        return math.sqrt(2.0 * energy_au * (1.0 + 0.5 * energy_au / C_SQ))


# ---------------------------------------------------------------------------
# Sommerfeld fine-structure formula
# ---------------------------------------------------------------------------


def hydrogenic_dirac_coulomb_expectation(n: int, kappa: int, Z: float) -> float:
    """Exact point-Coulomb <−Z/r> in Hartree, from Z*dE_Dirac/dZ.

    With a=Z*alpha, gamma=sqrt(kappa²−a²), and N=n−|kappa|+gamma,
    the Hellmann–Feynman derivative is −Z²*(N+a²/gamma)/(N²+a²)^(3/2).
    In particular, 1s gives −Z²/gamma rather than its nonrelativistic limit.
    """
    ell = kappa if kappa > 0 else -kappa-1
    if kappa == 0 or n <= ell or not math.isfinite(Z) or Z <= 0 or abs(kappa) <= ALPHA*Z:
        raise ValueError('invalid bound point-Coulomb state')
    a2=(ALPHA*Z)**2
    gamma=math.sqrt(kappa*kappa-a2)
    principal=n-abs(kappa)+gamma
    return -Z*Z*(principal+a2/gamma)/(principal*principal+a2)**1.5


def hydrogenic_dirac_energy(n: int, kappa: int, Z: int) -> float:
    """Exact Dirac energy eigenvalue for a hydrogenic ion.

    ``E_{n,κ} = c² · {[1 + (αZ)² / (n_r + γ)²]^{−1/2} − 1}``

    with  ``γ = √(κ² − (αZ)²)`` and  ``n_r = n − |κ|``.

    Parameters
    ----------
    n : int
        Principal quantum number (``n ≥ 1``).
    kappa : int
        Dirac quantum number (``κ ≠ 0``): ``κ = −(l+1)`` for ``j = l+½``,
        ``κ = +l`` for ``j = l−½``.
    Z : int
        Nuclear charge.

    Returns
    -------
    E : float
        Binding energy in Hartree (negative for bound states).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if kappa == 0:
        raise ValueError("kappa cannot be zero")

    aZ = ALPHA * Z
    gamma = np.sqrt(kappa * kappa - aZ * aZ)
    n_r = n - abs(kappa)

    if n_r < 0:
        raise ValueError(
            f"Invalid quantum numbers: n={n}, kappa={kappa} gives n_r = {n_r} < 0"
        )

    denom = n_r + gamma
    ratio_sq = (aZ / denom) ** 2
    root = np.sqrt(1.0 + ratio_sq)
    # Rationalize the rest-energy subtraction, especially for weak binding.
    return -C_SQ * ratio_sq / (root * (root + 1.0))


def list_hydrogenic_states(n_max: int, Z: int):
    """Enumerate hydrogenic Dirac bound states up to ``n_max``.

    Returns a list of ``{n, l, j, kappa, energy_hartree}`` dicts.
    """
    states = []
    for n in range(1, n_max + 1):
        for l in range(n):
            # j = l + 1/2  →  κ = −(l+1)
            kappa_minus = -(l + 1)
            states.append({
                "n": n, "l": l, "j": l + 0.5, "kappa": kappa_minus,
                "energy_hartree": hydrogenic_dirac_energy(n, kappa_minus, Z),
            })
            # j = l − 1/2  →  κ = +l   (only l ≥ 1)
            if l >= 1:
                kappa_plus = l
                states.append({
                    "n": n, "l": l, "j": l - 0.5, "kappa": kappa_plus,
                    "energy_hartree": hydrogenic_dirac_energy(n, kappa_plus, Z),
                })
    return states


__all__ = [
    "Dirac",
    "hydrogenic_dirac_energy",
    "list_hydrogenic_states",
]
