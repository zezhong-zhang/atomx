"""Breit–Pauli perturbative operators (non-relativistic-limit reduction).

Closed-form expressions for the three leading ``(Zα)²`` corrections
to non-relativistic orbital energies:

* **Mass-velocity** — relativistic kinetic-energy shift from the
  ``−p⁴ / (8 m³ c²)`` term in the Foldy–Wouthuysen reduction.
* **Darwin** — zitterbewegung contact term ``(π Z α² / 2) δ³(r)``,
  non-zero only for s-orbitals.
* **Spin-orbit** — ``(α² / 2) · (1/r) (dV/dr) · L·S``; splits l > 0
  shells into fine-structure sub-levels ``j = l ± ½``.

These are *hydrogenic* point-nucleus expressions in ``Z_eff``; they
are the operators the :class:`~atomx.methods.BreitPauli` method
applies on top of a converged non-relativistic Hartree–Fock
calculation.  The formulae are the ones tabulated in Cowan,
*The Theory of Atomic Structure and Spectra*, §7.5.

Using per-orbital expectation values requires ``Z_eff`` (an effective
charge screened by the other electrons).  Cowan recommends using the
nuclear Z unless running a fully relativistic follow-up — see the
module docstring of :mod:`atomx.methods.relativistic_hf` for the
precise choice atomx makes.
"""

from __future__ import annotations

from atomx.constants import ALPHA

ALPHA_SQ = ALPHA * ALPHA


def mass_velocity_shift(n: int, l: int, Z_eff: float) -> float:
    """Mass-velocity hydrogenic shift for an (n, l) orbital.

    ``ΔE_MV = −α² · Z_eff⁴ / (2 n³) · [1 / (l + ½) − 3 / (4 n)]``.

    Returns
    -------
    float (Hartree) — always non-positive.
    """
    return -ALPHA_SQ * Z_eff ** 4 / (2.0 * n ** 3) * (
        1.0 / (l + 0.5) - 3.0 / (4.0 * n)
    )


def darwin_shift(n: int, l: int, Z_eff: float) -> float:
    """Darwin contact-term shift.  Non-zero only for ``l == 0``.

    ``ΔE_D = α² · Z_eff⁴ / (2 n³)`` for s-waves; 0 otherwise.
    """
    if l != 0:
        return 0.0
    return ALPHA_SQ * Z_eff ** 4 / (2.0 * n ** 3)


def spin_orbit_xi(n: int, l: int, Z_eff: float) -> float:
    """Spin-orbit radial factor ``ξ_{n,l}`` in Hartree.

    ``ξ = α² · Z_eff⁴ / [4 n³ l (l + ½)(l + 1)]``.

    The fine-structure splitting for the two ``j = l ± ½`` levels is
    then ``E_so_+ = +ξ · l/2`` and ``E_so_− = −ξ · (l+1)/2``.
    Returns 0 for ``l == 0`` (no spin-orbit).
    """
    if l == 0:
        return 0.0
    return ALPHA_SQ * Z_eff ** 4 / (4.0 * n ** 3 * l * (l + 0.5) * (l + 1))


def spin_orbit_shifts(l: int, xi: float) -> tuple[float, float]:
    """Return ``(E_so_+, E_so_−)`` for ``j = l + ½`` and ``j = l − ½``."""
    if l == 0:
        return (0.0, 0.0)
    return (+xi * l / 2.0, -xi * (l + 1) / 2.0)


__all__ = [
    "mass_velocity_shift",
    "darwin_shift",
    "spin_orbit_xi",
    "spin_orbit_shifts",
]
