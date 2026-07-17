"""The :class:`Equation` protocol.

Every concrete equation (Schrödinger, Dirac, Breit-Pauli-corrected
Schrödinger, …) advertises a small set of invariant properties that
numerical representations and methods can dispatch on without
knowing which equation they were given.

The protocol is intentionally minimal: only facts that are needed to
*decide* code paths live here, not anything that a particular
representation would compute.  For example, whether the equation has
a small component (and thus needs a P,Q block structure) is a
flag here; the matrix elements themselves live in
:mod:`atomx.numerics`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Equation(Protocol):
    """Minimal invariants of a radial atomic equation.

    Attributes
    ----------
    name
        Short identifier ("schrodinger", "dirac", …).
    is_relativistic
        ``True`` for equations that carry an explicit ``c`` and a
        small radial component.
    n_components
        1 for a single radial function (Schrödinger), 2 for the
        coupled P / Q pair of the Dirac equation.
    has_small_component
        Alias of ``n_components == 2`` used at call sites that read
        more naturally.
    """

    name: str
    is_relativistic: bool
    n_components: int

    @property
    def has_small_component(self) -> bool: ...

    def eigenvalue_floor(self) -> float:
        """Lower edge of the physical spectrum in Hartree.

        Equals ``0`` for the Schrödinger equation and ``−m_e c² = −c²``
        (in atomic units with c = 1/α) for the Dirac equation — the
        Dirac-sea / positive-energy boundary.
        """
        ...

    def continuum_wave_number(self, energy_au: float) -> float:
        """Asymptotic wave number ``k`` of a free continuum wave at
        kinetic energy ``E`` (Hartree, ``E > 0``).

        Non-relativistic dispersion gives ``k = √(2E)``; the Dirac
        equation gives the relativistic ``k = √(2E(1 + α²E/2))``
        (equivalently ``(cℏk)² = E(E + 2mc²)``), which the two-component
        continuum waves oscillate with asymptotically.
        """
        ...


__all__ = ["Equation"]
