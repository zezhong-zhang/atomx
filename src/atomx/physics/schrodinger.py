"""Non-relativistic Schrödinger equation invariants.

The radial Schrödinger equation in Hartree units reads

    [-½ u'' + l(l+1)/(2r²) u + V(r) u]  =  E u,   u(r) = r R(r).

This module defines the :class:`Schrodinger` equation object —
a tiny dataclass that tells the rest of the code how to dispatch.
Numerical matrix elements and ODE integrators live in
:mod:`atomx.numerics.bspline` and :mod:`atomx.numerics.shooting`
respectively.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schrodinger:
    """Single-component non-relativistic radial equation."""

    name: str = "schrodinger"
    is_relativistic: bool = False
    n_components: int = 1

    @property
    def has_small_component(self) -> bool:
        return False

    def eigenvalue_floor(self) -> float:
        """Physical lower bound on the spectrum (``0`` in atomic units)."""
        return 0.0


__all__ = ["Schrodinger"]
