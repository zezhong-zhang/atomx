"""Electron–electron interaction kernels (physics definitions).

This module collects the *operator definitions* for two-electron
interactions:

* **Coulomb**:  ``1 / r_{12}``  → Slater–Condon multipole expansion
  with radial integrals ``R^k(a,b; c,d)``.
* **Breit (magnetic + retardation)**:  from the Pauli-equation
  reduction of the photon propagator, decomposed by Grant & Pyper
  (J. Phys. B 9, 761, 1976) into angular coefficients S(1)…S(8)
  at ranks ``ν = L−1, L, L+1``.
* **Magnetic Gaunt**:  the ``−α_1·α_2 / r_{12}`` piece alone — the
  low-frequency Breit limit.
* **Retardation**:  the remainder of Breit after the Gaunt term.

Only angular / operator-level metadata lives here; the numerical
radial integral machinery stays in :mod:`atomx.numerics.bspline`
(cell-based Slater convolutions) and :mod:`atomx.corrections.breit`
(the current Breit evaluator).  Consolidating the radial machinery
is tracked for a follow-up PR; the purpose of this module today is
to provide a single import point for angular-coefficient helpers
when new callers (MBPT, CC) come online.

Re-exports
----------
:func:`atomx.corrections.breit._breit_smu_coefficients`
    The 8 Breit angular coefficients S(1)…S(8) at rank ν for a
    given quadruple (κ_a, κ_b, κ_c, κ_d) and multipole L.
"""

from __future__ import annotations

from atomx.corrections.breit import _breit_smu_coefficients as breit_angular_coefficients  # noqa: F401


__all__ = ["breit_angular_coefficients"]
