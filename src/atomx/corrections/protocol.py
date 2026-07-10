"""Unified protocol for physics corrections (QED, Breit, mass-shift, …).

This module defines the shared interface that every named correction in
:mod:`atomx.corrections` and adjacent physics modules should implement.
Corrections are strategies: given an :class:`~atomx.results.AtomicResult`
they return a :class:`CorrectionResult` describing the energy shift, any
per-orbital breakdown, and provenance metadata.

Design goals
------------
1. **Flavor-awareness via** :meth:`Correction.supports`.  Each concrete
   backend declares whether it can act on a given result (e.g. it needs
   Dirac small components, or a B-spline workspace, or a hydrogenic
   reference that can be projected).  :class:`CorrectionStack` uses
   this gate to skip backends that do not apply rather than raise.

2. **No side-effects on input**.  A correction reads from the result
   and returns a :class:`CorrectionResult`; it never mutates solver
   state.  This makes corrections safe to cache, compose, and compare.

3. **Provenance first**.  Each result carries a ``provenance`` dict
   naming the data table, paper, or fallback rule that fed the shift.
   This is what lets a user see, at a glance, which corrections were
   applied and which were silently skipped (and why).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atomx.results import AtomicResult


@dataclass
class CorrectionResult:
    """Outcome of applying a single named correction to a result.

    Attributes
    ----------
    name
        Short identifier of the correction ("uehling_vp", "breit", …).
    energy_shift
        Total energy shift in Hartree (sum over orbitals, weighted by
        occupation where applicable).
    per_orbital
        Optional ``{label: shift_per_electron}`` breakdown.  May be
        empty for corrections that only produce a total.
    provenance
        Free-form metadata for traceability: which reference,
        which fallback, which data table was used.  Callers can
        include keys like ``"reference"``, ``"fallback"``, ``"data"``.
    """

    name: str
    energy_shift: float
    per_orbital: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Correction(Protocol):
    """Common interface for a single named physics correction.

    Concrete implementations live across :mod:`atomx.corrections`
    (Breit, VP, SE backends) and :mod:`atomx.properties.isotope_shift` (NMS, SMS).
    They share this shape so that :class:`CorrectionStack` can compose
    them uniformly.
    """

    #: Short registration name.
    name: str

    #: Human-readable one-line description.
    description: str

    def supports(self, result: "AtomicResult") -> bool:
        """Return whether this correction can act on ``result``.

        Implementations answer "yes" only when every prerequisite is
        met: relativistic small components for Breit/SE, a B-spline
        workspace for integral-based backends, orbitals whose quantum
        numbers intersect the backend's table, and so on.  The stack
        skips (rather than fails) for unsupported results, so it is
        safe to query many backends and take whichever apply.
        """

    def evaluate(self, result: "AtomicResult", **options: Any) -> CorrectionResult:
        """Compute the correction energy shift for ``result``.

        Parameters
        ----------
        result
            Converged :class:`~atomx.results.AtomicResult` (any flavor —
            Schrödinger/DHF/DKS/MCDHF/Shooting — provided
            :meth:`supports` returned True).
        options
            Backend-specific switches (e.g. whether to include an
            overlap correction, the name of an auxiliary reference).

        Returns
        -------
        CorrectionResult
        """


__all__ = ["Correction", "CorrectionResult"]
