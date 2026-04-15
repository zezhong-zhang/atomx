"""Flavour-aware orchestrator for a bundle of physics corrections.

:class:`CorrectionStack` composes several :class:`Correction` backends
and, for a given :class:`~atomx.results.AtomicResult`, runs each one
whose :meth:`Correction.supports` gate returns True.  Unsupported
backends are recorded under ``skipped`` with a one-line reason instead
of raising — this is what makes a single stack usable across
Schrödinger, DHF, DKS, MCDHF and shooting results.

Example
-------
>>> stack = CorrectionStack(
...     se_backend="shabaev_local",
...     include_breit=True,
...     include_vp=True,
... )
>>> applied = stack.apply(result)
>>> applied["self_energy[shabaev_local]"].energy_shift
-8.4e-4
>>> stack.total(applied)
-6.2e-3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from atomx.corrections.adapters import (
    BreitCorrection,
    NormalMassShiftCorrection,
    SelfEnergyCorrection,
    UehlingVPCorrection,
    WichmannKrollVPCorrection,
)
from atomx.corrections.protocol import Correction, CorrectionResult

if TYPE_CHECKING:
    from atomx.results import AtomicResult


@dataclass
class CorrectionStack:
    """Declarative bundle of physics corrections.

    Each flag toggles a canonical named correction on or off;
    :meth:`build` returns the concrete :class:`Correction` list, and
    :meth:`apply` runs each supported backend on the target result.

    Parameters
    ----------
    se_backend
        Self-energy backend name — any key from
        :func:`atomx.corrections.qedmod.available_backends`.  Pass
        ``None`` to disable the self-energy rung.  Default
        ``"shabaev_local"``.
    include_vp
        Include Uehling (second-order) vacuum polarisation.
    include_wk_vp
        Include Wichmann–Kroll higher-order VP.
    include_breit
        Include the full Breit interaction.
    nuclear_mass_me
        If set, apply the normal-mass-shift correction with the given
        nuclear mass in units of the electron mass.
    """

    se_backend: Optional[str] = "shabaev_local"
    include_vp: bool = True
    include_wk_vp: bool = False
    include_breit: bool = True
    nuclear_mass_me: Optional[float] = None

    _custom: List[Correction] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------

    def add(self, correction: Correction) -> "CorrectionStack":
        """Append an ad-hoc user-supplied :class:`Correction`.

        Returns ``self`` for chaining.
        """
        if not isinstance(correction, Correction):
            raise TypeError(
                f"{correction!r} does not satisfy the Correction protocol."
            )
        self._custom.append(correction)
        return self

    def build(self) -> List[Correction]:
        """Return the concrete list of :class:`Correction` backends."""
        backends: List[Correction] = []
        if self.se_backend is not None:
            backends.append(SelfEnergyCorrection(self.se_backend))
        if self.include_vp:
            backends.append(UehlingVPCorrection())
        if self.include_wk_vp:
            backends.append(WichmannKrollVPCorrection())
        if self.include_breit:
            backends.append(BreitCorrection())
        if self.nuclear_mass_me is not None:
            backends.append(NormalMassShiftCorrection(self.nuclear_mass_me))
        backends.extend(self._custom)
        return backends

    # ------------------------------------------------------------------

    def apply(self, result: "AtomicResult") -> Dict[str, CorrectionResult]:
        """Run every supported :class:`Correction` on ``result``.

        Returns a dict keyed on ``correction.name`` with one
        :class:`CorrectionResult` per backend that passed
        :meth:`Correction.supports`.  Skipped backends are recorded
        via the special key ``"_skipped"`` whose value is a
        ``{name: reason}`` dict, so a caller can see why a backend was
        silently omitted.
        """
        applied: Dict[str, CorrectionResult] = {}
        skipped: Dict[str, str] = {}
        for corr in self.build():
            if not corr.supports(result):
                skipped[corr.name] = self._explain_skip(corr, result)
                continue
            applied[corr.name] = corr.evaluate(result)

        if skipped:
            applied["_skipped"] = CorrectionResult(
                name="_skipped",
                energy_shift=0.0,
                provenance=dict(skipped),
            )
        return applied

    # ------------------------------------------------------------------

    @staticmethod
    def total(applied: Dict[str, CorrectionResult]) -> float:
        """Sum the energy shifts of applied corrections (excluding
        the bookkeeping ``_skipped`` entry)."""
        return sum(
            r.energy_shift for k, r in applied.items() if k != "_skipped"
        )

    @staticmethod
    def _explain_skip(corr: Correction, result: "AtomicResult") -> str:
        """Return a short human-readable reason the correction was skipped."""
        if getattr(result, "_bsp_data", None) is None:
            return f"{corr.name} requires a B-spline workspace (got none)."
        if getattr(result, "orbitals_raw", None) is None:
            return f"{corr.name} requires retained orbitals (got none)."
        if getattr(result, "Z", None) is None:
            return f"{corr.name} requires result.Z (got None)."
        return f"{corr.name} does not support this AtomicResult flavour."


__all__ = ["CorrectionStack"]
