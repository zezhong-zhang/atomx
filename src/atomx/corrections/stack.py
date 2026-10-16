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

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from atomx.corrections.adapters import (
    BreitCorrection,
    HydrogenicMassRemainderCorrection,
    NormalMassShiftCorrection,
    SelfEnergyCorrection,
    SpecificMassShiftCorrection,
    UehlingVPCorrection,
    WichmannKrollVPCorrection,
)
from atomx.corrections.protocol import (
    Correction,
    CorrectionApplication,
    CorrectionResult,
)

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
        Include static Breit or the selected finite-frequency interaction.
    breit_mode
        ``"dbsr_hf"`` (default) applies the full static Breit
        configuration average; ``"operator"`` applies the full static Breit matrix
        element including static retardation;
        ``"scalar_legacy"`` opts into the approximate configuration
        average used by earlier DHF ledgers.
    breit_frequency
        ``"static"`` (default) or ``"orbital_differences"``. The latter
        replaces static Breit with the real transverse operator averaged at
        the two vertex energy differences, retaining the same ensemble/CI
        objective. Legacy scalar modes cannot supply this operator.
    breit_orbital_energies
        Optional ``(n, kappa) -> Hartree`` mapping. Ordinary DHF can use its
        converged orbital energies; CSF methods require an explicit map.
        Frequencies are fixed during CI; orbital relaxation is not included.
    nuclear_mass_me
        If set, apply the normal-mass-shift correction with the given
        nuclear mass in units of the electron mass.
    nms_mode
        ``"operator"`` (default) evaluates the one-body Breit recoil
        operator. ``"virial"`` explicitly selects the historical estimate.
    include_sms
        Add first-order two-electron Breit recoil. Requires nuclear_mass_me;
        disabled by default while the all-element generation is revalidated.
    include_hydrogenic_mass_remainder
        Add the exact nonrelativistic one-electron reduced-mass remainder
        beyond first order. Requires operator NMS and a nuclear mass.
        Zero for multielectron DHF; unsupported for MCDHF correlation labels.
        Does not supply higher-order relativistic or radiative recoil.
    strict
        Require every requested correction to support the result before
        evaluating any. Defaults to False for existing exploratory callers;
        production calculations can require complete correction coverage.
    """

    se_backend: Optional[str] = "shabaev_local"
    include_vp: bool = True
    include_wk_vp: bool = False
    include_breit: bool = True
    nuclear_mass_me: Optional[float] = None
    breit_mode: str = "dbsr_hf"
    breit_frequency: str = "static"
    breit_orbital_energies: Optional[dict] = None
    nms_mode: str = "operator"
    include_sms: bool = False
    include_hydrogenic_mass_remainder: bool = False
    strict: bool = False

    _custom: List[Correction] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.breit_frequency not in ('static', 'orbital_differences'):
            raise ValueError("breit_frequency must be 'static' or 'orbital_differences'")
        if self.breit_frequency != 'static':
            if not self.include_breit:
                raise ValueError('finite-frequency Breit requires include_breit=True')
            if self.breit_mode not in ('dbsr_hf', 'operator'):
                raise ValueError('finite-frequency Breit cannot use a legacy scalar mode')
        if self.breit_orbital_energies is not None:
            if self.breit_frequency == 'static':
                raise ValueError('breit_orbital_energies requires orbital_differences frequency')
            from atomx.physics.csf.transverse import validate_orbital_energies
            self.breit_orbital_energies = validate_orbital_energies(
                self.breit_orbital_energies, self.breit_orbital_energies)
        if self.include_sms and self.nuclear_mass_me is None:
            raise ValueError("include_sms requires nuclear_mass_me")
        if self.include_hydrogenic_mass_remainder and (
                self.nuclear_mass_me is None or self.nms_mode != 'operator'):
            raise ValueError('hydrogenic mass remainder requires nuclear_mass_me and operator NMS')

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
            backends.append(BreitCorrection(mode=self.breit_mode,
                frequency=self.breit_frequency, orbital_energies=self.breit_orbital_energies))
        if self.nuclear_mass_me is not None:
            backends.append(NormalMassShiftCorrection(self.nuclear_mass_me, self.nms_mode))
        if self.include_sms:
            backends.append(SpecificMassShiftCorrection(self.nuclear_mass_me))
        if self.include_hydrogenic_mass_remainder:
            backends.append(HydrogenicMassRemainderCorrection(self.nuclear_mass_me))
        backends.extend(self._custom)
        names = [backend.name for backend in backends]
        if any(not isinstance(name, str) or not name or name.startswith('_') for name in names):
            raise ValueError('correction names must be nonempty and cannot start with an underscore')
        if len(names) != len(set(names)):
            raise ValueError('duplicate correction names would overwrite an energy contribution')
        return backends

    # ------------------------------------------------------------------

    def validate_addition(self, result):
        """Reject operators already present before adding anything to energies.

        ``apply`` remains a read-only evaluator for diagnostic comparisons.
        Finalization uses this gate before calling any correction backend.
        """
        backends = self.build()
        operators = [corr.name.split('[', 1)[0] for corr in backends]
        repeated = sorted({name for name in operators if operators.count(name) > 1})
        if repeated:
            raise ValueError('overlapping operator requests: ' + ', '.join(repeated))
        # A different evaluator for the same operator is still an overlap.
        # Only explicit application records establish inclusion; requested
        # correction flags and zero-valued energy components do not.
        included = dict(getattr(result, 'operator_stages', ()))
        ledger = getattr(result, 'correction_application', None)
        if ledger is not None:
            included.update((name.split('[', 1)[0], ledger.stage)
                            for name, _ in ledger.shifts)
        overlaps = [(corr.name, included[corr.name.split('[', 1)[0]])
                    for corr in backends if corr.name.split('[', 1)[0] in included]
        if overlaps:
            raise ValueError('requested operators already included: ' + '; '.join(
                f'{name} at {stage}' for name, stage in overlaps))

    def apply(self, result: "AtomicResult", *, strict: Optional[bool] = None) -> Dict[str, CorrectionResult]:
        """Run every supported :class:`Correction` on ``result``.

        Returns a dict keyed on ``correction.name`` with one
        :class:`CorrectionResult` per backend that passed
        :meth:`Correction.supports`.  Skipped backends are recorded
        via the special key ``"_skipped"`` whose CorrectionResult.provenance
        contains ``{name: reason}``. With strict=True (or self.strict), all
        requested backends must be supported before any is evaluated.
        """
        applied: Dict[str, CorrectionResult] = {}
        skipped: Dict[str, str] = {}
        supported = []
        backends = self.build()
        for corr in backends:
            if not corr.supports(result):
                skipped[corr.name] = self._explain_skip(corr, result)
                continue
            supported.append(corr)
        if (self.strict if strict is None else strict) and skipped:
            raise ValueError('unsupported requested corrections: ' + '; '.join(
                f'{name}: {reason}' for name, reason in skipped.items()))
        for corr in supported:
            value = corr.evaluate(result)
            if not isinstance(value, CorrectionResult) or value.name != corr.name:
                raise ValueError(f'{corr.name} returned an inconsistent correction result')
            if (not math.isfinite(value.energy_shift)
                    or any(not math.isfinite(v) for v in value.per_orbital.values())):
                raise ValueError(f'{corr.name} returned a non-finite energy correction')
            applied[corr.name] = value

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
        if getattr(result, "_equation", "dirac") != "dirac":
            return (f"{corr.name} requires Dirac orbitals "
                    f"(got equation={result._equation!r}).")
        if (getattr(result, "_bsp_data", None) is None
                and getattr(result, "_grid", None) is None):
            return (f"{corr.name} requires a B-spline workspace or "
                    f"grid-tabulated orbitals (got neither).")
        if (getattr(result, "orbitals_raw", None) is None
                and getattr(result, "_pa", None) is None):
            return f"{corr.name} requires retained orbitals (got none)."
        if getattr(result, "Z", None) is None:
            return f"{corr.name} requires result.Z (got None)."
        return f"{corr.name} does not support this AtomicResult flavour."


def finalize_with_corrections(result: "AtomicResult",
                              stack: Optional["CorrectionStack"]):
    """Apply ``stack`` to a converged result — the single post-SCF seam.

    Every backend the stack supports adds its ``energy_shift`` to
    ``result.total_energy``, with the legacy ``energy_components`` keys
    (``E_Breit``, ``E_SE``, ``E_VP``, ``E_WK``, ``E_QED_total``,
    numeric shifts) preserved for backward compatibility. The self-energy
    method is recorded in ``solver_diagnostics["se_method"]``. Skipped
    backends leave no component keys (their reasons remain in
    ``result._correction_results["_skipped"].provenance``).

    Shared by :meth:`DFSolver._finalize_with_corrections` (B-spline)
    and the ``Atom`` shooting path, so both engines report corrections
    through identical keys.  Returns ``result`` for chaining;
    ``stack=None`` is a no-op.

    ``E_Breit`` is the shift actually added to ``total_energy``. Its
    provenance records whether DBSR_HF average Breit, full static Breit, or a legacy
    scalar was applied.
    """
    if stack is None:
        return result
    if "_correction_results" in result.__dict__:
        raise ValueError("corrections already finalized; use an uncorrected result")
    stack.validate_addition(result)
    applied = stack.apply(result)
    if getattr(result, 'calculation_spec', None) is not None:
        from dataclasses import replace

        from atomx.calculation_spec import resolved_correction_spec
        result.calculation_spec = replace(result.calculation_spec,
            corrections=resolved_correction_spec(stack))
    ledger = CorrectionApplication(
        base_energy=float(result.total_energy),
        shifts=tuple((name, float(cr.energy_shift)) for name, cr in applied.items()
                     if not name.startswith('_')),
        skipped=tuple((str(name), str(reason))
                      for name, reason in applied.get('_skipped',
                          CorrectionResult('_skipped', 0.)).provenance.items()))
    # Preserve the actual evaluation and skipped reasons. Re-running a
    # potentially expensive backend is not a provenance retrieval mechanism.
    result._correction_results = applied
    result.correction_application = ledger
    result.operator_stages = (
        *getattr(result, 'operator_stages', ()),
        *((name.split('[', 1)[0], ledger.stage) for name, _ in ledger.shifts),
    )
    comps = result.energy_components
    e_qed = 0.0
    has_qed = False
    for name, cr in applied.items():
        if name.startswith("_"):
            continue
        if name == "breit":
            comps["E_Breit"] = cr.energy_shift
            result.solver_diagnostics["_breit_provenance"] = dict(cr.provenance)
        elif name.startswith("self_energy"):
            comps["E_SE"] = cr.energy_shift
            se_method = name.split("[", 1)[1].rstrip("]")
            result.solver_diagnostics["se_method"] = se_method
            if se_method == "mohr":
                # Solver-policy metadata, not an energy: it lives in the
                # declared diagnostics channel so energy_components stays
                # float-only (review E6).
                result.solver_diagnostics["_mohr_high_ns_policy"] = (
                    "codata_a41_a40_v1")
            e_qed += cr.energy_shift
            has_qed = True
        elif name == "uehling_vp":
            comps["E_VP"] = cr.energy_shift
            e_qed += cr.energy_shift
            has_qed = True
        elif name == "wichmann_kroll_vp":
            comps["E_WK"] = cr.energy_shift
            e_qed += cr.energy_shift
            has_qed = True
        elif name == 'hydrogenic_radiative_mass_leading':
            comps[name] = cr.energy_shift
            e_qed += cr.energy_shift
            has_qed = True
        elif name in ("nms", "sms", "hydrogenic_mass_remainder"):
            comps[name] = cr.energy_shift
            result.solver_diagnostics[f"_{name}_provenance"] = dict(cr.provenance)
        else:
            comps[name] = cr.energy_shift
        result.total_energy += cr.energy_shift
    if has_qed:
        comps["E_QED_total"] = e_qed
    return result


def breit_provenance(result) -> Dict:
    """Which Breit interaction ``result`` carries, and how good it is.

    The counterpart of :attr:`MCDHFResult.breit_provenance` for the
    results that have no such property (``SCFResult``, ``ShootingResult``
    — the single-configuration route).  That route applies the *scalar*
    removed scalar Breit, which was 3.03x
    DBSR_HF for Ne, so the returned dict is the only place a reader of
    such a result can learn that the applied Breit is not externally
    validated.  It is filled in by :func:`finalize_with_corrections`
    (``{}`` when Breit was off).

    Returns
    -------
    dict
        ``{"mode", "operator", "scalar_shift_applied",
        "externally_validated", "note"}`` for the scalar route; the
        operator route's own dict for ``MCDHFResult``.
    """
    prop = getattr(result, "breit_provenance", None)
    if prop:
        return dict(prop)
    return dict(result.solver_diagnostics.get("_breit_provenance") or {})


__all__ = ["CorrectionStack", "breit_provenance", "finalize_with_corrections"]
