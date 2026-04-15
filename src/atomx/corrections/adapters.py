"""Adapters that wrap the existing atomx correction functions as
:class:`~atomx.corrections.protocol.Correction` implementations.

The underlying physics is untouched — these adapters only restate the
legacy ``(orbitals, bsp_data, Z)``-taking functions in the flavour-aware
interface expected by :class:`~atomx.corrections.stack.CorrectionStack`.

Why a separate module?
----------------------
Each legacy function lives in its own physics module (``corrections.qed``,
``corrections.breit``, ``isotope.mass_shift``, …) and depends on a
specific internal data layout.  Collecting the adapters here keeps
those physics modules lean and gives the stack a single import surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from atomx.corrections.protocol import CorrectionResult

if TYPE_CHECKING:
    from atomx.results import AtomicResult


# ---------------------------------------------------------------------------
# Helpers shared by Dirac-based corrections
# ---------------------------------------------------------------------------

OrbitalTuple = Tuple  # (P, Q, occ, kappa, label)


def _flatten_scf_orbitals(result: "AtomicResult") -> List[OrbitalTuple]:
    """Extract the flat orbital tuples a B-spline correction expects.

    Mirrors ``DFSolver._flatten_orbitals`` but reads from the result
    rather than the solver, so it works on any converged SCF-flavour
    :class:`~atomx.results.AtomicResult`.
    """
    orb_list: List[OrbitalTuple] = []
    kappa_counter: Dict[int, int] = {}
    subshells = getattr(result, "subshells", None) or []
    orbitals = getattr(result, "orbitals_raw", None) or {}
    for sh in subshells:
        idx = kappa_counter.get(sh.kappa, 0)
        kappa_counter[sh.kappa] = idx + 1
        orbs = orbitals.get(sh.kappa, [])
        if idx < len(orbs):
            P, Q, occ = orbs[idx]
            orb_list.append((P, Q, occ, sh.kappa, sh.label))
    return orb_list


def _has_bspline_dirac_data(result: "AtomicResult") -> bool:
    """True iff ``result`` carries the B-spline workspace a Dirac
    correction backend needs."""
    bsp = getattr(result, "_bsp_data", None)
    raw = getattr(result, "orbitals_raw", None)
    Z = getattr(result, "Z", None)
    return bool(bsp) and bool(raw) and Z is not None


# ---------------------------------------------------------------------------
# Vacuum polarisation (Uehling)
# ---------------------------------------------------------------------------

class UehlingVPCorrection:
    """Second-order Uehling vacuum-polarisation energy shift."""

    name = "uehling_vp"
    description = "Uehling VP  ⟨a|V_VP|a⟩ with V_VP = −(2Zα/3π)/r · K₁(2cr)."

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_vp_energy

        orbitals = _flatten_scf_orbitals(result)
        E = float(compute_vp_energy(orbitals, result._bsp_data, result.Z))
        return CorrectionResult(
            name=self.name,
            energy_shift=E,
            provenance={"method": "Uehling point-nucleus, folded over P²+Q²"},
        )


class WichmannKrollVPCorrection:
    """Simplified Wichmann–Kroll higher-order VP (Flambaum–Ginges form)."""

    name = "wichmann_kroll_vp"
    description = "Higher-order VP (Flambaum–Ginges 2005 eq. 12)."

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_wk_vp_energy

        orbitals = _flatten_scf_orbitals(result)
        E = float(compute_wk_vp_energy(orbitals, result._bsp_data, result.Z))
        return CorrectionResult(
            name=self.name,
            energy_shift=E,
            provenance={"method": "Flambaum–Ginges WK fit", "reference": "PRA 72, 052115 (2005)"},
        )


# ---------------------------------------------------------------------------
# Self-energy (delegates to the qedmod backend registry)
# ---------------------------------------------------------------------------

class SelfEnergyCorrection:
    """Pluggable self-energy correction.

    Dispatches to one of the registered :class:`SEBackend` instances
    (``mohr``, ``welton``, ``shabaev_local``, ``flambaum_ginges``, …).
    All existing backends are hydrogenic-anchored; the amplitude in
    each case is calibrated from a hydrogenic Dirac reference but the
    *application* already uses the converged SCF orbitals, so the
    shift flavour-adapts automatically for DHF, DKS, and MCDHF.
    """

    def __init__(self, backend: str = "shabaev_local"):
        self.name = f"self_energy[{backend}]"
        self._backend_name = backend
        self.description = (
            f"Self-energy via the {backend!r} backend "
            f"(see atomx.corrections.qedmod)."
        )

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qedmod import get_backend

        backend = get_backend(self._backend_name)
        orbitals = _flatten_scf_orbitals(result)
        out = backend.compute(orbitals, result._bsp_data, result.Z)
        return CorrectionResult(
            name=self.name,
            energy_shift=float(out.E_SE),
            per_orbital=dict(out.per_orbital),
            provenance={"backend": self._backend_name, "method": out.method},
        )


# ---------------------------------------------------------------------------
# Breit interaction
# ---------------------------------------------------------------------------

class BreitCorrection:
    """Full Breit interaction energy (Grant–Pyper multipole expansion)."""

    name = "breit"
    description = "Full Breit interaction (magnetic + retardation)."

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.breit import compute_breit_energy

        orbitals = _flatten_scf_orbitals(result)
        E = float(compute_breit_energy(orbitals, result._bsp_data, result.Z))
        return CorrectionResult(
            name=self.name,
            energy_shift=E,
            provenance={"reference": "Grant & Pyper, J.Phys.B 9, 761 (1976)"},
        )


# ---------------------------------------------------------------------------
# Normal Mass Shift
# ---------------------------------------------------------------------------

class NormalMassShiftCorrection:
    """Breit–Salpeter normal mass-shift correction."""

    name = "nms"
    description = "Normal mass shift  ΔE = −(m_e/M)·⟨T_e⟩ (virial estimator)."

    def __init__(self, nuclear_mass_me: float):
        if nuclear_mass_me <= 0.0:
            raise ValueError(
                "nuclear_mass_me must be positive (units of electron mass)."
            )
        self._M = float(nuclear_mass_me)

    def supports(self, result: "AtomicResult") -> bool:
        return hasattr(result, "total_energy")

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.prop.isotope_shift.mass_shift import normal_mass_shift

        shift = normal_mass_shift(
            total_energy=result.total_energy,
            nuclear_mass_me=self._M,
        )
        return CorrectionResult(
            name=self.name,
            energy_shift=float(shift),
            provenance={
                "estimator": "virial theorem ⟨T_e⟩ = −E_total",
                "nuclear_mass_me": self._M,
            },
        )


__all__ = [
    "UehlingVPCorrection",
    "WichmannKrollVPCorrection",
    "SelfEnergyCorrection",
    "BreitCorrection",
    "NormalMassShiftCorrection",
    "_flatten_scf_orbitals",
    "_has_bspline_dirac_data",
]
