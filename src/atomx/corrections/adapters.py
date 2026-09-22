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


def _has_radial_grid_data(result: "AtomicResult") -> bool:
    """True iff ``result`` carries grid-tabulated Dirac orbitals.

    This is the ODE-shooting flavour
    (:class:`atomx.numerics.shooting.solver.ShootingResult`): the radial
    grid ``_grid``, the integration weights ``_difr`` (integrals are
    ``sum(f * difr)``), the per-shell ``(P, Q)`` waves ``_pa``/``_qa``,
    and the subshell metadata (kappa, occupation, label).  Schrödinger
    shooting results are rejected — Breit/SE/VP act on Dirac spinors.
    """
    if getattr(result, "_equation", "dirac") != "dirac":
        return False
    if getattr(result, "Z", None) is None:
        return False
    subshells = getattr(result, "subshells", None)
    if not subshells:
        return False
    for attr in ("_grid", "_difr", "_pa", "_qa"):
        if getattr(result, attr, None) is None:
            return False
    return len(result._pa) >= len(subshells)


def _flatten_grid_orbitals(result: "AtomicResult") -> List[OrbitalTuple]:
    """Flat orbital tuples for a grid-flavour (shooting) result.

    Rows of ``_pa``/``_qa`` are ordered as ``result.subshells`` (the
    solvers fill them in subshell order).  The leading ``r <= 0`` grid
    points (Salvat's SGRID starts at r = 0, where every P, Q vanish)
    are sliced off so downstream potentials (-Z/r, Uehling, Y^k/r^k+1)
    stay finite; :func:`_grid_workspace_of` slices identically.
    """
    i0 = _origin_offset(result._grid)
    return [
        (result._pa[i][i0:], result._qa[i][i0:],
         float(sh.occupation), sh.kappa, sh.label)
        for i, sh in enumerate(result.subshells)
    ]


def _origin_offset(r) -> int:
    """Number of leading non-positive radial nodes to slice off."""
    import numpy as np
    return int(np.searchsorted(r, 0.0, side="right"))


def _grid_workspace_of(result: "AtomicResult"):
    """Radial-grid workspace of a shooting result (origin point sliced)."""
    from atomx.corrections.workspace import make_grid_workspace
    i0 = _origin_offset(result._grid)
    return make_grid_workspace(result._grid[i0:], result._difr[i0:])


def _dirac_orbital_payload(result: "AtomicResult"):
    """``(orbitals, workspace)`` for whichever Dirac flavour ``result`` is.

    B-spline results yield coefficient-vector orbitals plus the B-spline
    workspace; grid (shooting) results yield grid-array orbitals plus a
    :func:`~atomx.corrections.workspace.make_grid_workspace` dict.  The
    correction physics functions accept either through the
    :mod:`atomx.corrections.workspace` seam.
    """
    if _has_bspline_dirac_data(result):
        return _flatten_scf_orbitals(result), result._bsp_data
    return _flatten_grid_orbitals(result), _grid_workspace_of(result)


# ---------------------------------------------------------------------------
# Vacuum polarisation (Uehling)
# ---------------------------------------------------------------------------

class UehlingVPCorrection:
    """Second-order Uehling vacuum-polarisation energy shift.

    ``description`` (a user-facing string) said ``K₁(2cr)`` until
    2026-09 — the exact kernel defect fixed in ``200d657``, which made
    the VP 3.9x too large.  The history is kept *here* rather than in
    the string so that grepping the description for the kernel name
    cannot turn up the wrong one; ``tests/test_qed.py`` gates it.
    """

    name = "uehling_vp"
    description = ("Uehling VP  ⟨a|V_VP|a⟩ with V_VP = −(2Zα/3π)/r · I(2cr), "
                   "I = the Uehling integral (kernel fixed in 200d657).")

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_vp_energy

        orbitals, workspace = _dirac_orbital_payload(result)
        E = float(compute_vp_energy(orbitals, workspace, result.Z))
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
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_wk_vp_energy

        orbitals, workspace = _dirac_orbital_payload(result)
        E = float(compute_wk_vp_energy(orbitals, workspace, result.Z))
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
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qedmod import get_backend

        backend = get_backend(self._backend_name)
        orbitals, workspace = _dirac_orbital_payload(result)
        out = backend.compute(orbitals, workspace, result.Z)
        return CorrectionResult(
            name=self.name,
            energy_shift=float(out.E_SE),
            per_orbital=dict(out.per_orbital),
            provenance={"backend": self._backend_name, "method": out.method},
        )


# ---------------------------------------------------------------------------
# Breit interaction
# ---------------------------------------------------------------------------

class BreitCorrectionError(RuntimeError):
    """The Breit correction was requested but cannot be applied.

    Raised rather than substituting anything: the scalar
    configuration-average Breit was removed in 2026-09 (3.03x DBSR_HF for
    Ne), so there is no fallback left to fall back to.
    """


class BreitCorrection:
    """The Breit interaction: fixed scalar here, Gaunt operator cross-check.

    Both implementations are live and they are **complementary** (see
    ``corrections/breit.py``):

    * this rung *applies* ``corrections.breit.compute_breit_energy``, the
      scalar configuration-average sum — the better closed-shell total
      (He/Ne/Ar within 0.13 %/0.33 %/0.66 % of DBSR_HF) because it keeps
      the ``v != L`` retardation channels;
    * the method layer also stashes the exact Gaunt operator's value for
      the same orbitals (``breit_ci.attach_breit_operator_energy`` ->
      ``result._E_Breit_operator``), which is what the MCDHF route uses
      because it carries the non-scalar content that splits levels.  It
      is reported here as a cross-check: the two share no angular kernel,
      so their agreement is independent evidence, and their difference
      should be the documented omitted retardation (4-6 % at Ne/Ar).

    Neither is silently substituted for the other.
    """

    name = "breit"
    description = ("Breit interaction: scalar configuration-average "
                   "Grant-Pyper sum (v=L with 1/(2L+1)); Gaunt operator "
                   "cross-check on the same orbitals.")

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.breit import (
            SCALAR_BREIT_NOTE,
            compute_breit_energy,
        )

        orbitals, workspace = _dirac_orbital_payload(result)
        E = float(compute_breit_energy(orbitals, workspace, result.Z))
        operator = result.__dict__.get("_E_Breit_operator")
        provenance = {
            "reference": "Grant & Pyper, J.Phys.B 9, 761 (1976)",
            "implementation": "corrections/breit.py scalar AOC "
                              "(v=L normalised by 1/(2L+1))",
            "externally_validated": True,
            "validation": SCALAR_BREIT_NOTE,
        }
        if operator is not None:
            provenance["gaunt_operator_cross_check"] = float(operator)
            provenance["operator_vs_scalar_ratio"] = (
                float(operator) / E if E else None)
        return CorrectionResult(
            name=self.name,
            energy_shift=E,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Normal Mass Shift
# ---------------------------------------------------------------------------

class NormalMassShiftCorrection:
    """Breit–Salpeter normal mass-shift correction."""

    name = "nms"
    description = ("Normal mass shift  ΔE = +⟨T_e⟩/M = −E_total/M "
                   "(virial estimator, M in electron masses).")

    def __init__(self, nuclear_mass_me: float):
        if nuclear_mass_me <= 0.0:
            raise ValueError(
                "nuclear_mass_me must be positive (units of electron mass)."
            )
        self._M = float(nuclear_mass_me)

    def supports(self, result: "AtomicResult") -> bool:
        return hasattr(result, "total_energy")

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.properties.isotope_shift.mass_shift import normal_mass_shift

        # ``normal_mass_shift`` takes the nuclear mass as ``mass_me`` and
        # returns a ``MassShift`` dataclass; this call previously passed a
        # nonexistent ``nuclear_mass_me`` keyword and then ran ``float()``
        # on the dataclass, so it raised TypeError for every caller.
        shift = normal_mass_shift(
            total_energy=result.total_energy,
            mass_me=self._M,
        )
        return CorrectionResult(
            name=self.name,
            energy_shift=float(shift.E_NMS_Ha),
            provenance={
                "estimator": "virial theorem ⟨T_e⟩ = −E_total",
                "nuclear_mass_me": self._M,
            },
        )


__all__ = [
    "BreitCorrectionError",
    "UehlingVPCorrection",
    "WichmannKrollVPCorrection",
    "SelfEnergyCorrection",
    "BreitCorrection",
    "NormalMassShiftCorrection",
    "_flatten_scf_orbitals",
    "_flatten_grid_orbitals",
    "_grid_workspace_of",
    "_has_bspline_dirac_data",
    "_has_radial_grid_data",
]
