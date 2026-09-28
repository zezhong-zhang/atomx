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

def _has_correlated_radial_data(result):
    return callable(getattr(result, "one_body_density_matrix", None)) and (
        bool(getattr(result, "bsp_data", None))
        or (getattr(result, "_grid", None) is not None
            and getattr(result, "_difr", None) is not None))


def _correlated_local_energy(result, potential=None, *, radial_builder=None):
    """Contract a local potential with each ground state's full density."""
    import numpy as np

    from atomx.corrections.qed import short_range_radial_matrix
    from atomx.corrections.workspace import make_grid_workspace
    workspace = result.bsp_data
    i0 = 0
    if not workspace:
        i0 = _origin_offset(result._grid)
        workspace = make_grid_workspace(result._grid[i0:], result._difr[i0:])
    states = np.flatnonzero(abs(result.state_energies-result.state_energies[0]) < 1e-10)
    values = []
    for state in states:
        keys, density = result.one_body_density_matrix(state)
        orbitals = dict(result.orbitals_dict)
        block = (int(result.state_J2[state]), int(result.state_parity[state]))
        orbitals.update(result.per_block_orbitals.get(block, {}))
        waves = [(orbitals[k][0][i0:], orbitals[k][1][i0:], k[1]) for k in keys]
        radial = (radial_builder(keys, waves, workspace) if radial_builder is not None
                  else short_range_radial_matrix(waves, workspace, potential))
        values.append(float(np.sum(density*radial)))
    return float(np.mean(values))


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
        return (_has_correlated_radial_data(result) or _has_bspline_dirac_data(result)
                or _has_radial_grid_data(result))

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_vp_energy, uehling_potential

        if _has_correlated_radial_data(result):
            E = _correlated_local_energy(result, lambda r: uehling_potential(r, result.Z))
        else:
            orbitals, workspace = _dirac_orbital_payload(result)
            E = float(compute_vp_energy(orbitals, workspace, result.Z))
        return CorrectionResult(
            name=self.name,
            energy_shift=E,
            provenance={"method": "Uehling point-nucleus, full local one-body expectation"},
        )


class WichmannKrollVPCorrection:
    """Simplified Wichmann–Kroll higher-order VP (Flambaum–Ginges form)."""

    name = "wichmann_kroll_vp"
    description = "Higher-order VP (Flambaum–Ginges 2005 eq. 12)."

    def supports(self, result: "AtomicResult") -> bool:
        return (_has_correlated_radial_data(result) or _has_bspline_dirac_data(result)
                or _has_radial_grid_data(result))

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qed import compute_wk_vp_energy
        from atomx.corrections.qedmod.flambaum_ginges import V_wichmann_kroll

        if _has_correlated_radial_data(result):
            E = _correlated_local_energy(result, lambda r: V_wichmann_kroll(r, result.Z))
        else:
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
        return (_has_correlated_radial_data(result) or _has_bspline_dirac_data(result)
                or _has_radial_grid_data(result))

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.qedmod import get_backend

        if _has_correlated_radial_data(result):
            from atomx.corrections.qed import self_energy_radial_matrix
            from atomx.methods.dirac_fock import Subshell

            def radial_builder(keys, waves, workspace):
                return self_energy_radial_matrix(
                    [(p, q, k, Subshell(*key, 1.).label)
                     for key, (p, q, k) in zip(keys, waves, strict=True)],
                    result.Z, workspace, method=self._backend_name)

            energy = _correlated_local_energy(result, radial_builder=radial_builder)
            return CorrectionResult(
                name=self.name, energy_shift=energy,
                provenance={"backend": self._backend_name,
                            "evaluation": "correlated one-body density contraction"})
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
    """The requested operator matrix element is unavailable."""


class BreitCorrection:
    """Apply DBSR_HF average Breit, Gaunt, or an explicit legacy scalar.

    DBSR_HF is the default for a DHF configuration energy: it retains all
    eight radial/angle terms and the L-1,L,L+1 channels, including static
    retardation. It remains a configuration average, so it does not split
    levels. ``operator`` applies the level-capable Gaunt matrix element but
    omits retardation. ``scalar_legacy`` reproduces historical ledgers.
    """

    name = "breit"
    description = "DBSR_HF Breit average, including static retardation."

    def __init__(self, mode: str = "dbsr_hf") -> None:
        if mode not in {"dbsr_hf", "operator", "scalar_pairs", "scalar_legacy"}:
            raise ValueError(
                "Breit mode must be 'dbsr_hf', 'operator', 'scalar_pairs', "
                "or 'scalar_legacy'")
        self.mode = mode

    def supports(self, result: "AtomicResult") -> bool:
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        from atomx.corrections.breit import (
            SCALAR_BREIT_NOTE,
            compute_breit_energy,
        )

        operator = result.__dict__.get("_E_Breit_operator")
        if self.mode == "dbsr_hf":
            from atomx.corrections.dbsr_breit import compute_dbsr_breit_energy

            orbitals, workspace = _dirac_orbital_payload(result)
            configs = result.__dict__.get("_jj_configs")
            energy = float(compute_dbsr_breit_energy(
                orbitals, workspace, jj_configs=configs))
            provenance = {
                "mode": "dbsr_hf",
                "operator": "average_configuration",
                "scalar_shift_applied": True,
                "retardation_included": True,
                "pair_weighting": ("exact_jj_ensemble" if configs is not None
                                   else "integer_configuration"),
                "reference": "DBSR_HF hf_MOD_energy.f90 and hf_energy_BR.f90",
                "implementation": "corrections/dbsr_breit.py",
                "note": "Full static Breit configuration-average energy on the "
                        "converged orbitals; resolved J-dependent shifts "
                        "require a Breit operator in CI.",
            }
            if operator is not None:
                provenance["gaunt_operator_cross_check"] = float(operator)
            return CorrectionResult(
                name=self.name, energy_shift=energy, provenance=provenance)
        if self.mode == "operator":
            if operator is None:
                raise BreitCorrectionError(
                    "Gaunt operator value is absent; calculate its matrix "
                    "element before applying the default Breit correction")
            return CorrectionResult(
                name=self.name,
                energy_shift=float(operator),
                provenance={
                    "mode": "operator",
                    "operator": "Gaunt",
                    "scalar_shift_applied": False,
                    "retardation_included": False,
                    "reference": "two-electron Gaunt operator in MCDHF CI basis",
                    "implementation": "methods/mcdhf/breit_ci.py",
                    "note": "The Gaunt matrix element is evaluated exactly in the "
                            "selected orbital/configuration space; the "
                            "frequency-independent Breit retardation term "
                            "is not yet implemented as an operator.",
                },
            )

        orbitals, workspace = _dirac_orbital_payload(result)
        configs = (result.__dict__.get("_jj_configs")
                   if self.mode == "scalar_pairs" else None)
        E = float(compute_breit_energy(
            orbitals, workspace, result.Z, jj_configs=configs))
        provenance = {
            "mode": self.mode,
            "operator": "none",
            "scalar_shift_applied": True,
            "reference": "Grant & Pyper, J.Phys.B 9, 761 (1976)",
            "implementation": "corrections/breit.py scalar AOC "
                              "(v=L normalised by 1/(2L+1))",
            "externally_validated": False,
            "validation": SCALAR_BREIT_NOTE,
            "pair_weighting": ("exact_jj_ensemble" if configs is not None
                               else "product_of_mean_occupations"),
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
    """One-body Breit recoil; the historical virial estimate is opt-in."""

    name = "nms"
    description = ("Normal mass shift from the one-body Breit recoil operator; "
                   "nonrelativistic limit +⟨p²/2⟩/M. The optional virial "
                   "estimate is −E_total/M.")

    def __init__(self, nuclear_mass_me: float, mode: str = "operator"):
        import math
        if not math.isfinite(nuclear_mass_me) or nuclear_mass_me <= 0.0:
            raise ValueError(
                "nuclear_mass_me must be positive (units of electron mass)."
            )
        self._M = float(nuclear_mass_me)
        if mode not in ("operator", "virial"):
            raise ValueError("nms mode must be 'operator' or 'virial'")
        self.mode = mode

    def supports(self, result: "AtomicResult") -> bool:
        if self.mode == "virial":
            return hasattr(result, "total_energy")
        if callable(getattr(result, "one_body_density_matrix", None)):
            return bool(getattr(result, "orbitals_dict", None)) and (
                bool(getattr(result, "bsp_data", None))
                or (getattr(result, "_grid", None) is not None
                    and getattr(result, "_difr", None) is not None))
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        if self.mode == "operator":
            import numpy as np

            from atomx.corrections.recoil import recoil_provenance, recoil_radial_matrix
            if callable(getattr(result, "one_body_density_matrix", None)):
                keys, density = result.one_body_density_matrix()
                orbitals = [(*result.orbitals_dict[key], key[1]) for key in keys]
                workspace = result.bsp_data
                if not workspace:
                    from atomx.corrections.workspace import make_grid_workspace
                    i0 = _origin_offset(result._grid)
                    workspace = make_grid_workspace(result._grid[i0:], result._difr[i0:])
                    orbitals = [(p[i0:], q[i0:], k) for p, q, k in orbitals]
            else:
                payload, workspace = _dirac_orbital_payload(result)
                orbitals = [(p, q, k) for p, q, _, k, _ in payload]
                density = np.diag([occ for _, _, occ, _, _ in payload])
            matrix = recoil_radial_matrix(orbitals, workspace, result.Z)
            return CorrectionResult(
                name=self.name, energy_shift=float(np.sum(density * matrix) / self._M),
                provenance=recoil_provenance(self._M))
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


class HydrogenicMassRemainderCorrection:
    """Nonrelativistic recoil beyond 1/M for a one-electron DHF state.

    Multielectron results receive zero with explicit scope provenance.
    Correlation-orbital labels are not accepted as hydrogenic n values.
    """

    name = 'hydrogenic_mass_remainder'
    description = 'Exact one-electron nonrelativistic reduced-mass remainder beyond first order.'

    def __init__(self, nuclear_mass_me: float):
        self._M = NormalMassShiftCorrection(nuclear_mass_me)._M

    def supports(self, result):
        return (not callable(getattr(result, 'one_body_density_matrix', None))
                and bool(getattr(result, 'subshells', None)))

    def evaluate(self, result, **_):
        from atomx.corrections.recoil import hydrogenic_mass_remainder

        if not self.supports(result):
            raise ValueError('Hydrogenic mass remainder requires identified single-configuration shells')
        shells = [s for s in result.subshells if s.occupation > 1e-12]
        electrons = sum(s.occupation for s in shells)
        applied = abs(electrons-1.) < 1e-10
        if applied and len({s.n for s in shells}) != 1:
            raise ValueError('Hydrogenic mass remainder requires one identified principal shell')
        shift = (sum(s.occupation*hydrogenic_mass_remainder(result.Z, s.n, self._M)
                     for s in shells) if applied else 0.)
        return CorrectionResult(name=self.name, energy_shift=shift, provenance={
            'operator': self.description, 'nuclear_mass_me': self._M,
            'applied': applied, 'scope': 'one-electron point-Coulomb nonrelativistic remainder',
            'first_order_subtracted': True,
            'omitted': ['higher-order relativistic recoil', 'finite-size recoil',
                        'radiative recoil', 'multielectron higher-order recoil'],
            'reference': 'https://physics.nist.gov/cuu/pdf/CODATA_RMP2016.pdf'})


class SpecificMassShiftCorrection:
    """First-order point-Coulomb two-electron Breit recoil (GRASP RIS4)."""

    name = "sms"
    description = "Two-electron Breit recoil with VINTI and relativistic SMS2/SMS3 terms."

    def __init__(self, nuclear_mass_me: float):
        # Reuse the common positive, finite nuclear-mass validation.
        self._M = NormalMassShiftCorrection(nuclear_mass_me)._M

    def supports(self, result):
        return (_has_correlated_radial_data(result) or _has_bspline_dirac_data(result)
                or _has_radial_grid_data(result))

    def evaluate(self, result, **_):
        import numpy as np

        from atomx.corrections.specific_recoil import (
            configuration_average_recoil,
            momentum_recoil_parameters,
            specific_recoil_provenance,
        )
        if _has_correlated_radial_data(result):
            if result.per_block_orbitals:
                raise NotImplementedError("two-body recoil with per-block orbitals is not implemented")
            workspace = result.bsp_data
            orbitals = result.orbitals_dict
            if not workspace:
                from atomx.corrections.workspace import make_grid_workspace
                i0 = _origin_offset(result._grid)
                workspace = make_grid_workspace(result._grid[i0:], result._difr[i0:])
                orbitals = {k: (p[i0:], q[i0:]) for k, (p, q) in orbitals.items()}
            states = abs(result.state_energies-result.state_energies[0]) < 1e-10
            parameters = momentum_recoil_parameters(
                result.csfs, result.mixing_coefficients[:, states], orbitals,
                workspace, Z=result.Z)
            parameter = float(np.mean(parameters))
        else:
            orbitals, workspace = _dirac_orbital_payload(result)
            parameter = configuration_average_recoil(
                orbitals, workspace, Z=result.Z,
                jj_configs=result.__dict__.get('_jj_configs'))
        return CorrectionResult(name=self.name, energy_shift=parameter/self._M,
                                provenance=specific_recoil_provenance(self._M))
