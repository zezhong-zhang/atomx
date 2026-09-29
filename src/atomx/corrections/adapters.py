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
from atomx.radial_state import native_radial_state, radial_workspace_of

if TYPE_CHECKING:
    from atomx.results import AtomicResult


# ---------------------------------------------------------------------------
# Helpers shared by Dirac-based corrections
# ---------------------------------------------------------------------------

OrbitalTuple = Tuple  # (P, Q, occ, kappa, label)


def _flatten_scf_orbitals(result: "AtomicResult") -> List[OrbitalTuple]:
    return list(native_radial_state(result).occupied)


def _supports_native_dirac(result, representation):
    if getattr(result, 'Z', None) is None:
        return False
    try:
        state = native_radial_state(result)
    except (AttributeError, ValueError, TypeError, KeyError):
        return False
    return (state.equation == 'dirac' and state.representation == representation
            and bool(state.occupied))


def _has_bspline_dirac_data(result):
    return _supports_native_dirac(result, 'bspline')


def _has_radial_grid_data(result):
    return _supports_native_dirac(result, 'grid')


def _flatten_grid_orbitals(result):
    return list(native_radial_state(result).occupied)


def _origin_offset(r):
    import numpy as np
    return int(np.searchsorted(r, 0., side='right'))


def _grid_workspace_of(result):
    return radial_workspace_of(result)[0]


def _dirac_orbital_payload(result):
    state = native_radial_state(result)
    return list(state.occupied), state.correction_workspace()


# ---------------------------------------------------------------------------
# Vacuum polarisation (Uehling)
# ---------------------------------------------------------------------------

def _has_correlated_radial_data(result):
    if not callable(getattr(result, 'one_body_density_matrix', None)):
        return False
    try:
        state = native_radial_state(result)
    except (AttributeError, ValueError, TypeError, KeyError):
        return False
    return state.equation == 'dirac' and bool(state.orbitals)


def _correlated_local_energy(result, potential=None, *, radial_builder=None):
    """Contract a local potential with each ground state's full density."""
    import numpy as np

    from atomx.corrections.qed import short_range_radial_matrix
    workspace, i0 = radial_workspace_of(result)
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
            from atomx.config.subshell import Subshell

            def radial_builder(keys, waves, workspace):
                return self_energy_radial_matrix(
                    [(p, q, k, Subshell(*key, 1.).label)
                     for key, (p, q, k) in zip(keys, waves, strict=True)],
                    result.Z, workspace, method=self._backend_name)

            energy = _correlated_local_energy(result, radial_builder=radial_builder)
            provenance={"backend": self._backend_name,
                        "evaluation": "correlated one-body density contraction"}
            if self._backend_name in ('mohr','mss'):
                provenance['screening_reference'] = 'exact_point_dirac_coulomb_expectation_v1'
            return CorrectionResult(
                name=self.name, energy_shift=energy,
                provenance=provenance)
        backend = get_backend(self._backend_name)
        orbitals, workspace = _dirac_orbital_payload(result)
        out = backend.compute(orbitals, workspace, result.Z)
        provenance={"backend": self._backend_name, "method": out.method}
        if out.method == 'mohr':
            provenance['screening_reference'] = 'exact_point_dirac_coulomb_expectation_v1'
        return CorrectionResult(
            name=self.name,
            energy_shift=float(out.E_SE),
            per_orbital=dict(out.per_orbital),
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Breit interaction
# ---------------------------------------------------------------------------

class BreitCorrectionError(RuntimeError):
    """The requested operator matrix element is unavailable."""


class BreitCorrection:
    """Apply DBSR_HF average Breit, a static Breit operator, or an explicit legacy scalar.

    DBSR_HF is the default for a DHF configuration energy: it retains all
    eight radial/angle terms and the L-1,L,L+1 channels, including static
    retardation. It remains a configuration average, so it does not split
    levels. ``operator`` applies the level-capable static Breit matrix element,
    including Gaunt and retardation. ``scalar_legacy`` reproduces historical ledgers.
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
                "radial_integration": ("gauss_cell_polynomial" if workspace['gr'].ndim == 2
                                       else workspace.get('integration_rule', 'trapezoidal')),
                "note": "Full static Breit configuration-average energy on the "
                        "converged orbitals; resolved J-dependent shifts "
                        "require a Breit operator in CI.",
            }
            if operator is not None:
                provenance["static_breit_operator_cross_check"] = float(operator)
            return CorrectionResult(
                name=self.name, energy_shift=energy, provenance=provenance)
        if self.mode == "operator":
            if operator is None:
                raise BreitCorrectionError(
                    "Static Breit operator value is absent; calculate its matrix "
                    "element before applying the default Breit correction")
            return CorrectionResult(
                name=self.name,
                energy_shift=float(operator),
                provenance={
                    "mode": "operator",
                    "operator": "Gaunt plus static retardation",
                    "scalar_shift_applied": False,
                    "retardation_included": True,
                    "reference": "two-electron static Breit operator in MCDHF CI basis",
                    "implementation": "methods/mcdhf/breit_ci.py",
                    "note": "Complete frequency-independent Breit operator in the "
                            "selected orbital/configuration space. Frequency-dependent "
                            "transverse exchange is not included.",
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
            provenance["static_breit_operator_cross_check"] = float(operator)
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
            return _has_correlated_radial_data(result)
        return _has_bspline_dirac_data(result) or _has_radial_grid_data(result)

    def evaluate(self, result: "AtomicResult", **_: Any) -> CorrectionResult:
        if self.mode == "operator":
            import numpy as np

            from atomx.corrections.recoil import recoil_provenance, recoil_radial_matrix
            if callable(getattr(result, "one_body_density_matrix", None)):
                energy = _correlated_local_energy(result, radial_builder=(
                    lambda keys, orbitals, workspace: recoil_radial_matrix(
                        orbitals, workspace, result.Z)))
                return CorrectionResult(self.name, energy/self._M,
                                        provenance=recoil_provenance(self._M))
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
    "HydrogenicRadiativeMassLeadingCorrection",
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


class HydrogenicRadiativeMassLeadingCorrection:
    """Explicit alpha*(Z alpha)^4 reduced-mass QED addition for one electron."""

    name = 'hydrogenic_radiative_mass_leading'
    description = 'Leading-order finite-mass self-energy and vacuum-polarization addition.'

    def __init__(self, nuclear_mass_me):
        self._M = NormalMassShiftCorrection(nuclear_mass_me)._M

    def supports(self, result):
        return (not callable(getattr(result,'one_body_density_matrix',None))
                and bool(getattr(result,'subshells',None))
                and abs(sum(s.occupation for s in result.subshells)-1.)<1e-10)

    def evaluate(self, result, **_):
        from atomx.corrections.radiative_mass import hydrogenic_radiative_mass_leading
        if not self.supports(result):
            raise ValueError('radiative mass addition requires identified one-electron shells')
        shifts={}
        components={'self_energy_mass_ha':0.,'vacuum_polarization_mass_ha':0.}
        for shell in result.subshells:
            if shell.occupation<=0:continue
            values=hydrogenic_radiative_mass_leading(result.Z,shell.n,shell.kappa,self._M)
            shifts[shell.label]=sum(values.values())
            for key,value in values.items():components[key]+=shell.occupation*value
        return CorrectionResult(name=self.name,energy_shift=sum(components.values()),
            per_orbital=shifts,provenance=dict(nuclear_mass_me=self._M,
                order='alpha*(Z alpha)^4, including leading logarithm',
                mass_treatment='finite minus infinite nuclear mass; ordinary recoil excluded',
                components=components,reference='https://doi.org/10.1063/5.0064853',
                omitted=['higher powers of Z alpha','finite nuclear size in radiative mass terms',
                         'two-loop mass terms','additional radiative-recoil operators',
                         'many-electron radiative recoil']))


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
            workspace, i0 = radial_workspace_of(result)
            orbitals = {k: (p[i0:], q[i0:])
                        for k, (p, q) in native_radial_state(result).orbitals.items()}
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
