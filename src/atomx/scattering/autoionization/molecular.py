"""Molecular Auger initial-state entry points.

This module is intentionally provider-contract only: it consumes an
``ElectronicStructurePayload`` and never imports a back-end such as PySCF.
The current HYP-910.A surface stops at the explicitly named core-hole
initial state; Auger-rate kernels are wired in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from itertools import combinations_with_replacement
from numbers import Integral, Real
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from atomx.providers.contract import (
    Capability,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    Provenance,
    ProviderContractError,
    ensure_hartree,
)

_CORE_HOLE_IDENTITY_KEYS = frozenset(("orbital_label", "core_hole", "core_orbital"))
_MAX_MOLECULAR_AUGER_ACTIVE_ORBITALS = 16
_MAX_MOLECULAR_AUGER_AO_TO_MO_ORBITALS = 48
_MAX_MOLECULAR_AUGER_AO_TO_MO_AOS = 88
_MAX_MOLECULAR_AUGER_AO_TO_MO_ELEMENTS = (
    _MAX_MOLECULAR_AUGER_AO_TO_MO_AOS**4
)
_MAX_MOLECULAR_AUGER_AO_TO_MO_INPUT_BYTES = 512 * 1024**2
_MAX_MOLECULAR_AUGER_AO_TO_MO_OUTPUT_BYTES = 512 * 1024**2

__all__ = [
    "ActiveMolecularOrbital",
    "LocalizedValenceActiveSpace",
    "TwoHoleCSF",
    "TwoHoleFinalState",
    "CoreHoleState",
    "AugerStick",
    "AugerTransitionAmplitude",
    "ContinuumChannel",
    "DiscreteMOContinuumChannel",
    "SpectrumResult",
    "compute_molecular_auger",
    "enumerate_two_hole_csfs",
    "local_valence_active_space",
    "molecular_auger_spectrum",
    "molecular_auger_transition_amplitude",
    "molecular_auger_final_states",
    "molecular_auger_initial_state",
]


@dataclass(frozen=True)
class ActiveMolecularOrbital:
    """MO selected for the local valence active space."""

    label: str
    index: int
    energy: float
    occupation: float
    core_atom_population: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ProviderContractError("active MO label must be non-empty")
        object.__setattr__(self, "energy", ensure_hartree(self.energy))


@dataclass(frozen=True)
class LocalizedValenceActiveSpace:
    """Local valence active-space selection with an ADR Q2 transform contract.

    HYP-910.B deliberately starts with Mulliken-on-canonical-MO selection on
    the core-hole atom center.  The localization transform is therefore the
    identity in the selected MO block, but it is still recorded explicitly so
    downstream two-electron kernels know the parent AO ordering, S metric and
    residual they must match.
    """

    orbitals: tuple[ActiveMolecularOrbital, ...]
    parent_ao_labels: tuple[str, ...]
    overlap_s: NDArray[np.float64]
    parent_mo_indices: tuple[int, ...]
    localization_method: str
    localization_transform: NDArray[np.float64]
    population_threshold: float
    occupation_threshold: float
    orthonormality_residual: float
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "overlap_s", np.asarray(self.overlap_s, dtype=float))
        transform = np.asarray(self.localization_transform, dtype=float)
        object.__setattr__(self, "localization_transform", transform)
        if not self.orbitals:
            raise ProviderContractError("local valence active space is empty")
        if (
            self.overlap_s.ndim != 2
            or self.overlap_s.shape[0] != self.overlap_s.shape[1]
        ):
            raise ProviderContractError("active-space S metric must be square")
        n = len(self.orbitals)
        if transform.shape != (n, n):
            raise ProviderContractError(
                "localization transform must be square in the active MO block"
            )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(orb.label for orb in self.orbitals)


@dataclass(frozen=True)
class TwoHoleCSF:
    """Non-central-field two-hole CSF over molecular orbital labels."""

    hole_orbitals: tuple[str, str]
    spin_multiplicity: int
    spin_symmetry: str
    label: str

    def __post_init__(self) -> None:
        if len(self.hole_orbitals) != 2 or not all(self.hole_orbitals):
            raise ProviderContractError("two-hole CSF requires two MO labels")
        if self.spin_multiplicity not in (1, 3):
            raise ProviderContractError("two-hole spin multiplicity must be 1 or 3")
        if (
            self.hole_orbitals[0] == self.hole_orbitals[1]
            and self.spin_multiplicity != 1
        ):
            raise ProviderContractError("same-orbital two-hole CSF must be singlet")


@dataclass(frozen=True)
class TwoHoleFinalState:
    """Explicit molecular Auger final state (ADR-0001 I2/I4/I5)."""

    hole_orbitals: tuple[str, str]
    spin_multiplicity: int
    spin_symmetry: str
    final_energy: float
    provenance: Provenance
    active_space: LocalizedValenceActiveSpace
    energy_model: str = "independent-hole-koopmans"
    coulomb_correction: float = 0.0

    def __post_init__(self) -> None:
        if len(self.hole_orbitals) != 2 or not all(self.hole_orbitals):
            raise ProviderContractError("final state requires two hole MO labels")
        if not self.provenance.final_state:
            raise ProviderContractError(
                "final-state provenance.final_state is required"
            )
        if not self.provenance.energy_zero:
            raise ProviderContractError(
                "final-state provenance.energy_zero is required"
            )
        if not self.provenance.approximations:
            raise ProviderContractError("final-state approximations must be stamped")
        if not self.energy_model:
            raise ProviderContractError("final-state energy_model must be non-empty")
        object.__setattr__(self, "final_energy", ensure_hartree(self.final_energy))


@dataclass(frozen=True)
class CoreHoleState:
    """Explicit molecular core-hole initial state (ADR-0001 I2/I4).

    The hole is named by ``orbital_label``.  The optional provider-side MO
    index stays out of this dataclass so downstream kernels cannot depend on
    implicit array position for the physical state identity.
    """

    orbital_label: str
    occupation_before: float
    occupation_after: float
    initial_energy: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if not self.orbital_label:
            raise ProviderContractError("core-hole orbital_label must be non-empty")
        if not self.provenance.core_hole:
            raise ProviderContractError("core-hole provenance.core_hole is required")
        object.__setattr__(self, "initial_energy", ensure_hartree(self.initial_energy))

    @property
    def occupation_delta(self) -> float:
        """Occupation change, ``occupation_after - occupation_before``."""
        return float(self.occupation_after - self.occupation_before)


@runtime_checkable
class ContinuumChannel(Protocol):
    """Replaceable continuum-electron channel for molecular Auger kernels.

    The v1 implementations deliberately expose coefficients in the provider
    MO basis; later D-1 residual-ion continuum solvers can satisfy the same
    Protocol while producing physically energy-normalized scattering waves.
    """

    label: str
    approximations: tuple[str, ...]

    def coefficients(
        self,
        payload: ElectronicStructurePayload,
        kinetic_energy: float,
    ) -> NDArray[np.float64]:
        """Return continuum coefficients in the payload MO basis."""
        ...


@dataclass(frozen=True)
class DiscreteMOContinuumChannel:
    """Unit-normalized virtual-MO placeholder for the outgoing Auger electron.

    This is a degradation path, not a scattering solution.  It is useful for
    wiring and testing the ERI/direct-exchange contract until the residual-ion
    energy-normalized continuum solver is available.
    """

    mo_index: int | None = None
    label: str = "discrete-mo-continuum-v1"
    approximations: tuple[str, ...] = (
        "continuum:discrete-mo-v1-replaceable",
        "continuum:not-energy-normalized",
    )

    def coefficients(
        self,
        payload: ElectronicStructurePayload,
        kinetic_energy: float,
    ) -> NDArray[np.float64]:
        if kinetic_energy <= 0.0:
            raise ProviderContractError("continuum kinetic energy must be positive")
        if payload.orbitals is None:
            raise ProviderContractError("MO continuum channel requires MO coefficients")
        n_mo = payload.orbitals.n_mo
        index = self.mo_index
        if index is None:
            virtual = np.flatnonzero(payload.orbitals.occupations <= 1e-8)
            if virtual.size == 0:
                raise ProviderContractError(
                    "discrete MO continuum channel requires an unoccupied MO "
                    "or an explicit mo_index"
                )
            index = int(virtual[0])
        if index < 0 or index >= n_mo:
            raise ProviderContractError("continuum mo_index is outside the MO range")
        coeff = np.zeros(n_mo, dtype=float)
        coeff[index] = 1.0
        return coeff


@dataclass(frozen=True)
class AugerTransitionAmplitude:
    """Direct/exchange components and antisymmetrized Auger amplitude."""

    direct: float
    exchange: float
    amplitude: float
    selection_rule: str = "allowed"

    @property
    def rate_sum_intensity(self) -> float:
        """Incorrect incoherent reference, exposed only for tests/audits."""
        return float(self.direct * self.direct + self.exchange * self.exchange)

    @property
    def coherent_intensity(self) -> float:
        """Correct I3 intensity: ``|direct - exchange|^2``."""
        return float(self.amplitude * self.amplitude)


@dataclass(frozen=True)
class AugerStick:
    """One kinetic-energy Auger stick line."""

    kinetic_energy: float
    intensity: float
    raw_intensity: float
    amplitude: AugerTransitionAmplitude
    initial_state: CoreHoleState
    final_state: TwoHoleFinalState
    continuum_label: str
    provenance: Provenance


@dataclass(frozen=True)
class SpectrumResult:
    """Auger kinetic-energy stick spectrum with provenance aggregation."""

    lines: tuple[AugerStick, ...]
    dropped: tuple[dict[str, str], ...]
    provenance: tuple[Provenance, ...]
    axis: str = "kinetic_energy"
    axis_unit: str = "hartree"
    intensity_unit: str = "sum_rule_normalized_probability"

    @property
    def kinetic_energies(self) -> tuple[float, ...]:
        return tuple(line.kinetic_energy for line in self.lines)

    @property
    def intensities(self) -> tuple[float, ...]:
        return tuple(line.intensity for line in self.lines)


def compute_molecular_auger(
    provider: Any,
    core_hole_spec: str | int | dict[str, Any] | None = None,
    *,
    continuum_channel: ContinuumChannel | None = None,
    normalize: bool = True,
    population_threshold: float = 0.2,
    occupation_threshold: float = 1.5,
    residual_tolerance: float = 1e-7,
) -> SpectrumResult:
    """Run the provider-to-stick molecular Auger pipeline.

    The entrypoint accepts either a validated
    :class:`ElectronicStructurePayload` that already carries the explicit
    core-hole state, or a provider exposing ``build_core_hole_payload``.  It
    fails closed on missing core-hole, MO, overlap, or ERI capabilities before
    delegating to the existing initial-state, final-state, amplitude, and
    stick-spectrum kernels.  Normalization is the global stick sum-rule
    normalization used by :func:`molecular_auger_spectrum`; no per-line
    empirical scaling is applied.
    """

    validated_spec = _validate_core_hole_spec(core_hole_spec)
    population_threshold, occupation_threshold, residual_tolerance = (
        _validate_active_space_parameters(
            population_threshold,
            occupation_threshold,
            residual_tolerance,
        )
    )
    payload = _compute_payload(provider, validated_spec)
    _assert_molecular_auger_eri_budget_before_validation(payload)
    payload.validate()
    _assert_requested_core_hole(payload, validated_spec)
    payload.require(
        Capability.CORE_HOLE_SPEC,
        Capability.MO_COEFFICIENTS,
        Capability.OVERLAP_S,
        Capability.TWO_ELECTRON_INTEGRALS,
    )
    return molecular_auger_spectrum(
        payload,
        continuum_channel=continuum_channel,
        normalize=normalize,
        population_threshold=population_threshold,
        occupation_threshold=occupation_threshold,
        residual_tolerance=residual_tolerance,
    )


def _compute_payload(
    provider: Any,
    core_hole_spec: str | int | dict[str, Any] | None,
) -> ElectronicStructurePayload:
    if isinstance(provider, ElectronicStructurePayload):
        return provider

    if core_hole_spec is not None and hasattr(provider, "build_core_hole_payload"):
        payload = _call_core_hole_builder(
            provider.build_core_hole_payload,
            core_hole_spec,
        )
    elif hasattr(provider, "build_payload"):
        payload = provider.build_payload()
    else:
        raise ProviderContractError(
            "compute_molecular_auger requires an ElectronicStructurePayload "
            "or a provider with build_payload/build_core_hole_payload"
        )
    if not isinstance(payload, ElectronicStructurePayload):
        raise ProviderContractError("provider did not return an ElectronicStructurePayload")
    return payload


def _assert_molecular_auger_eri_budget_before_validation(
    payload: ElectronicStructurePayload,
) -> None:
    if (
        payload.two_electron is not None
        and payload.two_electron.basis == "ao"
        and payload.orbitals is not None
    ):
        _assert_ao_to_mo_eri_budget(payload.two_electron.eri, payload.orbitals)


def _validate_core_hole_spec(
    spec: str | int | dict[str, Any] | None,
) -> str | int | dict[str, Any] | None:
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ProviderContractError("core_hole_spec MO index must be an integer")
    if isinstance(spec, int):
        if spec < 0:
            raise ProviderContractError("core_hole_spec MO index must be non-negative")
        return spec
    if isinstance(spec, str):
        if not spec.strip():
            raise ProviderContractError("core_hole_spec label must be non-empty")
        return spec
    if not isinstance(spec, dict):
        raise ProviderContractError(
            "core_hole_spec must be None, a non-empty string, a non-negative "
            "MO index, or a one-key dict"
        )
    if len(spec) != 1:
        raise ProviderContractError(
            "core_hole_spec dict must contain exactly one supported identity key"
        )
    key, value = next(iter(spec.items()))
    if key not in _CORE_HOLE_IDENTITY_KEYS:
        supported = ", ".join(sorted(_CORE_HOLE_IDENTITY_KEYS))
        raise ProviderContractError(
            f"unsupported core_hole_spec key {key!r}; supported keys: {supported}"
        )
    _validate_core_hole_identity_value(value)
    return spec


def _validate_core_hole_identity_value(value: Any) -> None:
    if isinstance(value, bool):
        raise ProviderContractError("core_hole_spec MO index must be an integer")
    if isinstance(value, int):
        if value < 0:
            raise ProviderContractError("core_hole_spec MO index must be non-negative")
        return
    if isinstance(value, str) and value.strip():
        return
    raise ProviderContractError(
        "core_hole_spec identity value must be a non-empty string or "
        "non-negative MO index"
    )


def _validate_active_space_parameters(
    population_threshold: float,
    occupation_threshold: float,
    residual_tolerance: float,
) -> tuple[float, float, float]:
    raw_values = (
        ("population_threshold", population_threshold, False),
        ("occupation_threshold", occupation_threshold, False),
        ("residual_tolerance", residual_tolerance, True),
    )
    values: list[tuple[str, float, bool]] = []
    for name, raw_value, strictly_positive in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ProviderContractError(f"{name} must be a real number")
        try:
            value = float(raw_value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ProviderContractError(
                f"{name} must be a finite real number"
            ) from exc
        values.append((name, value, strictly_positive))
    for name, value, strictly_positive in values:
        if not np.isfinite(value):
            raise ProviderContractError(f"{name} must be finite")
        if strictly_positive:
            if value <= 0.0:
                raise ProviderContractError(f"{name} must be greater than zero")
        elif value < 0.0:
            raise ProviderContractError(f"{name} must be non-negative")
    return values[0][1], values[1][1], values[2][1]


def _assert_active_space_budget(active_orbital_count: int) -> None:
    if active_orbital_count > _MAX_MOLECULAR_AUGER_ACTIVE_ORBITALS:
        raise ProviderContractError(
            "molecular Auger active space exceeds the supported budget: "
            f"{active_orbital_count} active MOs > "
            f"{_MAX_MOLECULAR_AUGER_ACTIVE_ORBITALS}; reduce thresholds or "
            "split the calculation"
        )


def _core_hole_metadata(
    payload: ElectronicStructurePayload,
    *,
    n_mo: int | None = None,
    require_tracked_mo: bool = False,
) -> dict[str, Any]:
    spec = payload.metadata.get("core_hole")
    if not isinstance(spec, dict):
        raise ProviderContractError("payload metadata lacks core_hole specification")
    missing = [
        key
        for key in (
            "occupation_before",
            "occupation_after",
            "initial_state_energy_hartree",
        )
        if key not in spec
    ]
    if missing:
        raise ProviderContractError(
            "core_hole specification missing required fields: "
            + ", ".join(sorted(missing))
        )
    if "orbital_label" in spec and (
        not isinstance(spec["orbital_label"], str) or not spec["orbital_label"].strip()
    ):
        raise ProviderContractError("core_hole.orbital_label must be a non-empty string")
    occupation_before = _validate_core_hole_occupation(
        spec["occupation_before"],
        "core_hole.occupation_before",
    )
    occupation_after = _validate_core_hole_occupation(
        spec["occupation_after"],
        "core_hole.occupation_after",
    )
    if occupation_after >= occupation_before:
        raise ProviderContractError(
            "core_hole.occupation_after must be less than occupation_before"
        )
    initial_energy = _validate_core_hole_real(
        spec["initial_state_energy_hartree"],
        "core_hole.initial_state_energy_hartree",
    )
    validated = dict(spec)
    validated["occupation_before"] = occupation_before
    validated["occupation_after"] = occupation_after
    validated["initial_state_energy_hartree"] = initial_energy
    if require_tracked_mo or "tracked_mo_index" in spec:
        validated["tracked_mo_index"] = _validate_tracked_core_mo_index(
            spec,
            n_mo,
        )
    return validated


def _validate_core_hole_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProviderContractError(f"{field} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ProviderContractError(f"{field} must be finite")
    return numeric


def _validate_core_hole_occupation(value: Any, field: str) -> float:
    occupation = _validate_core_hole_real(value, field)
    if occupation < 0.0 or occupation > 2.0:
        raise ProviderContractError(f"{field} must be between 0 and 2")
    return occupation


def _validate_tracked_core_mo_index(
    core_spec: dict[str, Any],
    n_mo: int | None,
) -> int:
    if "tracked_mo_index" not in core_spec:
        raise ProviderContractError(
            "core_hole.tracked_mo_index is required to exclude the core MO "
            "from the valence active space"
        )
    value = core_spec["tracked_mo_index"]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProviderContractError("core_hole.tracked_mo_index must be an integer")
    tracked_core = int(value)
    if tracked_core < 0:
        raise ProviderContractError(
            "core_hole.tracked_mo_index is outside the MO coefficient range"
        )
    if n_mo is not None and tracked_core >= n_mo:
        raise ProviderContractError(
            "core_hole.tracked_mo_index is outside the MO coefficient range"
        )
    return tracked_core


def _call_core_hole_builder(
    builder: Any,
    core_hole_spec: str | int | dict[str, Any],
) -> ElectronicStructurePayload:
    sig = signature(builder)
    params = sig.parameters
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    if any(p.kind == Parameter.VAR_POSITIONAL for p in params.values()):
        args.append(core_hole_spec)
    else:
        for name in ("core_hole_spec", "core_orbital", "core_hole"):
            if name in params:
                kwargs[name] = core_hole_spec
                break
        else:
            positional = [
                p
                for p in params.values()
                if p.kind
                in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is Parameter.empty
            ]
            if len(positional) == 1:
                args.append(core_hole_spec)
            else:
                raise ProviderContractError(
                    "provider build_core_hole_payload cannot accept core_hole_spec"
                )

    if "with_eri" in params or any(
        p.kind == Parameter.VAR_KEYWORD for p in params.values()
    ):
        kwargs["with_eri"] = True
    return cast(ElectronicStructurePayload, builder(*args, **kwargs))


def _requested_core_hole_label(
    spec: str | int | dict[str, Any] | None,
) -> str | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return f"MO{spec}"
    if isinstance(spec, str):
        return spec
    for key in ("orbital_label", "core_hole", "core_orbital"):
        if key in spec:
            value = spec[key]
            return f"MO{value}" if isinstance(value, int) else str(value)
    return None


def _payload_core_hole_label(payload: ElectronicStructurePayload) -> str:
    spec = payload.metadata.get("core_hole")
    if isinstance(spec, dict) and spec.get("orbital_label"):
        return str(spec["orbital_label"])
    return payload.provenance.core_hole


def _assert_requested_core_hole(
    payload: ElectronicStructurePayload,
    core_hole_spec: str | int | dict[str, Any] | None,
) -> None:
    requested = _requested_core_hole_label(core_hole_spec)
    if requested is None:
        return
    actual = _payload_core_hole_label(payload)
    if actual != requested:
        raise ProviderContractError(
            "provider returned a different core-hole payload: "
            f"requested {requested!r}, got {actual!r}"
        )


def molecular_auger_initial_state(
    payload: ElectronicStructurePayload,
) -> CoreHoleState:
    """Build the molecular Auger core-hole initial state from a payload.

    Required capabilities fail closed before any state is accepted:
    ``CORE_HOLE_SPEC``, ``MO_COEFFICIENTS`` and ``OVERLAP_S``.
    """

    payload.require(
        Capability.CORE_HOLE_SPEC,
        Capability.MO_COEFFICIENTS,
        Capability.OVERLAP_S,
    )
    spec = _core_hole_metadata(payload)

    return CoreHoleState(
        orbital_label=str(spec.get("orbital_label") or payload.provenance.core_hole),
        occupation_before=spec["occupation_before"],
        occupation_after=spec["occupation_after"],
        initial_energy=spec["initial_state_energy_hartree"],
        provenance=payload.provenance,
    )


def _mo_label(index: int) -> str:
    return f"MO{index}"


def _core_atom_symbol(core_label: str) -> str:
    parts = core_label.split()
    if len(parts) < 2:
        raise ProviderContractError(
            "core-hole orbital label must include an atom and shell, e.g. 'O 1s'"
        )
    return parts[0]


@dataclass(frozen=True)
class _AOLabel:
    center: str
    atom: str


def _ao_label_identity(ao_label: str) -> _AOLabel:
    parts = ao_label.split()
    if len(parts) >= 3:
        return _AOLabel(center=parts[0], atom=parts[1])
    raise ProviderContractError(
        f"AO label {ao_label!r} must have '<center> <element> <shell>' format"
    )


def _basis_labels(payload: ElectronicStructurePayload) -> tuple[str, ...]:
    orbitals = payload.orbitals
    overlap = payload.overlap
    labels = None
    if overlap is not None:
        labels = overlap.basis_labels
    if labels is None and orbitals is not None:
        labels = orbitals.basis_labels
    if labels is None:
        if orbitals is None:
            return ()
        return tuple(f"AO{i}" for i in range(orbitals.n_ao))
    return tuple(str(label) for label in labels)


def _selected_coefficients(
    orbitals: OrbitalCoefficients,
    active_indices: tuple[int, ...],
    localization_transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        orbitals.coefficients[:, active_indices] @ localization_transform,
    )


def _active_residual(
    payload: ElectronicStructurePayload,
    active_indices: tuple[int, ...],
    localization_transform: NDArray[np.float64],
) -> float:
    if payload.orbitals is None:
        raise ProviderContractError("active-space residual requires MO coefficients")
    coeff = _selected_coefficients(
        payload.orbitals, active_indices, localization_transform
    )
    S = (
        payload.overlap.matrix
        if payload.overlap is not None
        else np.eye(coeff.shape[0])
    )
    gram = coeff.T @ S @ coeff
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def _core_hole_center(core_spec: object) -> str | None:
    if not isinstance(core_spec, dict):
        return None
    for key in ("atom_center", "center_index", "atom_index"):
        if key in core_spec:
            return str(core_spec[key])
    return None


def _core_atom_ao_indices(
    labels: tuple[str, ...],
    core_atom: str,
    core_center: str | None,
) -> tuple[int, ...]:
    parsed = tuple(_ao_label_identity(label) for label in labels)
    matching_centers = {
        label.center for label in parsed if label.atom.lower() == core_atom.lower()
    }
    if not matching_centers:
        raise ProviderContractError(f"no AO labels match core atom {core_atom!r}")
    if core_center is None:
        if len(matching_centers) > 1:
            raise ProviderContractError(
                "core-hole active-space selection is ambiguous for multiple "
                f"{core_atom!r} centers; provide core_hole atom_center/center_index"
            )
        core_center = next(iter(matching_centers))
    selected = tuple(
        index
        for index, label in enumerate(parsed)
        if label.atom.lower() == core_atom.lower() and label.center == core_center
    )
    if not selected:
        raise ProviderContractError(
            f"no AO labels match core atom {core_atom!r} at center {core_center!r}"
        )
    return selected


def _tracked_core_mo_index(core_spec: object, n_mo: int) -> int:
    if not isinstance(core_spec, dict):
        raise ProviderContractError(
            "core_hole.tracked_mo_index is required to exclude the core MO "
            "from the valence active space"
        )
    return _validate_tracked_core_mo_index(core_spec, n_mo)


def local_valence_active_space(
    payload: ElectronicStructurePayload,
    core_hole: CoreHoleState | None = None,
    *,
    population_threshold: float = 0.2,
    occupation_threshold: float = 1.5,
    residual_tolerance: float = 1e-7,
) -> LocalizedValenceActiveSpace:
    """Select local occupied valence MOs around the core-hole atom.

    The first implementation uses Mulliken population analysis on one
    core-hole atom center and keeps canonical MOs as the active orbitals.  AO
    labels must have ``<center> <element> <shell>`` format.  If the core-hole
    spec does not identify a center, selection only proceeds when the core
    element appears on exactly one center.  The core-hole spec must also carry
    ``tracked_mo_index`` so the selected active space can fail closed rather
    than accidentally treat the core MO as a valence orbital.

    To keep two-hole enumeration and downstream ERI contractions bounded, the
    selected local valence active space is capped at 16 MOs.
    """

    population_threshold, occupation_threshold, residual_tolerance = (
        _validate_active_space_parameters(
            population_threshold,
            occupation_threshold,
            residual_tolerance,
        )
    )
    payload.require(
        Capability.CORE_HOLE_SPEC,
        Capability.MO_COEFFICIENTS,
        Capability.OVERLAP_S,
    )
    orbitals = payload.orbitals
    overlap = payload.overlap
    if orbitals is None or overlap is None:
        raise ProviderContractError("local active-space selection requires C and S")
    labels = _basis_labels(payload)
    if len(labels) != orbitals.n_ao:
        raise ProviderContractError("AO label count must match coefficient rows")
    core_label = (
        core_hole.orbital_label
        if core_hole is not None
        else payload.provenance.core_hole
    )
    core_atom = _core_atom_symbol(core_label)
    core_spec = _core_hole_metadata(
        payload,
        n_mo=orbitals.n_mo,
        require_tracked_mo=True,
    )
    atom_indices = np.asarray(
        _core_atom_ao_indices(labels, core_atom, _core_hole_center(core_spec)),
        dtype=int,
    )

    C = orbitals.coefficients
    SC = overlap.matrix @ C
    populations = np.einsum("mi,mi->i", C[atom_indices, :], SC[atom_indices, :])
    tracked_core = _tracked_core_mo_index(core_spec, orbitals.n_mo)

    active = []
    for index, (energy, occupation, population) in enumerate(
        zip(orbitals.energies, orbitals.occupations, populations, strict=True)
    ):
        if tracked_core is not None and index == tracked_core:
            continue
        if occupation < occupation_threshold or population < population_threshold:
            continue
        if not np.isclose(occupation, 2.0, atol=1e-6):
            raise ProviderContractError(
                "two-hole closed-shell active space requires doubly occupied "
                f"valence MOs; {_mo_label(index)} has occupation {occupation:.6g}"
            )
        active.append(
            ActiveMolecularOrbital(
                label=_mo_label(index),
                index=index,
                energy=float(energy),
                occupation=float(occupation),
                core_atom_population=float(population),
            )
        )
    if not active:
        raise ProviderContractError(
            "no occupied valence MOs passed the local Mulliken thresholds"
        )
    _assert_active_space_budget(len(active))

    active_indices = tuple(orb.index for orb in active)
    transform = np.eye(len(active_indices))
    residual = _active_residual(payload, active_indices, transform)
    if residual > residual_tolerance:
        raise ProviderContractError(
            "active-space MOs fail C_active^T S C_active = I: "
            f"residual={residual:.3e}, tolerance={residual_tolerance:.3e}"
        )
    provenance = _copy_provenance_with(
        payload.provenance,
        approximations=(
            "mulliken-active-space",
            "single-core-center-active-space",
            "canonical-mo-localization-identity",
        ),
    )
    return LocalizedValenceActiveSpace(
        orbitals=tuple(active),
        parent_ao_labels=labels,
        overlap_s=overlap.matrix.copy(),
        parent_mo_indices=active_indices,
        localization_method="mulliken-canonical-mo",
        localization_transform=transform,
        population_threshold=float(population_threshold),
        occupation_threshold=float(occupation_threshold),
        orthonormality_residual=residual,
        provenance=provenance,
    )


def enumerate_two_hole_csfs(
    active_space: LocalizedValenceActiveSpace,
) -> tuple[TwoHoleCSF, ...]:
    """Enumerate singlet/triplet two-hole CSFs over active MO labels.

    For a closed-shell active space with ``n`` doubly occupied spatial MOs this
    yields ``n`` same-orbital singlets plus ``2*C(n, 2)`` distinct-orbital
    singlet/triplet CSFs.  MO labels are retained; no central-field quantum
    numbers are invented.
    """

    csfs: list[TwoHoleCSF] = []
    for left, right in combinations_with_replacement(active_space.labels, 2):
        if left == right:
            csfs.append(
                TwoHoleCSF(
                    hole_orbitals=(left, right),
                    spin_multiplicity=1,
                    spin_symmetry="singlet",
                    label=f"{left}^-2 singlet",
                )
            )
            continue
        for multiplicity, symmetry in ((1, "singlet"), (3, "triplet")):
            csfs.append(
                TwoHoleCSF(
                    hole_orbitals=(left, right),
                    spin_multiplicity=multiplicity,
                    spin_symmetry=symmetry,
                    label=f"{left}^-1 {right}^-1 {symmetry}",
                )
            )
    return tuple(csfs)


def molecular_auger_final_states(
    payload: ElectronicStructurePayload,
    core_hole: CoreHoleState | None = None,
    *,
    population_threshold: float = 0.2,
    occupation_threshold: float = 1.5,
    residual_tolerance: float = 1e-7,
    coulomb_corrections: dict[tuple[str, str], float] | None = None,
) -> tuple[TwoHoleFinalState, ...]:
    """Build independent-hole molecular Auger final states.

    Energies use the same declared zero as the core-hole initial state and name
    the two-valence-hole final state after core refilling.  The first-stage
    independent-hole model estimates ``E_final = E_core_hole + epsilon_core -
    epsilon_i - epsilon_j + U_ij`` using the tracked core MO energy and optional
    caller-provided Coulomb correction.  Correlated Cini-Sawatzky/cluster-CI
    final states are intentionally left to HYP-918.
    """

    population_threshold, occupation_threshold, residual_tolerance = (
        _validate_active_space_parameters(
            population_threshold,
            occupation_threshold,
            residual_tolerance,
        )
    )
    initial = core_hole or molecular_auger_initial_state(payload)
    active_space = local_valence_active_space(
        payload,
        initial,
        population_threshold=population_threshold,
        occupation_threshold=occupation_threshold,
        residual_tolerance=residual_tolerance,
    )
    orbitals = payload.orbitals
    if orbitals is None:
        raise ProviderContractError("final-state energies require MO coefficients")
    tracked_core = _tracked_core_mo_index(
        payload.metadata.get("core_hole"), orbitals.n_mo
    )
    core_energy = float(orbitals.energies[tracked_core])
    energy_by_label = {orb.label: orb.energy for orb in active_space.orbitals}
    corrections = coulomb_corrections or {}
    final_states: list[TwoHoleFinalState] = []
    for csf in enumerate_two_hole_csfs(active_space):
        holes = csf.hole_orbitals
        correction = float(corrections.get(holes, corrections.get(holes[::-1], 0.0)))
        final_label = csf.label
        final_energy = (
            initial.initial_energy
            + core_energy
            - energy_by_label[holes[0]]
            - energy_by_label[holes[1]]
            + correction
        )
        provenance = _copy_provenance_with(
            payload.provenance,
            final_state=final_label,
            approximations=(
                "mulliken-active-space",
                "single-core-center-active-space",
                "canonical-mo-localization-identity",
                "independent-hole-koopmans",
            ),
        )
        final_states.append(
            TwoHoleFinalState(
                hole_orbitals=holes,
                spin_multiplicity=csf.spin_multiplicity,
                spin_symmetry=csf.spin_symmetry,
                final_energy=final_energy,
                provenance=provenance,
                active_space=active_space,
                coulomb_correction=correction,
            )
        )
    return tuple(final_states)


def _mo_index_from_label(label: str) -> int:
    if not label.startswith("MO"):
        raise ProviderContractError(f"expected an MO label, got {label!r}")
    try:
        return int(label[2:])
    except ValueError as exc:
        raise ProviderContractError(f"invalid MO label {label!r}") from exc


def _assert_ao_to_mo_eri_budget(
    eri: NDArray[np.float64],
    orbitals: OrbitalCoefficients,
) -> None:
    n_ao = orbitals.n_ao
    n_mo = orbitals.n_mo
    if eri.shape != (n_ao, n_ao, n_ao, n_ao):
        raise ProviderContractError(
            "ao-basis ERI shape must match orbitals.n_ao on every axis: "
            f"expected {(n_ao, n_ao, n_ao, n_ao)}, got {eri.shape}"
        )
    c_logical_bytes = orbitals.coefficients.size * orbitals.coefficients.itemsize
    eri_logical_bytes = eri.size * eri.itemsize
    c_stride_span_bytes = _array_stride_span_bytes(orbitals.coefficients)
    eri_stride_span_bytes = _array_stride_span_bytes(eri)
    output_bytes = (n_mo**4) * eri.dtype.itemsize
    logical_input_bytes = c_logical_bytes + eri_logical_bytes
    stride_span_input_bytes = c_stride_span_bytes + eri_stride_span_bytes
    if (
        n_mo > _MAX_MOLECULAR_AUGER_AO_TO_MO_ORBITALS
        or n_ao > _MAX_MOLECULAR_AUGER_AO_TO_MO_AOS
        or eri.size > _MAX_MOLECULAR_AUGER_AO_TO_MO_ELEMENTS
        or logical_input_bytes > _MAX_MOLECULAR_AUGER_AO_TO_MO_INPUT_BYTES
        or stride_span_input_bytes > _MAX_MOLECULAR_AUGER_AO_TO_MO_INPUT_BYTES
        or output_bytes > _MAX_MOLECULAR_AUGER_AO_TO_MO_OUTPUT_BYTES
    ):
        raise ProviderContractError(
            "AO-to-MO ERI transformation exceeds the molecular Auger budget: "
            f"n_mo={n_mo} <= {_MAX_MOLECULAR_AUGER_AO_TO_MO_ORBITALS}, "
            f"n_ao={n_ao} <= {_MAX_MOLECULAR_AUGER_AO_TO_MO_AOS}, "
            "dense AO ERI elements="
            f"{eri.size} <= {_MAX_MOLECULAR_AUGER_AO_TO_MO_ELEMENTS}, "
            "estimated logical input bytes="
            f"{logical_input_bytes} <= "
            f"{_MAX_MOLECULAR_AUGER_AO_TO_MO_INPUT_BYTES}, "
            "estimated stride-span input bytes="
            f"{stride_span_input_bytes} <= "
            f"{_MAX_MOLECULAR_AUGER_AO_TO_MO_INPUT_BYTES}, "
            "estimated MO ERI output bytes="
            f"{output_bytes} <= {_MAX_MOLECULAR_AUGER_AO_TO_MO_OUTPUT_BYTES}"
        )


def _array_stride_span_bytes(array: NDArray[np.float64]) -> int:
    if array.size == 0:
        return 0
    low = 0
    high = 0
    for dimension, stride in zip(array.shape, array.strides, strict=False):
        extent = (dimension - 1) * stride
        low += min(0, extent)
        high += max(0, extent)
    return high - low + array.itemsize


def _assert_eri_finite(eri: NDArray[np.float64], basis: str) -> None:
    if not np.all(np.isfinite(eri)):
        raise ProviderContractError(f"{basis}-basis ERI contains non-finite values")


def _mo_eri(payload: ElectronicStructurePayload) -> NDArray[np.float64]:
    payload.require(Capability.TWO_ELECTRON_INTEGRALS, Capability.MO_COEFFICIENTS)
    if payload.two_electron is None or payload.orbitals is None:
        raise ProviderContractError("molecular Auger amplitudes require ERIs and MOs")
    eri = payload.two_electron.eri
    if payload.two_electron.notation != "chemist":
        raise ProviderContractError("molecular Auger requires chemist ERI notation")
    if payload.two_electron.basis == "mo":
        _assert_eri_finite(eri, "mo")
        return cast(NDArray[np.float64], eri)
    if payload.two_electron.basis != "ao":
        raise ProviderContractError("two-electron integral basis must be 'ao' or 'mo'")
    _assert_ao_to_mo_eri_budget(eri, payload.orbitals)
    _assert_eri_finite(eri, "ao")
    C = payload.orbitals.coefficients
    return cast(
        NDArray[np.float64],
        np.einsum("pa,qb,rc,sd,pqrs->abcd", C, C, C, C, eri, optimize=True),
    )


def _line_provenance(
    provenance: Provenance,
    final_state: str,
    continuum_channel: ContinuumChannel,
    *,
    sum_rule_normalized: bool,
) -> Provenance:
    approximations = (
        *continuum_channel.approximations,
        "direct-minus-exchange-amplitude",
        "independent-particle-auger-strength",
    )
    if sum_rule_normalized:
        approximations = (*approximations, "sum-rule-total-stick-normalization")
    return _copy_provenance_with(
        provenance,
        final_state=final_state,
        approximations=approximations,
    )


def molecular_auger_transition_amplitude(
    payload: ElectronicStructurePayload,
    initial_state: CoreHoleState,
    final_state: TwoHoleFinalState,
    continuum_channel: ContinuumChannel | None = None,
) -> AugerTransitionAmplitude:
    """Compute one molecular Auger amplitude from provider ERIs.

    Direct and exchange are combined as ``direct - exchange`` before squaring
    (ADR I3).  Triplet two-hole states are currently spin-forbidden in the
    closed-shell non-SOC v1 path and return an exactly zero amplitude rather
    than an incoherent rate.
    """

    if final_state.spin_multiplicity != 1:
        return AugerTransitionAmplitude(
            direct=0.0,
            exchange=0.0,
            amplitude=0.0,
            selection_rule="spin-forbidden-non-soc-v1",
        )
    eri = _mo_eri(payload)
    return _molecular_auger_transition_amplitude_from_mo_eri(
        payload,
        initial_state,
        final_state,
        cast(ContinuumChannel, continuum_channel or DiscreteMOContinuumChannel()),
        eri,
    )


def _molecular_auger_transition_amplitude_from_mo_eri(
    payload: ElectronicStructurePayload,
    initial_state: CoreHoleState,
    final_state: TwoHoleFinalState,
    channel: ContinuumChannel,
    eri: NDArray[np.float64],
) -> AugerTransitionAmplitude:
    kinetic_energy = initial_state.initial_energy - final_state.final_energy
    coeff = channel.coefficients(payload, kinetic_energy)
    orbitals = payload.orbitals
    if orbitals is None:
        raise ProviderContractError("molecular Auger amplitudes require MOs")
    if eri.shape != (orbitals.n_mo, orbitals.n_mo, orbitals.n_mo, orbitals.n_mo):
        raise ProviderContractError(
            "mo-basis ERI shape must match orbitals.n_mo on every axis: "
            f"expected {(orbitals.n_mo,) * 4}, got {eri.shape}"
        )
    if coeff.shape != (orbitals.n_mo,):
        raise ProviderContractError(
            "continuum coefficient vector must have length orbitals.n_mo"
        )
    core = _tracked_core_mo_index(payload.metadata.get("core_hole"), orbitals.n_mo)
    hole_i, hole_j = (
        _mo_index_from_label(label) for label in final_state.hole_orbitals
    )
    direct = float(np.einsum("e,e->", coeff, eri[core, :, hole_i, hole_j]))
    exchange = float(np.einsum("e,e->", coeff, eri[core, :, hole_j, hole_i]))
    return AugerTransitionAmplitude(
        direct=direct,
        exchange=exchange,
        amplitude=direct - exchange,
    )


def _dropped_line(final_state: TwoHoleFinalState, reason: str) -> dict[str, str]:
    return {
        "final_state": final_state.provenance.final_state,
        "reason": reason,
    }


def _stick_from_final_state(
    payload: ElectronicStructurePayload,
    initial: CoreHoleState,
    final: TwoHoleFinalState,
    channel: ContinuumChannel,
    mo_eri: NDArray[np.float64] | None,
    *,
    normalize: bool,
) -> AugerStick | dict[str, str]:
    kinetic_energy = initial.initial_energy - final.final_energy
    if kinetic_energy <= 0.0:
        return _dropped_line(final, "non_positive_kinetic_energy")
    if final.spin_multiplicity != 1:
        amplitude = AugerTransitionAmplitude(
            direct=0.0,
            exchange=0.0,
            amplitude=0.0,
            selection_rule="spin-forbidden-non-soc-v1",
        )
    else:
        if mo_eri is None:
            raise ProviderContractError("molecular Auger amplitudes require ERIs")
        amplitude = _molecular_auger_transition_amplitude_from_mo_eri(
            payload,
            initial,
            final,
            channel,
            mo_eri,
        )
    raw_intensity = amplitude.coherent_intensity
    if raw_intensity <= 0.0:
        reason = (
            amplitude.selection_rule
            if amplitude.selection_rule != "allowed"
            else "zero_coherent_intensity"
        )
        return _dropped_line(final, reason)
    provenance = _line_provenance(
        final.provenance,
        final.provenance.final_state,
        channel,
        sum_rule_normalized=normalize,
    )
    return AugerStick(
        kinetic_energy=kinetic_energy,
        intensity=raw_intensity,
        raw_intensity=raw_intensity,
        amplitude=amplitude,
        initial_state=initial,
        final_state=final,
        continuum_label=channel.label,
        provenance=provenance,
    )


def _normalize_sticks(
    raw_lines: list[AugerStick],
    normalize: bool,
) -> tuple[AugerStick, ...]:
    total = sum(line.raw_intensity for line in raw_lines)
    if not normalize or total <= 0.0:
        return tuple(raw_lines)
    return tuple(
        AugerStick(
            kinetic_energy=line.kinetic_energy,
            intensity=line.raw_intensity / total,
            raw_intensity=line.raw_intensity,
            amplitude=line.amplitude,
            initial_state=line.initial_state,
            final_state=line.final_state,
            continuum_label=line.continuum_label,
            provenance=line.provenance,
        )
        for line in raw_lines
    )


def molecular_auger_spectrum(
    payload: ElectronicStructurePayload,
    *,
    core_hole: CoreHoleState | None = None,
    final_states: tuple[TwoHoleFinalState, ...] | None = None,
    continuum_channel: ContinuumChannel | None = None,
    normalize: bool = True,
    population_threshold: float = 0.2,
    occupation_threshold: float = 1.5,
    residual_tolerance: float = 1e-7,
) -> SpectrumResult:
    """Synthesize an Auger kinetic-energy stick spectrum.

    Lines with non-positive kinetic energy or zero/forbidden coherent
    intensity are dropped and reported.  Normalization, when enabled, is a
    single total-stick sum-rule normalization; no empirical per-peak scaling is
    applied.
    """

    population_threshold, occupation_threshold, residual_tolerance = (
        _validate_active_space_parameters(
            population_threshold,
            occupation_threshold,
            residual_tolerance,
        )
    )
    initial = core_hole or molecular_auger_initial_state(payload)
    finals = final_states or molecular_auger_final_states(
        payload,
        initial,
        population_threshold=population_threshold,
        occupation_threshold=occupation_threshold,
        residual_tolerance=residual_tolerance,
    )
    if finals:
        _assert_active_space_budget(
            max(len(final.active_space.orbitals) for final in finals)
        )
    channel = cast(ContinuumChannel, continuum_channel or DiscreteMOContinuumChannel())
    mo_eri: NDArray[np.float64] | None = None
    raw_lines: list[AugerStick] = []
    dropped: list[dict[str, str]] = []
    for final in finals:
        kinetic_energy = initial.initial_energy - final.final_energy
        if kinetic_energy > 0.0 and final.spin_multiplicity == 1 and mo_eri is None:
            mo_eri = _mo_eri(payload)
        item = _stick_from_final_state(
            payload, initial, final, channel, mo_eri, normalize=normalize
        )
        if isinstance(item, AugerStick):
            raw_lines.append(item)
        else:
            dropped.append(item)
    lines = _normalize_sticks(raw_lines, normalize)
    return SpectrumResult(
        lines=lines,
        dropped=tuple(dropped),
        provenance=tuple(line.provenance for line in lines),
        intensity_unit="sum_rule_normalized_probability"
        if normalize
        else "coherent_auger_strength",
    )


def _copy_provenance_with(
    provenance: Provenance,
    *,
    final_state: str | None = None,
    approximations: tuple[str, ...] = (),
) -> Provenance:
    seen = set()
    merged_approximations = []
    for item in [*provenance.approximations, *approximations]:
        if item not in seen:
            seen.add(item)
            merged_approximations.append(item)
    return Provenance(
        provider=provenance.provider,
        code_version=provenance.code_version,
        method=provenance.method,
        basis=provenance.basis,
        theory_level=provenance.theory_level,
        approximations=merged_approximations,
        core_hole=provenance.core_hole,
        final_state=provenance.final_state if final_state is None else final_state,
        energy_zero=provenance.energy_zero,
        is_ml=provenance.is_ml,
        ood_status=provenance.ood_status,
        uncertainty_summary=provenance.uncertainty_summary,
    )
