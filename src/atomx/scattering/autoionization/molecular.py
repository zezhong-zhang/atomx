"""Molecular Auger initial-state entry points.

This module is intentionally provider-contract only: it consumes an
``ElectronicStructurePayload`` and never imports a back-end such as PySCF.
The current HYP-910.A surface stops at the explicitly named core-hole
initial state; Auger-rate kernels are wired in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import cast

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

__all__ = [
    "ActiveMolecularOrbital",
    "LocalizedValenceActiveSpace",
    "TwoHoleCSF",
    "TwoHoleFinalState",
    "CoreHoleState",
    "enumerate_two_hole_csfs",
    "local_valence_active_space",
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
        if self.overlap_s.ndim != 2 or self.overlap_s.shape[0] != self.overlap_s.shape[1]:
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
        if self.hole_orbitals[0] == self.hole_orbitals[1] and self.spin_multiplicity != 1:
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
            raise ProviderContractError("final-state provenance.final_state is required")
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

    return CoreHoleState(
        orbital_label=str(spec.get("orbital_label") or payload.provenance.core_hole),
        occupation_before=float(spec["occupation_before"]),
        occupation_after=float(spec["occupation_after"]),
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
    S = payload.overlap.matrix if payload.overlap is not None else np.eye(coeff.shape[0])
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
        label.center
        for label in parsed
        if label.atom.lower() == core_atom.lower()
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
    element appears on exactly one center.
    """

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
        core_hole.orbital_label if core_hole is not None else payload.provenance.core_hole
    )
    core_atom = _core_atom_symbol(core_label)
    core_spec = payload.metadata.get("core_hole")
    atom_indices = np.asarray(
        _core_atom_ao_indices(labels, core_atom, _core_hole_center(core_spec)),
        dtype=int,
    )

    C = orbitals.coefficients
    SC = overlap.matrix @ C
    populations = np.einsum("mi,mi->i", C[atom_indices, :], SC[atom_indices, :])
    tracked_core = None
    if isinstance(core_spec, dict) and "tracked_mo_index" in core_spec:
        tracked_core = int(core_spec["tracked_mo_index"])

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

    Energies use the same declared zero as the core-hole initial state.  The
    first-stage model removes two occupied valence electrons from the core-hole
    state using Koopmans orbital energies plus an optional caller-provided
    Coulomb correction.  Correlated Cini-Sawatzky/cluster-CI final states are
    intentionally left to HYP-918.
    """

    initial = core_hole or molecular_auger_initial_state(payload)
    active_space = local_valence_active_space(
        payload,
        initial,
        population_threshold=population_threshold,
        occupation_threshold=occupation_threshold,
        residual_tolerance=residual_tolerance,
    )
    energy_by_label = {orb.label: orb.energy for orb in active_space.orbitals}
    corrections = coulomb_corrections or {}
    final_states: list[TwoHoleFinalState] = []
    for csf in enumerate_two_hole_csfs(active_space):
        holes = csf.hole_orbitals
        correction = float(corrections.get(holes, corrections.get(holes[::-1], 0.0)))
        final_label = csf.label
        final_energy = (
            initial.initial_energy
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
