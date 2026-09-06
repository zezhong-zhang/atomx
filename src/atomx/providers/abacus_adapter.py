"""ABACUS LCAO result adapter for periodic Auger MVPs.

The adapter consumes imported ABACUS LCAO arrays; it does not run ABACUS and
does not make ABACUS an Atomx dependency.  The contract boundary validates the
matrix properties that the downstream two-hole kernels rely on: Hermitian
Hamiltonian/overlap matrices, positive-definite overlap, S-orthonormal MOs,
normalized k weights, and explicit core-hole metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from atomx.providers.contract import (
    Capability,
    DensityMatrix,
    ElectronicStructurePayload,
    KPointBlock,
    OrbitalCoefficients,
    OverlapBlock,
    PeriodicCell,
    ProviderContractError,
    Provenance,
    SpinBlock,
    TwoElectronIntegrals,
    ensure_hartree,
)
from atomx.providers.pyscf_adapter import MOLabel

__all__ = [
    "ABACUSKPointData",
    "ABACUSValidationLimits",
    "ABACUSLCAOProvider",
    "from_abacus_lcao",
]

_RESERVED_METADATA_KEYS = frozenset(
    {
        "core_hole",
        "kpoint_index",
        "kpoint",
        "k_weight",
        "local_projection_weight",
        "hamiltonian",
    }
)


@dataclass(frozen=True)
class ABACUSValidationLimits:
    """Resource limits enforced before expensive ABACUS validation work."""

    max_kpoints: int = 512
    max_ao: int = 4096
    max_mo: int = 4096
    max_array_elements: int = 20_000_000
    max_input_bytes: int = 512_000_000
    max_eri_elements: int = 10_000_000

    def validate(self) -> "ABACUSValidationLimits":
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ProviderContractError(f"{name} must be a positive integer")
        return self


@dataclass(frozen=True)
class ABACUSKPointData:
    """One k-point of imported ABACUS LCAO data."""

    coordinate: Iterable[float]
    weight: float
    hamiltonian: np.ndarray
    overlap: np.ndarray
    coefficients: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    density: np.ndarray | None = None
    two_electron: np.ndarray | None = None
    local_projection_weight: float = 1.0

    def validate(
        self,
        *,
        tol: float = 1e-8,
        limits: ABACUSValidationLimits | None = None,
    ) -> "ABACUSKPointData":
        limits = (limits or ABACUSValidationLimits()).validate()
        coord = _real_array(
            _collect_bounded(
                self.coordinate,
                3,
                "ABACUS k-point coordinate must have length 3",
            ),
            "ABACUS k-point coordinate",
        )
        if coord.shape != (3,):
            raise ProviderContractError("ABACUS k-point coordinate must have length 3")
        weight = _real_float(self.weight, "ABACUS k-point weight")
        if weight < 0.0:
            raise ProviderContractError("ABACUS k-point weight must be non-negative")
        H = _real_matrix(self.hamiltonian, "ABACUS Hamiltonian")
        S = _real_matrix(self.overlap, "ABACUS overlap")
        if H.shape != S.shape:
            raise ProviderContractError("ABACUS H and S shapes must match")
        if H.shape[0] > limits.max_ao:
            raise ProviderContractError("ABACUS n_ao exceeds validation limit")
        _enforce_array_limits(limits, H, S)
        if not np.allclose(H, H.T, atol=tol, rtol=0.0):
            raise ProviderContractError("ABACUS Hamiltonian must be Hermitian")
        if not np.allclose(S, S.T, atol=tol, rtol=0.0):
            raise ProviderContractError("ABACUS overlap must be Hermitian")
        eig = np.linalg.eigvalsh(S)
        if float(np.min(eig)) <= tol:
            raise ProviderContractError("ABACUS overlap must be positive definite")
        C = _real_array(self.coefficients, "ABACUS MO coefficients")
        if C.ndim != 2 or C.shape[0] != S.shape[0]:
            raise ProviderContractError("ABACUS MO coefficients must have shape (n_ao, n_mo)")
        n_mo = C.shape[1]
        if n_mo > limits.max_mo:
            raise ProviderContractError("ABACUS n_mo exceeds validation limit")
        energies = _real_array(self.energies, "ABACUS energies")
        occupations = _real_array(self.occupations, "ABACUS occupations")
        if energies.shape != (n_mo,) or occupations.shape != (n_mo,):
            raise ProviderContractError("ABACUS energies/occupations must have length n_mo")
        if np.any((occupations < 0.0) | (occupations > 2.0)):
            raise ProviderContractError("ABACUS occupations must be in [0, 2]")
        _enforce_array_limits(limits, C, energies, occupations)
        gram = C.T @ S @ C
        residual = float(np.max(np.abs(gram - np.eye(n_mo))))
        if residual > tol:
            raise ProviderContractError(
                f"ABACUS MO coefficients fail C^T S C = I: residual={residual:.3e}"
            )
        if self.density is not None:
            P = _real_array(self.density, "ABACUS density")
            if P.shape != S.shape:
                raise ProviderContractError("ABACUS density shape must match overlap")
            _enforce_array_limits(limits, P)
        if self.two_electron is not None:
            eri = _real_array(self.two_electron, "ABACUS two-electron tensor")
            if eri.ndim != 4 or len(set(eri.shape)) != 1 or eri.shape[0] != n_mo:
                raise ProviderContractError("ABACUS MVP two-electron tensor must be MO-basis (n_mo,)^4")
            if eri.size > limits.max_eri_elements:
                raise ProviderContractError("ABACUS ERI tensor exceeds validation limit")
            _enforce_array_limits(limits, eri)
        projection = _real_float(
            self.local_projection_weight, "local projection weight"
        )
        if not 0.0 <= projection <= 1.0:
            raise ProviderContractError("local projection weight must be in [0, 1]")
        return self


class ABACUSLCAOProvider:
    """Imported ABACUS LCAO data exposed as Atomx provider payloads."""

    def __init__(
        self,
        *,
        Z: int,
        kpoints: Iterable[ABACUSKPointData],
        ao_labels: Iterable[str],
        core_hole: dict[str, Any],
        basis: str = "abacus-lcao",
        method: str = "ABACUS-LCAO",
        code_version: str = "abacus imported",
        theory_level: str = "periodic-dft-lcao",
        cell: PeriodicCell | None = None,
        spin_soc: bool = False,
        metadata: dict[str, Any] | None = None,
        validation_tolerance: float = 1e-8,
        validation_limits: ABACUSValidationLimits | None = None,
    ) -> None:
        self.Z = int(Z)
        validation_tolerance = _real_float(
            validation_tolerance, "ABACUS validation tolerance"
        )
        if validation_tolerance <= 0.0:
            raise ProviderContractError("ABACUS validation tolerance must be positive")
        limits = (validation_limits or ABACUSValidationLimits()).validate()
        raw_kpoints = _collect_bounded_kpoints(kpoints, limits.max_kpoints)
        _enforce_multikpoint_resource_limits(raw_kpoints, limits)
        self.kpoint_data = tuple(
            k.validate(tol=validation_tolerance, limits=limits) for k in raw_kpoints
        )
        if not self.kpoint_data:
            raise ProviderContractError("ABACUS provider requires at least one k-point")
        self.ao_labels = tuple(
            str(label)
            for label in _collect_bounded(
                ao_labels,
                limits.max_ao,
                "ABACUS AO label count exceeds validation limit",
            )
        )
        n_ao = self.kpoint_data[0].overlap.shape[0]
        if len(self.ao_labels) != n_ao:
            raise ProviderContractError("ABACUS AO label count must match H/S shape")
        for kdata in self.kpoint_data:
            if kdata.overlap.shape[0] != n_ao:
                raise ProviderContractError("all ABACUS k-points must share n_ao")
        self.core_hole = _normalize_core_hole(core_hole)
        self.provenance = Provenance(
            provider="abacus",
            code_version=code_version,
            method=method,
            basis=basis,
            theory_level=theory_level,
            approximations=[
                "periodic-lcao-import",
                "abacus-runtime-not-invoked",
                "continuum:atomx-residual-ion-required",
            ],
            core_hole=str(self.core_hole["orbital_label"]),
            energy_zero="core_hole_total_energy",
            is_ml=False,
        )
        self.cell = cell
        self.spin_soc = bool(spin_soc)
        self.metadata = dict(metadata or {})
        collision = _RESERVED_METADATA_KEYS & self.metadata.keys()
        if collision:
            raise ProviderContractError(
                "ABACUS metadata may not override reserved fields: "
                + ", ".join(sorted(collision))
            )
        self._k_coordinates = _real_array(
            [k.coordinate for k in self.kpoint_data], "ABACUS k-point coordinates"
        )
        self._k_weights = KPointBlock(
            coordinates=self._k_coordinates,
            weights=_real_array(
                [k.weight for k in self.kpoint_data], "ABACUS k-point weights"
            ),
        ).weights

    def capabilities(self):
        return self.build_payload().capabilities()

    def supports(self, *requested: Capability) -> bool:
        return self.build_payload().supports(*requested)

    def build_payload(self) -> ElectronicStructurePayload:
        """Return the first k-point payload for Protocol compatibility."""

        return self.kpoint_payload(0)

    def iter_kpoint_payloads(self) -> tuple[ElectronicStructurePayload, ...]:
        return tuple(self.kpoint_payload(i) for i in range(len(self.kpoint_data)))

    def kpoint_payload(self, index: int) -> ElectronicStructurePayload:
        kdata = self.kpoint_data[index]
        weight = float(self._k_weights[index])
        energies = np.asarray([ensure_hartree(e) for e in kdata.energies], dtype=float)
        occupations = np.asarray(kdata.occupations, dtype=float)
        orbitals = OrbitalCoefficients(
            coefficients=np.asarray(kdata.coefficients, dtype=float),
            energies=energies,
            occupations=occupations,
            basis_labels=list(self.ao_labels),
            spin="restricted",
        )
        overlap = OverlapBlock(np.asarray(kdata.overlap, dtype=float), list(self.ao_labels))
        density = (
            DensityMatrix(np.asarray(kdata.density, dtype=float), basis="ao")
            if kdata.density is not None
            else None
        )
        two_electron = (
            TwoElectronIntegrals(np.asarray(kdata.two_electron, dtype=float), basis="mo")
            if kdata.two_electron is not None
            else None
        )
        payload = ElectronicStructurePayload(
            Z=self.Z,
            provenance=self.provenance,
            subshells=[MOLabel(f"MO{i}", occ) for i, occ in enumerate(occupations)],
            orbital_energies={f"MO{i}": float(e) for i, e in enumerate(energies)},
            overlap=overlap,
            orbitals=orbitals,
            density=density,
            two_electron=two_electron,
            spin=SpinBlock(soc=self.spin_soc),
            kpoints=KPointBlock(
                coordinates=self._k_coordinates,
                weights=self._k_weights,
                selected_index=index,
            ),
            cell=self.cell,
            metadata={
                "core_hole": self.core_hole,
                "kpoint_index": int(index),
                "kpoint": tuple(float(x) for x in kdata.coordinate),
                "k_weight": weight,
                "local_projection_weight": float(kdata.local_projection_weight),
                "hamiltonian": np.asarray(kdata.hamiltonian, dtype=float),
                **self.metadata,
            },
        )
        payload.require(Capability.CORE_HOLE_SPEC, Capability.KPOINTS)
        return payload.validate()


def from_abacus_lcao(**kwargs: Any) -> ABACUSLCAOProvider:
    """Build an :class:`ABACUSLCAOProvider` from imported ABACUS arrays."""

    return ABACUSLCAOProvider(**kwargs)


def _real_matrix(value: np.ndarray, name: str) -> np.ndarray:
    arr = _real_array(value, name)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ProviderContractError(f"{name} must be a square matrix")
    return arr


def _real_float(value: Any, name: str) -> float:
    try:
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            if arr.shape != ():
                raise TypeError("expected scalar")
            if not np.isclose(float(arr.imag), 0.0, atol=0.0, rtol=0.0):
                raise ProviderContractError(f"{name} must be real in the MVP adapter")
            result = float(arr.real)
        else:
            result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ProviderContractError(f"{name} must be finite")
    return result


def _real_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if np.iscomplexobj(arr) and not np.allclose(arr.imag, 0.0, atol=0.0, rtol=0.0):
        raise ProviderContractError(f"{name} must be real in the MVP adapter")
    arr = np.asarray(arr.real if np.iscomplexobj(arr) else arr, dtype=float)
    _require_finite_array(arr, name)
    return arr


def _require_finite_array(value: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ProviderContractError(f"{name} must contain only finite values")


def _enforce_array_limits(
    limits: ABACUSValidationLimits,
    *arrays: np.ndarray,
) -> None:
    elements = sum(int(arr.size) for arr in arrays)
    bytes_ = sum(int(arr.nbytes) for arr in arrays)
    if elements > limits.max_array_elements:
        raise ProviderContractError("ABACUS input arrays exceed element limit")
    if bytes_ > limits.max_input_bytes:
        raise ProviderContractError("ABACUS input arrays exceed byte limit")


def _collect_bounded(
    items: Iterable[Any],
    limit: int,
    message: str,
) -> tuple[Any, ...]:
    """Materialize at most ``limit`` items from an unknown-length iterable.

    The (``limit`` + 1)-th pull raises immediately and the remaining tail is
    never consumed, so an oversized or infinite iterable cannot exhaust
    memory or hang before the contracted limit applies.
    """
    collected: list[Any] = []
    for count, item in enumerate(items, start=1):
        if count > limit:
            raise ProviderContractError(message)
        collected.append(item)
    return tuple(collected)


def _collect_bounded_kpoints(
    kpoints: Iterable[ABACUSKPointData],
    max_kpoints: int,
) -> tuple[ABACUSKPointData, ...]:
    return _collect_bounded(
        kpoints, max_kpoints, "ABACUS n_k exceeds validation limit"
    )


_FLOAT64_ITEMSIZE = np.dtype(np.float64).itemsize


@dataclass
class _ResourceAccount:
    """Streaming resource quota for the ABACUS preflight.

    The preflight must reject over-budget payloads before any expensive
    coercion or allocation, so it inspects only genuine ``np.ndarray``
    metadata (``size``/``dtype.itemsize``) and contracted scalar footprints.
    It never calls ``np.asarray`` and never triggers an input's ``__array__``:
    a non-``ndarray`` field is refused outright rather than coerced for
    counting, so an attacker-controlled wrapper cannot allocate inside the
    advertised preflight or under-report its footprint.
    """

    limits: ABACUSValidationLimits
    elements: int = 0
    bytes_: int = 0
    eri_elements: int = 0

    def _tally(self, size: int, bytes_: int, *, is_eri: bool) -> None:
        # Python ``int`` tallies cannot overflow; reject the moment any
        # advertised cap is crossed so later fields/k-points are never converted.
        self.elements += size
        self.bytes_ += bytes_
        if self.elements > self.limits.max_array_elements:
            raise ProviderContractError("ABACUS input arrays exceed element limit")
        if self.bytes_ > self.limits.max_input_bytes:
            raise ProviderContractError("ABACUS input arrays exceed byte limit")
        if is_eri:
            self.eri_elements += size
            if self.eri_elements > self.limits.max_eri_elements:
                raise ProviderContractError("ABACUS ERI tensor exceeds validation limit")

    def add_array(self, value: Any, *, is_eri: bool = False) -> None:
        """Account one genuine ``np.ndarray`` field without coercing it."""

        if not isinstance(value, np.ndarray):
            raise ProviderContractError(
                "ABACUS preflight requires np.ndarray array fields; "
                "non-ndarray inputs are rejected before any conversion or allocation"
            )
        size = int(value.size)
        itemsize = int(value.dtype.itemsize)
        # Validation coerces every array to float64, so account the
        # float64-equivalent footprint as a lower bound on post-coercion memory.
        bytes_ = max(size * itemsize, size * _FLOAT64_ITEMSIZE)
        self._tally(size, bytes_, is_eri=is_eri)

    def add_scalar(self, *, count: int = 1) -> None:
        """Account ``count`` float64-sized scalars (coordinates, weights, ...)."""

        self._tally(count, count * _FLOAT64_ITEMSIZE, is_eri=False)


def _enforce_multikpoint_resource_limits(
    kpoints: tuple[ABACUSKPointData, ...],
    limits: ABACUSValidationLimits,
) -> None:
    account = _ResourceAccount(limits)
    for kdata in kpoints:
        # The coordinate is a length-3 iterable per contract; account its
        # contracted float64 footprint directly instead of materializing or
        # inspecting the iterable, so a pathological coordinate cannot
        # allocate or run conversion code inside the preflight.
        account.add_scalar(count=3)
        account.add_scalar()  # k-point weight
        account.add_array(kdata.hamiltonian)
        account.add_array(kdata.overlap)
        account.add_array(kdata.coefficients)
        account.add_array(kdata.energies)
        account.add_array(kdata.occupations)
        account.add_scalar()  # local_projection_weight
        if kdata.density is not None:
            account.add_array(kdata.density)
        if kdata.two_electron is not None:
            account.add_array(kdata.two_electron, is_eri=True)


def _normalize_core_hole(core_hole: dict[str, Any]) -> dict[str, Any]:
    required = {
        "orbital_label",
        "occupation_before",
        "occupation_after",
        "tracked_mo_index",
        "initial_state_energy_hartree",
    }
    missing = required - set(core_hole)
    if missing:
        raise ProviderContractError(
            "ABACUS core-hole metadata missing required fields: "
            + ", ".join(sorted(missing))
        )
    normalized = dict(core_hole)
    normalized["occupation_before"] = _real_float(
        normalized["occupation_before"], "core-hole occupation_before"
    )
    normalized["occupation_after"] = _real_float(
        normalized["occupation_after"], "core-hole occupation_after"
    )
    if not 0.0 <= normalized["occupation_before"] <= 2.0:
        raise ProviderContractError("core-hole occupation_before must be in [0, 2]")
    if not 0.0 <= normalized["occupation_after"] <= 2.0:
        raise ProviderContractError("core-hole occupation_after must be in [0, 2]")
    normalized["tracked_mo_index"] = int(normalized["tracked_mo_index"])
    if normalized["tracked_mo_index"] < 0:
        raise ProviderContractError("core-hole tracked_mo_index must be non-negative")
    normalized["initial_state_energy_hartree"] = ensure_hartree(
        normalized["initial_state_energy_hartree"]
    )
    _real_float(
        normalized["initial_state_energy_hartree"],
        "core-hole initial_state_energy_hartree",
    )
    return normalized
