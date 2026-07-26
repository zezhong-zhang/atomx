"""Periodic LCAO Auger stick spectra.

This module consumes provider payloads only.  It reuses the molecular
direct-minus-exchange two-hole machinery at each k-point and then applies
Brillouin-zone and local-projection weights.  The result is explicitly a
matrix-element-weighted Auger stick spectrum, not a DOS/EELS surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from atomx.providers.contract import (
    Capability,
    ElectronicStructurePayload,
    Provenance,
    ProviderContractError,
)
from atomx.scattering.autoionization.molecular import (
    AugerStick,
    ContinuumChannel,
    molecular_auger_spectrum,
)

__all__ = [
    "PeriodicAugerLine",
    "PeriodicAugerSpectrum",
    "periodic_auger_spectrum",
]


@dataclass(frozen=True)
class PeriodicAugerLine:
    """One k-weighted Auger stick line from a periodic LCAO payload."""

    kinetic_energy: float
    intensity: float
    raw_intensity: float
    kpoint: tuple[float, float, float]
    k_weight: float
    local_projection_weight: float
    source_line: AugerStick
    provenance: Provenance


@dataclass(frozen=True)
class PeriodicAugerSpectrum:
    """Aggregated periodic Auger spectrum."""

    lines: tuple[PeriodicAugerLine, ...]
    dropped: tuple[dict[str, str], ...]
    provenance: tuple[Provenance, ...]
    axis: str = "kinetic_energy"
    axis_unit: str = "hartree"
    intensity_unit: str = "sum_rule_normalized_k_weighted_probability"

    @property
    def kinetic_energies(self) -> tuple[float, ...]:
        return tuple(line.kinetic_energy for line in self.lines)

    @property
    def intensities(self) -> tuple[float, ...]:
        return tuple(line.intensity for line in self.lines)


def periodic_auger_spectrum(
    payloads: Iterable[ElectronicStructurePayload],
    *,
    continuum_channel: ContinuumChannel | None = None,
    normalize: bool = True,
    population_threshold: float = 0.2,
    occupation_threshold: float = 1.5,
    residual_tolerance: float = 1e-7,
) -> PeriodicAugerSpectrum:
    """Synthesize a k-weighted periodic Auger spectrum.

    Each payload represents one k-point from an imported periodic LCAO
    provider.  The underlying molecular spectrum is evaluated with raw
    coherent intensities; this wrapper applies ``k_weight`` and
    ``local_projection_weight`` before optional total-stick normalization.
    The MVP supports real Gamma/LCAO inputs only; non-real arrays fail closed
    instead of silently discarding imaginary parts.
    """

    _validate_runtime_scalars(
        population_threshold=population_threshold,
        occupation_threshold=occupation_threshold,
        residual_tolerance=residual_tolerance,
    )
    payload_tuple = tuple(payloads)
    k_contracts = [_payload_kpoint_contract(payload) for payload in payload_tuple]
    _validate_k_weight_conservation(k_contracts)
    raw_lines: list[PeriodicAugerLine] = []
    dropped: list[dict[str, str]] = []
    for payload, (kpoint, k_weight, projection) in zip(
        payload_tuple, k_contracts, strict=True
    ):
        payload.require(
            Capability.CORE_HOLE_SPEC,
            Capability.MO_COEFFICIENTS,
            Capability.OVERLAP_S,
            Capability.KPOINTS,
        )
        _require_real_finite_payload_arrays(payload)
        spectrum = molecular_auger_spectrum(
            payload,
            continuum_channel=continuum_channel,
            normalize=False,
            population_threshold=population_threshold,
            occupation_threshold=occupation_threshold,
            residual_tolerance=residual_tolerance,
        )
        for item in spectrum.dropped:
            dropped.append({"kpoint": str(kpoint), **item})
        for line in spectrum.lines:
            if not np.isfinite(line.raw_intensity) or line.raw_intensity < 0.0:
                raise ProviderContractError(
                    "molecular Auger raw intensity must be finite and non-negative"
                )
            weighted = line.raw_intensity * k_weight * projection
            if weighted <= 0.0:
                dropped.append(
                    {
                        "kpoint": str(kpoint),
                        "final_state": line.final_state.provenance.final_state,
                        "reason": "zero_k_or_projection_weight",
                    }
                )
                continue
            raw_lines.append(
                PeriodicAugerLine(
                    kinetic_energy=line.kinetic_energy,
                    intensity=weighted,
                    raw_intensity=weighted,
                    kpoint=kpoint,
                    k_weight=k_weight,
                    local_projection_weight=projection,
                    source_line=line,
                    provenance=_periodic_provenance(line.provenance),
                )
            )
    total = sum(line.raw_intensity for line in raw_lines)
    if normalize and total > 0.0:
        lines = tuple(
            PeriodicAugerLine(
                kinetic_energy=line.kinetic_energy,
                intensity=line.raw_intensity / total,
                raw_intensity=line.raw_intensity,
                kpoint=line.kpoint,
                k_weight=line.k_weight,
                local_projection_weight=line.local_projection_weight,
                source_line=line.source_line,
                provenance=line.provenance,
            )
            for line in raw_lines
        )
    else:
        lines = tuple(raw_lines)
    return PeriodicAugerSpectrum(
        lines=lines,
        dropped=tuple(dropped),
        provenance=tuple(line.provenance for line in lines),
        intensity_unit="sum_rule_normalized_k_weighted_probability"
        if normalize
        else "k_weighted_coherent_auger_strength",
    )


def _validate_runtime_scalars(
    *,
    population_threshold: float,
    occupation_threshold: float,
    residual_tolerance: float,
) -> None:
    population_threshold = _finite_float(
        population_threshold, "population_threshold"
    )
    occupation_threshold = _finite_float(
        occupation_threshold, "occupation_threshold"
    )
    residual_tolerance = _finite_float(
        residual_tolerance, "residual_tolerance"
    )
    if population_threshold < 0.0:
        raise ProviderContractError("population_threshold must be non-negative")
    if not 0.0 <= occupation_threshold <= 2.0:
        raise ProviderContractError("occupation_threshold must be in [0, 2]")
    if residual_tolerance <= 0.0:
        raise ProviderContractError("residual_tolerance must be positive")


def _payload_kpoint_contract(
    payload: ElectronicStructurePayload,
) -> tuple[tuple[float, float, float], float, float]:
    if payload.kpoints is None:
        raise ProviderContractError("periodic Auger payload lacks typed kpoints")
    coordinates = np.asarray(payload.kpoints.coordinates)
    weights = np.asarray(payload.kpoints.weights)
    _require_real_finite_array(coordinates, "k-point coordinates")
    _require_real_finite_array(weights, "k-point weights")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ProviderContractError("k-point coordinates must have shape (n_k, 3)")
    if weights.shape != (coordinates.shape[0],):
        raise ProviderContractError("k-point weights must have length n_k")
    if np.any(weights < 0.0):
        raise ProviderContractError("k-point weights must be non-negative")
    if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-12, rtol=0.0):
        raise ProviderContractError("k-point weights must sum to one")

    index = _typed_kpoint_index(payload, upper=coordinates.shape[0])
    typed_kpoint = tuple(float(x) for x in coordinates[index])
    k_weight = float(weights[index])
    if "kpoint_index" in payload.metadata:
        metadata_index = _optional_metadata_index(payload, "kpoint_index")
        if metadata_index is not None and metadata_index != index:
            raise ProviderContractError(
                "periodic Auger metadata 'kpoint_index' conflicts with typed kpoints"
            )
    metadata_kpoint = _optional_metadata_tuple(payload, "kpoint", length=3)
    if metadata_kpoint is not None and not np.allclose(
        metadata_kpoint, typed_kpoint, atol=1e-12, rtol=0.0
    ):
        raise ProviderContractError(
            "periodic Auger metadata 'kpoint' conflicts with typed kpoints"
        )
    if "k_weight" in payload.metadata:
        metadata_weight = _finite_float(
            payload.metadata["k_weight"], "periodic Auger metadata 'k_weight'"
        )
        if not np.isclose(metadata_weight, k_weight, atol=1e-12, rtol=0.0):
            raise ProviderContractError(
                "periodic Auger metadata 'k_weight' conflicts with typed kpoints"
            )
    projection = _metadata_probability(payload, "local_projection_weight")
    return typed_kpoint, k_weight, projection


def _validate_k_weight_conservation(
    contracts: list[tuple[tuple[float, float, float], float, float]],
) -> None:
    total = float(sum(weight for _, weight, _ in contracts))
    if not np.isclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise ProviderContractError("periodic Auger payload k weights must sum to one")


def _typed_kpoint_index(payload: ElectronicStructurePayload, *, upper: int) -> int:
    index = payload.kpoints.selected_index
    if index is None:
        raise ProviderContractError("periodic Auger payload lacks typed selected k-point index")
    if not 0 <= index < upper:
        raise ProviderContractError("typed selected k-point index is out of range")
    return int(index)


def _optional_metadata_index(
    payload: ElectronicStructurePayload,
    key: str,
) -> int | None:
    value = payload.metadata.get(key)
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(
            f"periodic Auger metadata {key!r} is out of range"
        ) from exc
    if result != value:
        raise ProviderContractError(f"periodic Auger metadata {key!r} is out of range")
    return result


def _optional_metadata_tuple(
    payload: ElectronicStructurePayload,
    key: str,
    *,
    length: int,
) -> tuple[float, ...] | None:
    value = payload.metadata.get(key)
    if value is None:
        return None
    result = tuple(float(x) for x in value)
    if not all(np.isfinite(x) for x in result):
        raise ProviderContractError(f"periodic Auger metadata {key!r} must be finite")
    if len(result) != length:
        raise ProviderContractError(f"periodic Auger metadata {key!r} has wrong length")
    return result


def _metadata_probability(payload: ElectronicStructurePayload, key: str) -> float:
    if key not in payload.metadata:
        raise ProviderContractError(f"periodic Auger payload lacks {key!r}")
    value = _finite_float(payload.metadata[key], f"periodic Auger metadata {key!r}")
    if not 0.0 <= value <= 1.0:
        raise ProviderContractError(f"periodic Auger metadata {key!r} must be in [0, 1]")
    return value


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ProviderContractError(f"{name} must be finite")
    return result


def _require_real_finite_payload_arrays(payload: ElectronicStructurePayload) -> None:
    if payload.overlap is not None:
        _require_real_finite_array(payload.overlap.matrix, "overlap S")
    if payload.orbitals is not None:
        _require_real_finite_array(payload.orbitals.coefficients, "MO coefficients")
        _require_real_finite_array(payload.orbitals.energies, "MO energies")
        _require_real_finite_array(payload.orbitals.occupations, "MO occupations")
    if payload.density is not None:
        _require_real_finite_array(payload.density.matrix, "density matrix")
    if payload.two_electron is not None:
        _require_real_finite_array(payload.two_electron.eri, "two-electron integrals")
    if "hamiltonian" in payload.metadata:
        _require_real_finite_array(payload.metadata["hamiltonian"], "Hamiltonian")


def _require_real_finite_array(value: object, name: str) -> None:
    arr = np.asarray(value)
    if np.iscomplexobj(arr) and not np.allclose(arr.imag, 0.0, atol=0.0, rtol=0.0):
        raise ProviderContractError(f"{name} must be real in the periodic Auger MVP")
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ProviderContractError(f"{name} must contain only finite values")


def _periodic_provenance(provenance: Provenance) -> Provenance:
    approximations = [
        *provenance.approximations,
        "k-weighted-periodic-lcao",
        "local-projection-weighted",
        "matrix-element-weighted-not-dos",
    ]
    seen = set()
    unique = []
    for item in approximations:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return Provenance(
        provider=provenance.provider,
        code_version=provenance.code_version,
        method=provenance.method,
        basis=provenance.basis,
        theory_level=provenance.theory_level,
        approximations=unique,
        core_hole=provenance.core_hole,
        final_state=provenance.final_state,
        energy_zero=provenance.energy_zero,
        is_ml=provenance.is_ml,
        ood_status=provenance.ood_status,
        uncertainty_summary=provenance.uncertainty_summary,
    )
