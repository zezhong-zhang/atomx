"""Periodic LCAO Auger stick spectra.

This module consumes provider payloads only.  It reuses the molecular
direct-minus-exchange two-hole machinery at each k-point and then applies
Brillouin-zone and local-projection weights.  The result is explicitly a
matrix-element-weighted Auger stick spectrum, not a DOS/EELS surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    """

    raw_lines: list[PeriodicAugerLine] = []
    dropped: list[dict[str, str]] = []
    for payload in tuple(payloads):
        payload.require(
            Capability.CORE_HOLE_SPEC,
            Capability.MO_COEFFICIENTS,
            Capability.OVERLAP_S,
            Capability.KPOINTS,
        )
        spectrum = molecular_auger_spectrum(
            payload,
            continuum_channel=continuum_channel,
            normalize=False,
            population_threshold=population_threshold,
            occupation_threshold=occupation_threshold,
            residual_tolerance=residual_tolerance,
        )
        kpoint = _metadata_tuple(payload, "kpoint", length=3)
        k_weight = _metadata_nonnegative(payload, "k_weight")
        projection = _metadata_nonnegative(payload, "local_projection_weight")
        for item in spectrum.dropped:
            dropped.append({"kpoint": str(kpoint), **item})
        for line in spectrum.lines:
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


def _metadata_tuple(
    payload: ElectronicStructurePayload,
    key: str,
    *,
    length: int,
) -> tuple[float, ...]:
    value = payload.metadata.get(key)
    if value is None:
        raise ProviderContractError(f"periodic Auger payload lacks {key!r}")
    result = tuple(float(x) for x in value)
    if len(result) != length:
        raise ProviderContractError(f"periodic Auger metadata {key!r} has wrong length")
    return result


def _metadata_nonnegative(payload: ElectronicStructurePayload, key: str) -> float:
    if key not in payload.metadata:
        raise ProviderContractError(f"periodic Auger payload lacks {key!r}")
    value = float(payload.metadata[key])
    if value < 0.0:
        raise ProviderContractError(f"periodic Auger metadata {key!r} is negative")
    return value


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
