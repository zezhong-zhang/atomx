"""Molecular Auger initial-state entry points.

This module is intentionally provider-contract only: it consumes an
``ElectronicStructurePayload`` and never imports a back-end such as PySCF.
The current HYP-910.A surface stops at the explicitly named core-hole
initial state; Auger-rate kernels are wired in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.providers.contract import (
    Capability,
    ElectronicStructurePayload,
    Provenance,
    ProviderContractError,
    ensure_hartree,
)

__all__ = ["CoreHoleState", "molecular_auger_initial_state"]


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

    return CoreHoleState(
        orbital_label=str(spec.get("orbital_label") or payload.provenance.core_hole),
        occupation_before=float(spec["occupation_before"]),
        occupation_after=float(spec["occupation_after"]),
        initial_energy=spec["initial_state_energy_hartree"],
        provenance=payload.provenance,
    )
