"""Unified electronic-structure provider layer (ADR-0001, epic HYP-902).

This package is the thin adapter/contract edge that feeds external
electronic-structure back-ends (PySCF, ABACUS, DeePTB) into the atomx
physics kernels through one stable contract — without any kernel ever
importing a back-end.

* :mod:`atomx.providers.contract` — the Protocol, capability enum,
  payload, provenance, optional blocks and fail-closed errors.
* :mod:`atomx.providers.native_adapter` — the native atomx solvers as
  the reference provider.
* :mod:`atomx.providers.pyscf_adapter` — the PySCF molecular adapter
  (optional dependency).

The PySCF adapter is imported lazily; importing this package never
requires PySCF to be installed.
"""

from __future__ import annotations

from atomx.providers.contract import (
    Capability,
    CapabilityError,
    DensityMatrix,
    ElectronicStructurePayload,
    ElectronicStructureProvider,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
    SpinBlock,
    TwoElectronIntegrals,
    UncertaintyBlock,
    UncertaintyGateError,
    UnitError,
    ensure_hartree,
)
from atomx.providers.native_adapter import NativeProvider, from_atomic_result

__all__ = [
    "Capability",
    "CapabilityError",
    "DensityMatrix",
    "ElectronicStructurePayload",
    "ElectronicStructureProvider",
    "OrbitalCoefficients",
    "OverlapBlock",
    "Provenance",
    "ProviderContractError",
    "SpinBlock",
    "TwoElectronIntegrals",
    "UncertaintyBlock",
    "UncertaintyGateError",
    "UnitError",
    "ensure_hartree",
    "NativeProvider",
    "from_atomic_result",
]
