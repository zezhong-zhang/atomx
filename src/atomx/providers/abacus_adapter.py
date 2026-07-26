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
    "ABACUSLCAOProvider",
    "from_abacus_lcao",
]


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

    def validate(self, *, tol: float = 1e-8) -> "ABACUSKPointData":
        coord = np.asarray(tuple(self.coordinate), dtype=float)
        if coord.shape != (3,):
            raise ProviderContractError("ABACUS k-point coordinate must have length 3")
        if float(self.weight) < 0.0:
            raise ProviderContractError("ABACUS k-point weight must be non-negative")
        H = _real_matrix(self.hamiltonian, "ABACUS Hamiltonian")
        S = _real_matrix(self.overlap, "ABACUS overlap")
        if H.shape != S.shape:
            raise ProviderContractError("ABACUS H and S shapes must match")
        if not np.allclose(H, H.T, atol=tol):
            raise ProviderContractError("ABACUS Hamiltonian must be Hermitian")
        if not np.allclose(S, S.T, atol=tol):
            raise ProviderContractError("ABACUS overlap must be Hermitian")
        eig = np.linalg.eigvalsh(S)
        if float(np.min(eig)) <= tol:
            raise ProviderContractError("ABACUS overlap must be positive definite")
        C = np.asarray(self.coefficients, dtype=float)
        if C.ndim != 2 or C.shape[0] != S.shape[0]:
            raise ProviderContractError("ABACUS MO coefficients must have shape (n_ao, n_mo)")
        n_mo = C.shape[1]
        energies = np.asarray(self.energies, dtype=float)
        occupations = np.asarray(self.occupations, dtype=float)
        if energies.shape != (n_mo,) or occupations.shape != (n_mo,):
            raise ProviderContractError("ABACUS energies/occupations must have length n_mo")
        gram = C.T @ S @ C
        residual = float(np.max(np.abs(gram - np.eye(n_mo))))
        if residual > tol:
            raise ProviderContractError(
                f"ABACUS MO coefficients fail C^T S C = I: residual={residual:.3e}"
            )
        if self.density is not None:
            P = np.asarray(self.density, dtype=float)
            if P.shape != S.shape:
                raise ProviderContractError("ABACUS density shape must match overlap")
        if self.two_electron is not None:
            eri = np.asarray(self.two_electron, dtype=float)
            if eri.ndim != 4 or len(set(eri.shape)) != 1 or eri.shape[0] != n_mo:
                raise ProviderContractError("ABACUS MVP two-electron tensor must be MO-basis (n_mo,)^4")
        if float(self.local_projection_weight) < 0.0:
            raise ProviderContractError("local projection weight must be non-negative")
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
    ) -> None:
        self.Z = int(Z)
        self.kpoint_data = tuple(k.validate(tol=validation_tolerance) for k in kpoints)
        if not self.kpoint_data:
            raise ProviderContractError("ABACUS provider requires at least one k-point")
        self.ao_labels = tuple(str(label) for label in ao_labels)
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
        self._kblock = KPointBlock(
            coordinates=np.asarray([k.coordinate for k in self.kpoint_data]),
            weights=np.asarray([k.weight for k in self.kpoint_data]),
        )

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
        weight = float(self._kblock.weights[index])
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
            kpoints=self._kblock,
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
    arr = np.asarray(value)
    if np.iscomplexobj(arr) and not np.allclose(arr.imag, 0.0):
        raise ProviderContractError(f"{name} must be real in the MVP adapter")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ProviderContractError(f"{name} must be a square matrix")
    return arr


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
    normalized["occupation_before"] = float(normalized["occupation_before"])
    normalized["occupation_after"] = float(normalized["occupation_after"])
    normalized["tracked_mo_index"] = int(normalized["tracked_mo_index"])
    normalized["initial_state_energy_hartree"] = ensure_hartree(
        normalized["initial_state_energy_hartree"]
    )
    return normalized
