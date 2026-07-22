"""Molecular Auger initial-state contract tests (HYP-910.A)."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.providers.contract import (
    Capability,
    CapabilityError,
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
)
from atomx.scattering.autoionization.molecular import (
    CoreHoleState,
    molecular_auger_initial_state,
)


def _payload(*, core_hole: str = "O 1s", overlap=True, metadata=True):
    C = np.eye(2)
    return ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="sto-3g",
            theory_level="RHF",
            core_hole=core_hole,
            energy_zero="delta_scf_total_energy",
        ),
        orbital_energies={"MO0": -20.0, "MO1": -1.0},
        orbitals=OrbitalCoefficients(
            coefficients=C,
            energies=np.array([-20.0, -1.0]),
            occupations=np.array([1.0, 2.0]),
        ),
        overlap=OverlapBlock(np.eye(2)) if overlap else None,
        density=DensityMatrix(np.diag([1.0, 2.0])),
        metadata={
            "core_hole": {
                "orbital_label": "O 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "initial_state_energy_hartree": 19.5,
            }
        }
        if metadata
        else {},
    )


@pytest.mark.parametrize(
    "payload",
    [
        _payload(core_hole=""),
        _payload(overlap=False),
        ElectronicStructurePayload(
            Z=8,
            provenance=Provenance(
                provider="mock",
                code_version="mock 0",
                method="RHF",
                basis="sto-3g",
                theory_level="RHF",
                core_hole="O 1s",
            ),
            orbital_energies={"MO0": -20.0},
            orbitals_raw={1: []},
        ),
    ],
)
def test_molecular_entry_requires_core_mo_and_overlap(payload):
    with pytest.raises(CapabilityError):
        molecular_auger_initial_state(payload)


def test_molecular_entry_builds_explicit_core_hole_state():
    state = molecular_auger_initial_state(_payload())
    assert isinstance(state, CoreHoleState)
    assert state.orbital_label == "O 1s"
    assert state.occupation_before == pytest.approx(2.0)
    assert state.occupation_after == pytest.approx(1.0)
    assert state.occupation_delta == pytest.approx(-1.0)
    assert state.initial_energy == pytest.approx(19.5)
    assert state.provenance.core_hole == "O 1s"
    assert state.provenance.energy_zero == "delta_scf_total_energy"


def test_molecular_entry_rejects_missing_core_hole_metadata():
    payload = _payload(metadata=False)
    with pytest.raises(ProviderContractError):
        molecular_auger_initial_state(payload)
