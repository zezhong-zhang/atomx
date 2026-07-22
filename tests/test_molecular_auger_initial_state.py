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
    TwoHoleFinalState,
    enumerate_two_hole_csfs,
    local_valence_active_space,
    molecular_auger_final_states,
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


def _active_payload():
    C = np.eye(4)
    return ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="sto-3g",
            theory_level="RHF",
            approximations=["nonrel-gto"],
            core_hole="O 1s",
            energy_zero="delta_scf_total_energy",
        ),
        orbital_energies={"MO0": -20.0, "MO1": -1.0, "MO2": -0.5, "MO3": 0.2},
        orbitals=OrbitalCoefficients(
            coefficients=C,
            energies=np.array([-20.0, -1.0, -0.5, 0.2]),
            occupations=np.array([1.0, 2.0, 2.0, 0.0]),
            basis_labels=["0 O 1s", "0 O 2s", "0 O 2p", "1 H 1s"],
        ),
        overlap=OverlapBlock(
            np.eye(4),
            basis_labels=["0 O 1s", "0 O 2s", "0 O 2p", "1 H 1s"],
        ),
        density=DensityMatrix(np.diag([1.0, 2.0, 2.0, 0.0])),
        metadata={
            "core_hole": {
                "orbital_label": "O 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            }
        },
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


def test_local_valence_active_space_records_transform_contract():
    active = local_valence_active_space(_active_payload(), population_threshold=0.5)

    assert active.labels == ("MO1", "MO2")
    assert active.parent_ao_labels == ("0 O 1s", "0 O 2s", "0 O 2p", "1 H 1s")
    assert active.parent_mo_indices == (1, 2)
    assert active.localization_method == "mulliken-canonical-mo"
    np.testing.assert_allclose(active.localization_transform, np.eye(2))
    np.testing.assert_allclose(active.overlap_s, np.eye(4))
    assert active.orthonormality_residual <= 1e-12
    assert "mulliken-active-space" in active.provenance.approximations


def test_two_hole_csf_count_matches_closed_shell_spin_coupling_formula():
    active = local_valence_active_space(_active_payload(), population_threshold=0.5)
    csfs = enumerate_two_hole_csfs(active)

    n_active = len(active.orbitals)
    expected = n_active + 2 * (n_active * (n_active - 1) // 2)
    assert len(csfs) == expected
    assert [csf.spin_symmetry for csf in csfs] == [
        "singlet",
        "singlet",
        "triplet",
        "singlet",
    ]
    assert all("MO" in csf.label for csf in csfs)


def test_two_hole_final_states_stamp_i2_i4_i5_provenance_and_energy():
    final_states = molecular_auger_final_states(
        _active_payload(),
        population_threshold=0.5,
        coulomb_corrections={("MO1", "MO2"): 0.1},
    )

    pair = next(
        state
        for state in final_states
        if state.hole_orbitals == ("MO1", "MO2")
        and state.spin_symmetry == "singlet"
    )
    assert pair.final_energy == pytest.approx(19.5 - (-1.0) - (-0.5) + 0.1)
    assert pair.provenance.final_state == "MO1^-1 MO2^-1 singlet"
    assert pair.provenance.energy_zero == "delta_scf_total_energy"
    assert "independent-hole-koopmans" in pair.provenance.approximations
    assert "mulliken-active-space" in pair.provenance.approximations


def test_two_hole_final_state_rejects_missing_approximation_stamp():
    active = local_valence_active_space(_active_payload(), population_threshold=0.5)
    with pytest.raises(ProviderContractError):
        TwoHoleFinalState(
            hole_orbitals=("MO1", "MO2"),
            spin_multiplicity=1,
            spin_symmetry="singlet",
            final_energy=20.0,
            provenance=Provenance(
                provider="mock",
                code_version="mock 0",
                method="RHF",
                basis="sto-3g",
                theory_level="RHF",
                core_hole="O 1s",
                final_state="MO1^-1 MO2^-1 singlet",
            ),
            active_space=active,
        )
