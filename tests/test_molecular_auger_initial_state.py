"""Molecular Auger initial-state contract tests (HYP-910.A)."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.providers.contract import (
    CapabilityError,
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
)
from atomx.scattering.autoionization.molecular import (
    ActiveMolecularOrbital,
    CoreHoleState,
    TwoHoleCSF,
    TwoHoleFinalState,
    enumerate_two_hole_csfs,
    local_valence_active_space,
    molecular_auger_final_states,
    molecular_auger_initial_state,
)

_NON_REAL_THRESHOLD_VALUES = (
    True,
    "0.5",
    b"0.5",
    object(),
)
_HUGE_REAL_THRESHOLD_VALUE = 10**10000


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


def _active_payload(
    *,
    ao_labels=None,
    coefficients=None,
    overlap_matrix=None,
    occupations=None,
    metadata_extra=None,
):
    C = np.eye(4) if coefficients is None else np.asarray(coefficients, dtype=float)
    labels = ao_labels or ["0 O 1s", "0 O 2s", "0 O 2p", "1 H 1s"]
    overlap = (
        np.eye(4) if overlap_matrix is None else np.asarray(overlap_matrix, dtype=float)
    )
    occ = (
        np.array([1.0, 2.0, 2.0, 0.0])
        if occupations is None
        else np.asarray(occupations, dtype=float)
    )
    core_metadata = {
        "orbital_label": "O 1s",
        "occupation_before": 2.0,
        "occupation_after": 1.0,
        "tracked_mo_index": 0,
        "initial_state_energy_hartree": 19.5,
    }
    if metadata_extra:
        core_metadata.update(metadata_extra)
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
            occupations=occ,
            basis_labels=labels,
        ),
        overlap=OverlapBlock(
            overlap,
            basis_labels=labels,
        ),
        density=DensityMatrix(np.diag(occ)),
        metadata={"core_hole": core_metadata},
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


def test_molecular_entry_rejects_incomplete_core_hole_metadata():
    payload = _payload()
    payload.metadata["core_hole"].pop("occupation_before")

    with pytest.raises(ProviderContractError, match="occupation_before"):
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
    assert "single-core-center-active-space" in active.provenance.approximations


@pytest.mark.parametrize(
    "bad_label",
    ["O", "O 1s"],
)
def test_local_valence_active_space_rejects_malformed_ao_label(bad_label):
    payload = _active_payload(ao_labels=[bad_label, "0 O 2s", "0 O 2p", "1 H 1s"])

    with pytest.raises(ProviderContractError, match="AO label"):
        local_valence_active_space(payload, population_threshold=0.5)


def test_local_valence_active_space_rejects_ambiguous_same_element_centers():
    payload = _active_payload(
        ao_labels=["0 O 1s", "0 O 2s", "1 O 2s", "1 O 2p"],
    )

    with pytest.raises(ProviderContractError, match="ambiguous"):
        local_valence_active_space(payload, population_threshold=0.5)


def test_local_valence_active_space_uses_explicit_core_hole_center():
    payload = _active_payload(
        ao_labels=["0 O 1s", "0 O 2s", "1 O 2s", "1 O 2p"],
        metadata_extra={"atom_center": "0"},
    )

    active = local_valence_active_space(payload, population_threshold=0.5)

    assert active.labels == ("MO1",)


def test_local_valence_active_space_excludes_tracked_core_mo_even_if_occupied():
    payload = _active_payload(occupations=np.array([2.0, 2.0, 2.0, 0.0]))

    active = local_valence_active_space(payload, population_threshold=0.5)

    assert "MO0" not in active.labels


def test_local_valence_active_space_requires_tracked_core_mo_index():
    payload = _active_payload(occupations=np.array([2.0, 2.0, 2.0, 0.0]))
    payload.metadata["core_hole"].pop("tracked_mo_index")

    with pytest.raises(ProviderContractError, match="tracked_mo_index"):
        local_valence_active_space(payload, population_threshold=0.5)


def test_final_states_require_tracked_core_mo_index():
    payload = _active_payload()
    payload.metadata["core_hole"].pop("tracked_mo_index")

    with pytest.raises(ProviderContractError, match="tracked_mo_index"):
        molecular_auger_final_states(payload, population_threshold=0.5)


def test_local_valence_active_space_rejects_open_shell_active_mo():
    payload = _active_payload(occupations=np.array([1.0, 1.0, 2.0, 0.0]))

    with pytest.raises(ProviderContractError, match="closed-shell"):
        local_valence_active_space(
            payload,
            population_threshold=0.5,
            occupation_threshold=0.5,
        )


def test_local_valence_active_space_rejects_empty_active_space():
    with pytest.raises(ProviderContractError, match="no occupied valence MOs"):
        local_valence_active_space(_active_payload(), population_threshold=1.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_threshold": float("nan")},
        {"population_threshold": float("inf")},
        {"population_threshold": -1e-6},
        {"occupation_threshold": float("nan")},
        {"occupation_threshold": -1e-6},
        {"residual_tolerance": float("nan")},
        {"residual_tolerance": float("inf")},
        {"residual_tolerance": 0.0},
        {"residual_tolerance": -1e-7},
    ],
)
def test_local_valence_active_space_rejects_bad_thresholds(kwargs):
    with pytest.raises(ProviderContractError):
        local_valence_active_space(_active_payload(), **kwargs)


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
@pytest.mark.parametrize("bad_value", _NON_REAL_THRESHOLD_VALUES)
def test_local_valence_active_space_rejects_non_real_thresholds_before_active_work(
    monkeypatch,
    parameter,
    bad_value,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Mulliken active-space population work was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.np.einsum",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match=parameter):
        local_valence_active_space(_active_payload(), **{parameter: bad_value})


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
def test_local_valence_active_space_rejects_huge_real_thresholds_before_active_work(
    monkeypatch,
    parameter,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Mulliken active-space population work was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.np.einsum",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match=parameter):
        local_valence_active_space(
            _active_payload(),
            **{parameter: _HUGE_REAL_THRESHOLD_VALUE},
        )


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
def test_molecular_auger_final_states_rejects_huge_real_thresholds_before_work(
    monkeypatch,
    parameter,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("initial-state or active-space work was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.molecular_auger_initial_state",
        fail_if_called,
    )
    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.local_valence_active_space",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match=parameter):
        molecular_auger_final_states(
            _active_payload(),
            **{parameter: _HUGE_REAL_THRESHOLD_VALUE},
        )


def test_local_valence_active_space_rejects_oversized_active_space_budget():
    n_mo = 18
    occupations = np.array([1.0, *([2.0] * 17)])
    labels = ["0 O 1s", *(f"0 O valence{i}" for i in range(1, n_mo))]
    payload = ElectronicStructurePayload(
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
        orbital_energies={f"MO{i}": float(i) for i in range(n_mo)},
        orbitals=OrbitalCoefficients(
            coefficients=np.eye(n_mo),
            energies=np.arange(n_mo, dtype=float),
            occupations=occupations,
            basis_labels=labels,
        ),
        overlap=OverlapBlock(np.eye(n_mo), basis_labels=labels),
        density=DensityMatrix(np.diag(occupations)),
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

    with pytest.raises(ProviderContractError, match="active space exceeds"):
        local_valence_active_space(payload, population_threshold=0.5)


def test_local_valence_active_space_rejects_nonorthonormal_active_block():
    payload = _active_payload(overlap_matrix=np.diag([1.0, 2.0, 1.0, 1.0]))

    with pytest.raises(ProviderContractError, match="C_active"):
        local_valence_active_space(payload, population_threshold=0.5)


def test_local_valence_active_space_accepts_nonidentity_overlap_contract():
    overlap = np.diag([1.0, 4.0, 9.0, 1.0])
    coefficients = np.diag([1.0, 0.5, 1.0 / 3.0, 1.0])
    payload = _active_payload(coefficients=coefficients, overlap_matrix=overlap)

    active = local_valence_active_space(payload, population_threshold=0.5)

    assert active.labels == ("MO1", "MO2")
    np.testing.assert_allclose(active.overlap_s, overlap)
    assert active.orthonormality_residual <= 1e-12


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


def test_two_hole_csf_rejects_invalid_spin_contracts():
    with pytest.raises(ProviderContractError, match="multiplicity"):
        TwoHoleCSF(
            hole_orbitals=("MO1", "MO2"),
            spin_multiplicity=2,
            spin_symmetry="doublet",
            label="MO1^-1 MO2^-1 doublet",
        )
    with pytest.raises(ProviderContractError, match="same-orbital"):
        TwoHoleCSF(
            hole_orbitals=("MO1", "MO1"),
            spin_multiplicity=3,
            spin_symmetry="triplet",
            label="MO1^-2 triplet",
        )


def test_active_molecular_orbital_rejects_empty_label():
    with pytest.raises(ProviderContractError, match="label"):
        ActiveMolecularOrbital(
            label="",
            index=1,
            energy=-1.0,
            occupation=2.0,
            core_atom_population=0.8,
        )


def test_two_hole_final_states_stamp_i2_i4_i5_provenance_and_energy():
    final_states = molecular_auger_final_states(
        _active_payload(),
        population_threshold=0.5,
        coulomb_corrections={("MO1", "MO2"): 0.1},
    )

    pair = next(
        state
        for state in final_states
        if state.hole_orbitals == ("MO1", "MO2") and state.spin_symmetry == "singlet"
    )
    assert pair.final_energy == pytest.approx(19.5 + (-20.0) - (-1.0) - (-0.5) + 0.1)
    assert pair.provenance.final_state == "MO1^-1 MO2^-1 singlet"
    assert pair.provenance.energy_zero == "delta_scf_total_energy"
    assert "independent-hole-koopmans" in pair.provenance.approximations
    assert "mulliken-active-space" in pair.provenance.approximations


def test_two_hole_final_states_reuse_same_orbital_csf_label():
    final_states = molecular_auger_final_states(
        _active_payload(), population_threshold=0.5
    )

    same_orbital = next(
        state for state in final_states if state.hole_orbitals == ("MO1", "MO1")
    )

    assert same_orbital.provenance.final_state == "MO1^-2 singlet"


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


def test_two_hole_final_state_rejects_missing_energy_zero():
    active = local_valence_active_space(_active_payload(), population_threshold=0.5)
    with pytest.raises(ProviderContractError, match="energy_zero"):
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
                approximations=["cluster-ci-final-state"],
                core_hole="O 1s",
                final_state="MO1^-1 MO2^-1 singlet",
                energy_zero="",
            ),
            active_space=active,
        )
