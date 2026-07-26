"""Molecular ERI Auger amplitude and stick-spectrum tests (HYP-910.C)."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.providers.contract import (
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    TwoElectronIntegrals,
)
from atomx.scattering.autoionization.molecular import (
    DiscreteMOContinuumChannel,
    TwoHoleFinalState,
    local_valence_active_space,
    molecular_auger_final_states,
    molecular_auger_initial_state,
    molecular_auger_spectrum,
    molecular_auger_transition_amplitude,
)


def _payload_with_eri(eri: np.ndarray) -> ElectronicStructurePayload:
    labels = ["0 O 1s", "0 O 2s", "0 O 2p", "1 H 1s"]
    occupations = np.array([1.0, 2.0, 2.0, 0.0])
    return ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="orthonormal-mo",
            theory_level="RHF",
            approximations=["nonrel-gto"],
            core_hole="O 1s",
            energy_zero="delta_scf_total_energy",
        ),
        orbital_energies={"MO0": -20.0, "MO1": -1.0, "MO2": -0.5, "MO3": 0.2},
        orbitals=OrbitalCoefficients(
            coefficients=np.eye(4),
            energies=np.array([-20.0, -1.0, -0.5, 0.2]),
            occupations=occupations,
            basis_labels=labels,
        ),
        overlap=OverlapBlock(np.eye(4), basis_labels=labels),
        density=DensityMatrix(np.diag(occupations)),
        two_electron=TwoElectronIntegrals(eri, basis="mo"),
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


def _pair_state(payload: ElectronicStructurePayload, spin: str = "singlet"):
    final_states = molecular_auger_final_states(payload, population_threshold=0.5)
    return next(
        state
        for state in final_states
        if state.hole_orbitals == ("MO1", "MO2") and state.spin_symmetry == spin
    )


def test_direct_exchange_combined_at_amplitude_level_not_rate_level():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = 3.0
    eri[0, 3, 2, 1] = 1.0
    payload = _payload_with_eri(eri)

    amplitude = molecular_auger_transition_amplitude(
        payload,
        molecular_auger_initial_state(payload),
        _pair_state(payload),
        DiscreteMOContinuumChannel(mo_index=3),
    )

    assert amplitude.direct == pytest.approx(3.0)
    assert amplitude.exchange == pytest.approx(1.0)
    assert amplitude.amplitude == pytest.approx(2.0)
    assert amplitude.coherent_intensity == pytest.approx(4.0)
    assert amplitude.rate_sum_intensity == pytest.approx(10.0)
    assert amplitude.coherent_intensity != pytest.approx(amplitude.rate_sum_intensity)


def test_triplet_two_hole_state_is_spin_forbidden_zero_amplitude():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = 3.0
    eri[0, 3, 2, 1] = 1.0
    payload = _payload_with_eri(eri)

    amplitude = molecular_auger_transition_amplitude(
        payload,
        molecular_auger_initial_state(payload),
        _pair_state(payload, spin="triplet"),
        DiscreteMOContinuumChannel(mo_index=3),
    )

    assert amplitude.amplitude == pytest.approx(0.0)
    assert amplitude.coherent_intensity == pytest.approx(0.0)
    assert amplitude.selection_rule == "spin-forbidden-non-soc-v1"


def test_molecular_auger_spectrum_uses_ke_axis_and_sum_rule_normalization():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 1] = 2.0
    eri[0, 3, 1, 2] = 3.0
    eri[0, 3, 2, 1] = 1.0
    payload = _payload_with_eri(eri)

    spectrum = molecular_auger_spectrum(
        payload,
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
    )

    assert spectrum.axis == "kinetic_energy"
    assert spectrum.axis_unit == "hartree"
    assert spectrum.intensity_unit == "sum_rule_normalized_probability"
    assert sum(spectrum.intensities) == pytest.approx(1.0)
    assert len(spectrum.lines) == 1
    assert spectrum.lines[0].raw_intensity == pytest.approx(4.0)
    assert spectrum.lines[0].intensity == pytest.approx(1.0)
    assert all(line.kinetic_energy > 0.0 for line in spectrum.lines)
    assert any(
        item["reason"] == "spin-forbidden-non-soc-v1" for item in spectrum.dropped
    )
    assert any(
        "continuum:discrete-mo-v1-replaceable" in line.provenance.approximations
        for line in spectrum.lines
    )
    assert all(
        "direct-minus-exchange-amplitude" in line.provenance.approximations
        for line in spectrum.lines
    )


def test_non_positive_kinetic_energy_line_is_dropped_and_reported():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = 3.0
    payload = _payload_with_eri(eri)
    initial = molecular_auger_initial_state(payload)
    active = local_valence_active_space(payload, population_threshold=0.5)
    final = TwoHoleFinalState(
        hole_orbitals=("MO1", "MO2"),
        spin_multiplicity=1,
        spin_symmetry="singlet",
        final_energy=initial.initial_energy + 1.0,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="orthonormal-mo",
            theory_level="RHF",
            approximations=["test-final-state"],
            core_hole="O 1s",
            final_state="unphysical high final",
            energy_zero="delta_scf_total_energy",
        ),
        active_space=active,
    )

    spectrum = molecular_auger_spectrum(
        payload,
        core_hole=initial,
        final_states=(final,),
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
    )

    assert spectrum.lines == ()
    assert spectrum.dropped == (
        {
            "final_state": "unphysical high final",
            "reason": "non_positive_kinetic_energy",
        },
    )
