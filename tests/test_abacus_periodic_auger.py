"""ABACUS LCAO adapter and periodic Auger MVP tests (HYP-912)."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.providers.abacus_adapter import ABACUSKPointData, ABACUSLCAOProvider
from atomx.providers.contract import Capability, ProviderContractError
from atomx.scattering.autoionization.molecular import DiscreteMOContinuumChannel
from atomx.scattering.autoionization.periodic import periodic_auger_spectrum


def _eri(value: float) -> np.ndarray:
    tensor = np.zeros((4, 4, 4, 4))
    tensor[0, 3, 1, 2] = value
    return tensor


def _kdata(weight: float, *, eri_value: float = 1.0) -> ABACUSKPointData:
    return ABACUSKPointData(
        coordinate=(0.0, 0.0, 0.0),
        weight=weight,
        hamiltonian=np.diag([-20.0, -1.0, -0.5, 0.2]),
        overlap=np.eye(4),
        coefficients=np.eye(4),
        energies=np.array([-20.0, -1.0, -0.5, 0.2]),
        occupations=np.array([1.0, 2.0, 2.0, 0.0]),
        density=np.diag([1.0, 2.0, 2.0, 0.0]),
        two_electron=_eri(eri_value),
        local_projection_weight=1.0,
    )


def _provider(*kpoints: ABACUSKPointData) -> ABACUSLCAOProvider:
    return ABACUSLCAOProvider(
        Z=13,
        kpoints=kpoints,
        ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
        core_hole={
            "orbital_label": "Al 1s",
            "occupation_before": 2.0,
            "occupation_after": 1.0,
            "tracked_mo_index": 0,
            "initial_state_energy_hartree": 19.5,
            "atom_center": "0",
        },
    )


def test_abacus_payload_exposes_periodic_lcao_contract():
    payload = _provider(_kdata(2.0)).build_payload()

    assert payload.supports(
        Capability.CORE_HOLE_SPEC,
        Capability.MO_COEFFICIENTS,
        Capability.OVERLAP_S,
        Capability.KPOINTS,
        Capability.TWO_ELECTRON_INTEGRALS,
    )
    assert payload.orthonormality_residual() == pytest.approx(0.0)
    assert payload.electron_count() == pytest.approx(5.0)
    assert payload.metadata["k_weight"] == pytest.approx(1.0)


def test_abacus_adapter_rejects_non_hermitian_hamiltonian():
    bad = _kdata(1.0)
    H = bad.hamiltonian.copy()
    H[0, 1] = 1.0

    with pytest.raises(ProviderContractError, match="Hamiltonian"):
        _provider(
            ABACUSKPointData(
                coordinate=bad.coordinate,
                weight=bad.weight,
                hamiltonian=H,
                overlap=bad.overlap,
                coefficients=bad.coefficients,
                energies=bad.energies,
                occupations=bad.occupations,
            )
        )


def test_abacus_adapter_rejects_non_positive_overlap():
    bad = _kdata(1.0)
    S = bad.overlap.copy()
    S[0, 0] = 0.0

    with pytest.raises(ProviderContractError, match="positive definite"):
        _provider(
            ABACUSKPointData(
                coordinate=bad.coordinate,
                weight=bad.weight,
                hamiltonian=bad.hamiltonian,
                overlap=S,
                coefficients=bad.coefficients,
                energies=bad.energies,
                occupations=bad.occupations,
            )
        )


def test_periodic_auger_spectrum_is_k_weighted_and_not_dos():
    k0 = _kdata(1.0, eri_value=1.0)
    k1 = ABACUSKPointData(
        coordinate=(0.5, 0.0, 0.0),
        weight=3.0,
        hamiltonian=np.diag([-20.0, -1.0, -0.5, 0.2]),
        overlap=np.eye(4),
        coefficients=np.eye(4),
        energies=np.array([-20.0, -1.0, -0.5, 0.2]),
        occupations=np.array([1.0, 2.0, 2.0, 0.0]),
        density=np.diag([1.0, 2.0, 2.0, 0.0]),
        two_electron=_eri(2.0),
        local_projection_weight=0.5,
    )
    payloads = _provider(k0, k1).iter_kpoint_payloads()

    spectrum = periodic_auger_spectrum(
        payloads,
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
        normalize=False,
    )

    assert len(spectrum.lines) == 2
    # raw coherent intensity = ERI^2, then multiplied by normalized k weight
    # and local projection weight: 1^2 * 1/4, 2^2 * 3/4 * 1/2.
    assert sorted(spectrum.intensities) == pytest.approx([0.25, 1.5])
    assert any(
        "matrix-element-weighted-not-dos" in line.provenance.approximations
        for line in spectrum.lines
    )


def test_periodic_auger_normalizes_total_k_weighted_intensity():
    payloads = _provider(_kdata(1.0, eri_value=1.0), _kdata(1.0, eri_value=2.0))

    spectrum = periodic_auger_spectrum(
        payloads.iter_kpoint_payloads(),
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
        normalize=True,
    )

    assert sum(spectrum.intensities) == pytest.approx(1.0)
    assert spectrum.intensity_unit == "sum_rule_normalized_k_weighted_probability"
