"""Molecular ERI Auger amplitude and stick-spectrum tests (HYP-910.C)."""

from __future__ import annotations

import numpy as np
import pytest

import atomx.scattering.autoionization.molecular as molecular
from atomx.providers.contract import (
    CapabilityError,
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
    TwoElectronIntegrals,
)
from atomx.scattering.autoionization.molecular import (
    DiscreteMOContinuumChannel,
    TwoHoleFinalState,
    compute_molecular_auger,
    local_valence_active_space,
    molecular_auger_final_states,
    molecular_auger_initial_state,
    molecular_auger_spectrum,
    molecular_auger_transition_amplitude,
)

_NON_REAL_THRESHOLD_VALUES = (
    True,
    "0.5",
    b"0.5",
    object(),
)
_HUGE_REAL_THRESHOLD_VALUE = 10**10000


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


def _five_mo_payload_with_eri(eri: np.ndarray) -> ElectronicStructurePayload:
    labels = ["0 O 1s", "0 O 2s", "0 O 2p", "0 O 3s", "1 H 1s"]
    occupations = np.array([1.0, 2.0, 2.0, 2.0, 0.0])
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
        orbital_energies={
            "MO0": -20.0,
            "MO1": -1.0,
            "MO2": -0.8,
            "MO3": -0.6,
            "MO4": 0.2,
        },
        orbitals=OrbitalCoefficients(
            coefficients=np.eye(5),
            energies=np.array([-20.0, -1.0, -0.8, -0.6, 0.2]),
            occupations=occupations,
            basis_labels=labels,
        ),
        overlap=OverlapBlock(np.eye(5), basis_labels=labels),
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


def _payload_with_core_metadata(**updates) -> ElectronicStructurePayload:
    payload = _payload_with_eri(np.zeros((4, 4, 4, 4)))
    payload.metadata["core_hole"].update(updates)
    return payload


def _large_ao_eri_payload(n_mo: int = 49) -> ElectronicStructurePayload:
    labels = [
        "0 O 1s",
        "0 O 2s",
        "0 O 2p",
        *(f"1 H aux{i}" for i in range(3, n_mo)),
    ]
    occupations = np.zeros(n_mo)
    occupations[0] = 1.0
    occupations[1:3] = 2.0
    return ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="orthonormal-ao",
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
        two_electron=TwoElectronIntegrals(
            np.zeros((n_mo, n_mo, n_mo, n_mo)),
            basis="ao",
        ),
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


def _asymmetric_ao_eri_payload(*, n_ao: int = 1000) -> ElectronicStructurePayload:
    n_mo = 4
    labels = [
        "0 O 1s",
        "0 O 2s",
        "0 O 2p",
        *(f"1 H aux{i}" for i in range(3, n_ao)),
    ]
    occupations = np.array([1.0, 2.0, 2.0, 0.0])
    coefficients = np.zeros((n_ao, n_mo))
    coefficients[:n_mo, :n_mo] = np.eye(n_mo)
    tiny_eri = np.zeros((1, 1, 1, 1), dtype=float)
    eri = np.lib.stride_tricks.as_strided(
        tiny_eri,
        shape=(n_ao, n_ao, n_ao, n_ao),
        strides=(0, 0, 0, 0),
        writeable=False,
    )
    two_electron = TwoElectronIntegrals(eri, basis="ao")
    return ElectronicStructurePayload(
        Z=8,
        provenance=Provenance(
            provider="mock",
            code_version="mock 0",
            method="RHF",
            basis="orthonormal-ao",
            theory_level="RHF",
            approximations=["nonrel-gto"],
            core_hole="O 1s",
            energy_zero="delta_scf_total_energy",
        ),
        orbital_energies={f"MO{i}": float(i) for i in range(n_mo)},
        orbitals=OrbitalCoefficients(
            coefficients=coefficients,
            energies=np.arange(n_mo, dtype=float),
            occupations=occupations,
            basis_labels=labels,
        ),
        overlap=OverlapBlock(np.eye(n_ao), basis_labels=labels),
        density=DensityMatrix(np.diag([*occupations, *np.zeros(n_ao - n_mo)])),
        two_electron=two_electron,
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


class _CoreHoleProvider:
    def __init__(self, payload: ElectronicStructurePayload) -> None:
        self.payload = payload
        self.calls: list[tuple[object, bool]] = []

    def build_core_hole_payload(
        self,
        core_hole_spec,
        *,
        with_eri: bool = False,
    ) -> ElectronicStructurePayload:
        self.calls.append((core_hole_spec, with_eri))
        return self.payload


class _LazyCoreHoleProvider:
    def __init__(self, payload_factory) -> None:
        self.payload_factory = payload_factory
        self.calls: list[tuple[object, bool]] = []

    def build_core_hole_payload(
        self,
        core_hole_spec,
        *,
        with_eri: bool = False,
    ) -> ElectronicStructurePayload:
        self.calls.append((core_hole_spec, with_eri))
        return self.payload_factory()


class _AdversarialERI:
    def __init__(self, *, shape=()):
        self._shape = shape
        self.shape_reads = 0
        self.array_calls = 0

    @property
    def shape(self):
        self.shape_reads += 1
        return self._shape

    def __array__(self, dtype=None):
        self.array_calls += 1
        raise AssertionError("ERI __array__ conversion was reached")


class _ShapeChangingERI(np.ndarray):
    def astype(self, *args, **kwargs):
        raise AssertionError("ERI ndarray subclass astype conversion was reached")


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


def test_ao_to_mo_eri_transform_rejects_oversized_mo_budget():
    payload = _large_ao_eri_payload()
    initial = molecular_auger_initial_state(payload)
    final = _pair_state(payload)

    with pytest.raises(ProviderContractError, match="AO-to-MO ERI"):
        molecular_auger_transition_amplitude(
            payload,
            initial,
            final,
            DiscreteMOContinuumChannel(mo_index=3),
        )


def test_ao_to_mo_eri_transform_rejects_asymmetric_ao_budget_before_einsum(
    monkeypatch,
):
    provider = _LazyCoreHoleProvider(_asymmetric_ao_eri_payload)
    original_isfinite = np.isfinite

    def fail_if_array_scanned(value, *args, **kwargs):
        if np.asarray(value).ndim > 0:
            raise AssertionError("ERI validation reached np.isfinite array scan")
        return original_isfinite(value, *args, **kwargs)

    def fail_if_einsum_called(*args, **kwargs):
        raise AssertionError("AO-to-MO transform reached np.einsum")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.np.einsum",
        fail_if_einsum_called,
    )
    monkeypatch.setattr(
        "atomx.providers.contract.np.isfinite",
        fail_if_array_scanned,
    )

    with pytest.raises(
        ProviderContractError,
        match="ERI construction exceeds the provider contract finite-validation budget",
    ):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        )
    assert provider.calls == [("O 1s", True)]


@pytest.mark.parametrize(
    "bad_eri_factory",
    [
        lambda: object(),
        lambda: _AdversarialERI(shape=(1, 1, 1, 1)),
        lambda: _AdversarialERI(shape=(1, -1, 1, 1)),
        lambda: np.zeros((1, 1, 1, 1)).view(_ShapeChangingERI),
    ],
)
def test_compute_molecular_auger_rejects_untrusted_eri_before_array_protocols(
    monkeypatch,
    bad_eri_factory,
):
    bad_eri = bad_eri_factory()
    provider = _LazyCoreHoleProvider(
        lambda: TwoElectronIntegrals(bad_eri, basis="ao")
    )
    original_isfinite = np.isfinite

    def fail_if_array_scanned(value, *args, **kwargs):
        if np.asarray(value).ndim > 0:
            raise AssertionError("ERI rejection reached np.isfinite array scan")
        return original_isfinite(value, *args, **kwargs)

    def fail_if_einsum_called(*args, **kwargs):
        raise AssertionError("ERI rejection reached np.einsum")

    monkeypatch.setattr(
        "atomx.providers.contract.np.isfinite",
        fail_if_array_scanned,
    )
    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.np.einsum",
        fail_if_einsum_called,
    )

    with pytest.raises(
        ProviderContractError,
        match="ERI must be provided as a base numpy.ndarray",
    ):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        )
    assert provider.calls == [("O 1s", True)]
    if isinstance(bad_eri, _AdversarialERI):
        assert bad_eri.shape_reads == 0
        assert bad_eri.array_calls == 0


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


def test_molecular_auger_spectrum_reuses_one_mo_eri_transform_for_many_singlets(
    monkeypatch,
):
    eri = np.zeros((5, 5, 5, 5))
    for hole_i in range(1, 4):
        for hole_j in range(hole_i, 4):
            eri[0, 4, hole_i, hole_j] = float(hole_i + hole_j)
    payload = _five_mo_payload_with_eri(eri)
    calls = 0
    original_mo_eri = molecular._mo_eri

    def counted_mo_eri(payload):
        nonlocal calls
        calls += 1
        return original_mo_eri(payload)

    monkeypatch.setattr(molecular, "_mo_eri", counted_mo_eri)

    spectrum = molecular_auger_spectrum(
        payload,
        continuum_channel=DiscreteMOContinuumChannel(mo_index=4),
        population_threshold=0.5,
    )

    assert calls == 1
    assert len(spectrum.lines) > 1


def test_compute_molecular_auger_orchestrates_provider_core_hole_payload():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = 3.0
    eri[0, 3, 2, 1] = 1.0
    provider = _CoreHoleProvider(_payload_with_eri(eri))

    spectrum = compute_molecular_auger(
        provider,
        "O 1s",
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
    )

    assert provider.calls == [("O 1s", True)]
    assert spectrum.axis == "kinetic_energy"
    assert len(spectrum.lines) == 1
    assert spectrum.lines[0].raw_intensity == pytest.approx(4.0)
    assert spectrum.lines[0].provenance.core_hole == "O 1s"
    assert "sum-rule-total-stick-normalization" in (
        spectrum.lines[0].provenance.approximations
    )


def test_compute_molecular_auger_accepts_prepared_payload():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = 3.0
    eri[0, 3, 2, 1] = 1.0

    spectrum = compute_molecular_auger(
        _payload_with_eri(eri),
        {"orbital_label": "O 1s"},
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
    )

    assert len(spectrum.lines) == 1
    assert sum(spectrum.intensities) == pytest.approx(1.0)


def test_molecular_auger_spectrum_rejects_nonfinite_mo_eri_before_stick(
    monkeypatch,
):
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 3, 1, 2] = np.nan

    def fail_stick(*args, **kwargs):
        raise AssertionError("AugerStick must not be emitted for non-finite ERI")

    monkeypatch.setattr(molecular, "AugerStick", fail_stick)

    with pytest.raises(ProviderContractError, match="ERI values must be finite"):
        _payload_with_eri(eri)


@pytest.mark.parametrize(
    "core_hole_spec",
    [
        {},
        {"orbital_label": "O 1s", "core_orbital": "O 1s"},
        {"unknown": "O 1s"},
        {"orbital_label": ""},
        {"core_orbital": -1},
        {"core_orbital": 1.5},
        {"core_orbital": False},
        -1,
        1.5,
        True,
    ],
)
def test_compute_molecular_auger_rejects_bad_core_hole_spec_before_provider(
    core_hole_spec,
):
    provider = _CoreHoleProvider(_payload_with_eri(np.zeros((4, 4, 4, 4))))

    with pytest.raises(ProviderContractError):
        compute_molecular_auger(provider, core_hole_spec, population_threshold=0.5)

    assert provider.calls == []


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
def test_compute_molecular_auger_rejects_bad_thresholds_before_provider(kwargs):
    provider = _CoreHoleProvider(_payload_with_eri(np.zeros((4, 4, 4, 4))))

    with pytest.raises(ProviderContractError):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            **kwargs,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
@pytest.mark.parametrize("bad_value", _NON_REAL_THRESHOLD_VALUES)
def test_compute_molecular_auger_rejects_non_real_thresholds_before_provider(
    parameter,
    bad_value,
):
    provider = _CoreHoleProvider(_payload_with_eri(np.zeros((4, 4, 4, 4))))

    with pytest.raises(ProviderContractError, match=parameter):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            **{parameter: bad_value},
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
def test_compute_molecular_auger_rejects_huge_real_thresholds_before_provider(
    parameter,
):
    provider = _CoreHoleProvider(_payload_with_eri(np.zeros((4, 4, 4, 4))))

    with pytest.raises(ProviderContractError, match=parameter):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            **{parameter: _HUGE_REAL_THRESHOLD_VALUE},
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
@pytest.mark.parametrize("bad_value", _NON_REAL_THRESHOLD_VALUES)
def test_molecular_auger_spectrum_rejects_non_real_thresholds_before_final_work(
    monkeypatch,
    parameter,
    bad_value,
):
    payload = _payload_with_eri(np.zeros((4, 4, 4, 4)))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("final-state or ERI work was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.molecular_auger_final_states",
        fail_if_called,
    )
    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular._mo_eri",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match=parameter):
        molecular_auger_spectrum(payload, **{parameter: bad_value})


@pytest.mark.parametrize(
    "parameter",
    ["population_threshold", "occupation_threshold", "residual_tolerance"],
)
def test_molecular_auger_spectrum_rejects_huge_real_thresholds_before_final_work(
    monkeypatch,
    parameter,
):
    payload = _payload_with_eri(np.zeros((4, 4, 4, 4)))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("final-state or ERI work was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.molecular_auger_final_states",
        fail_if_called,
    )
    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular._mo_eri",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match=parameter):
        molecular_auger_spectrum(
            payload,
            **{parameter: _HUGE_REAL_THRESHOLD_VALUE},
        )


@pytest.mark.parametrize(
    "tracked_mo_index",
    [
        1.5,
        "1",
        True,
        -1,
        4,
    ],
)
def test_compute_molecular_auger_rejects_bad_provider_tracked_mo_index_before_final_states(
    monkeypatch,
    tracked_mo_index,
):
    provider = _CoreHoleProvider(
        _payload_with_core_metadata(tracked_mo_index=tracked_mo_index)
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("two-hole final-state enumeration was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.enumerate_two_hole_csfs",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match="tracked_mo_index"):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )

    assert provider.calls == [("O 1s", True)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("occupation_before", float("nan")),
        ("occupation_before", float("inf")),
        ("occupation_before", "2"),
        ("occupation_before", True),
        ("occupation_before", -1.0),
        ("occupation_before", 2.5),
        ("occupation_after", float("nan")),
        ("occupation_after", float("-inf")),
        ("occupation_after", "1"),
        ("occupation_after", False),
        ("occupation_after", -1.0),
        ("initial_state_energy_hartree", float("nan")),
        ("initial_state_energy_hartree", float("inf")),
        ("initial_state_energy_hartree", "19.5"),
        ("initial_state_energy_hartree", True),
    ],
)
def test_compute_molecular_auger_rejects_bad_provider_core_hole_numbers_before_final_states(
    monkeypatch,
    field,
    value,
):
    provider = _CoreHoleProvider(_payload_with_core_metadata(**{field: value}))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("final-state construction was reached")

    monkeypatch.setattr(
        "atomx.scattering.autoionization.molecular.molecular_auger_final_states",
        fail_if_called,
    )

    with pytest.raises(ProviderContractError, match="core_hole"):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )

    assert provider.calls == [("O 1s", True)]


def test_compute_molecular_auger_rejects_non_core_hole_occupation_change():
    provider = _CoreHoleProvider(
        _payload_with_core_metadata(
            occupation_before=1.0,
            occupation_after=1.0,
        )
    )

    with pytest.raises(ProviderContractError, match="occupation_after"):
        compute_molecular_auger(
            provider,
            "O 1s",
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )


def test_compute_molecular_auger_fails_closed_without_eris():
    payload = _payload_with_eri(np.zeros((4, 4, 4, 4)))
    payload.two_electron = None

    with pytest.raises(CapabilityError, match="two_electron_integrals"):
        compute_molecular_auger(payload, "O 1s", population_threshold=0.5)


def test_compute_molecular_auger_rejects_core_hole_identity_mismatch():
    provider = _CoreHoleProvider(_payload_with_eri(np.zeros((4, 4, 4, 4))))

    with pytest.raises(ProviderContractError, match="different core-hole"):
        compute_molecular_auger(provider, "Ne 1s", population_threshold=0.5)


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
