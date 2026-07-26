"""ABACUS LCAO adapter and periodic Auger MVP tests (HYP-912)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import atomx.providers.abacus_adapter as abacus_adapter
from atomx.providers.abacus_adapter import (
    ABACUSKPointData,
    ABACUSLCAOProvider,
    ABACUSValidationLimits,
)
from atomx.providers.contract import Capability, KPointBlock, ProviderContractError
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


def _replace_kdata(kdata: ABACUSKPointData, **updates) -> ABACUSKPointData:
    values = {
        "coordinate": kdata.coordinate,
        "weight": kdata.weight,
        "hamiltonian": kdata.hamiltonian,
        "overlap": kdata.overlap,
        "coefficients": kdata.coefficients,
        "energies": kdata.energies,
        "occupations": kdata.occupations,
        "density": kdata.density,
        "two_electron": kdata.two_electron,
        "local_projection_weight": kdata.local_projection_weight,
    }
    values.update(updates)
    return ABACUSKPointData(**values)


def _integer_kdata() -> ABACUSKPointData:
    return _replace_kdata(
        _kdata(1.0),
        hamiltonian=np.diag([-20, -1, 0, 1]),
        overlap=np.eye(4, dtype=np.int64),
        coefficients=np.eye(4, dtype=np.int64),
        energies=np.array([-20, -1, 0, 1], dtype=np.int64),
        occupations=np.array([1, 2, 2, 0], dtype=np.int64),
        density=np.diag([1, 2, 2, 0]).astype(np.int64),
        two_electron=_eri(1.0).astype(np.int64),
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
            _replace_kdata(bad, hamiltonian=H)
        )


def test_abacus_adapter_rejects_high_amplitude_absolute_hermiticity_residual():
    bad = _kdata(1.0)
    H = np.diag([1e12, 2e12, 3e12, 4e12])
    H[0, 1] = 1e12
    H[1, 0] = 1e12 + 1e-4

    with pytest.raises(ProviderContractError, match="Hamiltonian"):
        _provider(_replace_kdata(bad, hamiltonian=H))


def test_abacus_adapter_rejects_non_positive_overlap():
    bad = _kdata(1.0)
    S = bad.overlap.copy()
    S[0, 0] = 0.0

    with pytest.raises(ProviderContractError, match="positive definite"):
        _provider(
            _replace_kdata(bad, overlap=S)
        )


def test_abacus_adapter_rejects_local_projection_outside_probability_contract():
    with pytest.raises(ProviderContractError, match="local projection weight"):
        _provider(_replace_kdata(_kdata(1.0), local_projection_weight=2.0))


@pytest.mark.parametrize(
    ("reserved_key", "value"),
    [
        ("core_hole", {}),
        ("kpoint_index", 0),
        ("kpoint", (0.25, 0.0, 0.0)),
        ("k_weight", 99.0),
        ("local_projection_weight", 0.25),
        ("hamiltonian", np.eye(4)),
    ],
)
def test_abacus_adapter_rejects_reserved_metadata_collisions(reserved_key, value):
    with pytest.raises(ProviderContractError, match="reserved fields"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            metadata={reserved_key: value},
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"coordinate": (np.nan, 0.0, 0.0)},
        {"weight": np.inf},
        {"hamiltonian": np.diag([np.nan, -1.0, -0.5, 0.2])},
        {
            "overlap": np.array(
                [
                    [np.inf, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        },
        {
            "coefficients": np.array(
                [
                    [np.nan, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        },
        {"energies": np.array([-20.0, -1.0, np.inf, 0.2])},
        {"occupations": np.array([1.0, 2.0, -0.1, 0.0])},
        {"density": np.diag([1.0, np.nan, 2.0, 0.0])},
        {"two_electron": np.full((4, 4, 4, 4), np.inf)},
        {"local_projection_weight": np.nan},
    ],
)
def test_abacus_adapter_rejects_nonfinite_and_physical_bounds(updates):
    with pytest.raises(ProviderContractError):
        _provider(_replace_kdata(_kdata(1.0), **updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"coordinate": (1j, 0.0, 0.0)},
        {"weight": 1.0 + 1j},
        {"coefficients": np.eye(4, dtype=complex) + 1j * np.eye(4)},
        {"energies": np.array([-20.0 + 1j, -1.0, -0.5, 0.2])},
        {"density": np.diag([1.0 + 1j, 2.0, 2.0, 0.0])},
        {"two_electron": _eri(1.0).astype(complex) + 1j},
    ],
)
def test_abacus_adapter_rejects_complex_import_fields_without_warning(updates):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ProviderContractError, match="real"):
            _provider(_replace_kdata(_kdata(1.0), **updates))

    assert not caught


def test_abacus_adapter_rejects_nonfinite_validation_tolerance():
    with pytest.raises(ProviderContractError, match="tolerance"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_tolerance=np.inf,
        )


def test_abacus_adapter_rejects_resource_limit_before_eigensolve():
    with pytest.raises(ProviderContractError, match="n_ao exceeds"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_ao=3),
        )


def test_abacus_adapter_rejects_kpoint_resource_limit():
    with pytest.raises(ProviderContractError, match="n_k exceeds"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0), _kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_kpoints=1),
        )


def test_abacus_adapter_rejects_total_array_element_resource_limit():
    with pytest.raises(ProviderContractError, match="element limit"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_array_elements=1),
        )


def test_abacus_adapter_rejects_aggregate_multik_array_element_limit():
    with pytest.raises(ProviderContractError, match="element limit"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0), _kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_array_elements=500),
        )


def test_abacus_adapter_preflights_aggregate_limit_before_coercion(monkeypatch):
    coercions: list[str] = []
    original_real_array = abacus_adapter._real_array

    def recording_real_array(value, name):
        coercions.append(name)
        return original_real_array(value, name)

    kpoint = _integer_kdata()
    monkeypatch.setattr(abacus_adapter, "_real_array", recording_real_array)

    with pytest.raises(ProviderContractError, match="element limit"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[kpoint, kpoint],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_array_elements=500),
        )

    assert coercions == []


def test_abacus_adapter_preflights_aggregate_byte_limit_before_coercion(monkeypatch):
    coercions: list[str] = []
    original_real_array = abacus_adapter._real_array

    def recording_real_array(value, name):
        coercions.append(name)
        return original_real_array(value, name)

    kpoint = _integer_kdata()
    monkeypatch.setattr(abacus_adapter, "_real_array", recording_real_array)

    with pytest.raises(ProviderContractError, match="byte limit"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[kpoint, kpoint],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_input_bytes=4_000),
        )

    assert coercions == []


def test_abacus_preflight_rejects_array_dunder_wrapper_without_invoking_it(monkeypatch):
    """Regression (HYP-1292): preflight accounting must not execute inputs.

    A malicious field advertising a huge trusted footprint but whose
    ``__array__`` returns a tiny array used to be coerced by the accounting
    step (``np.asarray``), which both allocated inside the advertised
    preflight and under-reported the payload so it was accepted. Accounting
    must now reject non-``np.ndarray`` fields outright, so ``__array__`` is
    invoked zero times and the over-limit advertised metadata is refused
    before any conversion or allocation.
    """

    array_calls: list[int] = []

    class _MaliciousArrayWrapper:
        def __init__(self) -> None:
            # Advertise a footprint far over any reasonable cap...
            self.shape = (10_000_000,)
            self.size = 10_000_000
            self.ndim = 1
            self.nbytes = 10_000_000 * 8
            self.dtype = np.dtype(np.float64)

        def __array__(self, dtype=None):  # pragma: no cover - must never run
            array_calls.append(1)
            # ...then lie by returning a tiny array to fool byte accounting.
            return np.zeros(1, dtype=np.float64)

    # Unit level: _ResourceAccount.add_array refuses the wrapper without
    # triggering __array__, even though it advertises an over-limit footprint.
    account = abacus_adapter._ResourceAccount(
        ABACUSValidationLimits(max_array_elements=2, max_input_bytes=16, max_eri_elements=2)
    )
    with pytest.raises(ProviderContractError, match="np.ndarray"):
        account.add_array(_MaliciousArrayWrapper())
    assert array_calls == []

    # Integration level: building a provider whose hamiltonian is the wrapper
    # fails during the preflight, before _real_array coercion and before the
    # expensive validate() eigensolve / matrix products run.
    real_array_calls: list[str] = []
    original_real_array = abacus_adapter._real_array

    def recording_real_array(value, name):
        real_array_calls.append(name)
        return original_real_array(value, name)

    monkeypatch.setattr(abacus_adapter, "_real_array", recording_real_array)

    with pytest.raises(ProviderContractError, match="np.ndarray"):
        _provider(_replace_kdata(_kdata(1.0), hamiltonian=_MaliciousArrayWrapper()))

    assert array_calls == []
    assert real_array_calls == []


def test_abacus_adapter_rejects_eri_resource_limit():
    with pytest.raises(ProviderContractError, match="ERI tensor exceeds"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_eri_elements=1),
        )


def test_abacus_adapter_rejects_aggregate_eri_resource_limit():
    with pytest.raises(ProviderContractError, match="ERI tensor exceeds"):
        ABACUSLCAOProvider(
            Z=13,
            kpoints=[_kdata(1.0), _kdata(1.0)],
            ao_labels=["0 Al 1s", "0 Al 3s", "0 Al 3p", "0 Al cont"],
            core_hole={
                "orbital_label": "Al 1s",
                "occupation_before": 2.0,
                "occupation_after": 1.0,
                "tracked_mo_index": 0,
                "initial_state_energy_hartree": 19.5,
            },
            validation_limits=ABACUSValidationLimits(max_eri_elements=300),
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


def test_periodic_projection_weights_are_independent_local_fractions():
    k0 = _replace_kdata(_kdata(1.0, eri_value=1.0), local_projection_weight=0.25)
    k1 = _replace_kdata(_kdata(1.0, eri_value=2.0), local_projection_weight=0.25)
    payloads = _provider(k0, k1).iter_kpoint_payloads()

    spectrum = periodic_auger_spectrum(
        payloads,
        continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
        population_threshold=0.5,
        normalize=False,
    )

    assert sum(spectrum.intensities) == pytest.approx((1.0**2 + 2.0**2) * 0.5 * 0.25)


def test_periodic_auger_rejects_tampered_k_weight_metadata():
    payloads = list(_provider(_kdata(1.0)).iter_kpoint_payloads())
    payloads[0].metadata["k_weight"] = 2.0

    with pytest.raises(ProviderContractError, match="k_weight"):
        periodic_auger_spectrum(
            payloads,
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )


def test_periodic_auger_rejects_unconserved_payload_weight_subset():
    payloads = list(_provider(_kdata(1.0), _kdata(3.0)).iter_kpoint_payloads())

    with pytest.raises(ProviderContractError, match="sum to one"):
        periodic_auger_spectrum(
            payloads[:1],
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )


def test_periodic_auger_rejects_symmetric_metadata_index_swap_tampering():
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
    payloads = list(_provider(k0, k1).iter_kpoint_payloads())
    for key in ("kpoint_index", "kpoint", "k_weight"):
        payloads[0].metadata[key], payloads[1].metadata[key] = (
            payloads[1].metadata[key],
            payloads[0].metadata[key],
        )

    with pytest.raises(ProviderContractError, match="conflicts with typed kpoints"):
        periodic_auger_spectrum(
            payloads,
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
            normalize=False,
        )


def test_periodic_auger_rejects_nonfinite_typed_k_weight():
    payload = _provider(_kdata(1.0)).build_payload()
    payload.kpoints = KPointBlock(
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        weights=np.array([1.0]),
        selected_index=0,
    )
    payload.kpoints.weights[0] = np.nan

    with pytest.raises(ProviderContractError, match="weights"):
        periodic_auger_spectrum(
            [payload],
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
        )


def test_periodic_auger_rejects_complex_coefficients_at_public_boundary():
    payload = _provider(_kdata(1.0)).build_payload()
    payload.orbitals.coefficients = payload.orbitals.coefficients.astype(complex)
    payload.orbitals.coefficients[0, 0] += 1j

    with pytest.raises(ProviderContractError, match="real"):
        periodic_auger_spectrum(
            [payload],
            continuum_channel=DiscreteMOContinuumChannel(mo_index=3),
            population_threshold=0.5,
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
