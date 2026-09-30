"""Contract-layer tests for the ElectronicStructureProvider (HYP-904).

Uses a mock provider (no SCF) to exercise capability negotiation,
fail-closed validation, the mandatory provenance record, the ML gate,
and the orthogonal/non-orthogonal MO consistency check — independent of
any external back-end.
"""

from __future__ import annotations

import importlib.util
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import atomx.providers.contract as contract_module
from atomx.constants import HARTREE_EV
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
    TwoElectronIntegrals,
    UncertaintyBlock,
    UncertaintyGateError,
    UnitError,
    ensure_hartree,
)
from atomx.state_protocols import AtomicResultLike

#: ``pint`` is an optional extra (``units``): the two unit-coercion nodes are
#: the only coverage of ``ensure_hartree`` on a Quantity.  An explicit marker
#: names the extra instead of dropping the nodes with a bare in-body skip.
#: ``pint>=0.21`` is in the ``dev`` extra.
needs_pint = pytest.mark.skipif(
    importlib.util.find_spec("pint") is None,
    reason="the optional `units` extra is not installed (pint>=0.21); "
           "install with `pip install -e .[dev]`",
)

# ── helpers ────────────────────────────────────────────────────────────


def _good_provenance(**kw):
    base = dict(
        provider="mock",
        code_version="mock 0.0",
        method="MOCK",
        basis="mock-sto",
        theory_level="MOCK",
    )
    base.update(kw)
    return Provenance(**base)


def _mo_block(n_ao=3, n_mo=3, occ=(2.0, 0.0, 0.0)):
    rng = np.random.default_rng(0)
    C = rng.standard_normal((n_ao, n_mo))
    return OrbitalCoefficients(
        coefficients=C,
        energies=np.linspace(-1.0, 1.0, n_mo),
        occupations=np.asarray(occ, dtype=float),
    )


def _minimal_payload(**kw):
    """A payload that just clears the MUST contract via the MO surface."""
    defaults = dict(
        Z=2,
        provenance=_good_provenance(),
        subshells=[],
        orbital_energies={"MO0": -0.5, "MO1": 0.6},
        orbitals=_mo_block(n_mo=2, occ=(2.0, 0.0)),
    )
    defaults.update(kw)
    return ElectronicStructurePayload(**defaults)


# ── unit coercion (I1) ─────────────────────────────────────────────────


def test_ensure_hartree_passes_floats_through():
    assert ensure_hartree(-0.5) == -0.5


@needs_pint
def test_ensure_hartree_converts_pint_quantity():
    from atomx.units import ureg

    q = HARTREE_EV * ureg.electron_volt
    assert ensure_hartree(q) == pytest.approx(1.0, rel=1e-6)


@needs_pint
def test_orbital_energies_coerced_at_construction():
    from atomx.units import ureg

    payload = _minimal_payload(
        orbital_energies={"MO0": HARTREE_EV * ureg.electron_volt}
    )
    assert payload.orbital_energies["MO0"] == pytest.approx(1.0, rel=1e-6)


def test_unit_error_on_uncoercible():
    with pytest.raises(UnitError):
        ensure_hartree(object())


def test_unit_error_does_not_embed_unbounded_value_repr():
    class Noisy:
        def __repr__(self):
            return "x" * 1000

    with pytest.raises(UnitError) as exc_info:
        ensure_hartree(Noisy())

    message = str(exc_info.value)
    assert "x" * 100 not in message
    assert "type=Noisy" in message


def test_unit_error_does_not_embed_unbounded_coercion_exception_text():
    attacker_text = "attacker-controlled-text-" * 10_000

    class BadFloat:
        def __float__(self):
            raise ValueError(attacker_text)

    with pytest.raises(UnitError) as exc_info:
        ensure_hartree(BadFloat())

    message = str(exc_info.value)
    assert len(message) < 200
    assert "attacker-controlled-text" not in message
    assert "type=BadFloat" in message
    assert "reason=unsupported-or-invalid-value" in message
    assert exc_info.value.__cause__ is not None


def test_unit_error_public_text_is_bounded_when_coercion_raises_long_marker():
    marker = "ATTACKER_MARKER"

    class LongMarkerFloat:
        def __float__(self):
            raise RuntimeError(marker * 20_000)

    with pytest.raises(UnitError) as exc_info:
        ensure_hartree(LongMarkerFloat())

    assert len(str(exc_info.value)) < 200
    assert marker not in str(exc_info.value)
    assert marker in str(exc_info.value.__cause__)


def test_unit_error_survives_hostile_metaclass_type_name_lookup():
    attacker_text = "attacker-controlled\r\ntext-" + ("x" * 200_000)

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name == "__name__":
                raise RuntimeError(attacker_text)
            return super().__getattribute__(name)

    class HostileFloat(metaclass=HostileMeta):
        def __float__(self):
            raise ValueError(attacker_text)

    value = HostileFloat()
    with pytest.raises(UnitError) as exc_info:
        ensure_hartree(value)

    message = str(exc_info.value)
    assert len(message) < 200
    assert "\r" not in message
    assert "\n" not in message
    assert "attacker-controlled" not in message
    assert "type=HostileFloat" in message
    assert "reason=unsupported-or-invalid-value" in message
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert exc_info.value.__cause__.args == (attacker_text,)


# ── capability negotiation ─────────────────────────────────────────────


def test_capabilities_report_present_blocks():
    payload = _minimal_payload(
        overlap=OverlapBlock(np.eye(3)),
        density=DensityMatrix(np.eye(3)),
    )
    caps = payload.capabilities()
    assert Capability.MO_COEFFICIENTS in caps
    assert Capability.OVERLAP_S in caps
    assert Capability.DENSITY_MATRIX in caps
    assert Capability.ORBITAL_ENERGIES in caps
    assert Capability.OCCUPATIONS in caps
    assert Capability.TWO_ELECTRON_INTEGRALS not in caps


def test_supports_and_require():
    payload = _minimal_payload()
    assert payload.supports(Capability.MO_COEFFICIENTS)
    assert not payload.supports(Capability.KPOINTS)
    payload.require(Capability.MO_COEFFICIENTS)  # no raise
    with pytest.raises(CapabilityError):
        payload.require(Capability.KPOINTS)


# ── fail-closed validation ─────────────────────────────────────────────


def test_validate_ok_on_minimal_payload():
    assert _minimal_payload().validate() is not None


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf, complex(1.0, np.nan)]
)
@pytest.mark.parametrize(
    "block_factory",
    [
        lambda bad: OrbitalCoefficients(
            coefficients=np.array([[bad]]),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(1),
            energies=np.array([bad]),
            occupations=np.array([2.0]),
        ),
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(1),
            energies=np.array([-0.5]),
            occupations=np.array([bad]),
        ),
        lambda bad: OverlapBlock(np.array([[bad]])),
        lambda bad: DensityMatrix(np.array([[bad]])),
        lambda bad: TwoElectronIntegrals(np.full((1, 1, 1, 1), bad)),
    ],
)
def test_numeric_blocks_reject_non_finite_values(block_factory, bad_value):
    with pytest.raises(ProviderContractError, match="finite"):
        block_factory(bad_value)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_orbital_energy_mapping_rejects_non_finite_values(bad_value):
    with pytest.raises(UnitError, match="finite"):
        _minimal_payload(orbital_energies={"MO0": bad_value})


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_radial_orbitals_reject_non_finite_arrays_before_validate(bad_value):
    with pytest.raises(ProviderContractError, match="finite"):
        ElectronicStructurePayload(
            Z=2,
            provenance=_good_provenance(),
            subshells=[SimpleNamespace(occupation=2.0)],
            orbital_energies={"1s": -0.5},
            orbitals_raw={-1: [(np.array([0.0, bad_value]), np.zeros(2), 2.0)]},
        )


def test_orbitals_raw_hostile_kappa_error_is_bounded():
    marker = "HOSTILE_KAPPA_MARKER"

    class HostileKappa:
        def __int__(self):
            raise RuntimeError(marker * 10_000)

        def __repr__(self):
            return marker * 10_000

    with pytest.raises(ProviderContractError) as exc_info:
        ElectronicStructurePayload(
            Z=2,
            provenance=_good_provenance(),
            subshells=[SimpleNamespace(occupation=2.0)],
            orbital_energies={"1s": -0.5},
            orbitals_raw={HostileKappa(): []},
        )

    message = str(exc_info.value)
    assert len(message) < 200
    assert marker not in message
    assert "orbitals_raw key" in message
    assert "integer kappa" in message
    assert "type=HostileKappa" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert marker in str(exc_info.value.__cause__)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_radial_subshell_occupations_reject_non_finite_values(bad_value):
    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        subshells=[SimpleNamespace(occupation=bad_value)],
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: [(np.array([0.0]), np.array([0.0]), 2.0)]},
    )

    with pytest.raises(ProviderContractError, match="finite"):
        payload.validate()


@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_subshell_generator_is_replayable_and_validate_checks_all_positions(
    bad_index,
):
    def subshells():
        for index, occupation in enumerate([2.0, 1.0, 0.0]):
            yield SimpleNamespace(
                occupation=np.inf if index == bad_index else occupation
            )

    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        subshells=subshells(),
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: [(np.array([0.0]), np.array([0.0]), 2.0)]},
    )

    assert payload.capabilities()
    assert isinstance(payload.subshells, tuple)
    with pytest.raises(ProviderContractError, match=f"subshells\\[{bad_index}\\]"):
        payload.validate()


def test_validate_normalizes_mutated_subshell_generator_before_capabilities():
    payload = _minimal_payload()
    payload.orbitals = None
    payload.subshells = (
        SimpleNamespace(occupation=np.nan if index == 1 else 1.0)
        for index in range(3)
    )
    payload.orbitals_raw = {-1: [(np.array([0.0]), np.array([0.0]), 1.0)]}

    with pytest.raises(ProviderContractError, match=r"subshells\[1\]"):
        payload.validate()
    assert isinstance(payload.subshells, tuple)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_orbitals_raw_one_shot_iterables_are_normalized_and_replayable(count):
    def orbital_stream():
        for index in range(count):
            yield (
                np.array([float(index)]),
                np.array([float(index + 1)]),
                1.0,
            )

    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        subshells=[SimpleNamespace(occupation=2.0)],
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: iter(orbital_stream())},
        orbitals=None if count else _mo_block(n_mo=2, occ=(2.0, 0.0)),
    )

    assert isinstance(payload.orbitals_raw[-1], list)
    assert len(payload.orbitals_raw[-1]) == count
    assert payload.validate() is payload
    assert payload.validate() is payload
    if count:
        assert Capability.RADIAL_ORBITALS in payload.capabilities()
    else:
        assert Capability.RADIAL_ORBITALS not in payload.capabilities()


def test_orbitals_raw_infinite_iterable_fails_with_bounded_budget(monkeypatch):
    monkeypatch.setattr(contract_module, "_MAX_ORBITALS_RAW_PER_KAPPA", 2)

    def infinite_orbitals():
        while True:
            yield (np.array([0.0]), np.array([1.0]), 1.0)

    with pytest.raises(ProviderContractError, match="maximum item budget"):
        ElectronicStructurePayload(
            Z=2,
            provenance=_good_provenance(),
            subshells=[SimpleNamespace(occupation=2.0)],
            orbital_energies={"1s": -0.5},
            orbitals_raw={-1: infinite_orbitals()},
        )


def test_orbitals_raw_entry_iterable_is_normalized_once_and_revalidated():
    entry = iter((np.array([0.0]), np.array([1.0]), 1.0))
    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        subshells=[SimpleNamespace(occupation=2.0)],
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: iter([entry])},
    )

    assert payload.validate() is payload
    assert payload.validate() is payload
    p_component, q_component, occupation = payload.orbitals_raw[-1][0]
    assert isinstance(p_component, np.ndarray)
    assert isinstance(q_component, np.ndarray)
    assert occupation == pytest.approx(1.0)


def test_non_finite_coefficients_fail_before_orthonormality_residual():
    with pytest.raises(ProviderContractError, match="finite"):
        OrbitalCoefficients(
            coefficients=np.array([[np.nan]]),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        )


def test_finite_complex_contract_arrays_remain_supported():
    phase = np.exp(0.25j)
    payload = _minimal_payload(
        orbitals=OrbitalCoefficients(
            coefficients=np.array([[phase]], dtype=complex),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        overlap=OverlapBlock(np.array([[1.0 + 0.0j]])),
        density=DensityMatrix(np.array([[2.0 + 0.0j]])),
        two_electron=TwoElectronIntegrals(
            np.ones((1, 1, 1, 1), dtype=complex), basis="mo"
        ),
        orbital_energies={"MO0": -0.5},
    )

    assert payload.validate() is payload
    assert payload.orbitals.coefficients.dtype.kind == "c"
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-12)


def test_object_dtype_contract_arrays_fail_closed_even_for_finite_complex():
    with pytest.raises(ProviderContractError, match="non-object dtype"):
        OrbitalCoefficients(
            coefficients=np.array([[1.0 + 0.0j]], dtype=object),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        )


@pytest.mark.parametrize(
    "block_factory",
    [
        lambda: OverlapBlock(np.array([[1.0]], dtype=object)),
        lambda: DensityMatrix(np.array([[1.0]], dtype=object)),
        lambda: TwoElectronIntegrals(
            np.ones((1, 1, 1, 1), dtype=object)
        ),
    ],
)
def test_object_dtype_optional_blocks_fail_closed(block_factory):
    with pytest.raises(ProviderContractError, match="non-object dtype"):
        block_factory()


@pytest.mark.parametrize(
    "mutate, call, error_type, match",
    [
        (
            lambda payload: payload.orbitals.coefficients.__setitem__(
                (0, 0), np.nan
            ),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbital coefficients",
        ),
        (
            lambda payload: payload.orbitals.energies.__setitem__(0, np.inf),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbital energies",
        ),
        (
            lambda payload: payload.orbitals.occupations.__setitem__(
                0, -np.inf
            ),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbital occupations",
        ),
        (
            lambda payload: payload.overlap.matrix.__setitem__((0, 0), np.nan),
            lambda payload: payload.orthonormality_residual(),
            ProviderContractError,
            "overlap S",
        ),
        (
            lambda payload: payload.density.matrix.__setitem__((0, 0), np.nan),
            lambda payload: payload.electron_count(),
            ProviderContractError,
            "density matrix",
        ),
        (
            lambda payload: payload.two_electron.eri.__setitem__(
                (0, 0, 0, 0), np.inf
            ),
            lambda payload: payload.validate(),
            ProviderContractError,
            "ERI",
        ),
        (
            lambda payload: payload.orbital_energies.__setitem__("MO0", np.nan),
            lambda payload: payload.validate(),
            UnitError,
            "finite Hartree",
        ),
        (
            lambda payload: payload.orbitals_raw[-1][0][0].__setitem__(0, np.nan),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbitals_raw",
        ),
        (
            lambda payload: payload.orbitals_raw[-1][0][1].__setitem__(0, np.inf),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbitals_raw",
        ),
        (
            lambda payload: payload.orbitals_raw[-1].__setitem__(
                0, (np.array([0.0]), np.array([1.0]), -np.inf)
            ),
            lambda payload: payload.validate(),
            ProviderContractError,
            "orbitals_raw",
        ),
    ],
)
def test_payload_validate_rejects_public_numeric_mutation(
    mutate, call, error_type, match
):
    payload = _minimal_payload(
        orbitals=_mo_block(n_ao=2, n_mo=2, occ=(2.0, 0.0)),
        overlap=OverlapBlock(np.eye(2)),
        density=DensityMatrix(np.eye(2)),
        two_electron=TwoElectronIntegrals(np.zeros((2, 2, 2, 2)), basis="mo"),
        orbitals_raw={-1: [(np.array([0.0]), np.array([1.0]), 1.0)]},
    )

    mutate(payload)

    with pytest.raises(error_type, match=match):
        call(payload)


def test_density_block_revalidates_mutated_overlap_before_einsum():
    density = DensityMatrix(np.eye(2))
    overlap = OverlapBlock(np.eye(2))
    overlap.matrix[0, 0] = np.nan

    with pytest.raises(ProviderContractError, match="overlap S"):
        density.n_electrons(overlap)


def test_density_block_revalidates_mutated_matrix_before_trace():
    density = DensityMatrix(np.eye(2))
    density.matrix[0, 0] = np.nan

    with pytest.raises(ProviderContractError, match="density matrix"):
        density.n_electrons()


def test_validate_rejects_no_orbital_representation():
    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        orbital_energies={"MO0": -0.5},
        subshells=[],  # no radial orbitals, no MO block
    )
    with pytest.raises(ProviderContractError):
        payload.validate()


def test_validate_rejects_incomplete_provenance():
    payload = _minimal_payload(provenance=_good_provenance(basis=""))
    with pytest.raises(ProviderContractError):
        payload.validate()


def test_provenance_default_energy_zero_passes():
    assert _good_provenance().validate() is not None


def test_provenance_empty_energy_zero_rejected():
    with pytest.raises(ProviderContractError):
        _good_provenance(energy_zero="").validate()


def test_provenance_whitespace_energy_zero_rejected():
    with pytest.raises(ProviderContractError):
        _good_provenance(energy_zero="   ").validate()


@pytest.mark.parametrize("bad_value", [None, 42])
@pytest.mark.parametrize(
    "field_name", ["provider", "code_version", "method", "basis", "energy_zero"]
)
def test_provenance_mandatory_fields_must_be_strings(field_name, bad_value):
    with pytest.raises(ProviderContractError):
        _good_provenance(**{field_name: bad_value}).validate()


def test_ml_payload_without_uncertainty_fails_closed():
    payload = _minimal_payload(
        provenance=_good_provenance(is_ml=True, ood_status="in-domain")
    )
    with pytest.raises(UncertaintyGateError):
        payload.validate()


def test_ml_payload_with_uncertainty_passes():
    payload = _minimal_payload(
        provenance=_good_provenance(is_ml=True, ood_status="in-domain"),
        uncertainty=UncertaintyBlock(ood_score=0.1),
    )
    assert payload.validate() is not None
    assert Capability.UNCERTAINTY in payload.capabilities()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_uncertainty_rejects_non_finite_per_orbital_sigma(bad_value):
    with pytest.raises(
        ProviderContractError,
        match=r"UncertaintyBlock\.per_orbital_sigma\[MO0\]",
    ):
        UncertaintyBlock(per_orbital_sigma={"MO0": bad_value})


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_uncertainty_rejects_non_finite_ood_score(bad_value):
    with pytest.raises(ProviderContractError, match="UncertaintyBlock.ood_score"):
        UncertaintyBlock(ood_score=bad_value)


@pytest.mark.parametrize(
    "summary, match",
    [
        ({"level": {"sigma": np.nan}}, "UncertaintyBlock.summary.level.sigma"),
        ({"values": [0.1, np.inf]}, r"UncertaintyBlock.summary.values\[1\]"),
    ],
)
def test_uncertainty_summary_rejects_nested_non_finite_numeric_values(
    summary, match
):
    with pytest.raises(ProviderContractError, match=match):
        UncertaintyBlock(summary=summary)


def test_uncertainty_summary_allows_finite_nested_numeric_values():
    block = UncertaintyBlock(
        summary={
            "level": {"sigma": 0.2 + 0.1j},
            "values": np.array([1.0 + 0.0j]),
            "flags": [True, "ok", None],
        }
    )
    assert block.validate() is block


def test_uncertainty_summary_rejects_self_reference_cycle():
    summary = {}
    summary["self"] = summary

    with pytest.raises(
        ProviderContractError,
        match="UncertaintyBlock.summary.self contains a cycle",
    ):
        UncertaintyBlock(summary=summary)


def test_uncertainty_summary_rejects_indirect_cycle():
    parent = {}
    child = {"parent": parent}
    parent["child"] = child

    with pytest.raises(ProviderContractError, match="contains a cycle"):
        UncertaintyBlock(summary=parent)


def test_uncertainty_summary_rejects_excessive_depth(monkeypatch):
    monkeypatch.setattr(contract_module, "_MAX_NESTED_NUMERIC_DEPTH", 2)

    with pytest.raises(ProviderContractError, match="maximum nested depth"):
        UncertaintyBlock(summary={"a": {"b": {"c": 1.0}}})


def test_uncertainty_summary_rejects_excessive_node_budget(monkeypatch):
    monkeypatch.setattr(contract_module, "_MAX_NESTED_NUMERIC_NODES", 3)

    with pytest.raises(ProviderContractError, match="nested node budget"):
        UncertaintyBlock(summary={"values": [1.0, 2.0, 3.0]})


def test_uncertainty_summary_hostile_key_error_is_bounded():
    marker = "HOSTILE_SUMMARY_KEY"

    class HostileKey:
        def __str__(self):
            raise RuntimeError(marker * 3_000)

        def __repr__(self):
            return marker * 3_000

    with pytest.raises(ProviderContractError) as exc_info:
        UncertaintyBlock(summary={HostileKey(): np.inf})

    message = str(exc_info.value)
    assert len(message) < 200
    assert marker not in message
    assert "UncertaintyBlock.summary.<HostileKey>" in message
    assert "finite" in message


def test_uncertainty_summary_long_label_hook_return_is_not_public():
    marker = "HOSTILE_LABEL_RETURN"

    class HostileString(str):
        def __str__(self):
            return marker * 3_000

    with pytest.raises(ProviderContractError) as exc_info:
        UncertaintyBlock(summary={HostileString("stable-key"): np.inf})

    message = str(exc_info.value)
    assert len(message) < 200
    assert marker not in message
    assert "UncertaintyBlock.summary.stable-key" in message


def test_bounded_key_preserves_legal_int_and_string_mapping_keys():
    with pytest.raises(ProviderContractError, match="summary.shell.1"):
        UncertaintyBlock(summary={"shell": {1: np.inf}})


def test_provenance_uncertainty_summary_rejects_nested_non_finite_values():
    provenance = _good_provenance(uncertainty_summary={"ood": {"score": np.inf}})

    with pytest.raises(
        ProviderContractError,
        match="Provenance.uncertainty_summary.ood.score",
    ):
        provenance.validate()


def test_provenance_uncertainty_summary_rejects_cycle():
    summary = {}
    summary["self"] = summary
    provenance = _good_provenance(uncertainty_summary=summary)

    with pytest.raises(
        ProviderContractError,
        match="Provenance.uncertainty_summary.self contains a cycle",
    ):
        provenance.validate()


def test_ml_provenance_requires_ood_status():
    with pytest.raises(ProviderContractError):
        _good_provenance(is_ml=True).validate()


# ── orthogonal / non-orthogonal consistency ─────────────────────────────


def test_orthonormal_mo_residual_identity_overlap():
    """MOs built to be S-orthonormal give a ~0 residual (S = I here)."""
    C = np.eye(3)
    payload = _minimal_payload(
        orbitals=OrbitalCoefficients(
            coefficients=C,
            energies=np.array([-1.0, 0.0, 1.0]),
            occupations=np.array([2.0, 0.0, 0.0]),
        )
    )
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-12)


def test_orthonormal_mo_residual_nonorthogonal_basis():
    """Non-orthogonal AO basis (S != I) with S-orthonormal MOs → residual 0.

    Build S from a random overlap, then obtain MOs as C = S^{-1/2}, which
    are orthonormal in the S metric: C^T S C = I even though max|S-I| > 0.
    """
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 4 * np.eye(4)  # SPD, non-identity
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)  # unit-diagonal AO overlap
    # C = S^{-1/2} → columns orthonormal under S.
    evals, evecs = np.linalg.eigh(S)
    S_inv_half = evecs @ np.diag(evals ** -0.5) @ evecs.T
    payload = _minimal_payload(
        overlap=OverlapBlock(S),
        orbitals=OrbitalCoefficients(
            coefficients=S_inv_half,
            energies=np.linspace(-1, 1, 4),
            occupations=np.array([2.0, 2.0, 0.0, 0.0]),
        ),
    )
    assert not payload.overlap.is_orthonormal()
    assert payload.overlap.max_deviation_from_identity() > 0.1
    assert payload.orthonormality_residual() == pytest.approx(0.0, abs=1e-10)


def test_density_electron_count_uses_overlap():
    """Tr(P S) recovers the electron number for a non-orthogonal basis."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 3))
    S = A @ A.T + 3 * np.eye(3)
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    # One doubly-occupied MO c (S-normalized): P = 2 c c^T, Tr(P S) = 2.
    c = np.linalg.solve(S, rng.standard_normal(3))
    norm = np.sqrt(c @ S @ c)
    c = c / norm
    P = 2.0 * np.outer(c, c)
    payload = _minimal_payload(
        overlap=OverlapBlock(S), density=DensityMatrix(P)
    )
    assert payload.electron_count() == pytest.approx(2.0, abs=1e-9)


def _complex_warning_type():
    return getattr(np, "ComplexWarning", np.exceptions.ComplexWarning)


def test_complex_density_trace_electron_count_without_complex_warning():
    payload = _minimal_payload(
        orbitals=OrbitalCoefficients(
            coefficients=np.eye(2, dtype=complex),
            energies=np.array([-0.5, 0.1]),
            occupations=np.array([1.0, 1.0]),
        ),
        density=DensityMatrix(
            np.array(
                [[1.0 + 2e-12j, 0.0], [0.0, 1.0 - 1e-12j]],
                dtype=complex,
            )
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", _complex_warning_type())
        assert payload.electron_count() == pytest.approx(2.0, abs=1e-12)


def test_complex_density_overlap_electron_count_without_complex_warning():
    payload = _minimal_payload(
        orbitals=OrbitalCoefficients(
            coefficients=np.eye(2, dtype=complex),
            energies=np.array([-0.5, 0.1]),
            occupations=np.array([1.0, 1.0]),
        ),
        overlap=OverlapBlock(np.eye(2, dtype=complex)),
        density=DensityMatrix(
            np.array(
                [[1.0 + 2e-12j, 0.0], [0.0, 1.0 - 1e-12j]],
                dtype=complex,
            )
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", _complex_warning_type())
        assert payload.electron_count() == pytest.approx(2.0, abs=1e-12)


def test_complex_density_trace_electron_count_rejects_imaginary_part():
    payload = _minimal_payload(
        density=DensityMatrix(np.array([[2.0 + 1e-4j]], dtype=complex)),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )

    with pytest.raises(
        ProviderContractError, match="DensityMatrix.n_electrons.trace"
    ):
        payload.electron_count()


def test_complex_density_overlap_electron_count_rejects_imaginary_part():
    payload = _minimal_payload(
        density=DensityMatrix(np.array([[2.0 + 1e-4j]], dtype=complex)),
        overlap=OverlapBlock(np.array([[1.0 + 0.0j]], dtype=complex)),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )

    with pytest.raises(
        ProviderContractError, match="DensityMatrix.n_electrons.overlap"
    ):
        payload.electron_count()


def test_density_trace_revalidates_before_contraction(monkeypatch):
    payload = _minimal_payload(density=DensityMatrix(np.eye(2)))
    payload.density.matrix[0, 0] = np.nan

    def fail_trace(_matrix):
        raise AssertionError("trace should not run before validation")

    monkeypatch.setattr(np, "trace", fail_trace)

    with pytest.raises(ProviderContractError, match="density matrix"):
        payload.electron_count()


def test_density_overlap_revalidates_before_contraction(monkeypatch):
    payload = _minimal_payload(
        density=DensityMatrix(np.eye(2)),
        overlap=OverlapBlock(np.eye(2)),
    )
    payload.overlap.matrix[0, 0] = np.nan

    def fail_einsum(*_args, **_kwargs):
        raise AssertionError("einsum should not run before validation")

    monkeypatch.setattr(np, "einsum", fail_einsum)

    with pytest.raises(ProviderContractError, match="overlap S"):
        payload.electron_count()


def test_density_overlap_shape_mismatch_rejected_before_contraction(monkeypatch):
    payload = _minimal_payload(
        density=DensityMatrix(np.eye(2)),
        overlap=OverlapBlock(np.eye(3)),
        orbitals=None,
        subshells=[SimpleNamespace(occupation=2.0)],
        orbitals_raw={-1: [(np.array([0.0]), np.array([1.0]), 2.0)]},
    )

    def fail_einsum(*_args, **_kwargs):
        raise AssertionError("einsum must not run before shape check")

    monkeypatch.setattr(np, "einsum", fail_einsum)

    with pytest.raises(
        ProviderContractError, match="DensityMatrix.n_electrons.overlap"
    ):
        payload.electron_count()


def test_density_overlap_shape_mismatch_error_includes_shapes():
    payload = _minimal_payload(
        density=DensityMatrix(np.eye(2)),
        overlap=OverlapBlock(np.eye(3)),
        orbitals=None,
        subshells=[SimpleNamespace(occupation=2.0)],
        orbitals_raw={-1: [(np.array([0.0]), np.array([1.0]), 2.0)]},
    )

    with pytest.raises(ProviderContractError) as exc_info:
        payload.electron_count()

    message = str(exc_info.value)
    assert "DensityMatrix.n_electrons.overlap" in message
    assert "density (2, 2)" in message
    assert "overlap (3, 3)" in message


@pytest.mark.parametrize("use_overlap", [False, True])
def test_complex_density_electron_count_accepts_imaginary_tolerance_boundary(
    use_overlap,
):
    real = 2.0
    tolerance = 1e-10 + 1e-12 * abs(real)
    payload = _minimal_payload(
        density=DensityMatrix(
            np.array([[real + tolerance * 1j]], dtype=complex)
        ),
        overlap=(
            OverlapBlock(np.array([[1.0 + 0.0j]], dtype=complex))
            if use_overlap
            else None
        ),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", _complex_warning_type())
        assert payload.electron_count() == pytest.approx(real, abs=1e-12)


@pytest.mark.parametrize(
    "use_overlap, match",
    [
        (False, "DensityMatrix.n_electrons.trace"),
        (True, "DensityMatrix.n_electrons.overlap"),
    ],
)
def test_complex_density_electron_count_rejects_beyond_imaginary_tolerance(
    use_overlap, match
):
    real = 2.0
    tolerance = 1e-10 + 1e-12 * abs(real)
    imag = np.nextafter(tolerance, np.inf)
    payload = _minimal_payload(
        density=DensityMatrix(np.array([[real + imag * 1j]], dtype=complex)),
        overlap=(
            OverlapBlock(np.array([[1.0 + 0.0j]], dtype=complex))
            if use_overlap
            else None
        ),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )

    with pytest.raises(ProviderContractError, match=match):
        payload.electron_count()


def test_density_overlap_contraction_non_finite_result_fails_closed():
    payload = _minimal_payload(
        density=DensityMatrix(np.array([[2e154]])),
        overlap=OverlapBlock(np.array([[2e154]])),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )

    with pytest.raises(
        ProviderContractError, match="DensityMatrix.n_electrons.overlap.*finite"
    ):
        payload.electron_count()


# ── structural compatibility with AtomicResultLike ──────────────────────


def test_payload_is_atomic_result_like():
    payload = _minimal_payload()
    assert isinstance(payload, AtomicResultLike)


def test_payload_satisfies_provider_protocol_via_native(monkeypatch):
    """A trivial object with the three provider methods is a provider."""

    class _Prov:
        def capabilities(self):
            return set()

        def supports(self, *a):
            return True

        def build_payload(self):
            return _minimal_payload()

    assert isinstance(_Prov(), ElectronicStructureProvider)


# ── shape guards ────────────────────────────────────────────────────────


def test_orbital_block_shape_mismatch_rejected():
    with pytest.raises(ProviderContractError):
        OrbitalCoefficients(
            coefficients=np.zeros((3, 2)),
            energies=np.zeros(3),  # wrong: should be length 2
            occupations=np.zeros(2),
        )


def test_overlap_must_be_square():
    with pytest.raises(ProviderContractError):
        OverlapBlock(np.zeros((3, 2)))


def test_payload_overlap_shape_must_match_orbital_ao_count():
    payload = _minimal_payload(overlap=OverlapBlock(np.eye(4)))

    with pytest.raises(ProviderContractError):
        payload.validate()


def test_payload_density_shape_must_match_orbital_ao_count():
    payload = _minimal_payload(density=DensityMatrix(np.eye(4)))

    with pytest.raises(ProviderContractError):
        payload.validate()


def test_payload_ao_eri_shape_must_match_orbital_ao_count():
    payload = _minimal_payload(
        two_electron=TwoElectronIntegrals(np.zeros((4, 4, 4, 4)), basis="ao")
    )

    with pytest.raises(ProviderContractError):
        payload.validate()


def test_payload_unknown_eri_basis_rejected():
    payload = _minimal_payload(
        two_electron=TwoElectronIntegrals(np.zeros((3, 3, 3, 3)), basis="grid")
    )

    with pytest.raises(ProviderContractError):
        payload.validate()


def test_payload_small_eri_values_must_be_finite():
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = np.nan

    with pytest.raises(ProviderContractError, match="ERI values must be finite"):
        TwoElectronIntegrals(eri, basis="mo")


@pytest.mark.parametrize(
    "shape",
    [
        (np.int64(1), 1, 1, 1),
        (True, 1, 1, 1),
        (-1, 1, 1, 1),
    ],
)
def test_eri_shape_metadata_rejects_abnormal_dimensions(shape):
    with pytest.raises(
        ProviderContractError,
        match="ERI dimensions must be non-negative built-in integers",
    ):
        TwoElectronIntegrals._validated_shape(shape)


def test_radial_only_payload_unknown_eri_basis_rejected():
    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_good_provenance(),
        subshells=[SimpleNamespace(occupation=2.0)],
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: [(np.array([0.0]), np.array([1.0]), 1.0)]},
        two_electron=TwoElectronIntegrals(np.zeros((2, 2, 2, 2)), basis="grid"),
    )

    with pytest.raises(ProviderContractError):
        payload.validate()


def test_two_electron_integrals_must_be_rank_four():
    with pytest.raises(ProviderContractError):
        TwoElectronIntegrals(np.zeros((2, 2, 2)))


def test_two_electron_integrals_axes_must_be_equal_length():
    with pytest.raises(ProviderContractError):
        TwoElectronIntegrals(np.zeros((2, 2, 2, 3)))


def test_complex_eri_within_imaginary_tolerance_is_projected_to_real():
    eri = np.zeros((2, 2, 2, 2), dtype=complex)
    eri[0, 0, 0, 0] = 1.5 + 1e-12j

    block = TwoElectronIntegrals(eri, basis="ao")

    assert block.eri.dtype == float
    assert block.eri[0, 0, 0, 0] == pytest.approx(1.5)


def test_complex_eri_imaginary_tolerance_boundary_is_accepted():
    real = 1.5
    tolerance = 1e-10 + 1e-12 * abs(real)
    eri = np.full((2, 2, 2, 2), real + tolerance * 1j, dtype=complex)

    block = TwoElectronIntegrals(eri, basis="ao")

    assert block.eri.dtype == float
    assert np.all(block.eri == pytest.approx(real))


def test_complex_eri_large_imaginary_part_fails_closed():
    # HYP-1381 F1 probe: imag=1e6 was silently discarded by ``.real``.
    eri = np.zeros((2, 2, 2, 2), dtype=complex)
    eri[1, 1, 1, 1] = 0.25 + 1e6j

    with pytest.raises(
        ProviderContractError,
        match="complex ERI imaginary parts exceed the provider contract",
    ):
        TwoElectronIntegrals(eri, basis="ao")


def test_complex_eri_just_beyond_imaginary_tolerance_fails_closed():
    real = 1.5
    tolerance = 1e-10 + 1e-12 * abs(real)
    imag = np.nextafter(tolerance, np.inf)
    eri = np.full((2, 2, 2, 2), real + imag * 1j, dtype=complex)

    with pytest.raises(
        ProviderContractError,
        match="complex ERI imaginary parts exceed the provider contract",
    ):
        TwoElectronIntegrals(eri, basis="ao")


def test_complex_eri_non_finite_imaginary_part_still_rejected_as_non_finite():
    eri = np.zeros((2, 2, 2, 2), dtype=complex)
    eri[0, 0, 0, 0] = 1.0 + np.inf * 1j

    with pytest.raises(ProviderContractError, match="ERI values must be finite"):
        TwoElectronIntegrals(eri, basis="ao")


def test_payload_validate_catches_complex_eri_imaginary_violation():
    eri = np.zeros((2, 2, 2, 2), dtype=complex)
    eri[0, 1, 0, 1] = 0.5 + 10.0j
    payload = _minimal_payload(
        two_electron=TwoElectronIntegrals.__new__(TwoElectronIntegrals)
    )
    # Construct without __post_init__ validation to exercise payload.validate().
    payload.two_electron.eri = eri
    payload.two_electron.basis = "ao"
    payload.two_electron.notation = "chemist"

    with pytest.raises(ProviderContractError):
        payload.validate()
def _masked(matrix):
    # HYP-1381 F2 probe: the mask hides a finite sentinel (999) that
    # np.asarray would silently promote to active data.
    return np.ma.masked_array(matrix, mask=matrix == 0.0, fill_value=999.0)


@pytest.mark.parametrize(
    "block_factory",
    [
        lambda: OverlapBlock(_masked(np.eye(2))),
        lambda: DensityMatrix(_masked(np.eye(2))),
        lambda: OrbitalCoefficients(
            coefficients=_masked(np.eye(2)),
            energies=np.array([-0.5, 0.5]),
            occupations=np.array([2.0, 0.0]),
        ),
        lambda: OrbitalCoefficients(
            coefficients=np.eye(2),
            energies=_masked(np.array([-0.5, 0.5])),
            occupations=np.array([2.0, 0.0]),
        ),
        lambda: OrbitalCoefficients(
            coefficients=np.eye(2),
            energies=np.array([-0.5, 0.5]),
            occupations=_masked(np.array([2.0, 0.0])),
        ),
        lambda: TwoElectronIntegrals(
            _masked(np.zeros((2, 2, 2, 2))), basis="ao"
        ),
    ],
)
def test_masked_arrays_fail_closed_on_all_numeric_blocks(block_factory):
    with pytest.raises(
        ProviderContractError, match="base numpy.ndarray"
    ):
        block_factory()


def test_masked_scalar_fails_closed_in_electron_count_path():
    payload = _minimal_payload(
        density=DensityMatrix(np.array([[2.0]])),
        orbitals=OrbitalCoefficients(
            coefficients=np.ones((1, 1)),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        orbital_energies={"MO0": -0.5},
    )
    payload.density.matrix = np.ma.masked

    with pytest.raises(
        ProviderContractError, match="base numpy.ndarray"
    ):
        payload.electron_count()


class _TaggedArray(np.ndarray):
    """Minimal ndarray subclass used to probe the subclass gate."""


@pytest.mark.parametrize(
    "subclass_value",
    [
        np.matrix(np.eye(2)),
        np.eye(2).view(_TaggedArray),
    ],
)
def test_ndarray_subclasses_fail_closed_on_overlap_block(subclass_value):
    with pytest.raises(
        ProviderContractError, match="base numpy.ndarray"
    ):
        OverlapBlock(subclass_value)


def test_masked_array_subclass_name_in_error_is_bounded():
    try:
        OverlapBlock(_masked(np.eye(2)))
    except ProviderContractError as exc:
        message = str(exc)
    else:  # pragma: no cover - the parametrize above guarantees the raise
        raise AssertionError("masked overlap must fail closed")
    assert "MaskedArray" in message
    assert len(message) < 300


def test_plain_array_like_inputs_still_accepted_after_subclass_gate():
    block = OverlapBlock([[1.0, 0.0], [0.0, 1.0]])
    assert block.matrix.shape == (2, 2)
    density = DensityMatrix(np.eye(2))
    assert density.matrix.shape == (2, 2)
def _payload_with_eri_basis(basis):
    return _minimal_payload(
        two_electron=TwoElectronIntegrals(np.zeros((2, 2, 2, 2)), basis=basis)
    )


def test_unknown_eri_basis_echo_is_bounded_and_truncated():
    # HYP-1381 F3 probe: an adapter-controlled token-like string was echoed
    # unbounded into the error message and any downstream logs.
    token_like = "sk-live-TOKEN-" + "A" * 500
    payload = _payload_with_eri_basis(token_like)

    with pytest.raises(ProviderContractError) as excinfo:
        payload.validate()

    message = str(excinfo.value)
    assert token_like not in message
    assert "sk-live-TOKEN-" in message  # prefix retained for debuggability
    assert message.endswith("...'") or "...'" in message
    assert len(message) < 160


def test_unknown_eri_basis_echo_strips_control_characters():
    basis = "ao\x00\nsecret\ttoken\r\n"
    payload = _payload_with_eri_basis(basis)

    with pytest.raises(ProviderContractError) as excinfo:
        payload.validate()

    message = str(excinfo.value)
    assert "\n" not in message
    assert "\t" not in message
    assert "\x00" not in message


def test_unknown_eri_basis_non_string_type_echoes_bounded_type_name():
    class HostileBasis:
        pass

    payload = _payload_with_eri_basis(HostileBasis())

    with pytest.raises(ProviderContractError) as excinfo:
        payload.validate()

    message = str(excinfo.value)
    assert "HostileBasis" in message
    assert len(message) < 160


def test_unknown_eri_basis_short_string_still_echoed_for_debuggability():
    payload = _payload_with_eri_basis("grid")

    with pytest.raises(ProviderContractError, match="got 'grid'"):
        payload.validate()
