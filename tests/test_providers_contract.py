"""Contract-layer tests for the ElectronicStructureProvider (HYP-904).

Uses a mock provider (no SCF) to exercise capability negotiation,
fail-closed validation, the mandatory provenance record, the ML gate,
and the orthogonal/non-orthogonal MO consistency check — independent of
any external back-end.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

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
from atomx.scattering.protocols import AtomicResultLike


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


def test_ensure_hartree_converts_pint_quantity():
    pint = pytest.importorskip("pint")
    from atomx.units import ureg

    q = 27.211386245988 * ureg.electron_volt
    assert ensure_hartree(q) == pytest.approx(1.0, rel=1e-6)


def test_orbital_energies_coerced_at_construction():
    pint = pytest.importorskip("pint")
    from atomx.units import ureg

    payload = _minimal_payload(
        orbital_energies={"MO0": 27.211386245988 * ureg.electron_volt}
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
        summary={"level": {"sigma": 0.2}, "flags": [True, "ok", None]}
    )
    assert block.validate() is block


def test_provenance_uncertainty_summary_rejects_nested_non_finite_values():
    provenance = _good_provenance(uncertainty_summary={"ood": {"score": np.inf}})

    with pytest.raises(
        ProviderContractError,
        match="Provenance.uncertainty_summary.ood.score",
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
