"""Contract-layer tests for the ElectronicStructureProvider (HYP-904).

Uses a mock provider (no SCF) to exercise capability negotiation,
fail-closed validation, the mandatory provenance record, the ML gate,
and the orthogonal/non-orthogonal MO consistency check — independent of
any external back-end.
"""

from __future__ import annotations

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


def test_two_electron_integrals_must_be_rank_four():
    with pytest.raises(ProviderContractError):
        TwoElectronIntegrals(np.zeros((2, 2, 2)))


def test_two_electron_integrals_axes_must_be_equal_length():
    with pytest.raises(ProviderContractError):
        TwoElectronIntegrals(np.zeros((2, 2, 2, 3)))
