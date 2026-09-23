"""QA regression: provider fail-closed finite contract (HYP-1380).

Independent third-party verification of the finite-value gates merged to
``main`` via HYP-1376 / PR #59 (commit ``7b6bad2``) plus the follow-up ERI
hardening (HYP-1435/1438/1447).  Written by QA against the *behaviour
specification* in the HYP-1380 issue — deliberately not derived from the
author-side tests in ``test_providers_contract.py``.

Verifies, per field:

* NaN / +Inf / -Inf are rejected at construction time with a message that
  locates the offending field, *before* any ``matmul``/``einsum`` linear
  algebra (no ``RuntimeWarning: invalid value encountered`` may escape).
* A single non-finite element hidden in a large array is still caught.
* Finite complex data passes and keeps its complex dtype; a non-finite
  real *or* imaginary part is rejected.
* Valid real and complex payloads (all blocks) validate cleanly and the
  physics-adjacent helpers return finite floats.
* Indirect gates (radial P/Q/occupation, subshell occupations,
  ``UncertaintyBlock``, ``Provenance.uncertainty_summary``) are closed.

Two known residues are recorded as ``xfail`` probes (see the HYP-1380 QA
report): ``payload.metadata`` nested numerics are not gated, and the
``orbital_energies`` mapping error does not name the offending key.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from atomx.providers.contract import (
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    ProviderContractError,
    TwoElectronIntegrals,
    UncertaintyBlock,
    UnitError,
)
from atomx.providers.native_adapter import from_atomic_result

NON_FINITE_VALUES = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="plus-inf"),
    pytest.param(float("-inf"), id="minus-inf"),
]


# ── shared fixtures ────────────────────────────────────────────────────


def _provenance(**kw) -> Provenance:
    base = dict(
        provider="qa-mock",
        code_version="qa 0.0",
        method="QA",
        basis="qa-basis",
        theory_level="QA",
    )
    base.update(kw)
    return Provenance(**base)


def _complex_metric_blocks(n: int = 4, seed: int = 42):
    """Genuinely complex S (Hermitian PD) and C with ``C^H S C = I``.

    ``C`` is complex enough that the *non*-conjugated ``C.T @ S @ C``
    differs from the identity by O(1), so ``orthonormality_residual``
    only returns ~0 if it conjugates.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S = A @ A.conj().T + n * np.eye(n)
    L = np.linalg.cholesky(S)
    Q, _ = np.linalg.qr(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    C = np.linalg.inv(L).conj().T @ Q
    return S, C


def _real_payload(**kw) -> ElectronicStructurePayload:
    """Valid real payload exercising every optional block."""
    n_ao = 3
    C = np.eye(n_ao)
    S = np.eye(n_ao)
    P = np.diag([2.0, 2.0, 0.0])
    defaults = dict(
        Z=6,
        provenance=_provenance(),
        subshells=[SimpleNamespace(occupation=2.0)],
        orbital_energies={"MO0": -0.5, "MO1": -0.2, "MO2": 0.3},
        orbitals=OrbitalCoefficients(
            coefficients=C,
            energies=np.array([-0.5, -0.2, 0.3]),
            occupations=np.array([2.0, 2.0, 0.0]),
        ),
        overlap=OverlapBlock(S),
        density=DensityMatrix(P),
        two_electron=TwoElectronIntegrals(
            np.full((n_ao, n_ao, n_ao, n_ao), 0.05), basis="ao"
        ),
        uncertainty=UncertaintyBlock(
            per_orbital_sigma={"MO0": 0.01},
            ood_score=0.1,
            summary={"model": "qa-mock", "mae": 0.02},
        ),
    )
    defaults.update(kw)
    return ElectronicStructurePayload(**defaults)


class _NativeResultLike:
    """Minimal duck-typed ``AtomicResultLike`` for the native path."""

    Z = 2

    def __init__(self, p_values=None, occupation=2.0):
        p = np.asarray([0.1, 0.2, 0.3] if p_values is None else p_values)
        self.subshells = [SimpleNamespace(occupation=occupation)]
        self.orbital_energies = {"1s": -0.5}
        self.orbitals_raw = {-1: [(p, None, occupation)]}


# ── R1: per-field construction-time rejection, pre-linalg, locatable ──


FIELD_CASES = [
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.array([[bad]]),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        "orbital coefficients",
        id="coefficients",
    ),
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(1),
            energies=np.array([bad]),
            occupations=np.array([2.0]),
        ),
        "orbital energies",
        id="energies",
    ),
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(1),
            energies=np.array([-0.5]),
            occupations=np.array([bad]),
        ),
        "orbital occupations",
        id="occupations",
    ),
    pytest.param(
        lambda bad: OverlapBlock(np.array([[bad]])),
        "overlap S",
        id="overlap-matrix",
    ),
    pytest.param(
        lambda bad: DensityMatrix(np.array([[bad]])),
        "density matrix",
        id="density-matrix",
    ),
    pytest.param(
        lambda bad: TwoElectronIntegrals(np.full((1, 1, 1, 1), bad)),
        "ERI",
        id="eri",
    ),
]


@pytest.mark.parametrize("build,field_label", FIELD_CASES)
@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_block_construction_nonfinite_raises_locatable_error_without_warning(
    build, field_label, bad
):
    """Construction must fail closed, name the field, and stay silent.

    ``simplefilter("error", RuntimeWarning)`` turns any escaped NumPy
    warning (e.g. ``invalid value encountered``) into a test failure, so
    a pass proves the gate fires before linear algebra touches the data.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ProviderContractError, match=field_label) as exc:
            build(bad)
    assert "finite" in str(exc.value)


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_payload_orbital_energies_mapping_nonfinite_fails_at_construction(bad):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ProviderContractError, match="finite"):
            _real_payload(orbital_energies={"MO0": bad})


@pytest.mark.xfail(
    reason="HYP-1380 QA residue: UnitError from the orbital_energies "
    "mapping does not name the offending key — only the value type",
    strict=True,
)
def test_payload_orbital_energies_error_names_offending_key():
    with pytest.raises(UnitError, match="MO1"):
        _real_payload(
            orbital_energies={"MO0": -0.5, "MO1": float("nan")}
        )


# ── R2: single non-finite element inside a large array ─────────────────


LARGE_FIELD_CASES = [
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=_poison(np.zeros((40, 50)), bad, (7, 33)),
            energies=np.linspace(-1, 1, 50),
            occupations=np.zeros(50),
        ),
        "orbital coefficients",
        id="coefficients-40x50",
    ),
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(40, 50),
            energies=_poison(
                np.linspace(-1, 1, 50), bad, 29, scalar=True
            ),
            occupations=np.zeros(50),
        ),
        "orbital energies",
        id="energies-50",
    ),
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.eye(40, 50),
            energies=np.linspace(-1, 1, 50),
            occupations=_poison(np.zeros(50), bad, 3, scalar=True),
        ),
        "orbital occupations",
        id="occupations-50",
    ),
    pytest.param(
        lambda bad: OverlapBlock(
            _poison(np.eye(64), bad, (11, 52))
        ),
        "overlap S",
        id="overlap-64x64",
    ),
    pytest.param(
        lambda bad: DensityMatrix(_poison(np.eye(48), bad, (20, 21))),
        "density matrix",
        id="density-48x48",
    ),
    pytest.param(
        lambda bad: TwoElectronIntegrals(
            _poison(np.full((7, 7, 7, 7), 0.01), bad, (3, 5, 1, 6))
        ),
        "ERI",
        id="eri-7^4",
    ),
]


def _poison(array, bad, index, scalar: bool = False):
    poisoned = array.copy()
    if scalar:
        poisoned[index] = bad
    else:
        poisoned[index] = bad
    return poisoned


@pytest.mark.parametrize("build,field_label", LARGE_FIELD_CASES)
@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_large_array_single_nonfinite_element_is_caught(
    build, field_label, bad
):
    with pytest.raises(ProviderContractError, match=field_label) as exc:
        build(bad)
    assert "finite" in str(exc.value)


def test_large_array_all_finite_passes():
    """Control for the poison tests: the same large shapes must pass."""
    OrbitalCoefficients(
        coefficients=np.eye(40, 50),
        energies=np.linspace(-1, 1, 50),
        occupations=np.zeros(50),
    )
    OverlapBlock(np.eye(64))
    DensityMatrix(np.eye(48))
    TwoElectronIntegrals(np.full((7, 7, 7, 7), 0.01))


# ── R3: complex payloads — finite passes, re/im non-finite rejected ────


COMPLEX_BAD_VALUES = [
    pytest.param(complex(float("nan"), 1.0), id="nan-real"),
    pytest.param(complex(1.0, float("inf")), id="inf-imag"),
    pytest.param(complex(float("-inf"), float("nan")), id="inf-real-nan-imag"),
]

COMPLEX_FIELD_CASES = [
    pytest.param(
        lambda bad: OrbitalCoefficients(
            coefficients=np.array([[bad]]),
            energies=np.array([-0.5]),
            occupations=np.array([2.0]),
        ),
        "orbital coefficients",
        id="coefficients",
    ),
    pytest.param(
        lambda bad: OverlapBlock(np.array([[bad]])),
        "overlap S",
        id="overlap-matrix",
    ),
    pytest.param(
        lambda bad: DensityMatrix(np.array([[bad]])),
        "density matrix",
        id="density-matrix",
    ),
    pytest.param(
        lambda bad: TwoElectronIntegrals(np.full((1, 1, 1, 1), bad)),
        "ERI",
        id="eri",
    ),
]


@pytest.mark.parametrize("build,field_label", COMPLEX_FIELD_CASES)
@pytest.mark.parametrize("bad", COMPLEX_BAD_VALUES)
def test_complex_nonfinite_real_or_imaginary_part_rejected(
    build, field_label, bad
):
    with pytest.raises(ProviderContractError, match=field_label):
        build(bad)


def test_complex_finite_values_keep_complex_dtype():
    S, C = _complex_metric_blocks()
    orbitals = OrbitalCoefficients(
        coefficients=C,
        energies=np.array([-0.5, -0.4, -0.3, -0.2]),
        occupations=np.array([2.0, 2.0, 0.0, 0.0]),
    )
    overlap = OverlapBlock(S)
    density = DensityMatrix(np.diag([2.0 + 0.0j, 2.0 + 0.0j, 0.0, 0.0]))
    assert orbitals.coefficients.dtype == np.complex128
    assert overlap.matrix.dtype == np.complex128
    assert density.matrix.dtype == np.complex128


# ── R4: positive regression — valid real/complex payloads and helpers ──


def test_valid_real_payload_full_blocks_validates_and_helpers_finite():
    payload = _real_payload().validate()

    residual = payload.orthonormality_residual()
    electrons = payload.electron_count()
    deviation = payload.overlap.max_deviation_from_identity()

    assert isinstance(residual, float) and np.isfinite(residual)
    assert isinstance(electrons, float) and np.isfinite(electrons)
    assert isinstance(deviation, float) and np.isfinite(deviation)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert deviation == pytest.approx(0.0, abs=1e-12)
    assert electrons == pytest.approx(4.0)


def test_valid_complex_payload_validates_and_orthonormality_conjugates():
    """``C^H S C = I`` must hold for genuinely complex MOs.

    The fixture is built so the *non*-conjugated ``C.T @ S @ C`` misses
    the identity by O(1); a residual ~0 therefore proves the helper uses
    the conjugate transpose (the HYP-1376 fix), not a plain transpose.
    """
    n = 4
    S, C = _complex_metric_blocks(n=n)
    naive_residual = float(
        np.max(np.abs(C.T @ S @ C - np.eye(n)))
    )
    assert naive_residual > 0.1  # fixture sanity: plain transpose fails

    payload = ElectronicStructurePayload(
        Z=6,
        provenance=_provenance(),
        subshells=[SimpleNamespace(occupation=2.0)],
        orbital_energies={f"MO{i}": -0.5 + 0.1 * i for i in range(n)},
        orbitals=OrbitalCoefficients(
            coefficients=C,
            energies=np.array([-0.5 + 0.1 * i for i in range(n)]),
            occupations=np.array([2.0, 2.0, 0.0, 0.0]),
        ),
        overlap=OverlapBlock(S),
        density=DensityMatrix(
            np.diag([2.0 + 0.0j, 2.0 + 0.0j, 0.0, 0.0])
        ),
        two_electron=TwoElectronIntegrals(
            np.full((n, n, n, n), 0.05 + 0.0j), basis="ao"
        ),
    )
    assert payload.validate() is payload

    P = payload.density.matrix
    residual = payload.orthonormality_residual()
    electrons = payload.electron_count()
    assert isinstance(residual, float)
    assert residual == pytest.approx(0.0, abs=1e-10)
    # electron_count is Tr(P S) in the (non-identity) S metric.
    assert electrons == pytest.approx(
        float(np.real(np.einsum("ij,ji->", P, S)))
    )


def test_electron_count_complex_imag_within_tolerance_returns_real():
    payload = _real_payload(
        density=DensityMatrix(
            np.array(
                [
                    [2.0 + 1e-13j, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
    )
    payload.validate()
    assert payload.electron_count() == pytest.approx(4.0)


def test_electron_count_complex_imag_beyond_tolerance_rejected():
    payload = _real_payload(
        density=DensityMatrix(
            np.array(
                [
                    [2.0 + 0.5j, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
    )
    with pytest.raises(ProviderContractError, match="real within tolerance"):
        payload.electron_count()


def test_native_provider_payload_validates_and_finite_radial_passes():
    payload = from_atomic_result(_NativeResultLike())
    assert payload.validate() is payload
    assert payload.orbitals_raw[-1][0][0].dtype == np.float64


def test_native_provider_nonfinite_radial_fails_closed_through_adapter():
    with pytest.raises(ProviderContractError, match=r"orbitals_raw\[-1\]\[0\].P"):
        from_atomic_result(
            _NativeResultLike(p_values=[0.1, float("nan"), 0.3])
        )


# ── post-construction mutation: gate re-fires before the linear algebra ─


def test_mutated_coefficients_orthonormality_raises_before_matmul():
    payload = _real_payload()
    payload.orbitals.coefficients[0, 0] = float("nan")
    with pytest.raises(ProviderContractError, match="orbital coefficients"):
        payload.orthonormality_residual()


def test_mutated_overlap_orthonormality_raises_before_matmul():
    payload = _real_payload()
    payload.overlap.matrix[1, 1] = float("inf")
    with pytest.raises(ProviderContractError, match="overlap S"):
        payload.orthonormality_residual()


def test_mutated_density_electron_count_raises_before_einsum():
    payload = _real_payload()
    payload.density.matrix[2, 2] = float("nan")
    with pytest.raises(ProviderContractError, match="density matrix"):
        payload.electron_count()


# ── R5: indirect gates — radial, subshells, uncertainty, metadata ─────


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
@pytest.mark.parametrize(
    "component,label_part",
    [
        pytest.param(0, ".P", id="radial-P"),
        pytest.param(1, ".Q", id="radial-Q"),
        pytest.param(2, ".occupation", id="radial-occupation"),
    ],
)
def test_radial_component_nonfinite_rejected_at_payload_construction(
    bad, component, label_part
):
    entry = [np.array([0.1, 0.2]), np.array([0.05, 0.06]), 2.0]
    if component == 0:
        entry[0] = np.array([0.1, bad])
    elif component == 1:
        entry[1] = np.array([0.05, bad])
    else:
        entry[2] = bad
    with pytest.raises(
        ProviderContractError, match=r"orbitals_raw\[-1\]\[0\]" + label_part
    ):
        ElectronicStructurePayload(
            Z=2,
            provenance=_provenance(),
            subshells=[SimpleNamespace(occupation=2.0)],
            orbital_energies={"1s": -0.5},
            orbitals_raw={-1: [tuple(entry)]},
        )


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_subshell_occupation_nonfinite_rejected_at_validate(bad):
    payload = ElectronicStructurePayload(
        Z=2,
        provenance=_provenance(),
        subshells=[SimpleNamespace(occupation=2.0), SimpleNamespace(occupation=bad)],
        orbital_energies={"1s": -0.5},
        orbitals_raw={-1: [(np.array([0.1]), np.array([0.05]), 2.0)]},
    )
    with pytest.raises(
        ProviderContractError, match=r"subshells\[1\]\.occupation"
    ):
        payload.validate()


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_uncertainty_per_orbital_sigma_nonfinite_rejected(bad):
    with pytest.raises(
        ProviderContractError, match=r"UncertaintyBlock\.per_orbital_sigma"
    ):
        UncertaintyBlock(per_orbital_sigma={"MO0": bad})


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_uncertainty_ood_score_nonfinite_rejected(bad):
    with pytest.raises(
        ProviderContractError, match=r"UncertaintyBlock\.ood_score"
    ):
        UncertaintyBlock(ood_score=bad)


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param({"loss": float("nan")}, id="dict-nan-leaf"),
        pytest.param({"nested": {"deeper": [0.1, float("inf")]}}, id="list-inf-leaf"),
        pytest.param(np.array([1.0, float("-inf")]), id="ndarray-leaf"),
    ],
)
def test_uncertainty_summary_nested_nonfinite_rejected(summary):
    with pytest.raises(
        ProviderContractError, match=r"UncertaintyBlock\.summary"
    ):
        UncertaintyBlock(summary=summary)


def test_uncertainty_summary_allows_non_numeric_and_finite_numeric():
    block = UncertaintyBlock(
        summary={
            "model": "qa-mock",
            "calibrated": True,
            "note": None,
            "mae": 0.02,
            "history": [0.1, 0.2],
            "grid": np.array([1.0, 2.0]),
        }
    )
    assert block.validate() is block


@pytest.mark.parametrize("bad", NON_FINITE_VALUES)
def test_provenance_uncertainty_summary_nonfinite_rejected(bad):
    provenance = _provenance(uncertainty_summary={"mae": bad})
    with pytest.raises(
        ProviderContractError, match=r"Provenance\.uncertainty_summary"
    ):
        provenance.validate()


@pytest.mark.xfail(
    reason="HYP-1380 QA residue: ElectronicStructurePayload.metadata "
    "nested numeric values are not finite-gated (fail-open); see the "
    "HYP-1380 QA report for severity discussion",
    strict=True,
)
def test_payload_metadata_nested_nonfinite_rejected():
    payload = _real_payload(
        metadata={"downstream": {"energy_shift": float("nan")}}
    )
    with pytest.raises(ProviderContractError):
        payload.validate()
