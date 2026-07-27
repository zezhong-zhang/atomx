"""Unified ``ElectronicStructureProvider`` contract (ADR-0001, HYP-904).

This is the thin *contract / normalization* layer that sits between an
external electronic-structure back-end (PySCF, ABACUS, DeePTB) and the
atomx physics kernels (CSF/CI, two-electron ``R^k``, continuum coupling,
spectrum synthesis).  It defines:

* :class:`Capability` — the negotiable feature set (§5 of ADR-0001).
* :class:`Provenance` — the mandatory, structured reproducibility record.
* the optional data *blocks* (:class:`OverlapBlock`,
  :class:`OrbitalCoefficients`, :class:`DensityMatrix`,
  :class:`TwoElectronIntegrals`, :class:`SpinBlock`,
  :class:`UncertaintyBlock`).
* :class:`ElectronicStructurePayload` — the single object every adapter
  emits; it is *structurally* an
  :class:`~atomx.scattering.protocols.AtomicResultLike` (so the native
  atomx solvers remain the reference provider for free) and carries the
  extra AO/MO surface the molecular/periodic back-ends need.
* :class:`ElectronicStructureProvider` — the ``runtime_checkable``
  Protocol every adapter object satisfies (``capabilities`` /
  ``supports`` / ``build_payload``), mirroring the ``Correction`` pattern
  in :mod:`atomx.corrections.protocol`.

Design invariants enforced here (fail-closed, never silent):

* **I1** basis/units traceable — energies cross this boundary in Hartree
  atomic units only (:func:`ensure_hartree`); provenance names the basis.
* **I4** one energy/intensity convention — declared in provenance.
* **I5** every approximation explicit — degradations stamp
  ``provenance.approximations``.
* **I6** ML never bypasses the gate — an ``is_ml`` payload without an
  :class:`UncertaintyBlock` raises :class:`UncertaintyGateError`.

No physics lives here.  Adapters are the only place a non-atomic unit or
a back-end object may appear.

Reference: ``docs/adr/ADR-0001-electronic-structure-provider.md``.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from enum import Enum
from numbers import Number
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np

from atomx.units import to_hartree

__all__ = [
    "Capability",
    "MUST_CAPABILITIES",
    "ORBITAL_CAPABILITIES",
    "ProviderContractError",
    "CapabilityError",
    "UncertaintyGateError",
    "UnitError",
    "Provenance",
    "OverlapBlock",
    "OrbitalCoefficients",
    "DensityMatrix",
    "TwoElectronIntegrals",
    "SpinBlock",
    "UncertaintyBlock",
    "ElectronicStructurePayload",
    "ElectronicStructureProvider",
    "ensure_hartree",
]


# ---------------------------------------------------------------------------
# Capabilities (ADR-0001 §5)
# ---------------------------------------------------------------------------


class Capability(str, Enum):
    """Negotiable provider capability.

    A kernel asks the provider which of these it exposes *before*
    computing; a missing MUST capability, or a missing capability the
    kernel needs, triggers a documented degradation or a fail-closed
    :class:`CapabilityError` — never a number of unknown validity.
    """

    # --- MUST (a payload is rejected fail-closed without these) ---------
    ORBITAL_ENERGIES = "orbital_energies"
    OCCUPATIONS = "occupations"
    UNITS_DECLARED = "units_declared"
    PROVENANCE = "provenance"

    # --- orbital representation (a payload MUST expose at least one) -----
    RADIAL_ORBITALS = "radial_orbitals"      # native/atomic P,Q by kappa
    MO_COEFFICIENTS = "mo_coefficients"      # LCAO/GTO C in an AO basis

    # --- MAY (presence unlocks a code path; absence → degradation §10) --
    CORE_HOLE_SPEC = "core_hole_spec"
    OVERLAP_S = "overlap_S"
    DENSITY_MATRIX = "density_matrix"
    TWO_ELECTRON_INTEGRALS = "two_electron_integrals"
    HAMILTONIAN_MATRIX = "hamiltonian_matrix"
    KPOINTS = "kpoints"
    SPIN_SOC = "spin_soc"
    TWO_PARTICLE_RDM = "two_particle_rdm"
    ENERGY_NORMALIZED_CONTINUUM = "energy_normalized_continuum"
    UNCERTAINTY = "uncertainty"


#: The capabilities every conforming payload must expose (I1/I4/I5).
MUST_CAPABILITIES: Set[Capability] = {
    Capability.ORBITAL_ENERGIES,
    Capability.OCCUPATIONS,
    Capability.UNITS_DECLARED,
    Capability.PROVENANCE,
}

#: At least one orbital representation is required.
ORBITAL_CAPABILITIES: Set[Capability] = {
    Capability.RADIAL_ORBITALS,
    Capability.MO_COEFFICIENTS,
}


# ---------------------------------------------------------------------------
# Typed, fail-closed errors (ADR-0001 §7)
# ---------------------------------------------------------------------------


class ProviderContractError(Exception):
    """Base class for every contract violation (fail-closed)."""


class CapabilityError(ProviderContractError):
    """A required capability is absent — the pipeline stops and reports."""


class UncertaintyGateError(ProviderContractError):
    """An ML payload reached the boundary without an uncertainty block (I6)."""


class UnitError(ProviderContractError):
    """A quantity could not be coerced to Hartree atomic units (I1)."""


# ---------------------------------------------------------------------------
# Unit coercion (I1) — the only sanctioned energy entry point
# ---------------------------------------------------------------------------

_MAX_ERROR_TYPE_NAME = 80
_FALLBACK_ERROR_TYPE_NAME = "unknown"
_ELECTRON_COUNT_IMAG_ATOL = 1e-10
_ELECTRON_COUNT_IMAG_RTOL = 1e-12
_MAX_NESTED_NUMERIC_DEPTH = 16
_MAX_NESTED_NUMERIC_NODES = 10_000
_MAX_ORBITALS_RAW_KAPPAS = 128
_MAX_ORBITALS_RAW_PER_KAPPA = 10_000
_MAX_ORBITALS_RAW_TOTAL = 100_000


def _safe_type_name(value: Any) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
        if not isinstance(name, str):
            return _FALLBACK_ERROR_TYPE_NAME
    except Exception:
        return _FALLBACK_ERROR_TYPE_NAME
    sanitized = "".join(
        ch if ch.isprintable() and ch not in "\r\n\t" else "?"
        for ch in name
    )
    if not sanitized:
        return _FALLBACK_ERROR_TYPE_NAME
    if len(sanitized) > _MAX_ERROR_TYPE_NAME:
        return sanitized[: _MAX_ERROR_TYPE_NAME - 3] + "..."
    return sanitized


def ensure_hartree(value: Any) -> float:
    """Coerce a single energy to Hartree atomic units.

    Accepts a plain float (already Hartree) or a ``pint`` ``Quantity``
    (converted via :func:`atomx.units.to_hartree`, including spectroscopy
    context for cm⁻¹ / Hz).  Anything else raises :class:`UnitError` so a
    stray unit can never slip across the boundary silently.
    """
    try:
        hartree = to_hartree(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise UnitError(
            "cannot coerce energy value to Hartree atomic units "
            f"(type={_safe_type_name(value)}, reason=unsupported-or-invalid-value)"
        ) from exc
    if np.iscomplexobj(hartree) or not np.isscalar(hartree):
        raise UnitError(
            "cannot coerce energy value to a real Hartree scalar "
            f"(type={_safe_type_name(value)})"
        )
    if not np.isfinite(hartree):
        raise UnitError(
            "cannot coerce energy value to finite Hartree atomic units "
            f"(type={_safe_type_name(value)})"
        )
    return float(hartree)


def _ensure_hartree_mapping(mapping: Dict[str, Any]) -> Dict[str, float]:
    return {k: ensure_hartree(v) for k, v in mapping.items()}


def _bounded_key(key: Any) -> str:
    text = str(key)
    if len(text) > 40:
        return text[:37] + "..."
    return text


def _as_finite_numeric_array(name: str, value: Any) -> np.ndarray:
    """Coerce one public numeric array once and reject ambiguous object dtype.

    Object dtype is fail-closed because it can hide arbitrary Python objects,
    deferred conversions, or values whose numeric status changes after
    construction. This contract intentionally does not impose size caps; very
    large finite arrays remain a residual availability risk for adapters and
    runtime resource limits to control.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(f"{name} must be numeric") from exc
    if array.dtype == object:
        raise ProviderContractError(
            f"{name} must be numeric with non-object dtype"
        )
    dtype = complex if np.issubdtype(array.dtype, np.complexfloating) else float
    try:
        array = array.astype(dtype, copy=False)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ProviderContractError(f"{name} must contain only finite values")
    return array


def _ensure_finite_real_scalar(name: str, value: Any) -> float:
    array = _as_finite_numeric_array(name, value)
    if array.shape != () or np.iscomplexobj(array):
        raise ProviderContractError(f"{name} must be a finite real scalar")
    return float(array)


def _ensure_real_electron_count(name: str, value: Any) -> float:
    array = _as_finite_numeric_array(name, value)
    if array.shape != ():
        raise ProviderContractError(f"{name} must be a finite scalar")
    scalar = array.item()
    if np.iscomplexobj(array):
        real = float(np.real(scalar))
        imag = float(np.imag(scalar))
        tolerance = (
            _ELECTRON_COUNT_IMAG_ATOL
            + _ELECTRON_COUNT_IMAG_RTOL * abs(real)
        )
        if abs(imag) > tolerance:
            raise ProviderContractError(
                f"{name} must be real within tolerance; "
                f"imag={imag}, tolerance={tolerance}"
            )
        return real
    return float(scalar)


def _ensure_finite_nested_numeric_values(name: str, value: Any) -> None:
    """Reject non-finite numeric leaves while allowing non-numeric metadata."""
    seen: set[int] = set()
    nodes = 0

    def visit(label: str, item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NESTED_NUMERIC_NODES:
            raise ProviderContractError(
                f"{name} exceeds maximum nested node budget "
                f"({_MAX_NESTED_NUMERIC_NODES})"
            )
        if depth > _MAX_NESTED_NUMERIC_DEPTH:
            raise ProviderContractError(
                f"{label} exceeds maximum nested depth "
                f"({_MAX_NESTED_NUMERIC_DEPTH})"
            )
        if item is None:
            return
        if isinstance(item, MappingABC):
            item_id = id(item)
            if item_id in seen:
                raise ProviderContractError(f"{label} contains a cycle")
            seen.add(item_id)
            try:
                for key, child in item.items():
                    visit(f"{label}.{_bounded_key(key)}", child, depth + 1)
            finally:
                seen.remove(item_id)
            return
        if isinstance(item, (list, tuple)):
            item_id = id(item)
            if item_id in seen:
                raise ProviderContractError(f"{label} contains a cycle")
            seen.add(item_id)
            try:
                for index, child in enumerate(item):
                    visit(f"{label}[{index}]", child, depth + 1)
            finally:
                seen.remove(item_id)
            return
        if isinstance(item, np.ndarray):
            _as_finite_numeric_array(label, item)
            return
        if isinstance(item, Number) and not isinstance(item, bool):
            array = _as_finite_numeric_array(label, item)
            if array.shape != ():
                raise ProviderContractError(f"{label} must be a finite scalar")

    visit(name, value, 0)


def _as_bounded_list(value: Any, limit: int, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise ProviderContractError(f"{name} must be an iterable container")
    if not isinstance(value, IterableABC):
        raise ProviderContractError(f"{name} must be an iterable container")
    items: list[Any] = []
    for index, item in enumerate(value):
        if index >= limit:
            raise ProviderContractError(
                f"{name} exceeds maximum item budget ({limit})"
            )
        items.append(item)
    return items


def _normalize_orbitals_raw(
    orbitals_raw: MappingABC | None,
) -> Dict[int, List[Tuple[np.ndarray, Optional[np.ndarray], float]]]:
    if orbitals_raw is None:
        return {}
    if not isinstance(orbitals_raw, MappingABC):
        raise ProviderContractError("orbitals_raw must be a mapping")
    if len(orbitals_raw) > _MAX_ORBITALS_RAW_KAPPAS:
        raise ProviderContractError(
            "orbitals_raw exceeds maximum kappa budget "
            f"({_MAX_ORBITALS_RAW_KAPPAS})"
        )
    normalized: Dict[int, List[Tuple[np.ndarray, Optional[np.ndarray], float]]] = {}
    total = 0
    for kappa, orbitals in orbitals_raw.items():
        try:
            kappa_key = int(kappa)
        except (TypeError, ValueError) as exc:
            raise ProviderContractError(
                f"orbitals_raw key {kappa!r} must be an integer kappa"
            ) from exc
        orbital_items = _as_bounded_list(
            orbitals,
            _MAX_ORBITALS_RAW_PER_KAPPA,
            f"orbitals_raw[{kappa_key!r}]",
        )
        total += len(orbital_items)
        if total > _MAX_ORBITALS_RAW_TOTAL:
            raise ProviderContractError(
                "orbitals_raw exceeds maximum total orbital budget "
                f"({_MAX_ORBITALS_RAW_TOTAL})"
            )
        normalized_orbitals = []
        for index, orbital in enumerate(orbital_items):
            label = f"orbitals_raw[{_bounded_key(kappa_key)}][{index}]"
            components = _as_bounded_list(orbital, 4, label)
            if len(components) != 3:
                raise ProviderContractError(
                    f"{label} must contain exactly P, Q, occupation"
                )
            p_component, q_component, occupation = components
            p_array = _as_finite_numeric_array(f"{label}.P", p_component)
            q_array = (
                None
                if q_component is None
                else _as_finite_numeric_array(f"{label}.Q", q_component)
            )
            occ = _ensure_finite_real_scalar(f"{label}.occupation", occupation)
            normalized_orbitals.append((p_array, q_array, occ))
        normalized[kappa_key] = normalized_orbitals
    return normalized


def _ensure_finite_subshell_occupations(subshells: Iterable[Any]) -> None:
    for index, subshell in enumerate(subshells):
        occupation = getattr(subshell, "occupation", None)
        if occupation is not None:
            _as_finite_numeric_array(
                f"subshells[{index}].occupation", occupation
            )


# ---------------------------------------------------------------------------
# Provenance (ADR-0001 §6.3) — mandatory
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Mandatory reproducibility/audit record attached to every payload.

    Satisfies I1 (basis), I2 (core hole / final state), I4 (energy zero),
    I5 (approximations) and gates I6 (``is_ml``).
    """

    provider: str            # "pyscf" | "abacus" | "deeptb" | "atomx-native"
    code_version: str        # e.g. "pyscf 2.14.0"
    method: str              # "RHF" | "CASSCF(8,8)" | "PBE+SOC" | ...
    basis: str               # basis identity → I1
    theory_level: str        # free-form level label
    approximations: List[str] = field(default_factory=list)  # I5
    core_hole: str = ""      # initial hole subshell(s) → I2
    final_state: str = ""    # two-hole final config → I2
    energy_zero: str = "total_energy"  # declared reference → I4
    is_ml: bool = False      # gate trigger → I6
    ood_status: Optional[str] = None   # "in-domain" | "ood" | "unknown"
    uncertainty_summary: Optional[dict] = None

    def validate(self) -> "Provenance":
        """Raise :class:`ProviderContractError` if identity fields missing."""
        # ``theory_level`` is intentionally free-form and not an I1-I6 gate.
        for name in ("provider", "code_version", "method", "basis", "energy_zero"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProviderContractError(
                    f"Provenance.{name} is mandatory and must be a non-empty string"
                )
        if self.is_ml and self.ood_status is None:
            # An ML provider must at least declare its domain status.
            raise ProviderContractError(
                "ML provenance must declare ood_status "
                "('in-domain' | 'ood' | 'unknown')"
            )
        _ensure_finite_nested_numeric_values(
            "Provenance.uncertainty_summary", self.uncertainty_summary
        )
        return self


# ---------------------------------------------------------------------------
# Optional data blocks (ADR-0001 §6.1)
# ---------------------------------------------------------------------------


@dataclass
class OverlapBlock:
    """AO-basis overlap matrix ``S`` for a (possibly) non-orthogonal basis.

    ``None`` payload overlap ⇒ the basis is assumed orthonormal; a kernel
    that needs ``R^k`` must assert ``max|S - I| < tol`` (see
    :meth:`is_orthonormal`) or route through biorthogonalization first.
    """

    matrix: np.ndarray            # (n_ao, n_ao)
    basis_labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "OverlapBlock":
        self.matrix = _as_finite_numeric_array("overlap S", self.matrix)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ProviderContractError("overlap S must be a square 2-D matrix")
        return self

    def max_deviation_from_identity(self) -> float:
        self.validate()
        n = self.matrix.shape[0]
        return float(np.max(np.abs(self.matrix - np.eye(n))))

    def is_orthonormal(self, tol: float = 1e-8) -> bool:
        return self.max_deviation_from_identity() <= tol

    def is_symmetric(self, tol: float = 1e-8) -> bool:
        self.validate()
        return bool(np.allclose(self.matrix, self.matrix.T, atol=tol))


@dataclass
class OrbitalCoefficients:
    """Molecular/LCAO orbital coefficients in an AO basis.

    ``coefficients`` is the ``(n_ao, n_mo)`` matrix ``C`` such that MO ``i``
    is ``sum_mu C[mu, i] * phi_mu``.  ``energies`` and ``occupations`` are
    per-MO; ``energies`` are Hartree.  For a non-orthogonal AO basis the
    MOs satisfy ``C^T S C = I`` (checked in
    :meth:`ElectronicStructurePayload.orthonormality_residual`).
    """

    coefficients: np.ndarray      # (n_ao, n_mo)
    energies: np.ndarray          # (n_mo,) Hartree
    occupations: np.ndarray       # (n_mo,)
    basis_labels: Optional[List[str]] = None
    spin: str = "restricted"      # "restricted" | "alpha" | "beta"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "OrbitalCoefficients":
        self.coefficients = _as_finite_numeric_array(
            "orbital coefficients", self.coefficients
        )
        self.energies = _as_finite_numeric_array(
            "orbital energies", self.energies
        )
        self.occupations = _as_finite_numeric_array(
            "orbital occupations", self.occupations
        )
        if self.coefficients.ndim != 2:
            raise ProviderContractError("orbital coefficients must be 2-D")
        n_mo = self.coefficients.shape[1]
        if self.energies.shape != (n_mo,) or self.occupations.shape != (n_mo,):
            raise ProviderContractError(
                "orbital energies/occupations must have length n_mo "
                f"= {n_mo}"
            )
        return self

    @property
    def n_ao(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def n_mo(self) -> int:
        return int(self.coefficients.shape[1])


@dataclass
class DensityMatrix:
    """One-particle reduced density matrix in the AO basis."""

    matrix: np.ndarray            # (n_ao, n_ao)
    basis: str = "ao"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "DensityMatrix":
        self.matrix = _as_finite_numeric_array("density matrix", self.matrix)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ProviderContractError("density matrix must be square 2-D")
        return self

    def n_electrons(self, overlap: Optional[OverlapBlock] = None) -> float:
        """Trace of ``P`` (orthonormal) or ``P S`` (non-orthogonal AO)."""
        self.validate()
        if overlap is None:
            return _ensure_real_electron_count(
                "DensityMatrix.n_electrons.trace", np.trace(self.matrix)
            )
        overlap.validate()
        if self.matrix.shape != overlap.matrix.shape:
            raise ProviderContractError(
                "DensityMatrix.n_electrons.overlap requires density and "
                "overlap with identical shapes; got density "
                f"{self.matrix.shape} and overlap {overlap.matrix.shape}"
            )
        return _ensure_real_electron_count(
            "DensityMatrix.n_electrons.overlap",
            np.einsum("ij,ji->", self.matrix, overlap.matrix),
        )


@dataclass
class TwoElectronIntegrals:
    """Two-electron repulsion integrals (chemist notation ``(ij|kl)``)."""

    eri: np.ndarray
    basis: str = "ao"             # "ao" | "mo"
    notation: str = "chemist"     # "chemist" (ij|kl) | "physicist" <ij|kl>

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "TwoElectronIntegrals":
        self.eri = _as_finite_numeric_array("ERI", self.eri)
        if self.eri.ndim != 4:
            raise ProviderContractError(
                f"ERI must be rank-4 (ij|kl), got ndim={self.eri.ndim}"
            )
        if len(set(self.eri.shape)) != 1:
            raise ProviderContractError(
                "ERI must have equal dimensions in all four axes, "
                f"got shape={self.eri.shape}"
            )
        return self


@dataclass
class SpinBlock:
    """Spin / SOC treatment flag."""

    collinear: bool = True
    soc: bool = False
    coupling: str = "LS"          # "LS" | "jj"


@dataclass
class UncertaintyBlock:
    """Per-quantity uncertainty; required when ``provenance.is_ml``.

    ``per_orbital_sigma`` and ``ood_score`` are typed numeric fields and must
    be finite real scalars. Numeric leaves in ``summary`` must also be finite;
    descriptive strings, booleans and ``None`` are accepted.
    """

    per_orbital_sigma: Dict[str, float] = field(default_factory=dict)
    ood_score: Optional[float] = None
    summary: Optional[dict] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "UncertaintyBlock":
        for label, sigma in self.per_orbital_sigma.items():
            self.per_orbital_sigma[label] = _ensure_finite_real_scalar(
                f"UncertaintyBlock.per_orbital_sigma[{_bounded_key(label)}]",
                sigma,
            )
        if self.ood_score is not None:
            self.ood_score = _ensure_finite_real_scalar(
                "UncertaintyBlock.ood_score", self.ood_score
            )
        _ensure_finite_nested_numeric_values(
            "UncertaintyBlock.summary", self.summary
        )
        return self


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


@dataclass
class ElectronicStructurePayload:
    """Single object every adapter emits (ADR-0001 §6.1).

    Structurally an :class:`~atomx.scattering.protocols.AtomicResultLike`
    (``Z``, ``subshells``, ``orbital_energies``, ``orbitals_raw`` are always
    present) so native atomx solvers are the reference provider for free.
    Molecular/periodic back-ends leave ``orbitals_raw`` empty and supply the
    :attr:`orbitals` (AO/MO), :attr:`overlap`, :attr:`density` and
    :attr:`two_electron` blocks instead.

    Use :meth:`validate` (fail-closed) before handing the payload to a
    kernel, and :meth:`require` to assert the capabilities a kernel needs.
    """

    # --- AtomicResultLike core (always present; frozen cut-set C-α/C-β) --
    Z: int
    provenance: Provenance
    subshells: Iterable[Any] = field(default_factory=list)
    orbital_energies: Dict[str, float] = field(default_factory=dict)
    orbitals_raw: Dict[int, List[Tuple[Any, Any, float]]] = field(
        default_factory=dict
    )

    # --- optional blocks (each: value | None, "None ⇒ …" documented) -----
    overlap: Optional[OverlapBlock] = None
    orbitals: Optional[OrbitalCoefficients] = None
    density: Optional[DensityMatrix] = None
    two_electron: Optional[TwoElectronIntegrals] = None
    spin: Optional[SpinBlock] = None
    uncertainty: Optional[UncertaintyBlock] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # I1: coerce declared orbital energies to Hartree at construction.
        self.subshells = tuple(self.subshells)
        self.orbital_energies = _ensure_hartree_mapping(self.orbital_energies)
        self.orbitals_raw = _normalize_orbitals_raw(self.orbitals_raw)

    # -- capability negotiation --------------------------------------------

    def capabilities(self) -> Set[Capability]:
        """Return the set of capabilities this payload actually exposes."""
        caps: Set[Capability] = {
            Capability.UNITS_DECLARED,
            Capability.PROVENANCE,
        }
        if self.orbital_energies:
            caps.add(Capability.ORBITAL_ENERGIES)
        if self._has_occupations():
            caps.add(Capability.OCCUPATIONS)
        if any(self.orbitals_raw.values()):
            caps.add(Capability.RADIAL_ORBITALS)
        if self.orbitals is not None:
            caps.add(Capability.MO_COEFFICIENTS)
        if self.overlap is not None:
            caps.add(Capability.OVERLAP_S)
        if self.density is not None:
            caps.add(Capability.DENSITY_MATRIX)
        if self.two_electron is not None:
            caps.add(Capability.TWO_ELECTRON_INTEGRALS)
        if self.spin is not None and self.spin.soc:
            caps.add(Capability.SPIN_SOC)
        if self.uncertainty is not None:
            caps.add(Capability.UNCERTAINTY)
        if self.provenance.core_hole:
            caps.add(Capability.CORE_HOLE_SPEC)
        return caps

    def supports(self, *requested: Capability) -> bool:
        """True iff every requested capability is present."""
        return set(requested).issubset(self.capabilities())

    def require(self, *requested: Capability) -> "ElectronicStructurePayload":
        """Fail-closed: raise :class:`CapabilityError` on any missing cap."""
        missing = set(requested) - self.capabilities()
        if missing:
            names = ", ".join(sorted(c.value for c in missing))
            raise CapabilityError(
                f"provider '{self.provenance.provider}' is missing required "
                f"capabilities: {names}"
            )
        return self

    # -- validation (fail-closed) ------------------------------------------

    def validate(self) -> "ElectronicStructurePayload":
        """Enforce the MUST contract + ML gate; return ``self`` if valid.

        Raises
        ------
        ProviderContractError
            A MUST capability is absent, no orbital representation is
            present, or provenance is incomplete.
        UncertaintyGateError
            The provenance is flagged ML but no uncertainty block is
            attached (I6 — ML fails closed).
        """
        self.subshells = tuple(self.subshells)
        self.provenance.validate()
        self.orbital_energies = _ensure_hartree_mapping(self.orbital_energies)
        self.orbitals_raw = _normalize_orbitals_raw(self.orbitals_raw)
        if self.overlap is not None:
            self.overlap.validate()
        if self.orbitals is not None:
            self.orbitals.validate()
        if self.density is not None:
            self.density.validate()
        if self.two_electron is not None:
            self.two_electron.validate()
        if self.uncertainty is not None:
            self.uncertainty.validate()
        caps = self.capabilities()
        missing = MUST_CAPABILITIES - caps
        if missing:
            names = ", ".join(sorted(c.value for c in missing))
            raise ProviderContractError(
                f"payload missing MUST capabilities: {names}"
            )
        if not (caps & ORBITAL_CAPABILITIES):
            raise ProviderContractError(
                "payload exposes no orbital representation "
                "(need radial_orbitals or mo_coefficients)"
            )
        if self.provenance.is_ml and self.uncertainty is None:
            raise UncertaintyGateError(
                "ML provider payload lacks an uncertainty block; "
                "fails closed (I6)"
            )
        _ensure_finite_subshell_occupations(self.subshells)
        if self.orbitals is not None:
            n_ao = self.orbitals.n_ao
            if self.overlap is not None and self.overlap.matrix.shape != (n_ao, n_ao):
                raise ProviderContractError(
                    "overlap S shape must match orbitals.n_ao: "
                    f"expected {(n_ao, n_ao)}, got {self.overlap.matrix.shape}"
                )
            if self.density is not None and self.density.matrix.shape != (n_ao, n_ao):
                raise ProviderContractError(
                    "density matrix shape must match orbitals.n_ao: "
                    f"expected {(n_ao, n_ao)}, got {self.density.matrix.shape}"
                )
        if self.two_electron is not None:
            basis = self.two_electron.basis
            if basis not in ("ao", "mo"):
                raise ProviderContractError(
                    "two-electron integral basis must be 'ao' or 'mo', "
                    f"got {basis!r}"
                )
            if self.orbitals is not None:
                expected = n_ao if basis == "ao" else self.orbitals.n_mo
                actual = self.two_electron.eri.shape[0]
                if actual != expected:
                    raise ProviderContractError(
                        f"{basis}-basis ERI leading dimension must match "
                        f"orbitals.n_{basis}: expected {expected}, got {actual}"
                    )
        return self

    # -- physics-adjacent helpers (pure linear algebra, no back-end) -------

    def orthonormality_residual(self) -> Optional[float]:
        """``max|C^T S C - I|`` for the MO block, or ``None`` if N/A.

        This is the orthogonal/non-orthogonal *consistency* check: an
        LCAO/GTO basis has ``S != I`` yet its occupied+virtual MOs must be
        orthonormal in the ``S`` metric.  A native (orthonormal) payload
        with ``overlap=None`` treats ``S = I``.
        """
        self.validate()
        if self.orbitals is None:
            return None
        C = self.orbitals.coefficients
        S = (
            self.overlap.matrix
            if self.overlap is not None
            else np.eye(C.shape[0])
        )
        gram = C.conj().T @ S @ C
        return float(np.max(np.abs(gram - np.eye(gram.shape[1]))))

    def electron_count(self) -> Optional[float]:
        """Electron number from the density block (``Tr(P S)``)."""
        self.validate()
        if self.density is None:
            return None
        return self.density.n_electrons(self.overlap)

    def _has_occupations(self) -> bool:
        if self.orbitals is not None:
            return True
        for sh in self.subshells:
            if getattr(sh, "occupation", None) is not None:
                return True
        return False


# ---------------------------------------------------------------------------
# The provider Protocol (mirror of atomx.corrections.protocol.Correction)
# ---------------------------------------------------------------------------


@runtime_checkable
class ElectronicStructureProvider(Protocol):
    """Common interface for every back-end adapter.

    Concrete adapters (``NativeProvider``, ``PySCFProvider`` …) satisfy
    this structurally — no inheritance required.  The physics kernels
    depend on this Protocol, never on a back-end module.
    """

    def capabilities(self) -> Set[Capability]:
        """Capabilities this provider can put into a payload."""
        ...

    def supports(self, *requested: Capability) -> bool:
        """Whether every requested capability is available."""
        ...

    def build_payload(self) -> ElectronicStructurePayload:
        """Translate the back-end object into a validated payload."""
        ...
