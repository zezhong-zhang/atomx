"""Finite native electron attachment to a nonlinear correlated core.

The projected EOM operator is exp(-T) H(N+1) exp(T) - E_core.  Core T1/T2/T3
and attachment 1p/2p1h/3p2h ranks are distinct controls.  Both particle sectors
consume one immutable SpinorIntegrals operator and the shared finite cluster
machinery.  This small complete-Fock-space anchor is not a production atomic
FS-RCCSDT implementation or a reproduction of the 69-electron Yb paper.
"""

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import comb
from types import MappingProxyType

import numpy as np
from scipy.linalg import eig

from atomx.methods.coupled_cluster import (
    CoupledClusterResult,
    _cluster_generators,
    _exponential_action,
    _state_exponential_action,
)
from atomx.methods.coupled_cluster import (
    compute as core_compute,
)
from atomx.numerics.fermion import apply_terms
from atomx.physics.csf.spinor_integrals import SpinorIntegrals

__all__ = ["AttachmentResult", "compute"]


def _owned(array):
    value = np.asarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _freeze_identity(value):
    """Own plain producer metadata independently of mutable nested dictionaries."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_identity(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_identity(item) for item in value)
    if isinstance(value, np.ndarray):
        return _owned(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
        return value
    raise TypeError("finite attachment producer identity requires plain owned metadata")


@dataclass(frozen=True)
class AttachmentResult:
    """Complete projected attachment spectrum with separate physical residuals."""

    method: str
    core: CoupledClusterResult
    attachment_energies_Ha: np.ndarray
    total_energies_Ha: np.ndarray
    coefficients: np.ndarray
    projected_right_vectors: np.ndarray
    projected_operator_Ha: np.ndarray
    projected_residuals_Ha: np.ndarray
    full_eigenstate_residuals_Ha: np.ndarray
    determinant_masks: tuple
    projected_masks: tuple
    provenance: object

    def __post_init__(self):
        for name in (
            "attachment_energies_Ha", "total_energies_Ha", "coefficients",
            "projected_right_vectors", "projected_operator_Ha",
            "projected_residuals_Ha", "full_eigenstate_residuals_Ha",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "provenance", _freeze_identity(self.provenance))


def _masks(spinors, electrons):
    return tuple(sum(1 << i for i in row)
                 for row in itertools.combinations(range(spinors), electrons))


def _hamiltonian_terms(integrals):
    """Use one copy of each antisymmetric pair, preserving every nonzero term."""
    n = len(integrals.labels)
    terms = [(float(integrals.frozen_energy_Ha), ())]
    for p in range(n):
        for q in range(n):
            value = float(integrals.one_body_Ha[p, q].real)
            if value != 0:
                terms.append((value, ((p, True), (q, False))))
    pairs = tuple(itertools.combinations(range(n), 2))
    for p, q in pairs:
        for r, s in pairs:
            value = float(integrals.eri(p, q, r, s).real)
            if value != 0:
                terms.append((value, ((p, True), (q, True), (s, False), (r, False))))
    return tuple(terms)


def _matrix(terms, masks):
    lookup = {mask: i for i, mask in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)))
    for col, mask in enumerate(masks):
        for target, value in apply_terms(terms, {mask: 1.0}).items():
            if target not in lookup:
                raise RuntimeError("number-conserving operator left the complete sector")
            matrix[lookup[target], col] = value
    # Preserve the actual producer; finite real CC requires exact symmetry.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("common finite Hamiltonian is not exactly real symmetric")
    return matrix


def compute(integrals, *, core_method="CCSDT", attachment_particle_rank=3,
            determinant_cap=256, amplitude_cap=128, generator_memory_MiB=128,
            logical_term_action_cap=2_000_000, max_evaluations=2000,
            residual_tolerance_Ha=1e-10, imaginary_energy_tolerance_Ha=1e-10,
            continuation=(.25, .5, .75, 1.), action_representation="dense",
            cluster_term_application_cap=2_000_000):
    """Solve a resource-capped finite EOM-EA-CCSD/CCSDT common-operator anchor.

    Complete fixed-N and fixed-(N+1) magnetic spaces are retained.  The attached
    projected basis contains every kp/(k-1)h state through the explicit rank;
    there is no amplitude screening or experimental root selection.  This
    matrix is generally non-Hermitian and is never symmetrized.  The projected
    right-eigenstate residual is a gate; the physical full-H residual is a
    diagnostic because omitted attachment/core ranks can leave it nonzero.
    Only exactly real packed integrals are supported by this finite core engine.
    The memory estimate covers NumPy generator/workspace payloads; input
    integrals and Python/SciPy overhead are extra. Logical term-action limits
    cover construction of the two Hamiltonians, not nonlinear iterations,
    similarity transforms or diagonalization.
    Explicit state_action removes both dense generator banks and the attached
    T matrix while retaining every finite-space coordinate. Its separate
    term cap bounds each scalar action, not the full calculation.
    """
    if not isinstance(integrals, SpinorIntegrals):
        raise TypeError("owned packed SpinorIntegrals required for finite attachment")
    if core_method not in ("CCSD", "CCSDT"):
        raise NotImplementedError("finite core method must be CCSD or CCSDT")
    if action_representation not in ("dense", "state_action"):
        raise ValueError("finite action representation must be dense or state_action")
    for name, value in (
        ("attachment_particle_rank", attachment_particle_rank),
        ("determinant_cap", determinant_cap), ("amplitude_cap", amplitude_cap),
        ("logical_term_action_cap", logical_term_action_cap),
        ("max_evaluations", max_evaluations),
        ("cluster_term_application_cap", cluster_term_application_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"positive integer {name} required")
    if attachment_particle_rank not in (1, 2, 3):
        raise ValueError("attachment particle rank must be 1, 2 or 3")
    for name, value in (
        ("generator_memory_MiB", generator_memory_MiB),
        ("residual_tolerance_Ha", residual_tolerance_Ha),
        ("imaginary_energy_tolerance_Ha", imaginary_energy_tolerance_Ha),
    ):
        if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isrealobj(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"positive finite {name} required")
    n, occupied = len(integrals.labels), integrals.noccupied
    if n > 64:
        raise ValueError("finite proof engine permits at most 64 spinors")
    virtual = n - occupied
    nc, na = comb(n, occupied), comb(n, occupied + 1)
    rank = 2 if core_method == "CCSD" else 3
    amplitudes = sum(comb(occupied, k) * comb(virtual, k)
                     for k in range(1, min(rank, occupied, virtual) + 1))
    projected_count = sum(comb(occupied, k - 1) * comb(virtual, k)
                          for k in range(1, min(attachment_particle_rank, virtual) + 1)
                          if k - 1 <= occupied)
    # Covers both generator banks and resident/transient matrices conservatively.
    dense_generator_bytes = 8 * amplitudes * (nc * nc + na * na)
    generator_bytes = dense_generator_bytes if action_representation == "dense" else 0
    bytes_upper = generator_bytes + 8 * (16 * (nc * nc + na * na)
                       + 8 * projected_count * projected_count)
    term_upper = 1 + n * n + comb(n, 2)**2
    work_upper = term_upper * (nc + na)
    if max(nc, na) > determinant_cap:
        raise ValueError(f"complete particle sectors need {nc}/{na} determinants; cap exceeded")
    if amplitudes > amplitude_cap or bytes_upper > generator_memory_MiB * 1024**2:
        raise ValueError("complete core/attachment generator memory or amplitude cap exceeded")
    if work_upper > logical_term_action_cap:
        raise ValueError("complete common-operator term action cap exceeded")
    action = None
    if action_representation == "state_action":
        from atomx.numerics.cluster_actions import ClusterAction

        if min(occupied, virtual)*max(nc, na)*amplitudes > cluster_term_application_cap:
            raise ValueError("complete core/attachment exponential term application cap exceeded")
        action = ClusterAction(reference_mask=(1 << occupied)-1, spinor_count=n,
                               rank_limit=rank, sector_electrons=occupied+1,
                               determinant_cap=determinant_cap, amplitude_cap=amplitude_cap,
                               term_application_cap=cluster_term_application_cap)
    if (np.any(integrals.one_body_Ha.imag != 0)
            or np.any(integrals.packed_eri_Ha.imag != 0)):
        raise NotImplementedError("finite attachment currently requires exactly real integrals")
    producer_identity = _freeze_identity(integrals.identity)

    core_masks, attached_masks = _masks(n, occupied), _masks(n, occupied + 1)
    reference_mask = (1 << occupied) - 1
    terms = _hamiltonian_terms(integrals)
    hc, ha = _matrix(terms, core_masks), _matrix(terms, attached_masks)
    if abs(hc[0, 0] - integrals.reference_energy_Ha) > residual_tolerance_Ha:
        raise ValueError("core reference energy disagrees with common owned operator")
    digest = hashlib.sha256()
    for value in (integrals.one_body_Ha, integrals.packed_eri_Ha):
        digest.update(value.tobytes())
    digest.update(json.dumps((integrals.labels, occupied,
                              integrals.frozen_energy_Ha)).encode())
    identity = digest.hexdigest()
    core = core_compute(
        hc, core_masks, reference_mask, spinor_count=n,
        reference_identity="common-spinor-operator:" + identity, method=core_method,
        determinant_cap=determinant_cap, amplitude_cap=amplitude_cap,
        generator_memory_MiB=generator_memory_MiB, max_evaluations=max_evaluations,
        residual_tolerance_Ha=residual_tolerance_Ha, continuation=continuation,
        action_representation=action_representation,
        cluster_term_application_cap=cluster_term_application_cap,
    )
    cluster = None
    if action is None:
        generators = _cluster_generators(
            attached_masks, reference_mask, core.provenance["excitations"]
        )
        cluster = np.einsum("a,aij->ij", core.amplitudes, generators)
    elif action.excitations != core.provenance["excitations"]:
        raise RuntimeError("attachment/core state-action excitation census disagrees")
    max_order = min(occupied, virtual)

    def exponential(vector, sign):
        if action is None:
            return _exponential_action(cluster, vector, max_order, sign)
        return _state_exponential_action(action, attached_masks, core.amplitudes, vector, sign)

    holes = range(occupied)
    particles = range(occupied, n)
    projected_masks = []
    for k in range(1, min(attachment_particle_rank, virtual) + 1):
        for removed in itertools.combinations(holes, k - 1):
            for added in itertools.combinations(particles, k):
                projected_masks.append((reference_mask ^ sum(1 << h for h in removed))
                                       | sum(1 << a for a in added))
    if len(projected_masks) != projected_count or len(set(projected_masks)) != projected_count:
        raise RuntimeError("complete distinct attachment projection census failed")
    lookup = {mask: i for i, mask in enumerate(attached_masks)}
    rows = np.array([lookup[mask] for mask in projected_masks])
    transformed = np.empty((projected_count, projected_count))
    for col, row in enumerate(rows):
        vector = np.zeros(na)
        vector[row] = 1
        psi = exponential(vector, 1)
        acted = exponential(ha @ psi, -1)
        transformed[:, col] = acted[rows]
    transformed -= core.total_energy_Ha * np.eye(projected_count)
    energies, right = eig(transformed, check_finite=True)
    imaginary_max = float(np.max(abs(energies.imag)))
    if imaginary_max > imaginary_energy_tolerance_Ha:
        raise RuntimeError("projected finite attachment has non-real energy roots")
    order = np.argsort(energies.real, kind="stable")
    energies, right = energies[order].real, right[:, order]
    projected_residuals, full_residuals, states = [], [], []
    for energy, column in zip(energies, right.T, strict=True):
        projected_residuals.append(float(np.max(abs(transformed @ column - energy * column))))
        lifted = np.zeros(na, dtype=column.dtype)
        lifted[rows] = column
        psi = exponential(lifted, 1)
        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("invalid physical attachment wavefunction norm")
        psi /= norm
        states.append(psi)
        full_residuals.append(float(np.max(abs(ha @ psi - (core.total_energy_Ha + energy) * psi))))
    if not np.all(np.isfinite(energies)) or max(projected_residuals) > residual_tolerance_Ha:
        raise RuntimeError("finite attachment projected residual gate failed")
    provenance = dict(
        common_operator_sha256=identity, labels=tuple(integrals.labels),
        integral_producer_identity=producer_identity,
        correlated_core_electrons=occupied, attached_electrons=occupied + 1,
        core_connected_rank=rank, attachment_particle_rank=attachment_particle_rank,
        attachment_hole_rank=attachment_particle_rank - 1,
        core_determinants=nc, attached_determinants=na,
        projected_determinants=projected_count, core_amplitudes=amplitudes,
        generator_and_working_memory_upper_bytes=bytes_upper,
        generator_bytes=generator_bytes,
        dense_generator_bytes_if_materialized=dense_generator_bytes,
        action_representation=action_representation,
        cluster_term_application_cap=cluster_term_application_cap,
        cluster_term_application_scope="per scalar state action/exponential, not whole core/attachment solve",
        memory_scope="NumPy generators/workspace payload; input integrals and Python/SciPy overhead extra",
        logical_term_action_upper=work_upper,
        logical_term_action_scope="construction of both H matrices only; excludes CC/Jacobian/similarity/eig",
        logical_nonzero_term_action_count=len(terms) * (nc + na),
        core_hamiltonian_sha256=hashlib.sha256(hc.tobytes()).hexdigest(),
        attached_hamiltonian_sha256=hashlib.sha256(ha.tobytes()).hexdigest(),
        reference_mask=reference_mask, frozen_energy_Ha=integrals.frozen_energy_Ha,
        operator="same owned one-body plus antisymmetric two-body and frozen constant",
        imaginary_energy_tolerance_Ha=imaginary_energy_tolerance_Ha,
        largest_discarded_imaginary_energy_Ha=imaginary_max,
        energy_reference="attachment omega relative to the same correlated N-electron core",
        limitations="finite EOM-EA anchor; no 69e atomic FSCCSDT, root spectroscopy or dressed observables",
    )
    return AttachmentResult(
        f"finite-EOM-EA[core={core_method},R={attachment_particle_rank}p{attachment_particle_rank-1}h]",
        core, energies,
        energies + core.total_energy_Ha, np.column_stack(states), right, transformed,
        np.asarray(projected_residuals), np.asarray(full_residuals),
        attached_masks, tuple(projected_masks), provenance,
    )
