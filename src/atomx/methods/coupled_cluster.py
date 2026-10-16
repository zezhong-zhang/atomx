"""Finite determinant nonlinear CCSD/CCSDT with exact cluster exponentials.

Solves <SD[T]|exp(-T) H exp(T)|reference>=0 for T=T1+T2[+T3]. The exponential
retains disconnected higher excitations; this is not CISD. The complete
fixed-N determinant representation is deliberately capped and supplies small
physical/algorithm anchors. It is not the angular-compressed/integral-direct
atomic all-order method, FSCC or a large-Yb production solver. Private shared
generators/exponential actions also support explicit finite-sector consumers;
closed-core T3 is distinct from an attachment-sector 3p2h operator.

Reference: Bartlett and Musial, Rev. Mod. Phys. 79, 291 (2007),
https://doi.org/10.1103/RevModPhys.79.291 .
"""
import hashlib
import itertools
from dataclasses import dataclass
from math import comb
from types import MappingProxyType

import numpy as np
from scipy.optimize import root

from atomx.angular.csf_hamiltonian import _ops_sign

__all__ = ["CoupledClusterResult", "compute"]


def _owned(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


@dataclass(frozen=True)
class CoupledClusterResult:
    method: str
    correlation_energy: float
    total_energy_Ha: float
    reference_energy_Ha: float
    amplitudes: np.ndarray
    coefficients: np.ndarray
    amplitude_residual_Ha: float
    full_eigenstate_residual_Ha: float
    n_function_evaluations: int
    provenance: object
    triples_correction: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'amplitudes', _owned(self.amplitudes))
        object.__setattr__(self, 'coefficients', _owned(self.coefficients))
        object.__setattr__(self, 'provenance', MappingProxyType(dict(self.provenance)))


def _cluster_generators(masks, reference_mask, excitations):
    """Reference-normalized excitation operators in an aligned Fock sector.

    ``excitations`` contains ascending occupied/virtual index tuples. The
    reference may belong to a different electron sector, as for a core
    cluster acting on N+1 determinants. Callers preflight the complete sector,
    excitation list and dense generator memory before this allocation.
    Every nonzero operator action must remain in the supplied sector.
    """
    lookup = {int(mask): i for i, mask in enumerate(masks)}
    generators = np.zeros((len(excitations), len(masks), len(masks)))
    for t, (holes, particles) in enumerate(excitations):
        ops = tuple((a, True) for a in particles)+tuple((i, False) for i in reversed(holes))
        ref_phase = _ops_sign(int(reference_mask), ops)
        if not ref_phase:
            raise ValueError('cluster excitation must act nontrivially on the reference')
        remove = sum(1 << i for i in holes)
        add = sum(1 << a for a in particles)
        for col, mask in enumerate(masks):
            phase = _ops_sign(int(mask), ops)
            if phase:
                target = (int(mask) ^ remove) | add
                if target not in lookup:
                    raise ValueError('complete cluster-action determinant sector required')
                generators[t, lookup[target], col] = phase/ref_phase
    return generators


def _exponential_action(t_matrix, vector, max_order, sign):
    """Exact nilpotent excitation exponential, with caller-bound order/dtype."""
    out = vector.copy()
    term = vector.copy()
    for order in range(1, max_order+1):
        term = (sign/order)*(t_matrix@term)
        out += term
    return out


def _state_dictionary(masks, vector):
    """Canonical scalar state view; omit only exact zeros, preserving phases."""
    return {int(mask): value for mask, value in zip(masks, vector, strict=True) if value != 0}


def _state_array(masks, state, *, real):
    value = np.asarray([state.get(int(mask), 0.) for mask in masks])
    if real:
        if np.any(value.imag != 0):
            raise RuntimeError('real finite cluster action produced a complex coordinate')
        value = value.real
    return value


def _state_exponential_action(action, masks, amplitudes, vector, sign):
    """Array adapter around complete signed scalar actions, without a T matrix."""
    if vector.ndim == 2:
        return np.column_stack([
            _state_exponential_action(action, masks, amplitudes, column, sign)
            for column in vector.T])
    state = action.exponential(amplitudes, _state_dictionary(masks, vector), sign)
    return _state_array(masks, state, real=np.isrealobj(vector))


def _projected_jacobian(t_matrix, matrix, generators, psi, transformed,
                        projected, max_order):
    """Exact derivative for mutually commuting occupied-to-virtual operators.

    d(exp(-T) H exp(T)|ref>)/dt_a = exp(-T) H tau_a psi - tau_a Hbar|ref>.
    This preserves all coordinates, including amplitudes near an exact zero;
    no small finite-difference step or amplitude screening is used.
    """
    directions = (generators @ psi).T
    left = _exponential_action(t_matrix, matrix @ directions, max_order, -1.)
    return (left-(generators @ transformed).T)[projected]


def compute(hamiltonian_Ha, determinants, reference_mask, *, spinor_count,
            reference_identity, method='CCSD', residual_tolerance_Ha=1e-10,
            determinant_cap=256, amplitude_cap=128, generator_memory_MiB=128,
            max_evaluations=2000, continuation=(.25, .5, .75, 1.),
            action_representation='dense', cluster_term_application_cap=2_000_000):
    """Solve finite CCSD or iterative CCSDT on a complete fixed-N Fock space.

    ``determinants`` are occupation masks in canonical ascending spinor
    order, aligned with the real symmetric Hamiltonian. Occupied/virtual
    spinors are defined by the explicit reference mask. No coordinate is
    removed by amplitude magnitude or a denominator floor. Matrix/state/
    mask hashes and normalized wavefunction are returned. The projected
    SD or SDT residual gates convergence; the full eigenstate residual is
    reported separately because higher connected ranks may be omitted.
    CCSDT solves the connected T3 coordinates with T1/T2; it is not CCSD(T).
    ``generator_memory_MiB`` bounds the dense generators plus a conservative
    NumPy similarity/Jacobian workspace estimate; Python/SciPy overhead is
    not a certified process-memory bound. The exact excitation-commutator
    Jacobian avoids relative finite-difference steps on near-zero amplitudes.
    Explicit ``action_representation='state_action'`` uses complete signed
    scalar Fock actions without the dense excitation-generator bank. Both
    paths retain the same complete determinant space and equations. Its term
    cap applies separately to each scalar action/exponential, not the entire
    nonlinear solve; Python dictionary overhead is outside the array bound.

    The branch follows the declared off-diagonal-coupling continuation from
    the reference. CC energies are not variational and a root is not matched
    to experiment or automatically asserted to be the exact ground state.
    Raises on resource, resonance, nonlinear convergence or residual failure.
    Ranks above three, perturbative triples and FSCC remain unsupported.
    """
    if method not in ('CCSD', 'CCSDT'):
        raise NotImplementedError('finite engine supports CCSD/CCSDT; other ranks and Fock-space CC remain unsupported')
    if action_representation not in ('dense', 'state_action'):
        raise ValueError('finite action representation must be dense or state_action')
    max_rank = 3 if method == 'CCSDT' else 2
    rank_label = 'SDT' if max_rank == 3 else 'SD'
    for name, value in (('spinor_count', spinor_count), ('determinant_cap', determinant_cap),
                        ('amplitude_cap', amplitude_cap), ('max_evaluations', max_evaluations),
                        ('cluster_term_application_cap', cluster_term_application_cap)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'positive integer {name} required')
    if spinor_count > 64:
        raise ValueError('finite proof engine permits at most 64 spinors')
    if (isinstance(reference_mask, bool) or not isinstance(reference_mask, (int, np.integer))
            or not 0 < reference_mask < 1 << int(spinor_count)):
        raise ValueError('valid occupied reference mask required')
    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError('physical reference identity required')
    for name, value in (('residual tolerance', residual_tolerance_Ha), ('generator memory', generator_memory_MiB)):
        if np.ndim(value) != 0 or not np.isrealobj(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'positive finite {name} required')
    electrons = int(reference_mask).bit_count()
    count = comb(int(spinor_count), electrons)
    if count > determinant_cap:
        raise ValueError(f'complete fixed-N basis needs {count} determinants; cap={determinant_cap}')
    masks = tuple(determinants)
    if (len(masks) != count or len(set(masks)) != count
            or any(isinstance(m, bool) or not isinstance(m, (int, np.integer))
                   or not 0 < m < 1 << int(spinor_count) or int(m).bit_count() != electrons for m in masks)
            or reference_mask not in masks):
        raise ValueError('complete distinct fixed-N determinant basis required')
    h = np.asarray(hamiltonian_Ha)
    if (h.shape != (count, count) or not np.isrealobj(h) or not np.all(np.isfinite(h))
            or not np.array_equal(h, h.T)):
        raise ValueError('finite real symmetric Hamiltonian in determinant order required')
    stages = tuple(continuation)
    if (not stages or stages[-1] != 1. or any(np.ndim(v) != 0 or not np.isrealobj(v)
                                            or not np.isfinite(v) or not 0 < v <= 1 for v in stages)
            or any(a >= b for a, b in zip(stages, stages[1:], strict=False))):
        raise ValueError('strictly increasing continuation ending at one required')
    occupied = tuple(i for i in range(spinor_count) if reference_mask & (1 << i))
    virtual = tuple(i for i in range(spinor_count) if not reference_mask & (1 << i))
    rank_counts = tuple((r, comb(len(occupied), r)*comb(len(virtual), r))
                        for r in range(1, max_rank+1)
                        if r <= len(occupied) and r <= len(virtual))
    number = sum(value for _, value in rank_counts)
    dense_generator_bytes = number*count*count*8
    generator_bytes = dense_generator_bytes if action_representation == 'dense' else 0
    # Covers the dense T/similarity matrices, batched derivative actions,
    # output Jacobian and temporary columns. No allocation or rank reduction
    # occurs before this full-space admission. SciPy/Python overhead is extra.
    jacobian_workspace_bytes = 8*(2*count*count+10*count*number+4*number*number+4*count) if number else 0
    if number > amplitude_cap or generator_bytes+jacobian_workspace_bytes > generator_memory_MiB*1024**2:
        raise ValueError(f'complete {rank_label} needs {number} amplitudes, {generator_bytes} generator bytes '
                         f'and {jacobian_workspace_bytes} Jacobian workspace bytes; cap exceeded')
    action = None
    if action_representation == 'state_action':
        from atomx.numerics.cluster_actions import ClusterAction

        if min(electrons, len(virtual))*count*number > cluster_term_application_cap:
            raise ValueError('complete cluster exponential term application cap exceeded')
        action = ClusterAction(reference_mask, spinor_count, max_rank, electrons,
                               determinant_cap=determinant_cap, amplitude_cap=amplitude_cap,
                               term_application_cap=cluster_term_application_cap)
    h = _owned(h)
    lookup = {int(mask): i for i, mask in enumerate(masks)}
    ref = lookup[int(reference_mask)]
    excitations = []
    columns = []
    for rank, _ in rank_counts:
        for holes in itertools.combinations(occupied, rank):
            for particles in itertools.combinations(virtual, rank):
                remove = sum(1 << i for i in holes)
                add = sum(1 << a for a in particles)
                target = (int(reference_mask) ^ remove) | add
                excitations.append((holes, particles))
                columns.append(lookup[target])
    generators = None
    if action is None:
        generators = _cluster_generators(masks, reference_mask, excitations)
    elif action.excitations != tuple(excitations):
        raise RuntimeError('complete state-action excitation census disagrees with finite CC')
    projected = np.asarray(columns, dtype=int)
    reference = np.zeros(count)
    reference[ref] = 1.
    amplitudes = np.zeros(number)
    # Positive/negative exponentials both terminate after N powers because
    # every generator removes reference-occupied and adds virtual spinors.
    max_order = min(electrons, len(virtual))

    def evaluate(amplitudes, matrix):
        if action is None:
            t_matrix = np.einsum('a,aij->ij', amplitudes, generators)
            psi = _exponential_action(t_matrix, reference, max_order, 1.)
            transformed = _exponential_action(t_matrix, matrix@psi, max_order, -1.)
        else:
            psi = _state_exponential_action(action, masks, amplitudes, reference, 1.)
            transformed = _state_exponential_action(action, masks, amplitudes, matrix@psi, -1.)
        return transformed[projected], float(transformed[ref]), psi

    def jacobian(amplitudes, matrix):
        if action is not None:
            psi = _state_exponential_action(action, masks, amplitudes, reference, 1.)
            transformed = _state_exponential_action(action, masks, amplitudes, matrix@psi, -1.)
            psi_state, transformed_state = _state_dictionary(masks, psi), _state_dictionary(masks, transformed)
            directions = np.column_stack([
                _state_array(masks, action.unit_action(index, psi_state), real=True)
                for index in range(number)])
            left = _state_exponential_action(action, masks, amplitudes, matrix@directions, -1.)
            right = np.column_stack([
                _state_array(masks, action.unit_action(index, transformed_state), real=True)
                for index in range(number)])
            return (left-right)[projected]
        t_matrix = np.einsum('a,aij->ij', amplitudes, generators)
        psi = _exponential_action(t_matrix, reference, max_order, 1.)
        transformed = _exponential_action(t_matrix, matrix@psi, max_order, -1.)
        return _projected_jacobian(t_matrix, matrix, generators, psi, transformed,
                                   projected, max_order)

    evaluations = 0
    jacobian_evaluations = 0
    records = []
    if number:
        denominator = h[projected, projected]-h[ref, ref]
        if np.any((abs(denominator) < 1e-12) & (h[projected, ref] != 0)):
            raise ValueError(f'coupled degenerate {rank_label} reference; use a multireference method')
        diagonal = np.diag(np.diag(h))
        for stage in stages:
            matrix = diagonal+stage*(h-diagonal)
            solved = root(lambda a, matrix=matrix: evaluate(a, matrix)[0], amplitudes, method='hybr',
                          jac=lambda a, matrix=matrix: jacobian(a, matrix),
                          options={'xtol': 1e-11, 'maxfev': int(max_evaluations)})
            residual = float(np.max(abs(evaluate(solved.x, matrix)[0])))
            evaluations += solved.nfev
            jacobian_evaluations += solved.njev
            records.append((float(stage), int(solved.nfev), bool(solved.success), residual))
            if (not solved.success or not np.all(np.isfinite(solved.x))
                    or not np.isfinite(residual) or residual > residual_tolerance_Ha):
                raise RuntimeError(f'{method} stage={stage} failed: {solved.message}; projected residual={residual} Ha')
            amplitudes = solved.x
    residual, energy, psi = evaluate(amplitudes, h)
    amplitude_residual = float(np.max(abs(residual))) if number else 0.
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm <= 0 or amplitude_residual > residual_tolerance_Ha:
        raise RuntimeError(f'final {method} projected residual or wavefunction norm failed')
    vector = psi/norm
    full_residual = float(np.max(abs(h@vector-energy*vector)))
    provenance = dict(reference_identity=reference_identity, spinor_count=int(spinor_count),
        electrons=electrons, reference_mask=int(reference_mask), determinant_masks=tuple(map(int, masks)),
        determinants=count, amplitudes=number, excitations=tuple(excitations),
        excitation_rank_limit=max_rank, amplitudes_by_rank=rank_counts,
        generator_bytes=generator_bytes,
        dense_generator_bytes_if_materialized=dense_generator_bytes,
        action_representation=action_representation,
        cluster_term_application_cap=cluster_term_application_cap,
        cluster_term_application_scope='per scalar state action/exponential, not whole CC solve',
        jacobian_workspace_bytes_upper=jacobian_workspace_bytes,
        hamiltonian_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
        energy_units='Hartree', amplitude_convention='reference-determinant-normalized excitation generators',
        continuation_records=tuple(records), nonlinear_function_evaluations=evaluations,
        nonlinear_jacobian_evaluations=jacobian_evaluations,
        amplitude_jacobian='exact commuting-excitation similarity derivative',
        exponential=f'exact nilpotent T1+T2{"+T3" if max_rank == 3 else ""}, includes disconnected higher excitations',
        triples_treatment='iterative connected T3' if max_rank == 3 else 'omitted',
        limitations='finite proof/anchor engine; no production atomic RCCSDT, FSCC, left state or dressed observables')
    return CoupledClusterResult(method, energy-h[ref, ref], energy, float(h[ref, ref]),
                                amplitudes, vector, amplitude_residual, full_residual, evaluations, provenance)
