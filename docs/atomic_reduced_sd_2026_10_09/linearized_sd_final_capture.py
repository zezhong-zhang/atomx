"""Scalar closed-core linearized SD and exact rotational resource planning.

The finite native solver projects H + [H,T1+T2] onto all declared J=0
singles/doubles. It solves the linked linearized CC equations, not CISD
or nonlinear CCSD. The singles-times-singles subtraction is retained for
noncanonical orbitals. Native signed CI kernels supply H; no second
implementation of Coulomb/Breit or radial integration is introduced.

This dense reduced-CSF solver is a correctness anchor. The planner supports
large complete radial pools, but does not make their dense solve scalable.
Monovalent SD/Bloch equations and CI+all-order screening are separate work.
"""
import hashlib
from collections import Counter
from dataclasses import dataclass
from math import comb
from types import MappingProxyType

import numpy as np

from atomx.angular.csf_hamiltonian import _ops_sign
from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa

__all__ = ['ScalarSDPlan', 'LinearizedSDResult', 'plan_scalar_sd', 'solve', 'compute']


def _keys(values):
    keys = tuple(tuple(k) for k in values)
    if (not keys or len(set(keys)) != len(keys)
            or any(len(k) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
                                      for x in k) or k[0] < 1 or k[1] == 0 for k in keys)):
        raise ValueError('distinct positive-n nonzero-kappa radial keys required')
    return keys


def _coupled_pairs(keys):
    """Counts of normalized antisymmetric radial pairs at each (2J, parity)."""
    counts = Counter()
    radial_counts = Counter(k for _, k in keys)
    kappas = sorted(radial_counts)
    for i, left in enumerate(kappas):
        ja = twoj_of_kappa(left)
        for right in kappas[i:]:
            jb = twoj_of_kappa(right)
            parity = (-1)**(l_of_kappa(left)+l_of_kappa(right))
            distinct = (comb(radial_counts[left], 2) if left == right
                        else radial_counts[left]*radial_counts[right])
            for j in range(abs(ja-jb), ja+jb+1, 2):
                # Identical radial orbitals permit only even integer J.
                counts[j, parity] += distinct
                if left == right and j % 4 == 0:
                    counts[j, parity] += radial_counts[left]
    return counts


def _magnetic_pairs(keys):
    states = Counter((m, (-1)**l_of_kappa(k)) for _, k in keys
                     for m in range(-twoj_of_kappa(k), twoj_of_kappa(k)+1, 2))
    counts = Counter()
    groups = sorted(states)
    for i, a in enumerate(groups):
        for b in groups[i:]:
            count = comb(states[a], 2) if a == b else states[a]*states[b]
            counts[a[0]+b[0], a[1]*b[1]] += count
    return states, counts


@dataclass(frozen=True)
class ScalarSDPlan:
    occupied_keys: tuple
    virtual_keys: tuple
    singles: int
    doubles: int
    magnetic_symmetry_coordinates: int
    unrestricted_coordinates: int
    pair_channels: object

    def __post_init__(self):
        object.__setattr__(self, 'pair_channels', MappingProxyType(dict(self.pair_channels)))

    @property
    def amplitudes(self):
        return self.singles+self.doubles

    def resources(self):
        """One real scalar amplitude vector; dense anchor estimate, not HPC RSS."""
        n = self.amplitudes
        return dict(scalar_amplitudes=n, singles=self.singles, doubles=self.doubles,
                    real_amplitude_vector_bytes=8*n,
                    dense_anchor_matrix_working_bytes=8*(5*(n+1)**2+4*(n+1)),
                    magnetic_symmetry_coordinates=self.magnetic_symmetry_coordinates,
                    unrestricted_coordinates=self.unrestricted_coordinates,
                    scope='Exact scalar coordinates for closed-shell T1/T2; excludes integral/recoupling caches, valence sectors, triples and iterative intermediates')


def plan_scalar_sd(occupied_keys, virtual_keys):
    """Count exact independent scalar T1/T2 coordinates before any CSF build.

    Each radial key implies its complete magnetic multiplet. Occupied keys
    are the correlated closed core; frozen keys must be excluded explicitly.
    Doubles couple an occupied and virtual antisymmetric pair with the same
    J/parity to a scalar. No amplitude cutoff or principal-index cap occurs.
    This is an SU(2) count, not the count of redundant particle-hole rho_k.
    """
    occupied, virtual = _keys(occupied_keys), _keys(virtual_keys)
    if set(occupied) & set(virtual):
        raise ValueError('disjoint occupied and virtual radial keys required')
    oc, vc = Counter(k for _, k in occupied), Counter(k for _, k in virtual)
    singles = sum(count*vc[k] for k, count in oc.items())
    op, vp = _coupled_pairs(occupied), _coupled_pairs(virtual)
    channels = {key: (count, vp[key]) for key, count in sorted(op.items()) if count and vp[key]}
    doubles = sum(a*b for a, b in channels.values())
    om, omp = _magnetic_pairs(occupied)
    vm, vmp = _magnetic_pairs(virtual)
    magnetic = sum(count*vm[k] for k, count in om.items())+sum(count*vmp[k] for k, count in omp.items())
    o, v = sum(om.values()), sum(vm.values())
    return ScalarSDPlan(occupied, virtual, singles, doubles, magnetic,
                        o*v+comb(o, 2)*comb(v, 2), channels)


def _owned(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


@dataclass(frozen=True)
class LinearizedSDResult:
    reference_energy_Ha: float
    correlation_energy_Ha: float
    total_energy_Ha: float
    amplitudes: np.ndarray
    equation_residual_Ha: float
    equation_matrix_Ha: np.ndarray
    singles_product_subtraction_Ha: np.ndarray
    provenance: object

    def __post_init__(self):
        for name in ('amplitudes', 'equation_matrix_Ha', 'singles_product_subtraction_Ha'):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, 'provenance', MappingProxyType(dict(self.provenance)))


def solve(hamiltonian_Ha, vectors, reference_mask, *, reference_identity,
          residual_tolerance_Ha=1e-10, memory_bytes=128*1024**2):
    """Solve a finite signed SD projection of H+[H,T] in the given order.

    ``vectors`` are orthonormal signed determinant expansions, reference
    first, then reference-defined singles/doubles. The producer owns their
    physical completeness and angular sector; native ``compute`` verifies
    that completeness by independent scalar counting. H must be a genuine
    same-space one-/two-body Hamiltonian. No full fixed-N determinant space
    or cluster-generator matrices are allocated. The result is amplitudes
    and a linked energy, not a normalized CI or exponential CC eigenstate.
    """
    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError('physical reference identity required')
    if (isinstance(reference_mask, bool) or not isinstance(reference_mask, (int, np.integer))
            or reference_mask < 1):
        raise ValueError('positive occupied reference mask required')
    if (not np.isscalar(residual_tolerance_Ha) or not np.isrealobj(residual_tolerance_Ha)
            or not np.isfinite(residual_tolerance_Ha) or residual_tolerance_Ha <= 0
            or isinstance(memory_bytes, bool) or not isinstance(memory_bytes, (int, np.integer))
            or memory_bytes < 1):
        raise ValueError('positive finite residual tolerance and integer memory cap required')
    n = len(vectors)
    if n < 2:
        raise ValueError('nonempty excited SD space required')
    required = 8*(5*n*n+4*n)
    if required > memory_bytes:
        raise MemoryError(f'complete dense SD equations require at least {required} working bytes; cap={memory_bytes}')
    h = np.asarray(hamiltonian_Ha)
    if (h.shape != (n, n) or not np.isrealobj(h) or not np.all(np.isfinite(h))
            or not np.array_equal(h, h.T)):
        raise ValueError('finite real symmetric same-space Hamiltonian required')
    ref = int(reference_mask)
    rows = []
    determinant_columns = {}
    for column, vector in enumerate(vectors):
        row = dict(vector)
        if len(row) != len(vector) or not row:
            raise ValueError('nonempty distinct determinant coefficients required')
        ranks = set()
        for mask, coefficient in row.items():
            if (isinstance(mask, bool) or not isinstance(mask, (int, np.integer))
                    or mask < 1 or int(mask).bit_count() != ref.bit_count()
                    or not np.isscalar(coefficient) or not np.isrealobj(coefficient)
                    or not np.isfinite(coefficient) or coefficient == 0):
                raise ValueError('finite nonzero signed fixed-N determinant coefficients required')
            ranks.add((ref & ~int(mask)).bit_count())
            determinant_columns.setdefault(int(mask), []).append((column, float(coefficient)))
        if ranks != ({0} if column == 0 else {1}) and ranks != {2}:
            raise ValueError('each excited vector must have one reference-defined SD rank')
        if column == 0 and set(row) != {ref}:
            raise ValueError('first vector must be the unique closed reference')
        rows.append(row)
    overlap = np.zeros((n, n))
    for columns in determinant_columns.values():
        for i, a in columns:
            for j, b in columns:
                overlap[i, j] += a*b
    if np.max(abs(overlap-np.eye(n))) > 1e-10:
        raise ValueError('signed SD vectors must be orthonormal')
    reference_phase = rows[0][ref]
    # Only T1 acting on the singly excited part of H|reference> can end in
    # the SD projection. Every other nonconstant product has rank >=3.
    href_singles = {}
    for mask, columns in determinant_columns.items():
        if (ref & ~mask).bit_count() == 1:
            href_singles[mask] = sum(c*h[col, 0] for col, c in columns)
    subtraction = np.zeros((n-1, n-1))
    for column, row in enumerate(rows[1:]):
        if (ref & ~next(iter(row))).bit_count() != 1:
            continue
        for mask, coefficient in row.items():
            remove, add = ref & ~mask, mask & ~ref
            hole, particle = remove.bit_length()-1, add.bit_length()-1
            ops = ((particle, True), (hole, False))
            factor = coefficient/(reference_phase*_ops_sign(ref, ops))
            for source, value in href_singles.items():
                sign = _ops_sign(source, ops)
                if sign:
                    target = (source ^ remove) | add
                    for i, c in determinant_columns.get(target, ()):
                        if i:
                            subtraction[i-1, column] += c*factor*sign*value
    energy = float(h[0, 0])
    matrix = h[1:, 1:]-energy*np.eye(n-1)-subtraction
    amplitudes = np.linalg.solve(matrix, -h[1:, 0])
    residual = float(np.max(abs(h[1:, 0]+matrix@amplitudes)))
    correlation = float(h[0, 1:]@amplitudes)
    if (not np.all(np.isfinite(amplitudes)) or not np.isfinite(correlation)
            or residual > residual_tolerance_Ha):
        raise RuntimeError(f'linearized SD equation residual failed: {residual} Ha')
    return LinearizedSDResult(energy, correlation, energy+correlation, amplitudes,
        residual, matrix, subtraction, dict(reference_identity=reference_identity,
            method='finite projected linked linearized SD', amplitudes=n-1,
            working_matrix_estimate_bytes=required,
            equation='Q(H+[H,T1+T2])|reference>=0',
            hamiltonian_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
            reference_mask=ref,
            vector_sha256=hashlib.sha256(repr(tuple(tuple(sorted(row.items())) for row in rows)).encode()).hexdigest(),
            approximation='No nonlinear cluster products, connected triples, valence Bloch sectors or dressed properties'))


def compute(reference_configuration, target_orbitals, orbitals_dict, bsp_data,
            Z, ns, *, reference_identity, frozen_orbitals=(), include_breit=False,
            nuclear_potential=None, radial_snapshot=None, dimension_cap=512,
            max_determinants=2000, memory_bytes=128*1024**2,
            residual_tolerance_Ha=1e-10):
    """Native complete finite J=0 closed-core SD using the shared CI operator.

    All NR reference shells must be closed; the target NR shells must be
    disjoint and supply both spin-orbit components. Frozen NR shells stay
    in H and the reference zero while being excluded from amplitudes.
    DC or full shared static DCB is declared explicitly. The memory/CSF
    gates reject the whole request before vector/matrix materialization.
    """
    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    from atomx.config.csf import get_shared_orbital_basis
    from atomx.config.parsing import parse_nr_shells
    from atomx.config.subshell import L_LETTERS
    from atomx.methods.ci_sd import plan_sd_ci_space
    from atomx.physics.csf.effective_one_body import CIReferenceBinding
    from atomx.physics.csf.matrix import build_ci_matrix

    for name, value in (('dimension_cap', dimension_cap), ('max_determinants', max_determinants),
                        ('memory_bytes', memory_bytes)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'positive integer {name} required')
    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError('physical reference identity required')
    reference = parse_nr_shells(reference_configuration)
    if (not reference or len({(n, l) for n, l, _ in reference}) != len(reference)
            or any(q != 2*(2*l+1) for _, l, q in reference)):
        raise ValueError('one fully closed-shell NR reference required')
    frozen = tuple(frozen_orbitals)
    if not isinstance(include_breit, bool):
        raise ValueError('include_breit must be bool')
    targets = tuple(target_orbitals)
    occupied_labels = {f'{n}{L_LETTERS[l]}' for n, l, _ in reference}
    if occupied_labels & set(targets):
        raise ValueError('target shells must be disjoint from the closed reference')
    def relativistic(n, l):
        return [(n, -l-1)] if l == 0 else [(n, l), (n, -l-1)]
    occupied = [k for n, l, _ in reference if f'{n}{L_LETTERS[l]}' not in frozen
                for k in relativistic(n, l)]
    virtual = [k for label in targets for n, l, _ in parse_nr_shells(label)
               for k in relativistic(n, l)]
    reduced = plan_scalar_sd(occupied, virtual)
    if reduced.resources()['dense_anchor_matrix_working_bytes'] > memory_bytes:
        raise MemoryError('complete scalar SD dense anchor exceeds memory cap')
    if reduced.amplitudes+1 > dimension_cap:
        raise ValueError(f'complete scalar SD needs {reduced.amplitudes+1} CSFs; cap={dimension_cap}')
    plan = plan_sd_ci_space((reference_configuration,), targets, frozen_orbitals=frozen,
                           J2=0, parity=1, dimension_cap=dimension_cap,
                           max_determinants=max_determinants)
    if plan.n_csfs != reduced.amplitudes+1:
        raise RuntimeError('complete J0 CSF count disagrees with independent scalar SD count')
    csfs, p = plan.materialize()
    if len(p) != 1:
        raise RuntimeError('closed-shell reference must be unique')
    csfs = (csfs[p[0]],)+tuple(c for i, c in enumerate(csfs) if i != p[0])
    keys = get_shared_orbital_basis(csfs)
    binding = CIReferenceBinding.capture([(*orbitals_dict[key], key[1]) for key in keys],
                                         {}, bsp_data, Z, ns, nuclear_potential)
    h = build_ci_matrix(csfs, orbitals_dict, keys, bsp_data, Z, ns,
                        nuclear_potential=nuclear_potential, radial_snapshot=radial_snapshot,
                        include_breit=include_breit, max_determinants=max_determinants)
    vectors = tuple(csf_magnetic_vector(csf_signature(c), 0, tuple(keys)) for c in csfs)
    result = solve(h, vectors, vectors[0][0][0], reference_identity=reference_identity,
                   memory_bytes=memory_bytes, residual_tolerance_Ha=residual_tolerance_Ha)
    object.__setattr__(result, 'provenance', MappingProxyType(dict(result.provenance,
        scalar_plan=reduced.resources(), radial_binding=repr(binding),
        frozen_orbitals=frozen, hamiltonian_terms=('dirac-coulomb', 'static-breit')
        if include_breit else ('dirac-coulomb',),
        csf_identities=tuple(csf_signature(c).key for c in csfs))))
    return result
