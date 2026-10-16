"""Native spin-orbital tensor CCSDT residuals and projected EA actions.

Consumes the same owned atomic integrals as finite CC, without enumerating
any fixed-particle determinant sector. Complete antisymmetric SDT coordinates
and 1p/2p1h/3p2h attachment coordinates are retained. Dense integral/amplitude
tensors still impose explicit polynomial memory limits; this is not J-reduced
production Yb, valence-universal FSCC or a dressed-property implementation.
"""
import hashlib
import itertools
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator

from atomx.numerics.tensor_recipes import contract
from atomx.physics.csf import ccsdt_terms as equations
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'positive integer {name} required')
    return int(value)


def resource_plan(noccupied, nvirtual, *, attachment_rank=3, batch_size=1):
    """Scalar admission estimate only; no tensor or determinant allocation."""
    o = _positive_integer(noccupied, 'noccupied')
    v = _positive_integer(nvirtual, 'nvirtual')
    batch = _positive_integer(batch_size, 'batch_size')
    if attachment_rank not in (1, 2, 3):
        raise ValueError('attachment_rank must be 1, 2 or 3')
    core = sum(comb(o, k)*comb(v, k) for k in range(1, min(3, o, v)+1))
    attached = sum(comb(o, k-1)*comb(v, k)
                   for k in range(1, min(attachment_rank, v)+1) if k-1 <= o)
    core_dense = sum((v*o)**k for k in (1, 2, 3))
    ea_dense = sum(v**k*o**(k-1) for k in (1, 2, 3))
    # Real input only for this first consumer. Includes owned integral copy,
    # operator/solver vectors, antisymmetric maps, residuals, permutations,
    # einsum intermediates and ownership copies. Python/SciPy overhead extra.
    resident = 8*((o+v)**4+2*(o+v)**2)
    maps = 8*(6*core+5*attached)
    working = 8*(24*core_dense+16*batch*ea_dense+16*core+8*batch*attached)
    return dict(core_amplitudes=core, attachment_coordinates=attached,
                dense_core_coordinates=core_dense,
                dense_attachment_coordinates=ea_dense,
                resident_integral_bytes=resident,
                numerical_upper_bytes=resident+maps+working,
                attachment_matrix_bytes=8*attached**2,
                full_particle_sectors_enumerated=False,
                scope='real NumPy payload including contraction workspace; Python/SciPy overhead extra')


def _parity(permutation):
    return (-1)**sum(permutation[i] > permutation[j]
                     for i in range(len(permutation)) for j in range(i+1, len(permutation)))


class _Coordinates:
    def __init__(self, occupied, virtual, attachment):
        self.occupied, self.virtual = occupied, virtual
        self.blocks = []
        offset = 0
        for rank in (1, 2, 3):
            holes = rank-1 if attachment else rank
            rows = [(p, h) for h in itertools.combinations(range(occupied), holes)
                    for p in itertools.combinations(range(virtual), rank)]
            particle = np.array([p for p, _ in rows], dtype=int).reshape(len(rows), rank)
            hole = np.array([h for _, h in rows], dtype=int).reshape(len(rows), holes)
            shape = (virtual,)*rank+(occupied,)*holes
            self.blocks.append((slice(offset, offset+len(rows)), particle, hole, shape))
            offset += len(rows)
        self.size = offset

    def unpack(self, vector):
        vector = np.asarray(vector)
        if vector.ndim not in (1, 2) or vector.shape[0] != self.size or not np.all(np.isfinite(vector)):
            raise ValueError('finite complete canonical amplitude vector required')
        batch = () if vector.ndim == 1 else (vector.shape[1],)
        tensors = []
        for section, particle, hole, shape in self.blocks:
            tensor = np.zeros(shape+batch, dtype=vector.dtype)
            for pp in itertools.permutations(range(particle.shape[1])):
                for hh in itertools.permutations(range(hole.shape[1])):
                    index = tuple(particle[:, k] for k in pp)+tuple(hole[:, k] for k in hh)
                    tensor[index] = (_parity(pp)*_parity(hh))*vector[section]
            tensors.append(tensor)
        return tuple(tensors)

    def pack(self, tensors):
        values = []
        for tensor, (_, particle, hole, _) in zip(tensors, self.blocks, strict=True):
            index = tuple(particle[:, k] for k in range(particle.shape[1]))+tuple(
                hole[:, k] for k in range(hole.shape[1]))
            values.append(tensor[index])
        return np.concatenate(values, axis=0)


@dataclass(frozen=True)
class TensorCoreResult:
    amplitudes: np.ndarray
    total_energy_Ha: float
    correlation_energy_Ha: float
    amplitude_residual_Ha: float
    evaluations: int
    operator_sha256: str
    retained_rank: int = 3

    def __post_init__(self):
        value = np.asarray(self.amplitudes)
        object.__setattr__(self, 'amplitudes', np.frombuffer(value.tobytes(), dtype=value.dtype))


class CCSDTTensorOperator:
    """Explicit real native integral consumer, with no Fock-space matrices.

    Dense tensor storage scales polynomially but is not an atomic production
    scaling certificate. All core amplitudes through T3 and all attachment
    projections through R3 remain present. Full-H states/residuals are not
    available from these projected equations and are never inferred.
    """
    def __init__(self, integrals, *, memory_MiB=512, projection_storage='dense'):
        if not isinstance(integrals, SpinorIntegrals):
            raise TypeError('owned packed SpinorIntegrals required')
        if projection_storage not in ('dense', 'canonical'):
            raise ValueError('projection_storage must be dense or canonical')
        self.projection_storage = projection_storage
        self.labels = tuple(integrals.labels)
        if (isinstance(memory_MiB, bool) or not np.isfinite(memory_MiB)
                or memory_MiB <= 0):
            raise ValueError('positive finite memory_MiB required')
        self.o, self.v = integrals.noccupied, len(integrals.labels)-integrals.noccupied
        self.memory_bytes = int(memory_MiB*1024**2)
        self.plan = resource_plan(self.o, self.v)
        if self.plan['numerical_upper_bytes'] > self.memory_bytes:
            raise MemoryError(f'complete tensor workspace exceeds admission: {self.plan}')
        if any(np.any(a.imag != 0) for a in (integrals.one_body_Ha, integrals.fock_Ha, integrals.packed_eri_Ha)):
            raise NotImplementedError('first native tensor consumer requires exactly real integrals')
        n = self.o+self.v
        fock = np.asarray(integrals.fock_Ha.real).copy()
        eri = np.empty((n,)*4)
        for index in itertools.product(range(n), repeat=4):
            eri[index] = integrals.eri(*index).real
        self.f = np.frombuffer(fock.tobytes(), dtype=float).reshape(fock.shape)
        self.g = np.frombuffer(eri.tobytes(), dtype=float).reshape(eri.shape)
        self.reference_energy_Ha = float(integrals.reference_energy_Ha)
        self.operator_sha256 = hashlib.sha256(
            self.f.tobytes()+self.g.tobytes()+repr((integrals.labels,
                self.o, self.reference_energy_Ha, integrals.frozen_energy_Ha,
                dict(integrals.identity))).encode()).hexdigest()
        self.core = _Coordinates(self.o, self.v, False)
        self.attachment = _Coordinates(self.o, self.v, True)

    def _arrays(self, vector, coupling=1.):
        if not np.isrealobj(vector):
            raise NotImplementedError('real core amplitudes required')
        if not np.isfinite(coupling) or not 0 <= coupling <= 1:
            raise ValueError('coupling must lie in [0,1]')
        t1, t2, t3 = self.core.unpack(vector)
        diagonal = np.diag(np.diag(self.f))
        return dict(f=diagonal+coupling*(self.f-diagonal), g=coupling*self.g,
                    t1=t1, t2=t2, t3=t3)

    def _contract(self, terms, output, arrays, batch=False, coordinates=None):
        return contract(terms, output, arrays, self.o,
                        intermediate_bytes=self.memory_bytes//8,
                        attachment_batch=batch, output_indices=coordinates)

    @staticmethod
    def _indices(block):
        _, particle, hole, _ = block
        return tuple(particle[:, k] for k in range(particle.shape[1]))+tuple(
            hole[:, k] for k in range(hole.shape[1]))

    def core_residual(self, vector, *, coupling=1.):
        arrays = self._arrays(vector, coupling)
        tensors = [self._contract(terms, output, arrays,
                    coordinates=self._indices(self.core.blocks[k]) if self.projection_storage == 'canonical' else None)
                   for k, (terms, output) in enumerate((
            (equations.SINGLES, 'ai'), (equations.DOUBLES, 'abij'),
            (equations.TRIPLES, 'abcijk')))]
        return (np.concatenate(tensors) if self.projection_storage == 'canonical'
                else self.core.pack(tensors))

    def scalar_core_indices(self):
        """Exact M=0/even excitation coordinates for a closed spherical core.

        Require complete occupied and virtual multiplets and exactly conserved
        M/parity in every Fock/ERI entry. No amplitude magnitude selects a
        coordinate. The complete SDT residual is still checked after solving.
        """
        for labels in (self.labels[:self.o], self.labels[self.o:]):
            groups = {}
            for label in labels:
                if not isinstance(label, tuple) or len(label) != 3:
                    raise ValueError('spherical spinor labels required')
                n, k, m = label
                if k == 0:
                    raise ValueError('nonzero spherical kappa required')
                groups.setdefault((n, k), set()).add(m)
            if any(ms != set(range(-2*abs(k)+1, 2*abs(k), 2))
                   for (_, k), ms in groups.items()):
                raise ValueError('complete occupied/virtual magnetic multiplets required')
        m = np.array([x[2] for x in self.labels])
        parity = np.array([(x[1] if x[1] > 0 else -x[1]-1) % 2 for x in self.labels])
        invalid = (m[:, None] != m[None, :]) | (parity[:, None] != parity[None, :])
        if np.any(self.f[invalid] != 0):
            raise ValueError('Fock operator does not exactly conserve M/parity')
        invalid = ((m[:, None, None, None]+m[None, :, None, None]
                    != m[None, None, :, None]+m[None, None, None, :]) |
                   ((parity[:, None, None, None]+parity[None, :, None, None]
                     +parity[None, None, :, None]+parity[None, None, None, :]) % 2 != 0))
        if np.any(self.g[invalid] != 0):
            raise ValueError('ERI operator does not exactly conserve M/parity')
        active = []
        for section, particle, hole, _ in self.core.blocks:
            same_m = (np.sum(m[self.o+particle], axis=1)-np.sum(m[hole], axis=1)) == 0
            even = (np.sum(parity[self.o+particle], axis=1)+np.sum(parity[hole], axis=1)) % 2 == 0
            active.extend(section.start+np.flatnonzero(same_m & even))
        values = np.asarray(active, dtype=np.int64)
        return np.frombuffer(values.tobytes(), dtype=values.dtype)

    def core_energy(self, vector):
        arrays = self._arrays(vector)
        value = self._contract(equations.ENERGY, '', arrays)
        o = slice(0, self.o)
        reference = np.trace(self.f[o, o])-.5*np.einsum('ijij', self.g[o, o, o, o])
        return self.reference_energy_Ha+float(value-reference)

    def solve_core(self, *, tolerance_Ha=1e-10, max_iterations=100,
                   continuation=(.25, .5, .75, 1.), retained_rank=3,
                   scalar_sector=False, preconditioner=None, initial=None):
        if retained_rank not in (2, 3):
            raise ValueError('retained core rank must be 2 or 3')
        if (isinstance(tolerance_Ha, bool) or not np.isfinite(tolerance_Ha)
                or tolerance_Ha <= 0):
            raise ValueError('positive finite tolerance_Ha required')
        max_iterations = _positive_integer(max_iterations, 'max_iterations')
        if preconditioner not in (None, 'fock_diagonal'):
            raise ValueError('preconditioner must be None or fock_diagonal')
        if not isinstance(scalar_sector, bool):
            raise ValueError('scalar_sector must be boolean')
        steps = tuple(continuation)
        if (not steps or steps[-1] != 1 or any(not 0 < x <= 1 for x in steps)
                or any(a >= b for a, b in zip(steps, steps[1:], strict=False))):
            raise ValueError('strictly increasing continuation ending at one required')
        count = self.core.blocks[retained_rank-1][0].stop
        active = (self.scalar_core_indices() if scalar_sector else np.arange(count))
        active = active[active < count]
        if initial is not None:
            if not isinstance(initial, TensorCoreResult) or initial.operator_sha256 != self.operator_sha256:
                raise ValueError('initial amplitudes belong to a different tensor operator')
            if initial.amplitudes.shape != (self.core.size,) or np.any(initial.amplitudes[np.setdiff1d(np.arange(self.core.size), active)] != 0):
                raise ValueError('initial amplitudes leave the declared retained coordinate space')
            vector = initial.amplitudes[active].copy()
        else:
            vector = np.zeros(len(active))
        def complete(x):
            full = np.zeros(self.core.size)
            full[active] = x
            return full
        options = dict(fatol=tolerance_Ha/10, maxiter=max_iterations)
        if preconditioner == 'fock_diagonal':
            diagonal = np.diag(self.f)
            denominators = np.concatenate([
                np.sum(diagonal[self.o+particle], axis=1)-np.sum(diagonal[hole], axis=1)
                for _, particle, hole, _ in self.core.blocks])[active]
            if np.any(~np.isfinite(denominators)) or np.any(denominators == 0):
                raise ValueError('nonzero finite Fock excitation denominators required')
            options['jac_options'] = dict(inner_M=LinearOperator(
                (len(active), len(active)), matvec=lambda x: np.asarray(x).reshape(-1)/denominators, dtype=float))
        evaluations = 0
        for coupling in steps:
            def residual(x, coupling=coupling):
                nonlocal evaluations
                evaluations += 1
                return self.core_residual(complete(x), coupling=coupling)[active]
            if np.max(abs(residual(vector))) > tolerance_Ha:
                solved = root(residual, vector, method='krylov', options=options)
                vector = solved.x
            error = float(np.max(abs(residual(vector))))
            if not np.isfinite(error) or error > tolerance_Ha:
                raise RuntimeError(f'native tensor core equations failed at coupling={coupling}: {error}')
        vector = complete(vector)
        evaluations += 1
        error = float(np.max(abs(self.core_residual(vector)[:count])))
        if not np.isfinite(error) or error > tolerance_Ha:
            raise RuntimeError(f'complete retained core residual failed: {error}')
        energy = self.core_energy(vector)
        return TensorCoreResult(vector, energy, energy-self.reference_energy_Ha,
                                error, evaluations, self.operator_sha256, retained_rank)

    def attachment_action(self, core, vector):
        """Complete R3 projected (Hbar-E_core) action, optionally batched.

        R2 is an explicit restriction by the consuming caller; this action
        itself retains all R3 coordinates and all core T3 contributions.
        """
        if not isinstance(core, TensorCoreResult) or core.operator_sha256 != self.operator_sha256:
            raise ValueError('core result belongs to a different tensor operator')
        vector = np.asarray(vector)
        batch = vector.shape[1] if vector.ndim == 2 else 1
        if resource_plan(self.o, self.v, batch_size=batch)['numerical_upper_bytes'] > self.memory_bytes:
            raise MemoryError('complete batched attachment tensor workspace exceeds admission')
        arrays = self._arrays(core.amplitudes)
        r1, r2, r3 = self.attachment.unpack(vector)
        arrays.update(r1=r1, r2=r2, r3=r3)
        tensors = [self._contract(terms, output, arrays, vector.ndim == 2,
                    coordinates=self._indices(self.attachment.blocks[k]) if self.projection_storage == 'canonical' else None)
                   for k, (terms, output) in enumerate(((equations.EA1, 'a'),
                                         (equations.EA2, 'abi'),
                                         (equations.EA3, 'abcij')))]
        value = (np.concatenate(tensors, axis=0) if self.projection_storage == 'canonical'
                 else self.attachment.pack(tensors))
        o = slice(0, self.o)
        bare = np.trace(self.f[o, o])-.5*np.einsum('ijij', self.g[o, o, o, o])
        core_normal = bare+core.correlation_energy_Ha
        return value-core_normal*vector

    def attachment_matrix(self, core, *, batch_size=4):
        """Directed R3 pencil only; no physical state or full-H certificate."""
        batch_size = _positive_integer(batch_size, 'batch_size')
        plan = resource_plan(self.o, self.v, batch_size=batch_size)
        if plan['numerical_upper_bytes']+plan['attachment_matrix_bytes'] > self.memory_bytes:
            raise MemoryError('complete projected EA matrix exceeds tensor admission')
        count = self.attachment.size
        matrix = np.empty((count, count))
        for begin in range(0, count, batch_size):
            end = min(begin+batch_size, count)
            vectors = np.zeros((count, end-begin))
            vectors[np.arange(begin, end), np.arange(end-begin)] = 1
            matrix[:, begin:end] = self.attachment_action(core, vectors)
        return matrix
