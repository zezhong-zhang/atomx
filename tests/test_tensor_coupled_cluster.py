"""Tensor equations versus independent full fermion exponentials and FCI."""
import itertools
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from test_eom_attachment import independent_hamiltonian, integrals
from test_finite_ccsdt import _integrals

from atomx.methods.tensor_coupled_cluster import (
    CCSDTTensorOperator,
    TensorCoreResult,
    resource_plan,
)
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def lift(operator, vector, masks, attachment=False):
    """Occupation-list signs independent of native tensor permutation maps."""
    coordinate = operator.attachment if attachment else operator.core
    ref = (1 << operator.o)-1
    lookup = {m: i for i, m in enumerate(masks)}
    matrix = np.zeros((len(masks), 1 if attachment else len(masks)))
    targets, phases = [], []
    for section, particles, holes, _ in coordinate.blocks:
        for amplitude, particle, hole in zip(vector[section], particles, holes, strict=True):
            operations = [(operator.o+int(p), True) for p in particle]
            operations += [(int(h), False) for h in reversed(hole)]
            for col, mask in enumerate((ref,) if attachment else masks):
                occupied = [i for i in range(operator.o+operator.v) if mask & (1 << i)]
                sign = 1
                for orbital, create in reversed(operations):
                    if (orbital in occupied) == create:
                        break
                    sign *= (-1)**sum(i < orbital for i in occupied)
                    if create:
                        occupied.append(orbital)
                    else:
                        occupied.remove(orbital)
                else:
                    row = lookup[sum(1 << i for i in occupied)]
                    matrix[row, col] += amplitude*sign
                    if attachment or mask == ref:
                        targets.append(row)
                        phases.append(sign)
    return matrix, targets, phases


@pytest.mark.parametrize('spinors', [6, 7])
@pytest.mark.parametrize('storage', ['dense', 'canonical'])
def test_all_core_and_attachment_coordinates_against_full_exponentials(spinors, storage):
    obj, h, eri = integrals(spinors=spinors)
    op = CCSDTTensorOperator(obj, projection_storage=storage)
    rng = np.random.default_rng(331)
    vector = rng.normal(scale=.025, size=op.core.size)
    mc, hc = independent_hamiltonian(obj, h, eri, obj.noccupied)
    ma, ha = independent_hamiltonian(obj, h, eri, obj.noccupied+1)
    tc, core_rows, core_phase = lift(op, vector, mc)
    ta, _, _ = lift(op, vector, ma)
    hcbar, habar = expm(-tc)@hc@expm(tc), expm(-ta)@ha@expm(ta)
    ref = mc.index((1 << op.o)-1)
    np.testing.assert_allclose(op.core_residual(vector),
                               np.array(core_phase)*hcbar[core_rows, ref], atol=2e-13, rtol=0)
    energy = op.core_energy(vector)
    assert energy == pytest.approx(hcbar[ref, ref], abs=2e-13)
    core = TensorCoreResult(vector, energy, energy-obj.reference_energy_Ha,
                            np.max(abs(op.core_residual(vector))), 0, op.operator_sha256)
    _, rows, phase = lift(op, np.ones(op.attachment.size), ma, True)
    expected = np.array(phase)[:, None]*habar[np.ix_(rows, rows)]*np.array(phase)[None, :]
    expected -= energy*np.eye(len(rows))
    actual = op.attachment_matrix(core, batch_size=3)
    assert np.max(abs(actual-actual.T)) > 1e-4
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
    signed = rng.normal(size=(op.attachment.size, 2))
    np.testing.assert_allclose(op.attachment_action(core, signed), expected@signed, atol=3e-13, rtol=0)


def test_complete_three_electron_tensor_core_and_ea_match_fci():
    obj, h, eri = integrals()
    op = CCSDTTensorOperator(obj)
    result = op.solve_core()
    _, hc = independent_hamiltonian(obj, h, eri, 3)
    _, ha = independent_hamiltonian(obj, h, eri, 4)
    assert result.amplitude_residual_Ha < 1e-10
    assert result.total_energy_Ha == pytest.approx(np.linalg.eigvalsh(hc)[0], abs=1e-11)
    assert np.linalg.norm(result.amplitudes[op.core.blocks[2][0]]) > 1e-6
    matrix = op.attachment_matrix(result)
    roots = np.linalg.eigvals(matrix)
    assert np.max(abs(roots.imag)) < 1e-12
    np.testing.assert_allclose(np.sort(roots.real),
                               np.linalg.eigvalsh(ha)-result.total_energy_Ha, atol=1e-10, rtol=0)
    assert not result.amplitudes.flags.writeable
    with pytest.raises(ValueError):
        result.amplitudes.flags.writeable = True
    sd = op.solve_core(retained_rank=2)
    assert sd.retained_rank == 2
    assert np.all(sd.amplitudes[op.core.blocks[2][0]] == 0)
    assert abs(sd.total_energy_Ha-result.total_energy_Ha) > 1e-8


def test_four_electron_t3_equations_preserve_omitted_quadruple_defect():
    n, o = 8, 4
    one, pair = _integrals(n)
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    one = one[np.ix_(order, order)]
    pairs = tuple(itertools.combinations(range(n), 2))
    def eri(p, q, r, s):
        p, q, r, s = (order[k] for k in (p, q, r, s))
        if p == q or r == s:
            return 0.
        sign = (-1 if p > q else 1)*(-1 if r > s else 1)
        return sign*pair[pairs.index(tuple(sorted((p, q)))), pairs.index(tuple(sorted((r, s))))]
    packed = [eri(p, q, r, s) for p in range(n-1) for q in range(p+1, n)
              for r in range(p, n-1) for s in range(r+1, n)]
    fock = one+np.array([[sum(eri(p, i, q, i) for i in range(o))
                          for q in range(n)] for p in range(n)])
    reference = np.trace(one[:o, :o])+.5*sum(eri(i, j, i, j) for i in range(o) for j in range(o))
    obj = SpinorIntegrals(tuple(range(n)), o, one, fock, packed, 0., reference, {'anchor': '4e8'})
    op = CCSDTTensorOperator(obj)
    vector = np.random.default_rng(513).normal(scale=.025, size=op.core.size)
    masks, h = independent_hamiltonian(obj, one, eri, o)
    t, rows, phase = lift(op, vector, masks)
    hbar = expm(-t)@h@expm(t)
    reference_row = masks.index((1 << o)-1)
    np.testing.assert_allclose(op.core_residual(vector),
                               np.array(phase)*hbar[rows, reference_row], atol=3e-13, rtol=0)
    quadruple = masks.index(sum(1 << i for i in range(o, n)))
    assert abs(hbar[quadruple, reference_row]) > 1e-5
    assert quadruple not in rows


def test_tensor_radial_phase_covariance_and_energy_zero():
    obj, _, _ = integrals()
    phase = np.array([-1., 1., -1., 1., -1., 1.])
    changed, _, _ = integrals(phases=phase, frozen=-1.2)
    a, b = CCSDTTensorOperator(obj), CCSDTTensorOperator(changed)
    ca, cb = a.solve_core(), b.solve_core()
    assert cb.total_energy_Ha-ca.total_energy_Ha == pytest.approx(1.1, abs=1e-12)
    d = []
    for _, particles, holes, _ in a.attachment.blocks:
        d.extend(np.prod(phase[a.o+p])*np.prod(phase[h]) for p, h in zip(particles, holes, strict=True))
    d = np.array(d)
    np.testing.assert_allclose(b.attachment_matrix(cb),
                               d[:, None]*a.attachment_matrix(ca)*d[None, :], atol=1e-11, rtol=0)
    with pytest.raises(ValueError, match='different tensor operator'):
        b.attachment_action(ca, np.ones(b.attachment.size))


def test_admission_before_eri_or_coordinate_allocation(monkeypatch):
    obj, _, _ = integrals()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('integral evaluated after failed admission')
    monkeypatch.setattr(type(obj), 'eri', forbidden)
    with pytest.raises(MemoryError):
        CCSDTTensorOperator(obj, memory_MiB=.001)
    plan = resource_plan(68, 12)
    assert plan['core_amplitudes'] == 11176684
    assert plan['numerical_upper_bytes'] > 8*1024**3
    assert plan['full_particle_sectors_enumerated'] is False


def test_input_and_complete_rank_contracts():
    obj, _, _ = integrals()
    op = CCSDTTensorOperator(obj)
    for vector in (np.zeros(op.core.size-1), np.full(op.core.size, np.nan)):
        with pytest.raises(ValueError):
            op.core_residual(vector)
    with pytest.raises(ValueError):
        op.solve_core(continuation=(1., .5))
    with pytest.raises(ValueError):
        op.solve_core(tolerance_Ha=0)
    with pytest.raises(ValueError):
        resource_plan(True, 3)
    with pytest.raises(ValueError):
        resource_plan(3, 3, attachment_rank=4)
    complex_h = obj.one_body_Ha.copy()
    complex_h[0, 1] += .01j
    complex_h[1, 0] -= .01j
    complex_f = obj.fock_Ha+complex_h-obj.one_body_Ha
    rotated = replace(obj, one_body_Ha=complex_h, fock_Ha=complex_f)
    with pytest.raises(NotImplementedError, match='exactly real'):
        CCSDTTensorOperator(rotated)


def test_canonical_projection_composes_overlapping_signed_permutations():
    from atomx.numerics.tensor_recipes import contract
    rng = np.random.default_rng(7301)
    arrays = {'t3': (rng.normal(size=(3, 3, 3, 2, 2, 2))
                     +1j*rng.normal(size=(3, 3, 3, 2, 2, 2)))*1e-200}
    before = arrays['t3'].copy()
    terms = (('-0.250', 'P(a,b)', 'P(b,c)', 't3(a,b,c,i,j,k)'),)
    indices = tuple(rng.integers(dim, size=17) for dim in arrays['t3'].shape)
    dense = contract(terms, 'abcijk', arrays, 2, intermediate_bytes=1024**2)
    selected = contract(terms, 'abcijk', arrays, 2, intermediate_bytes=1024**2,
                         output_indices=indices)
    np.testing.assert_allclose(selected, dense[indices], atol=0, rtol=3e-15)
    assert np.max(abs(selected)) > 1e-201
    np.testing.assert_array_equal(arrays['t3'], before)


def test_preconditioned_core_and_explicit_warm_start_preserve_equations():
    obj, _, _ = integrals()
    dense = CCSDTTensorOperator(obj)
    op = CCSDTTensorOperator(obj, projection_storage='canonical')
    a = dense.solve_core()
    b = op.solve_core(preconditioner='fock_diagonal')
    assert b.total_energy_Ha == pytest.approx(a.total_energy_Ha, abs=1e-11)
    np.testing.assert_allclose(op.core_residual(b.amplitudes), 0, atol=1e-10, rtol=0)
    warmed = op.solve_core(initial=b, continuation=(1.,), preconditioner='fock_diagonal')
    assert warmed.total_energy_Ha == pytest.approx(a.total_energy_Ha, abs=1e-11)
    with pytest.raises(ValueError, match='complete occupied/virtual'):
        op.solve_core(scalar_sector=True)
    with pytest.raises(ValueError, match='different tensor operator'):
        op.solve_core(initial=replace(a, operator_sha256='wrong'))
    with pytest.raises(ValueError, match='preconditioner'):
        op.solve_core(preconditioner='unknown')


def test_exact_scalar_sector_preserves_complete_spherical_core_solution():
    # Whole s1/2 multiplets, with a real scalar one-body rotation and signed
    # number-number pair interactions; no omitted magnetic partners.
    labels = ((1,-1,-1),(1,-1,1),(2,-1,-1),(2,-1,1),(3,-1,-1),(3,-1,1))
    n, o = len(labels), 2
    h = np.diag([-2.,-2.,.2,.2,.7,.7])
    for p,q in [(0,2),(1,3),(2,4),(3,5)]:
        h[p,q]=h[q,p]=.03
    def eri(p,q,r,s):
        if p==q or r==s:
            return 0.
        return .02*((p==r and q==s)-(p==s and q==r))
    packed = [eri(p,q,r,s) for p in range(n-1) for q in range(p+1,n)
              for r in range(p,n-1) for s in range(r+1,n)]
    f = h+np.array([[sum(eri(p,i,q,i) for i in range(o)) for q in range(n)] for p in range(n)])
    reference = np.trace(h[:o,:o])+.5*sum(eri(i,j,i,j) for i in range(o) for j in range(o))
    obj = SpinorIntegrals(labels,o,h,f,packed,0.,reference,{'scope':'spherical scalar anchor'})
    op = CCSDTTensorOperator(obj,projection_storage='canonical')
    active = op.scalar_core_indices()
    assert 0 < len(active) < op.core.size
    all_coordinates = op.solve_core(preconditioner='fock_diagonal')
    selected = op.solve_core(scalar_sector=True,preconditioner='fock_diagonal')
    assert selected.total_energy_Ha == pytest.approx(all_coordinates.total_energy_Ha, abs=1e-12)
    np.testing.assert_allclose(op.core_residual(selected.amplitudes),0,atol=1e-10,rtol=0)
    assert np.all(selected.amplitudes[np.setdiff1d(np.arange(op.core.size),active)]==0)
