"""Connected finite triples against independent fermion/FCI cumulant oracles."""
import itertools

import numpy as np
import pytest

from atomx.methods import coupled_cluster


def _masks(spinors, electrons):
    return tuple(sum(1 << i for i in c)
                 for c in itertools.combinations(range(spinors), electrons))


def _act(mask, operations):
    """Independent ascending-occupation Fock sign; no Atomx fermion helper."""
    sign = 1
    for bit, create in reversed(operations):
        present = bool(mask & (1 << bit))
        if present == create:
            return None
        sign *= (-1)**(mask & ((1 << bit)-1)).bit_count()
        mask = mask | (1 << bit) if create else mask & ~(1 << bit)
    return mask, sign


def _hamiltonian(masks, one_body, pair_matrix):
    """One-/two-body strings assembled independently, both triangles together."""
    spinors = len(one_body)
    pairs = tuple(itertools.combinations(range(spinors), 2))
    lookup = {m: i for i, m in enumerate(masks)}
    h = np.zeros((len(masks), len(masks)))

    def add(value, operations, diagonal):
        for col, mask in enumerate(masks):
            result = _act(mask, operations)
            if result is not None:
                target, phase = result
                row = lookup[target]
                h[row, col] += value*phase
                if not diagonal:
                    h[col, row] += value*phase

    for p in range(spinors):
        for q in range(p+1):
            add(one_body[p, q], ((p, True), (q, False)), p == q)
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs[:i+1]):
            add(pair_matrix[i, j], ((p, True), (q, True), (s, False), (r, False)), i == j)
    assert np.array_equal(h, h.T)
    return h


def _integrals(spinors, seed=1729):
    rng = np.random.default_rng(seed)
    # Interleaved occupied/virtual spinors exercise nontrivial fermion signs.
    energies = np.array([-1.7, .4, -1.2, .9, -.8, 1.3, -.6, 1.6])[:spinors]
    one = rng.normal(scale=.02, size=(spinors, spinors))
    one = .5*(one+one.T)+np.diag(energies)
    pairs = spinors*(spinors-1)//2
    two = rng.normal(scale=.10, size=(pairs, pairs))
    return one, .5*(two+two.T)


def _fci_cluster(masks, reference, coefficients):
    """Independent T=log(1+C), retaining every rank in the finite Fock space."""
    spinors = max(masks).bit_length()
    occupied = [i for i in range(spinors) if reference & (1 << i)]
    virtual = [i for i in range(spinors) if not reference & (1 << i)]
    lookup = {m: i for i, m in enumerate(masks)}
    normalized = coefficients/coefficients[lookup[reference]]
    ci = np.zeros((len(masks), len(masks)))
    targets = {}
    for rank in range(1, min(len(occupied), len(virtual))+1):
        for holes in itertools.combinations(occupied, rank):
            for particles in itertools.combinations(virtual, rank):
                operations = tuple((a, True) for a in particles)+tuple(
                    (i, False) for i in reversed(holes))
                target, reference_phase = _act(reference, operations)
                targets[holes, particles] = lookup[target]
                amplitude = normalized[lookup[target]]/reference_phase
                for col, mask in enumerate(masks):
                    result = _act(mask, operations)
                    if result is not None:
                        end, phase = result
                        ci[lookup[end], col] += amplitude*phase
    cluster = np.zeros_like(ci)
    power = np.eye(len(masks))
    for rank in range(1, min(len(occupied), len(virtual))+1):
        power = power @ ci
        cluster += (-1)**(rank+1)*power/rank
    return {excitation: cluster[row, lookup[reference]] for excitation, row in targets.items()}


def test_connected_triples_two_body_fci_cumulant_and_signed_order():
    masks = _masks(6, 3)
    ref = sum(1 << i for i in (0, 2, 4))
    h = _hamiltonian(masks, *_integrals(6))
    opts = dict(spinor_count=6, reference_identity='independent-six-spinor-two-body')
    result = coupled_cluster.compute(h, masks, ref, method='CCSDT', **opts)
    energies, states = np.linalg.eigh(h)
    cumulants = _fci_cluster(masks, ref, states[:, 0])
    triple = ((0, 2, 4), (1, 3, 5))
    index = result.provenance['excitations'].index(triple)
    assert abs(cumulants[triple]) > 1e-5
    assert result.amplitudes[index] == pytest.approx(cumulants[triple], abs=1e-11)
    assert result.method == 'CCSDT'
    assert result.provenance['amplitudes_by_rank'] == ((1, 9), (2, 9), (3, 1))
    assert result.provenance['triples_treatment'] == 'iterative connected T3'
    assert result.total_energy_Ha == pytest.approx(energies[0], abs=1e-11)
    assert result.amplitude_residual_Ha < 1e-10
    assert result.full_eigenstate_residual_Ha < 1e-10
    np.testing.assert_allclose(np.outer(result.coefficients, result.coefficients),
                               np.outer(states[:, 0], states[:, 0]), atol=1e-11, rtol=0)
    ccsd = coupled_cluster.compute(h, masks, ref, **opts)
    assert ccsd.full_eigenstate_residual_Ha > 1e-6
    # Orbital phases and determinant storage order cannot change the answer.
    phases = np.array([(-1)**(m & 0b101010).bit_count() for m in masks])
    permutation = np.arange(len(masks))[::-1]
    transformed = phases[:, None]*h*phases[None, :]
    changed = coupled_cluster.compute(transformed[np.ix_(permutation, permutation)],
        tuple(masks[i] for i in permutation), ref, method='CCSDT', **opts)
    expected = (phases*result.coefficients)[permutation]
    assert changed.total_energy_Ha == pytest.approx(result.total_energy_Ha, abs=1e-11)
    np.testing.assert_allclose(np.outer(changed.coefficients, changed.coefficients),
                               np.outer(expected, expected), atol=1e-11, rtol=0)


def test_noninteracting_triple_and_single_have_disconnected_quadruples():
    one_a, pair_a = _integrals(6)
    masks_a = _masks(6, 3)
    h_a = _hamiltonian(masks_a, one_a, pair_a)
    energy_a, state_a = np.linalg.eigh(h_a)
    one_b = np.array([[-.8, .15], [.15, .9]])
    energy_b, state_b = np.linalg.eigh(one_b)
    one = np.zeros((8, 8))
    one[:6, :6], one[6:, 6:] = one_a, one_b
    pairs = tuple(itertools.combinations(range(8), 2))
    pairs_a = tuple(itertools.combinations(range(6), 2))
    pair = np.zeros((len(pairs), len(pairs)))
    where = [pairs.index(p) for p in pairs_a]
    pair[np.ix_(where, where)] = pair_a
    masks = _masks(8, 4)
    h = _hamiltonian(masks, one, pair)
    # Local number penalties select the N_A=3,N_B=1 ground sector without
    # discarding any other four-electron determinants or coupling coordinates.
    for i, m in enumerate(masks):
        na = (m & 0b111111).bit_count()
        h[i, i] += 5*((na-3)**2+(4-na-1)**2)
    ref = sum(1 << i for i in (0, 2, 4, 6))
    result = coupled_cluster.compute(h, masks, ref, spinor_count=8,
        reference_identity='noninteracting-three-plus-one', method='CCSDT')
    expected = np.zeros(len(masks))
    for ma, a in zip(masks_a, state_a[:, 0], strict=True):
        for mb, b in enumerate(state_b[:, 0]):
            expected[masks.index(ma | (1 << (6+mb)))] = a*b
    assert result.total_energy_Ha == pytest.approx(energy_a[0]+energy_b[0], abs=1e-11)
    assert result.full_eigenstate_residual_Ha < 1e-10
    np.testing.assert_allclose(np.outer(result.coefficients, result.coefficients),
                               np.outer(expected, expected), atol=1e-11, rtol=0)
    quadruple = sum(1 << i for i in (1, 3, 5, 7))
    assert abs(result.coefficients[masks.index(quadruple)]) > 1e-5
    # Connected T4 vanishes for this independent product, yet exp(T) keeps its
    # nonzero quadruple coefficient; a rank-three CI vector would lose it.
    cumulants = _fci_cluster(masks, ref, expected)
    assert abs(cumulants[((0, 2, 4, 6), (1, 3, 5, 7))]) < 1e-12
    sdt = [i for i, m in enumerate(masks) if (m ^ ref).bit_count() <= 6]
    assert np.linalg.eigvalsh(h[np.ix_(sdt, sdt)])[0]-result.total_energy_Ha > 1e-8


def test_rank_three_is_not_complete_four_electron_fci():
    masks = _masks(8, 4)
    ref = sum(1 << i for i in (0, 2, 4, 6))
    h = _hamiltonian(masks, *_integrals(8, seed=2718))
    result = coupled_cluster.compute(h, masks, ref, spinor_count=8,
        reference_identity='connected-quadruple-omission', method='CCSDT')
    assert result.amplitude_residual_Ha < 1e-10
    assert result.provenance['amplitudes_by_rank'] == ((1, 16), (2, 36), (3, 16))
    assert result.full_eigenstate_residual_Ha > 1e-8
    assert abs(result.total_energy_Ha-np.linalg.eigvalsh(h)[0]) > 1e-9


def test_two_electron_ccsdt_has_no_triple_coordinate():
    masks = _masks(4, 2)
    ref = (1 << 0) | (1 << 2)
    h = _hamiltonian(masks, *_integrals(4))
    opts = dict(spinor_count=4, reference_identity='no-possible-connected-triples')
    sd = coupled_cluster.compute(h, masks, ref, **opts)
    sdt = coupled_cluster.compute(h, masks, ref, method='CCSDT', **opts)
    assert sdt.provenance['amplitudes_by_rank'] == ((1, 4), (2, 1))
    assert len(sdt.amplitudes) == 5
    assert sdt.total_energy_Ha == pytest.approx(sd.total_energy_Ha, abs=1e-12)
    assert sdt.total_energy_Ha == pytest.approx(np.linalg.eigvalsh(h)[0], abs=1e-12)
    assert sdt.full_eigenstate_residual_Ha < 1e-10


def test_exact_triples_jacobian_matches_independent_finite_differences():
    masks = _masks(6, 3)
    ref = sum(1 << i for i in (0, 2, 4))
    h = _hamiltonian(masks, *_integrals(6))
    occupied, virtual = (0, 2, 4), (1, 3, 5)
    excitations = [(a, b) for rank in (1, 2, 3)
                   for a in itertools.combinations(occupied, rank)
                   for b in itertools.combinations(virtual, rank)]
    generators = coupled_cluster._cluster_generators(masks, ref, excitations)
    reference = np.eye(len(masks))[:, masks.index(ref)]
    projected = [masks.index(_act(ref, tuple((a, True) for a in particles)
                 +tuple((i, False) for i in reversed(holes)))[0])
                 for holes, particles in excitations]
    amplitudes = np.linspace(-.02, .03, len(excitations))

    def action(amplitudes):
        # Full matrix exponential is an independent implementation of the
        # finite similarity transformation, including disconnected products.
        from scipy.linalg import expm

        t = np.einsum('a,aij->ij', amplitudes, generators)
        psi = expm(t) @ reference
        return expm(-t) @ h @ psi, psi, t

    transformed, psi, t = action(amplitudes)
    analytic = coupled_cluster._projected_jacobian(t, h, generators, psi,
                                                 transformed, projected, 3)
    step = 1e-5
    numerical = np.column_stack([(action(amplitudes+step*np.eye(len(excitations))[i])[0]
                                -action(amplitudes-step*np.eye(len(excitations))[i])[0])[projected]
                                /(2*step) for i in range(len(excitations))])
    np.testing.assert_allclose(analytic, numerical, atol=1e-9, rtol=0)


def test_rank_three_caps_precede_generator_allocation_and_no_fake_methods(monkeypatch):
    masks = _masks(6, 3)
    h = _hamiltonian(masks, *_integrals(6))
    ref = sum(1 << i for i in (0, 2, 4))
    opts = dict(spinor_count=6, reference_identity='rank-and-resource-guard')

    def forbidden(*args, **kwargs):
        raise AssertionError('generator allocation entered before resource rejection')

    monkeypatch.setattr(coupled_cluster, '_cluster_generators', forbidden)
    for caps in [dict(determinant_cap=19), dict(amplitude_cap=18),
                 dict(generator_memory_MiB=.04), dict(generator_memory_MiB=.07)]:
        with pytest.raises(ValueError, match='cap'):
            coupled_cluster.compute(h, masks, ref, method='CCSDT', **opts, **caps)
    for method in ['CCSD(T)', 'CCSDTQ', 'FS-CCSDT']:
        with pytest.raises(NotImplementedError, match='Fock-space'):
            coupled_cluster.compute(h, masks, ref, method=method, **opts)
