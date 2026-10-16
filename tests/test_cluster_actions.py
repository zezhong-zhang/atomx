"""Sparse finite cluster action against independent signed dense/FCI oracles."""

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from atomx.numerics import cluster_actions
from atomx.numerics.cluster_actions import ClusterAction


def _masks(spinors, electrons):
    return tuple(sum(1 << i for i in occupied)
                 for occupied in itertools.combinations(range(spinors), electrons))


def _act(mask, operations):
    """Independent elementary Fock action, without Atomx sign/action helpers."""
    phase = 1
    for bit, create in reversed(operations):
        present = bool(mask & (1 << bit))
        if present == create:
            return None
        phase *= (-1)**(mask & ((1 << bit)-1)).bit_count()
        mask = mask | (1 << bit) if create else mask & ~(1 << bit)
    return mask, phase


def _matrix(terms, masks):
    lookup = {mask: i for i, mask in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)), dtype=complex)
    for col, mask in enumerate(masks):
        for value, operations in terms:
            result = _act(mask, operations)
            if result is not None:
                target, phase = result
                matrix[lookup[target], col] += value*phase
    return matrix


def _generators(masks, reference, excitations):
    generators = []
    for holes, particles in excitations:
        ops = (tuple((a, True) for a in particles)
               +tuple((i, False) for i in reversed(holes)))
        _, phase = _act(reference, ops)
        generators.append(_matrix(((1/phase, ops),), masks))
    return np.array(generators)


def _operator_terms(spinors):
    rng = np.random.default_rng(1729)
    energies = [-1.7, .4, -1.2, .9, -.8, 1.3]
    terms = [(.17, ())]
    for p in range(spinors):
        terms.append((energies[p], ((p, True), (p, False))))
        for q in range(p):
            value = rng.normal(scale=.02)
            terms.extend(((value, ((p, True), (q, False))),
                          (value, ((q, True), (p, False)))))
    pairs = tuple(itertools.combinations(range(spinors), 2))
    for i, (p, q) in enumerate(pairs):
        for r, s in pairs[:i+1]:
            value = rng.normal(scale=.10)
            terms.append((value, ((p, True), (q, True), (s, False), (r, False))))
            if (p, q) != (r, s):
                terms.append((value, ((r, True), (s, True), (q, False), (p, False))))
    return tuple(terms)


def _vector(state, masks):
    return np.array([state.get(mask, 0) for mask in masks], dtype=complex)


@pytest.mark.parametrize('electrons', [3, 4])
def test_complete_core_and_ea_actions_match_dense_exponential_and_commutators(electrons):
    ref = sum(1 << i for i in (0, 2, 4))
    action = ClusterAction(ref, 6, 3, electrons)
    expected_excitations = tuple((holes, particles) for rank in (1, 2, 3)
                                for holes in itertools.combinations((0, 2, 4), rank)
                                for particles in itertools.combinations((1, 3, 5), rank))
    assert action.excitations == expected_excitations
    assert action.amplitudes_by_rank == ((1, 9), (2, 9), (3, 1))
    masks = _masks(6, electrons)
    assert action.sector_dimension == len(masks)
    rng = np.random.default_rng(801+electrons)
    amplitudes = rng.normal(scale=.035, size=19)+1j*rng.normal(scale=.01, size=19)
    values = rng.normal(size=len(masks))+1j*rng.normal(size=len(masks))
    state = dict(zip(masks, values, strict=True))
    generators = _generators(masks, ref, expected_excitations)
    t = np.einsum('a,aij->ij', amplitudes, generators)
    np.testing.assert_allclose(_vector(action.apply(amplitudes, state), masks),
                               t@values, atol=1e-13, rtol=0)
    for index, generator in enumerate(generators):
        np.testing.assert_allclose(_vector(action.unit_action(index, state), masks),
                                   generator@values, atol=1e-13, rtol=0)
    assert np.max(abs(np.linalg.matrix_power(t, action.max_order+1))) < 1e-15
    for sign in (-1, 1):
        sparse = action.exponential(amplitudes, state, sign)
        np.testing.assert_allclose(_vector(sparse, masks), expm(sign*t)@values,
                                   atol=1e-12, rtol=0)
    inverse = action.exponential(amplitudes, action.exponential(amplitudes, state, 1), -1)
    np.testing.assert_allclose(_vector(inverse, masks), values, atol=1e-12, rtol=0)
    hterms = _operator_terms(6)
    h = _matrix(hterms, masks)
    hbar = expm(-t)@h@expm(t)
    sparse = action.similarity(amplitudes, hterms, state)
    np.testing.assert_allclose(_vector(sparse, masks), hbar@values, atol=1e-12, rtol=0)
    # Independent dense nested commutators for this explicitly two-body H.
    # The action engine itself assumes no BCH-depth truncation.
    bch, commutator = h.copy(), h.copy()
    factorial = 1
    for depth in range(1, 6):
        commutator = commutator@t-t@commutator
        factorial *= depth
        if depth <= 4:
            bch += commutator/factorial
    assert np.max(abs(commutator)) < 1e-14
    np.testing.assert_allclose(bch, hbar, atol=1e-12, rtol=0)


def test_complete_three_electron_fci_log_cluster_has_nonzero_triples():
    masks = _masks(6, 3)
    ref = sum(1 << i for i in (0, 2, 4))
    action = ClusterAction(ref, 6, 3, 3)
    hterms = _operator_terms(6)
    h = _matrix(hterms, masks)
    # Keep the actual independently summed H; do not symmetrize it to make
    # an input pass. Its Hermiticity defect is bounded, and general eig keeps
    # every represented entry in this independent complete-FCI oracle.
    assert np.max(abs(h-h.conj().T)) < 1e-14
    energies, states = np.linalg.eig(h)
    order = np.argsort(energies.real)
    energy, state = energies[order[0]], states[:, order[0]]
    assert abs(energy.imag) < 1e-12
    assert np.max(abs(h@state-energy*state)) < 1e-12
    state /= state[masks.index(ref)]
    generators = _generators(masks, ref, action.excitations)
    rows = []
    for holes, particles in action.excitations:
        target = (ref ^ sum(1 << i for i in holes)) | sum(1 << a for a in particles)
        rows.append(masks.index(target))
    ci = np.einsum('a,aij->ij', state[rows], generators)
    log_cluster = ci-ci@ci/2+ci@ci@ci/3
    amplitudes = log_cluster[rows, masks.index(ref)]
    assert abs(amplitudes[-1]) > 1e-5
    transformed = action.similarity(amplitudes, hterms, {ref: 1.0})
    residual = _vector(transformed, masks)
    assert residual[masks.index(ref)] == pytest.approx(energy, abs=1e-12)
    assert np.max(abs(residual[rows])) < 1e-11
    sparse_state = _vector(action.exponential(amplitudes, {ref: 1.0}, 1), masks)
    np.testing.assert_allclose(sparse_state, state, atol=1e-11, rtol=0)


def test_disconnected_quadruple_and_tiny_coordinate_are_not_screened():
    ref = sum(1 << i for i in (0, 2, 4, 6))
    action = ClusterAction(ref, 8, 3, 4)
    amplitudes = np.zeros(len(action.excitations))
    triple = action.excitations.index(((0, 2, 4), (1, 3, 5)))
    single = action.excitations.index(((6,), (7,)))
    amplitudes[triple], amplitudes[single] = .08, -.13
    quadruple = sum(1 << i for i in (1, 3, 5, 7))
    state = action.exponential(amplitudes, {ref: 1.0}, 1)
    assert state[quadruple] == pytest.approx(.08*(-.13), abs=1e-15)
    amplitudes[:] = 0
    amplitudes[triple] = 1e-200
    target = (ref ^ sum(1 << i for i in (0, 2, 4))) | sum(1 << a for a in (1, 3, 5))
    assert action.apply(amplitudes, {ref: 1.0})[target] == 1e-200


def test_complete_caps_and_input_guards_precede_fermion_actions(monkeypatch):
    ref = sum(1 << i for i in (0, 2, 4))
    action = ClusterAction(ref, 6, 3, 3, term_application_cap=380)
    amplitudes = np.zeros(19)

    def forbidden(*args, **kwargs):
        pytest.fail('fermion action entered before failed input/resource guard')

    monkeypatch.setattr(cluster_actions, 'apply_terms', forbidden)
    for caps in [dict(determinant_cap=19), dict(amplitude_cap=18),
                 dict(term_application_cap=379)]:
        with pytest.raises(ValueError, match='cap'):
            ClusterAction(ref, 6, 3, 3, **caps)
    with pytest.raises(ValueError, match='cap'):
        action.exponential(amplitudes, {ref: 1.0}, 1)
    with pytest.raises(ValueError, match='cap'):
        action.similarity(amplitudes, (), {ref: 1.0})
    with pytest.raises(ValueError, match='ordered amplitude'):
        action.apply(amplitudes[:-1], {ref: 1.0})
    with pytest.raises(ValueError, match='particle sector'):
        action.apply(amplitudes, {1: 1.0})
    with pytest.raises(ValueError, match='finite scalar'):
        action.apply(amplitudes, {ref: np.nan})
    invalid = amplitudes.copy()
    invalid[0] = np.inf
    with pytest.raises(ValueError, match='finite scalar'):
        action.apply(invalid, {ref: 1.0})
    with pytest.raises(ValueError, match='excitation index'):
        action.unit_action(19, {ref: 1.0})
    with pytest.raises(ValueError, match='exponential sign'):
        action.exponential(amplitudes, {ref: 1.0}, True)


def test_operator_number_and_index_coverage_are_explicit(monkeypatch):
    ref = sum(1 << i for i in (0, 2, 4))
    action = ClusterAction(ref, 6, 3, 3)

    def forbidden(*args, **kwargs):
        pytest.fail('invalid operator reached scientific action')

    monkeypatch.setattr(cluster_actions, 'apply_terms', forbidden)
    for terms, message in [(((1.0, ((1, True),)),), 'particle number'),
                           (((1.0, ((6, True), (1, False))),), 'in-range'),
                           (((np.nan, ()),), 'finite scalar'),
                           (((1.0, ((1, 1), (1, False))),), 'Boolean')]:
        with pytest.raises(ValueError, match=message):
            action.similarity(np.zeros(19), terms, {ref: 1.0})


def test_empty_cluster_is_identity_in_empty_and_filled_sectors():
    for reference, electrons in ((0, 0), (15, 4)):
        action = ClusterAction(reference, 4, 3, electrons)
        state = {reference: -2j}
        assert action.excitations == ()
        assert action.apply((), state) == {}
        assert action.exponential((), state, 1) == state
        assert action.similarity((), ((1.25, ()),), state) == {reference: -2.5j}
