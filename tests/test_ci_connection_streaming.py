"""Exact structural CI connections retain signed operator contractions.

The connectivity oracle exhausts determinant pairs and uses particle-number
distance only. It does not use spectator buckets or the production iterator.
"""

import tracemalloc
from collections.abc import Iterator
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from atomx.angular import csf_hamiltonian as ci
from atomx.physics.csf.breit_ci import breit_terms
from atomx.physics.csf.radial_symmetry import canonical_static_radial_key


def _mask(*occupied):
    return sum(1 << index for index in occupied)


def _brute_connections(dets, *, triangular=False):
    return [(row, col) for row, bra in enumerate(dets)
            for col, ket in enumerate(dets)
            if (not triangular or row <= col) and (bra ^ ket).bit_count() <= 4]


_CORE = _mask(0, 5, 65)
_CASES = {
    'empty': (),
    'vacuum': (0,),
    'all_frozen': (_mask(0, 1, 66),),
    'one_electron': tuple(1 << bit for bit in (0, 1, 3, 65)),
    'incomplete_occupations': tuple(_mask(*bits) for bits in
                                    ((0, 1), (0, 3), (1, 4), (3, 4), (7, 9))),
    'disconnected_three_electron': tuple(_mask(*bits) for bits in
                                         ((0, 1, 2), (0, 1, 3), (3, 4, 5))),
    'only_isolated_three_electron': tuple(_mask(*bits) for bits in
                                         ((0, 1, 2), (3, 4, 5), (6, 7, 8), (70, 71, 72))),
    'connected_cluster_and_isolate': tuple(_mask(*bits) for bits in
                                          ((0, 1, 2), (0, 1, 3), (0, 4, 5), (65, 66, 67))),
    'one_active_above_frozen_core': tuple(_CORE | (1 << bit) for bit in (1, 3, 71)),
    'high_bit_frozen_core': tuple(_CORE | _mask(*bits) for bits in
                                  ((2, 3), (2, 70), (3, 80), (90, 91))),
    'shuffled_hole_patterns': tuple(_mask(*bits) for bits in
                                    tuple(combinations(range(8), 6))[::3][::-1]),
}


@pytest.mark.parametrize('dets', _CASES.values(), ids=_CASES.keys())
@pytest.mark.parametrize('triangular', [False, True])
def test_structural_connections_equal_independent_pair_distance(dets, triangular):
    assert len(set(dets)) == len(dets)
    assert len({mask.bit_count() for mask in dets}) <= 1
    actual = list(ci._connected_pairs(dets, triangular=triangular))
    assert actual == _brute_connections(dets, triangular=triangular)
    assert actual == sorted(set(actual))


@pytest.mark.parametrize('dets', [(7, 7, 56), (0, 0), (7, 7, 7)])
@pytest.mark.parametrize('triangular', [False, True])
def test_duplicate_determinant_coordinates_preserve_exact_pair_graph(dets, triangular):
    # Physical determinant bases normally contain unique masks. This private
    # connectivity helper has also preserved distinct input row coordinates
    # when masks repeat, including an entirely common occupied determinant.
    actual = list(ci._connected_pairs(dets, triangular=triangular))
    assert actual == _brute_connections(dets, triangular=triangular)
    assert actual == sorted(set(actual))


@pytest.mark.parametrize('triangular', [False, True])
def test_dense_connection_graph_yields_before_materializing_all_pairs(triangular):
    # Every pair of these determinants is structurally connected. A retained
    # D-squared Python set/list costs tens of MiB; the index plus first row is
    # much smaller. The generous bound tests representation, not a speedup.
    dets = tuple(1 << bit for bit in range(512))
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    initial, _ = tracemalloc.get_traced_memory()
    try:
        pairs = ci._connected_pairs(dets, triangular=triangular)
        assert isinstance(pairs, Iterator)
        assert next(pairs) == (0, 0)
        allocated, peak = tracemalloc.get_traced_memory()
        assert allocated - initial < 4 * 1024**2
        if not already_tracing:
            assert peak - initial < 4 * 1024**2
        assert [next(pairs) for _ in range(3)] == [(0, 1), (0, 2), (0, 3)]
    finally:
        if not already_tracing:
            tracemalloc.stop()


def _static_integral(key):
    canonical, sign = canonical_static_radial_key(key)
    if not sign:
        return 0.
    if canonical[0] == 'h':
        _, na, kappa, nc = canonical
        return -sign / (na * nc * (1 + abs(kappa)))
    kind, rank, *orbitals = canonical
    scale = {'r': 1., 'g': .4, 't': .3}[kind]
    if kind == 'g':
        scale *= np.prod([.3 if leg[0] == 'Q' else 1. for leg in orbitals])
        orbitals = [leg[1:] for leg in orbitals]
    return sign * scale / (1 + rank + sum(n + abs(kappa) for n, kappa in orbitals))


@pytest.mark.parametrize('frozen_core', [False, True])
@pytest.mark.parametrize('radial_symmetry', [False, True])
@pytest.mark.parametrize('term_fn', [None, breit_terms], ids=['DC', 'static_Breit'])
def test_streamed_table_preserves_full_signed_hamiltonian(
        monkeypatch, frozen_core, radial_symmetry, term_fn):
    labels = tuple((n, kappa, m2) for n, kappa in ((1, -1), (2, -1), (3, -1), (2, 1))
                   for m2 in (-1, 1))
    if frozen_core:
        dets = tuple(_mask(0, 1, *active) for active in combinations(range(2, 8), 2))
    else:
        dets = tuple(_mask(*active) for active in combinations(range(8), 3))
    # Input ordering is not a numeric-mask ordering contract.
    dets = dets[::2] + dets[1::2][::-1]
    keys, table, coordinates = ci._coefficient_table(
        dets, labels, term_fn, radial_symmetry=radial_symmetry)
    structure = SimpleNamespace(dets=dets, keys=keys, table=table,
                                pairs=coordinates, radial_symmetry=radial_symmetry)
    actual = ci.evaluate_block(structure, _static_integral)

    # Evaluate every determinant pair directly, bypassing both the connectivity
    # filter and the frozen-core table contraction. Exact angular algebra is
    # intentionally shared: this increment changes representation, not physics.
    reference = np.asarray([
        [sum(value * _static_integral(key)
             for key, value in ci._pair_terms(bra, ket, labels, term_fn))
         for ket in dets] for bra in dets])
    np.testing.assert_allclose(actual, reference, atol=3e-14, rtol=0)
    np.testing.assert_allclose(actual, actual.T, atol=3e-14, rtol=0)
    off_diagonal = reference.copy()
    np.fill_diagonal(off_diagonal, 0.)
    assert np.max(np.abs(off_diagonal)) > 1e-4

    rng = np.random.default_rng(719)
    left, right = rng.normal(size=(2, len(dets)))
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    signed = left @ actual @ right
    assert signed == pytest.approx(left @ reference @ right, rel=0, abs=3e-14)
    assert abs(signed - np.abs(left) @ reference @ np.abs(right)) > 1e-5

    # Independently enumerated rows must also preserve first-encounter radial
    # key order and the exact sparse coefficients, not just one numeric H.
    monkeypatch.setattr(ci, '_connected_pairs', _brute_connections)
    expected_keys, expected_table, expected_coordinates = ci._coefficient_table(
        dets, labels, term_fn, radial_symmetry=radial_symmetry)
    assert keys == expected_keys
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(table.indptr, expected_table.indptr)
    np.testing.assert_array_equal(table.indices, expected_table.indices)
    np.testing.assert_array_equal(table.data, expected_table.data)


@pytest.mark.parametrize('dets, labels, term_fn', [
    ((), (), None),
    ((0,), (), None),
    ((1, 2), ((1, -1, -1), (1, -1, 1)), breit_terms),
])
@pytest.mark.parametrize('radial_symmetry', [False, True])
def test_empty_angular_tables_keep_shape_and_zero_operator(dets, labels, term_fn, radial_symmetry):
    keys, table, coordinates = ci._coefficient_table(
        dets, labels, term_fn, radial_symmetry=radial_symmetry)
    assert keys == ()
    assert table.shape == (0, 1)
    assert coordinates.shape == (0, 2)
    assert table.nnz == 0
    structure = SimpleNamespace(dets=dets, keys=keys, table=table,
                                pairs=coordinates, radial_symmetry=radial_symmetry)
    np.testing.assert_array_equal(ci.evaluate_block(structure, _static_integral),
                                  np.zeros((len(dets), len(dets))))


@pytest.mark.parametrize('radial_symmetry', [False, True])
def test_typed_table_matches_independent_coo_with_cancellation_and_key_reuse(
        monkeypatch, radial_symmetry):
    # Exercise sparse storage semantics directly, including exact zeros which
    # the old COO builder retained after duplicate summation. No angular cutoff
    # or implicit elimination of explicit zeros is part of the unsymmetric path.
    ka, kb = ('h', 1, -1, 2), ('h', 2, -1, 1)
    kc, kd = ('h', 2, -1, 2), ('h', 3, -1, 3)

    def raw_terms(bra, ket):
        if (bra, ket) == (3, 3):
            return ((ka, 1.), (kb, -1.))
        if bra != ket:
            return ((kc, 3.), (ka, -1.), (ka, .5), (ka, .5), (kd, 0.))
        return ()

    monkeypatch.setattr(ci, '_core_contracted_pair_terms', lambda *_args: raw_terms)
    keys, actual, coordinates = ci._coefficient_table(
        (3, 5), ((1, -1, -1), (1, -1, 1), (2, -1, -1)),
        radial_symmetry=radial_symmetry)
    if radial_symmetry:
        assert keys == (kc,)
        np.testing.assert_array_equal(coordinates, [[0, 1]])
        expected = csr_matrix([[3.]])
    else:
        assert keys == (ka, kb, kc, kd)
        np.testing.assert_array_equal(coordinates, [[0, 0], [0, 1], [1, 0]])
        expected = csr_matrix((
            [1., -1., 3., -1., .5, .5, 0., 3., -1., .5, .5, 0.],
            ([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             [0, 1, 2, 0, 0, 0, 3, 2, 0, 0, 0, 3])), shape=(3, 4))
        assert expected.nnz == 8
        assert np.count_nonzero(expected.data == 0.) == 4
    assert actual.has_canonical_format
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize('radial_symmetry', [False, True])
def test_real_angular_table_preserves_complex_radial_c_product(radial_symmetry):
    labels = ((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1))
    dets = tuple(_mask(*sites) for sites in combinations(range(4), 2))
    keys, table, coordinates = ci._coefficient_table(
        dets, labels, radial_symmetry=radial_symmetry)
    structure = SimpleNamespace(dets=dets, keys=keys, table=table,
                                pairs=coordinates, radial_symmetry=radial_symmetry)

    def radial(key):
        return (1. + .2j) * _static_integral(key)

    actual = ci.evaluate_block(structure, radial, integral_dtype=complex)
    expected = np.asarray([[sum(value * radial(key)
                                for key, value in ci._pair_terms(bra, ket, labels))
                            for ket in dets] for bra in dets])
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
    np.testing.assert_allclose(actual, actual.T, atol=3e-14, rtol=0)
    assert np.max(np.abs(actual - actual.conj().T)) > 1e-3


@pytest.mark.parametrize('complex_type', [complex, np.complex64, np.complex128])
@pytest.mark.parametrize('radial_symmetry', [False, True])
def test_typed_table_promotes_late_complex_coefficients_without_losing_phase(
        monkeypatch, complex_type, radial_symmetry):
    ka, kb, kc = (('h', n, -1, n) for n in (1, 2, 3))
    terms = {
        (3, 3): ((ka, 1.5), (kb, np.float32(-.5))),
        (3, 5): ((kc, complex_type(.25 + .5j)), (ka, -1j),
                 (kc, np.complex128(.5 - .25j))),
        (5, 3): ((kc, complex_type(.25 + .5j)), (ka, -1j),
                 (kc, np.complex128(.5 - .25j))),
        (5, 5): ((kb, 2.), (kc, -.25j)),
    }
    monkeypatch.setattr(ci, '_core_contracted_pair_terms',
                        lambda *_args: lambda bra, ket: terms[bra, ket])
    keys, actual, coordinates = ci._coefficient_table(
        (3, 5), ((1, -1, -1), (1, -1, 1), (2, -1, -1)),
        radial_symmetry=radial_symmetry)
    expected_coordinates = [(0, 0), (0, 1), (1, 1)] if radial_symmetry else [
        (0, 0), (0, 1), (1, 0), (1, 1)]
    columns, rows, values = [], [], []
    key_index = {ka: 0, kb: 1, kc: 2}
    for index, (row, col) in enumerate(expected_coordinates):
        for key, value in terms[(3, 5)[row], (3, 5)[col]]:
            rows.append(index)
            columns.append(key_index[key])
            values.append(value)
    expected = csr_matrix((values, (rows, columns)),
                          shape=(len(expected_coordinates), 3))
    assert keys == (ka, kb, kc)
    assert actual.dtype == np.dtype(complex)
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert np.max(np.abs(actual.data.imag)) == 1.
