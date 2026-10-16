"""Numeric streamed CI preserves the complete signed physical operator.

The small-Fock oracle enumerates all creation/annihilation products, without
the production connection filter, Slater--Condon cases or frozen-core
contraction. Angular matrix elements and CSF vectors remain shared; these
tests certify the new assembly/storage path, not an independent angular theory.
"""

import hashlib
from collections import Counter
from functools import lru_cache
from itertools import combinations, product

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from atomx.angular import csf_hamiltonian as exact
from atomx.config.csf import enumerate_jj_csfs
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf import matrix
from atomx.physics.csf.breit_ci import breit_terms
from atomx.physics.csf.radial_symmetry import canonical_static_radial_key


def _signatures(configs, j2):
    signatures = [exact.csf_signature(csf) for cfg in configs
                  for csf in enumerate_jj_csfs(cfg) if csf.total_J2 == j2]
    # Interleave configurations and reverse repeated-parentage columns.
    return signatures[::2] + signatures[1::2][::-1]


@lru_cache(None)
def _radial(key):
    canonical, sign = canonical_static_radial_key(key)
    if sign == 0:
        return 0.
    digest = hashlib.sha256(repr(canonical).encode()).digest()
    value = .2 + int.from_bytes(digest[:4], 'little') / 2**32
    return sign * (-value if key[0] == 'h' else .13 * value)


def _apply(occupied, operations):
    """Act right to left using sorted exterior products, independently."""
    occupied, phase = list(occupied), 1
    for site, creating in operations:
        position = sum(other < site for other in occupied)
        if (site in occupied) == creating:
            return None
        phase *= (-1)**position
        if creating:
            occupied.insert(position, site)
        else:
            occupied.pop(position)
    return tuple(occupied), phase


def _fock_oracle(signatures, *, breit=False, complex_radials=False):
    patterns = sorted({sig.pattern for sig in signatures})
    keys = sorted({(n, k) for pattern in patterns for n, k, _ in pattern})
    labels = tuple((n, k, m) for n, k in keys
                   for m in range(-(2*abs(k)-1), 2*abs(k), 2))
    sites = {key: [i for i, lab in enumerate(labels) if lab[:2] == key]
             for key in keys}
    determinants = []
    for pattern in patterns:
        occupants = (combinations(sites[n, k], q) for n, k, q in pattern)
        here = [tuple(sorted(sum(parts, ()))) for parts in product(*occupants)
                if sum(labels[i][2] for part in parts for i in part)
                == signatures[0].total_J2]
        determinants.extend(sorted(here, key=lambda occ: sum(1 << i for i in occ)))
    row_of = {occupied: i for i, occupied in enumerate(determinants)}
    factor = 1 + .23j if complex_radials else 1.

    @lru_cache(None)
    def two_body(a, b, c, d):
        na, ka, ma = labels[a]
        nb, kb, mb = labels[b]
        nc, kc, mc = labels[c]
        nd, kd, md = labels[d]
        terms = exact.two_body_angular(ka, ma, kb, mb, kc, mc, kd, md)
        value = sum(angular * _radial(('r', rank, (na, ka), (nc, kc),
                                             (nb, kb), (nd, kd)))
                    for rank, angular in terms)
        if breit:
            value += sum(angular * _radial(key)
                         for key, angular in breit_terms(a, b, c, d, 1., labels))
        return factor * value

    h = np.zeros((len(determinants), len(determinants)),
                 dtype=complex if complex_radials else float)
    for col, occupied in enumerate(determinants):
        for c in occupied:
            nc, kc, mc = labels[c]
            for a, (na, ka, ma) in enumerate(labels):
                acted = _apply(occupied, ((c, False), (a, True)))
                if acted is not None and acted[0] in row_of and (kc, mc) == (ka, ma):
                    h[row_of[acted[0]], col] += acted[1] * factor * _radial(
                        ('h', na, ka, nc))
        # H2 = 1/2 sum_abcd <ab|v|cd> a+_a a+_b a_d a_c.
        for c, d in product(occupied, repeat=2):
            if c == d:
                continue
            for a, b in product(range(len(labels)), repeat=2):
                acted = _apply(occupied, ((c, False), (d, False),
                                          (b, True), (a, True)))
                if acted is not None and acted[0] in row_of:
                    h[row_of[acted[0]], col] += .5 * acted[1] * two_body(a, b, c, d)
    groups = [[sig for sig in signatures if sig.pattern == p] for p in patterns]
    projection = exact.csf_block_matrix(patterns, signatures[0].total_J2, groups)
    grouped = [sig for group in groups for sig in group]
    order = [grouped.index(sig) for sig in signatures]
    projection = projection[:, order]
    np.testing.assert_allclose(projection.T @ projection, np.eye(len(signatures)),
                               atol=2e-13, rtol=0)
    return projection.T @ h @ projection


def _reject_tables(monkeypatch):
    def reject(*_args, **_kwargs):
        raise AssertionError('streamed numeric assembly requested a coefficient table')
    for name in ('_coefficient_table', 'block_terms', 'breit_block_terms'):
        monkeypatch.setattr(exact, name, reject)


@pytest.mark.parametrize('complex_radials', [False, True])
@pytest.mark.parametrize('radial_symmetry', [False, True])
@pytest.mark.parametrize('breit', [False, True], ids=['DC', 'static_Breit'])
def test_complete_signed_operator_matches_explicit_fock_sum(
        monkeypatch, complex_radials, radial_symmetry, breit):
    signatures = _signatures(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'), 0)
    reference = _fock_oracle(signatures, breit=breit, complex_radials=complex_radials)
    factor = 1 + .23j if complex_radials else 1.

    def evaluate(key):
        return factor * _radial(key)

    options = dict(integral_dtype=complex if complex_radials else float,
                   radial_symmetry=radial_symmetry,
                   breit_evaluator=evaluate if breit else None)
    cached = exact.exact_block_matrix(signatures, evaluate, **options)
    _reject_tables(monkeypatch)
    streamed = exact.exact_block_matrix(signatures, evaluate,
                                       angular_storage='streamed', **options)
    operator = exact.exact_block_matrix(signatures, evaluate, as_operator=True,
                                       angular_storage='streamed', **options)
    assert isinstance(operator, LinearOperator)
    for actual in (cached, streamed, operator @ np.eye(len(signatures))):
        np.testing.assert_allclose(actual, reference, atol=3e-13, rtol=0)
    np.testing.assert_allclose(operator.diagonal(), np.diag(reference), atol=3e-13, rtol=0)
    # A mutable diagonal view must not alter the operator's retained diagonal.
    operator.diagonal()[:] = 0
    np.testing.assert_allclose(operator.diagonal(), np.diag(reference), atol=3e-13, rtol=0)
    rng = np.random.default_rng(829)
    left, right = rng.normal(size=(2, len(signatures)))
    if complex_radials:
        left = left + .31j * right
    np.testing.assert_allclose(operator.rmatvec(left), reference.conj().T @ left,
                               atol=3e-13, rtol=0)
    np.testing.assert_allclose(left @ (operator @ right), left @ reference @ right,
                               atol=3e-13, rtol=0)
    assert abs(left @ reference @ right - abs(left) @ reference @ abs(right)) > 1e-4
    np.testing.assert_allclose(streamed, streamed.T, atol=3e-13, rtol=0)
    if complex_radials:
        assert np.max(abs(streamed - streamed.conj().T)) > .01


def test_multiple_parentages_and_reordered_columns_keep_full_projected_diagonal(monkeypatch):
    signatures = _signatures(('1s2 2s1 2p1 3s1', '1s2 2s1 2p1 4s1'), 1)
    assert max(Counter(sig.pattern for sig in signatures).values()) > 1
    reference = _fock_oracle(signatures)
    _reject_tables(monkeypatch)
    operator = exact.exact_block_matrix(signatures, _radial, as_operator=True,
                                       angular_storage='streamed')
    np.testing.assert_allclose(operator @ np.eye(len(signatures)), reference,
                               atol=3e-13, rtol=0)
    np.testing.assert_allclose(operator.diagonal(), np.diag(reference), atol=3e-13, rtol=0)


def test_each_radial_key_is_evaluated_once_per_operator_and_not_cached_between_trials(monkeypatch):
    signatures = _signatures(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'), 0)
    _reject_tables(monkeypatch)
    outputs, used = [], []
    for scale in (1., 1.7):
        calls = Counter()

        def evaluate(key, calls=calls, scale=scale):
            calls[key] += 1
            return scale * _radial(key)

        operator = exact.exact_block_matrix(signatures, evaluate, as_operator=True,
                                           angular_storage='streamed', radial_symmetry=True,
                                           breit_evaluator=evaluate)
        before = calls.copy()
        outputs.append(operator @ np.eye(len(signatures)))
        operator.rmatvec(np.ones(len(signatures)))
        operator.diagonal()
        assert calls == before
        assert calls and all(count == 1 for count in calls.values())
        assert all(canonical_static_radial_key(key) == (key, 1) for key in calls)
        used.append(set(calls))
    assert used[0] == used[1]
    np.testing.assert_allclose(outputs[1], 1.7 * outputs[0], atol=3e-13, rtol=0)


@pytest.mark.parametrize('complex_radials', [False, True])
def test_default_radial_contract_keeps_distinct_nonreciprocal_keys(monkeypatch, complex_radials):
    # An arbitrary evaluator has not promised permutation symmetries. It must
    # keep both orientations until the existing final transpose averaging.
    signatures = _signatures(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'), 0)
    calls = Counter()

    def evaluate(key):
        calls[key] += 1
        digest = hashlib.sha256(repr(key).encode()).digest()
        return (.1 + int.from_bytes(digest[:4], 'little') / 2**32) * (
            1 + .17j if complex_radials else 1.)

    options = dict(integral_dtype=complex if complex_radials else float,
                   breit_evaluator=evaluate)
    reference = exact.exact_block_matrix(signatures, evaluate, **options)
    calls.clear()
    _reject_tables(monkeypatch)
    operator = exact.exact_block_matrix(signatures, evaluate, angular_storage='streamed',
                                       as_operator=True, **options)
    np.testing.assert_allclose(operator @ np.eye(len(signatures)), reference,
                               atol=3e-13, rtol=0)
    np.testing.assert_allclose(operator.diagonal(), np.diag(reference), atol=3e-13, rtol=0)
    assert calls and all(count == 1 for count in calls.values())
    assert any(canonical_static_radial_key(key) != (key, 1) for key in calls)


@pytest.mark.parametrize('invalid', ['automatic', '', None, True, 1])
def test_invalid_storage_rejects_before_space_or_integral_work(monkeypatch, invalid):
    signatures = _signatures(('1s2',), 0)

    def reject(*_args, **_kwargs):
        raise AssertionError('invalid storage reached physical assembly')

    monkeypatch.setattr(exact, 'pattern_layout', reject)
    with pytest.raises(ValueError, match='angular_storage'):
        exact.exact_block_matrix(signatures, reject, angular_storage=invalid)


def test_raw_determinant_cap_rejects_before_pair_or_radial_work(monkeypatch):
    signatures = _signatures(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'), 0)

    def reject(*_args, **_kwargs):
        raise AssertionError('capped space reached pair or radial assembly')

    monkeypatch.setattr(exact, '_core_contracted_pair_terms', reject)
    with pytest.raises(exact.ExactCISpaceError, match='determinant|exceed'):
        exact.exact_block_matrix(signatures, reject, max_determinants=1,
                                 angular_storage='streamed')


@pytest.mark.parametrize('value', [np.nan, np.inf, np.array([1.]), 1. + .1j])
def test_invalid_radial_value_rejects_explicitly(value):
    signatures = _signatures(('1s2',), 0)
    with pytest.raises(ValueError, match='nonfinite|nonscalar|complex'):
        exact.exact_block_matrix(signatures, lambda _key: value, angular_storage='streamed')


@pytest.mark.parametrize('dtype', [int, np.float32, np.complex64])
def test_invalid_numeric_representation_rejects(dtype):
    with pytest.raises(ValueError, match='integral_dtype'):
        exact.exact_block_matrix([], _radial, integral_dtype=dtype, angular_storage='streamed')


@pytest.fixture(scope='module')
def radial_problem():
    """Actual common orthonormal RKB waves, without an SCF or external backend."""
    ns, charge = 35, 4
    data = prepare_bspline_data(make_knots(ns, 9, .005, 30), ns, 9,
                               kinetic_balance=True)
    configs = ('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2')
    csfs = [csf for cfg in configs for csf in enumerate_jj_csfs(cfg)][::-1]
    keys = sorted({(sh.n, sh.kappa) for csf in csfs for sh in csf.subshells})
    waves = {}
    for kappa in sorted({k for _, k in keys}):
        spectrum = solve_dirac(*build_dirac_matrices(ns, kappa, charge, data),
                               ns, kappa, nsp=data['nsp'], nsq=data['nsq'])
        ell = kappa if kappa > 0 else -kappa-1
        for n, k in keys:
            if k == kappa:
                p, q, _ = spectrum.get_orbital(n-ell-1)
                waves[n, k] = p, q
    return csfs, waves, keys, data, charge, ns


@pytest.mark.parametrize('include_breit', [False, True])
@pytest.mark.parametrize('as_operator', [False, True])
def test_public_builder_preserves_symmetry_blocks_and_actual_radial_phases(
        radial_problem, monkeypatch, include_breit, as_operator):
    csfs, waves, keys, data, charge, ns = radial_problem
    args = csfs, waves, keys, data, charge, ns
    expected = matrix.build_ci_matrix(*args, include_breit=include_breit)
    _reject_tables(monkeypatch)
    actual = matrix.build_ci_matrix(*args, include_breit=include_breit,
                                    as_operator=as_operator, angular_storage='streamed')
    if as_operator:
        np.testing.assert_allclose(actual.diagonal(), np.diag(expected), atol=2e-11, rtol=0)
        actual = actual @ np.eye(len(csfs))
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    for a, left in enumerate(csfs):
        for b, right in enumerate(csfs):
            if (left.total_J2, left.parity) != (right.total_J2, right.parity):
                assert actual[a, b] == 0.
    changed = dict(waves)
    changed[3, -1] = tuple(-part for part in waves[3, -1])
    rephased = matrix.build_ci_matrix(csfs, changed, keys, data, charge, ns,
                                      include_breit=include_breit, angular_storage='streamed')
    phases = np.asarray([(-1.)**sum(int(sh.occupation) for sh in csf.subshells
                                    if (sh.n, sh.kappa) == (3, -1)) for csf in csfs])
    np.testing.assert_allclose(rephased, phases[:, None] * expected * phases[None, :],
                               atol=2e-11, rtol=0)
    assert np.max(abs(rephased-expected)) > 1e-4


def test_public_invalid_storage_rejects_before_radial_reset(monkeypatch):
    def reject(*_args, **_kwargs):
        raise AssertionError('invalid storage reached radial reset')

    monkeypatch.setattr(matrix, '_reset_memo_for_ci_build', reject)
    with pytest.raises(ValueError, match='angular_storage'):
        matrix.build_ci_matrix([], {}, None, None, 1, 1, angular_storage='automatic')
