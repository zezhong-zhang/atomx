"""Complete signed radial bilinears without a retained angular table.

The independent small-Fock oracle enumerates creation/annihilation actions;
cached radial-key controls additionally check every coefficient, including
tiny signed terms, rather than just one total energy.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from test_ci_numeric_streaming import (  # noqa: F401
    _fock_oracle,
    _radial,
    radial_problem,
)

from atomx.angular import csf_hamiltonian as exact
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.physics.csf.radial_density import CIRadialDensity


def _csfs(configs):
    csfs = [csf for config in configs for csf in enumerate_jj_csfs(config)]
    return csfs[::2] + csfs[1::2][::-1]


def _reject_tables(monkeypatch):
    def reject(*_args, **_kwargs):
        raise AssertionError('streamed density requested an angular table')

    monkeypatch.setattr(exact, 'block_terms', reject)
    monkeypatch.setattr(exact, '_coefficient_table', reject)


def _compare(actual, reference, atol=1e-12):
    for key in actual.keys() | reference.keys():
        assert actual.get(key, 0.) == pytest.approx(reference.get(key, 0.), abs=atol, rel=0)


@pytest.mark.parametrize('configs', [
    ('2s2', '2s1 3s1', '2p2'),
    ('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'),
    ('1s2 2s1 2p1 3s1', '1s2 2s1 2p1 4s1'),
])
def test_every_signed_key_and_independent_Fock_bilinear(configs, monkeypatch):
    csfs = _csfs(configs)
    cached = CIRadialDensity(csfs)
    rng = np.random.default_rng(1008)
    left, right = rng.normal(size=(2, len(csfs), 3))
    left[:, 1] *= -2.1
    right[:, 2] *= .003
    expected = cached.contract_bilinears(left, right)
    _reject_tables(monkeypatch)
    streamed = CIRadialDensity(csfs, angular_storage='streamed')
    actual = streamed.contract_bilinears(left, right)
    for state, reference in zip(actual, expected, strict=True):
        _compare(state, reference)
    for symmetry in sorted({(csf.total_J2, csf.parity) for csf in csfs}):
        indices = [i for i, csf in enumerate(csfs)
                   if (csf.total_J2, csf.parity) == symmetry]
        signatures = [exact.csf_signature(csfs[i]) for i in indices]
        oracle = _fock_oracle(signatures)
        block_left, block_right = np.zeros_like(left), np.zeros_like(right)
        block_left[indices], block_right[indices] = left[indices], right[indices]
        weights = streamed.contract_bilinears(block_left, block_right)
        for k, terms in enumerate(weights):
            value = math.fsum(coefficient * _radial(key) for key, coefficient in terms.items())
            assert value == pytest.approx(left[indices, k] @ oracle @ right[indices, k],
                                          rel=0, abs=1e-11)
    # Removing coherent signs is measurably different for this full space.
    signed = streamed.contract_bilinear(left[:, 0], right[:, 0])
    unsigned = streamed.contract_bilinear(abs(left[:, 0]), abs(right[:, 0]))
    assert abs(math.fsum((signed.get(key, 0.) - unsigned.get(key, 0.)) * _radial(key)
                        for key in signed.keys() | unsigned.keys())) > 1e-4


def test_batch_traverses_each_symmetry_once_and_keeps_tiny_terms(monkeypatch):
    csfs = _csfs(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'))
    cached = CIRadialDensity(csfs)
    vector = np.random.default_rng(606).normal(size=len(csfs)) * 1e-11
    expected = cached.contract_bilinear(vector, vector)
    assert expected and max(abs(value) for value in expected.values()) < 1e-15
    _reject_tables(monkeypatch)
    original, traversals = exact._connected_pairs, []

    def counted(dets, **kwargs):
        traversals.append(len(dets))
        yield from original(dets, **kwargs)

    monkeypatch.setattr(exact, '_connected_pairs', counted)
    density = CIRadialDensity(csfs, angular_storage='streamed')
    terms = density.contract_bilinears(np.column_stack((vector, -3 * vector)),
                                      np.column_stack((vector, 2 * vector)))
    assert len(traversals) == len(density.blocks)
    for key, value in expected.items():
        assert terms[0][key] == pytest.approx(value, rel=2e-14, abs=1e-35)
        assert terms[1][key] == pytest.approx(-6 * value, rel=2e-14, abs=1e-35)
    assert density.contract_bilinear(np.zeros_like(vector), vector) == {}


@pytest.mark.parametrize('storage', ['cached', 'streamed'])
def test_explicit_ensemble_and_ground_weights_match_bilinears(storage):
    csfs = _csfs(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'))
    density = CIRadialDensity(csfs, angular_storage=storage)
    vectors = np.random.default_rng(92).normal(size=(len(csfs), 3))
    result = SimpleNamespace(state_energies=np.array([-3., -3., 1.]),
                             ground_energy=-3., mixing_coefficients=vectors)
    terms = density.contract_bilinears(vectors, vectors)
    for populations in (None, np.array([.2, .3, .5])):
        weights = np.array([.5, .5, 0.]) if populations is None else populations
        expected = {key: math.fsum(w * state.get(key, 0.)
                                   for w, state in zip(weights, terms, strict=True))
                    for key in set().union(*terms)}
        _compare(density.contract(result, populations), expected)


def test_native_full_radial_gradient_matches_cached(request, monkeypatch):
    csfs, waves, keys, data, charge, ns = request.getfixturevalue('radial_problem')
    cached = CSFOrbitalFunctional(csfs, data, charge)
    vector = np.random.default_rng(985).normal(size=len(csfs))
    reference_weights = cached.radial_density.contract_bilinear(vector, vector)
    reference_gradient = cached.gradient_from_weights(waves, reference_weights)
    _reject_tables(monkeypatch)
    streamed = CSFOrbitalFunctional(csfs, data, charge, density_storage='streamed')
    weights = streamed.radial_density.contract_bilinear(vector, vector)
    _compare(weights, reference_weights)
    gradient = streamed.gradient_from_weights(waves, weights)
    assert len(gradient) == len(keys)
    for key in keys:
        assert gradient[key].shape == (2 * ns,)
        np.testing.assert_allclose(gradient[key], reference_gradient[key], atol=1e-9, rtol=0)


@pytest.mark.parametrize('invalid', ['', 'automatic', None, 1, True])
def test_invalid_storage_rejects_before_basis_work(monkeypatch, invalid):
    _reject_tables(monkeypatch)
    monkeypatch.setattr(exact, '_block_basis', lambda *_a: pytest.fail('basis work'))
    with pytest.raises(ValueError, match='angular_storage'):
        CIRadialDensity([], angular_storage=invalid)


def test_cap_rejects_before_connections_or_projection(monkeypatch):
    csfs = _csfs(('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'))
    _reject_tables(monkeypatch)
    monkeypatch.setattr(exact, '_connected_pairs', lambda *_a: pytest.fail('pair work'))
    monkeypatch.setattr(exact, 'csf_block_matrix', lambda *_a, **_kw: pytest.fail('projection work'))
    with pytest.raises(exact.ExactCISpaceError, match='determinant'):
        CIRadialDensity(csfs, angular_storage='streamed', max_determinants=1)


@pytest.mark.parametrize('storage', ['cached', 'streamed'])
@pytest.mark.parametrize('invalid', [np.array([1.]), np.ones((1, 0)),
                                     np.ones((1, 1), dtype=complex), np.array([[np.nan]])])
def test_invalid_column_vectors_reject(storage, invalid):
    density = CIRadialDensity(enumerate_jj_csfs('1s2'), angular_storage=storage)
    with pytest.raises(ValueError, match='finite real CSF column arrays'):
        density.contract_bilinears(invalid, invalid)
