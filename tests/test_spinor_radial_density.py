"""Complete spinor densities versus radial-key and full small-Fock oracles."""

import math

import numpy as np
import pytest
from test_ci_numeric_streaming import _fock_oracle, _radial
from test_ci_numeric_streaming import radial_problem as native_fixture

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.physics.csf.radial_density import CIRadialDensity
from atomx.physics.csf.radial_symmetry import canonical_static_radial_key
from atomx.physics.csf.spinor_density import SpinorRadialDensity

radial_problem = native_fixture


def canonical(terms):
    result = {}
    for key, value in terms.items():
        key, sign = canonical_static_radial_key(key)
        result[key] = result.get(key, 0.)+sign*value
    return result


@pytest.mark.parametrize('configs', [
    ('2s2', '2s1 3s1', '2p2'),
    ('1s2 2s2', '1s2 2s1 3s1', '1s2 2p2'),
    ('2s2 2p2', '2s2 2p1 3p1', '2s2 3p2'),
    ('1s2',),
])
def test_every_canonical_key_signed_columns_and_independent_fock(configs):
    csfs = [c for cfg in configs for c in enumerate_jj_csfs(cfg)]
    csfs = csfs[::2]+csfs[1::2][::-1]
    reference = CIRadialDensity(csfs)
    density = SpinorRadialDensity(csfs)
    rng = np.random.default_rng(863)
    left, right = rng.normal(size=(2, len(csfs), 2))
    expected = reference.contract_bilinears(left, right)
    actual = density.contract_bilinears(left, right)
    for a, b in zip(actual, expected, strict=True):
        b = canonical(b)
        for key in a.keys() | b.keys():
            assert a.get(key, 0.) == pytest.approx(b.get(key, 0.), abs=2e-12, rel=2e-13)
    for _sector, _, indices in density.blocks:
        signatures = [density.signatures[i] for i in indices]
        oracle = _fock_oracle(signatures)
        block_l, block_r = np.zeros_like(left), np.zeros_like(right)
        block_l[list(indices)], block_r[list(indices)] = left[list(indices)], right[list(indices)]
        for k, row in enumerate(density.contract_bilinears(block_l, block_r)):
            value = math.fsum(v*_radial(key) for key, v in row.items())
            assert value == pytest.approx(left[list(indices), k]@oracle@right[list(indices), k], abs=2e-11)


def test_tiny_bilinears_and_actual_native_radial_gradient(request):
    csfs, waves, keys, data, Z, ns = request.getfixturevalue('radial_problem')
    reference = CSFOrbitalFunctional(csfs, data, Z)
    candidate = CSFOrbitalFunctional(csfs, data, Z, density_storage='spinor')
    vector = np.random.default_rng(381).normal(size=len(csfs))
    old = reference.radial_density.contract_bilinear(vector, vector)
    new = candidate.radial_density.contract_bilinear(vector, vector)
    old_g, new_g = reference.gradient_from_weights(waves, old), candidate.gradient_from_weights(waves, new)
    for key in keys:
        np.testing.assert_allclose(new_g[key], old_g[key], atol=1e-9, rtol=1e-12)
    tiny = candidate.radial_density.contract_bilinear(vector*1e-12, -2*vector*1e-12)
    for key, value in new.items():
        assert tiny.get(key, 0.) == pytest.approx(-2e-24*value, rel=2e-12, abs=1e-33)


def test_admission_and_invalid_bilinears():
    csfs = enumerate_jj_csfs('2p2')
    with pytest.raises(MemoryError, match='determinant'):
        SpinorRadialDensity(csfs, max_determinants=1)
    density = SpinorRadialDensity(csfs)
    with pytest.raises(ValueError):
        density.contract_bilinear(np.array([1.]), np.ones(len(csfs)))
    with pytest.raises(ValueError):
        SpinorRadialDensity([])
