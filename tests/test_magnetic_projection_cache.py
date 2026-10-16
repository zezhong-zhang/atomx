"""Exact bounded angular reuse through independent H and radial consumers."""

import numpy as np
import pytest
from test_ci_blocks import native_space as native_fixture

from atomx.angular.csf_hamiltonian import csf_signature
from atomx.angular.one_body_tensor import magnetic_csf_projection
from atomx.angular.projection_cache import MagneticCSFProjectionCache
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.blocked_functional import BlockedCSFOrbitalFunctional
from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian
from atomx.physics.csf.state_average import CIStateAverage

native_space = native_fixture


def test_basis_reuse_preserves_signed_root_changes_order_and_ownership():
    csfs = enumerate_jj_csfs('1s1 2s1')
    block = tuple(csf_signature(c) for c in csfs if c.total_J2 == 0)
    keys = ((2, -1), (1, -1))
    cache = MagneticCSFProjectionCache(1024**2)
    dets, index, basis = cache.project(block, 0, keys)
    expected_dets, _, expected = magnetic_csf_projection(block, 0, keys)
    assert dets == expected_dets
    np.testing.assert_array_equal(basis.toarray(), expected.toarray())
    for values in (basis.data, basis.indices, basis.indptr):
        with pytest.raises(ValueError):
            values.flags.writeable = True
    with pytest.raises(TypeError):
        index[dets[0]] = 99
    basis.data = np.zeros_like(basis.data)  # Wrapper changes cannot poison cache.
    coefficients = np.array([[1., -1e-20, 2j]])
    _, _, result = cache.project(block, 0, tuple(reversed(keys)), coefficients)
    np.testing.assert_array_equal(result, expected@coefficients)
    coefficients *= -1
    _, _, changed = cache.project(block, 0, keys, coefficients)
    np.testing.assert_array_equal(changed, -result)
    assert cache.diagnostics()['hits'] == 2 and cache.diagnostics()['misses'] == 1
    assert np.any(changed[:, 1] != 0)
    with pytest.raises(ValueError, match='complete signed'):
        cache.project(block, 0, keys, np.ones((2, 1)))


def test_bounded_cache_never_truncates_oversized_basis():
    block = tuple(csf_signature(c) for c in enumerate_jj_csfs('1s2'))
    cache = MagneticCSFProjectionCache(1)
    _, _, first = cache.project(block, 0, ((1, -1),))
    _, _, second = cache.project(block, 0, ((1, -1),))
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    assert cache.diagnostics() == dict(hits=0, misses=2, entries=0,
                                      accounted_resident_bytes=0, byte_budget=1)


def test_active_layout_matches_explicit_occupied_spinor_permutation():
    csfs = enumerate_jj_csfs('1s2 2p1 3d1')
    block = tuple(csf_signature(c) for c in csfs if c.total_J2 == 2)
    active = ((3, 2), (2, 1), (3, -3), (2, -2))
    frozen = ((1, -1),)
    keys = frozen+active
    def labels(order):
        return tuple((n, k, m) for n, k in order
                     for m in range(-2*abs(k)+1, 2*abs(k), 2))
    old = labels(sorted(keys))
    destination = {label: i for i, label in enumerate(labels(keys))}
    full_masks, _, full = magnetic_csf_projection(block, 0, keys)
    mapped, phases = [], []
    for mask in full_masks:
        occupied = [destination[label] for i, label in enumerate(old) if mask & (1 << i)]
        phases.append((-1)**sum(a > b for i, a in enumerate(occupied) for b in occupied[i+1:]))
        mapped.append(sum(1 << (i-2) for i in occupied if i >= 2))
    row_order = np.argsort(mapped)
    cache = MagneticCSFProjectionCache(1024**2)
    masks, basis = cache.project_active(block, 0, frozen_keys=frozen,
        active_keys=active, max_determinants=1000)
    assert masks == tuple(mapped[i] for i in row_order)
    expected = full.multiply(np.asarray(phases)[:, None]).tocsr()[row_order].toarray()
    np.testing.assert_array_equal(basis.toarray(), expected)
    with pytest.raises(ValueError, match='contracted frozen'):
        cache.project_active(block, 0, frozen_keys=((2, 1),),
            active_keys=((1, -1), (3, 2), (3, -3), (2, -2)), max_determinants=1000)


@pytest.mark.parametrize('phase', [1., -1.])
@pytest.mark.parametrize('breit', [False, True])
def test_every_native_signed_h_entry_matches_uncached_open_core(native_space, phase, breit):
    core, original, _ = native_space
    od = dict(original)
    od[2, -1] = tuple(phase*v for v in od[2, -1])
    csfs = tuple(c for config in ('1s2 2s2', '1s2 2p2', '1s1 2s1 2p2')
                 for c in enumerate_jj_csfs(config))
    kwargs = dict(reference_keys=((1, -1),), nuclear_potential=core._nuclear_potential,
                  include_breit=breit)
    ordinary = SpinorCIHamiltonian(csfs, od, core._bsp_data, 4, 30, **kwargs)
    cache = MagneticCSFProjectionCache(1024**2)
    cached = SpinorCIHamiltonian(csfs, od, core._bsp_data, 4, 30,
                                projection_cache=cache, **kwargs)
    for sector in sorted({(c.total_J2, c.parity) for c in csfs}):
        block = [c for c in csfs if (c.total_J2, c.parity) == sector]
        expected = ordinary.block_matrix(block, max_determinants=2000)
        for _ in range(2):
            np.testing.assert_array_equal(cached.block_matrix(block, max_determinants=2000), expected)
    assert cache.diagnostics()['hits'] > 0
    with pytest.raises(ValueError, match='determinant cap'):
        cached.block_matrix(csfs[:2], max_determinants=1)


def test_full_objective_and_every_gradient_coordinate_match_without_cache(native_space):
    core, od, csfs = native_space
    kwargs = dict(state_average=CIStateAverage(((0, 1, 0, 2.), (2, -1, 0, 1.))),
                  reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    ordinary = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4, projection_cache_bytes=0, **kwargs)
    cached = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4, **kwargs)
    e, derivative, trial = ordinary.evaluate(od)
    for _ in range(2):
        actual, gradient, current = cached.evaluate(od)
        assert actual == e and current.hamiltonian_hashes == trial.hamiltonian_hashes
        for key in derivative:
            np.testing.assert_array_equal(gradient[key], derivative[key])
    assert cached.projection_cache.diagnostics()['hits'] >= 6
    with pytest.raises(ValueError, match='cache budget'):
        BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4, projection_cache_bytes=-1, **kwargs)
    with pytest.raises(MemoryError):
        BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4, memory_bytes=1024, **kwargs)
