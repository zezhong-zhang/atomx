"""Dense continuum samples must retain the ODE solution and normalization."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn
from atomx.constants import C
from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_shooting, solve_dirac_continuum_many)


@pytest.mark.parametrize('energy', [.01, 10., 7500.])
@pytest.mark.parametrize('kappa', [-1, 1, -3, 5])
@pytest.mark.parametrize('points', [100, 1000])
def test_free_wave_dense_samples_against_analytic_bessel(energy, kappa, points):
    # Includes the singular-origin interval, an irregularly sampled exterior,
    # and coarse intervals requiring the adaptive fallback at high energy.
    target = np.r_[0., np.geomspace(1e-8, 20., 8000)]
    r, p, q, _ = solve_dirac_continuum_shooting(
        energy, kappa, lambda r: np.zeros_like(r), r_max=20., n_grid=points,
        interp_grid=target)
    np.testing.assert_array_equal(r, target)
    momentum = np.sqrt(energy*(2+energy/C**2))
    x = momentum*r
    ell = kappa if kappa > 0 else -kappa-1
    small_ell = ell-1 if kappa > 0 else ell+1
    factor = np.sqrt(energy/(energy+2*C*C))*(1 if kappa > 0 else -1)
    reference = np.array([x*spherical_jn(ell, x), factor*x*spherical_jn(small_ell, x)])
    amplitude = np.sqrt((2+energy/C**2)/(np.pi*momentum))
    actual = np.array([p, q])/amplitude
    sign = 1 if np.vdot(actual, reference) >= 0 else -1
    np.testing.assert_allclose(actual, sign*reference, rtol=0, atol=2e-9)


def test_threaded_dense_samples_match_scalar_and_return_output_grid():
    target = np.r_[0., np.geomspace(1e-7, 20., 2000)]
    def potential(r):
        return -(17.+9.*np.exp(-r/.3))/r
    requests = [(-1, .1), (1, 10.), (-3, 150.), (5, 10.)]
    r, waves = solve_dirac_continuum_many(
        requests, potential, r_max=20., n_grid=1000, Z_residual=17.,
        interp_grid=target, workers=2)
    np.testing.assert_array_equal(r, target)
    for k, e in requests:
        _, p, q, _ = solve_dirac_continuum_shooting(
            e, k, potential, r_max=20., n_grid=1000, Z_residual=17., interp_grid=target)
        np.testing.assert_allclose(waves[k, e], [p, q], rtol=2e-13, atol=2e-13)


def test_reusable_batch_preserves_waves_refreshes_field_and_releases_workers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    from atomx.numerics.shooting.dirac_continuum import DiracContinuumBatch
    pools = []

    class CountedPool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed_by_owner = False
            pools.append(self)

        def shutdown(self, *args, **kwargs):
            self.closed_by_owner = True
            return super().shutdown(*args, **kwargs)

    target = np.geomspace(1e-7, 20., 100)
    requests = [(-1, .1), (1, 10.), (-3, 150.)]
    charge = [2.]
    def potential(r):
        return -charge[0]/r

    options = dict(r_max=20., n_grid=500, workers=2, interp_grid=target)
    expected = {}
    for z in [2., 3.]:
        charge[0] = z
        expected[z] = solve_dirac_continuum_many(requests, potential, **options)[1]
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', CountedPool)
    with DiracContinuumBatch(potential, **options) as batch:
        for z in [2., 3.]:
            charge[0] = z
            _, actual = batch.solve_many(requests)
            for req in requests:
                np.testing.assert_array_equal(actual[req], expected[z][req])
        assert len(pools) == 1 and not pools[0].closed_by_owner
    assert pools[0].closed_by_owner
    with pytest.raises(RuntimeError, match='closed'):
        batch.solve_many(requests)


def test_reusable_batch_releases_workers_after_wave_failure(monkeypatch):
    import atomx.numerics.shooting.dirac_continuum as continuum
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    closed = []

    class CountedPool(ThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            closed.append(True)
            return super().shutdown(*args, **kwargs)

    def fail(*args, **kwargs):
        raise FloatingPointError('unresolved wave')

    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', CountedPool)
    monkeypatch.setattr(continuum, '_solve_prepared', fail)
    with pytest.raises(RuntimeError, match='kappa=-1, E=1.0 Ha'):
        with continuum.DiracContinuumBatch(lambda r: -2/r, workers=2) as batch:
            batch.solve_many([(-1, 1.), (1, 2.), (-3, 3.)])
    assert closed


@pytest.mark.parametrize('target', [[-.1, 1.], [0., 21.], [0., float('nan')], [2., 1.]])
def test_sampling_rejects_clamping_extrapolation_and_invalid_grids(target):
    with pytest.raises(ValueError, match='output grid'):
        solve_dirac_continuum_shooting(1., -1, lambda r: -1/r,
                                       r_max=20., interp_grid=target)
    with pytest.raises(ValueError, match='output grid'):
        solve_dirac_continuum_many([(-1, 1.)], lambda r: -1/r,
                                   r_max=20., interp_grid=target)


@pytest.mark.parametrize('workers', [1, 2])
def test_batch_preserves_wave_failure_context(monkeypatch, workers):
    import atomx.numerics.shooting.dirac_continuum as continuum
    def fail(*args, **kwargs):
        raise FloatingPointError('unresolved radial solution')
    monkeypatch.setattr(continuum, '_solve_prepared', fail)
    with pytest.raises(RuntimeError, match='kappa=-1, E=1.0 Ha') as error:
        continuum.solve_dirac_continuum_many(
            [(-1, 1.), (1, 2.), (-3, 3.)], lambda r: -1/r, workers=workers)
    assert isinstance(error.value.__cause__, FloatingPointError)


@pytest.mark.parametrize('kappa', [-65, 64])
@pytest.mark.parametrize('energy', [7.380170478971869e-5, 3.9410606013154137])
def test_high_partial_wave_dense_output_preserves_tiny_regular_solution(kappa, energy):
    from atomx.numerics.shooting.dirac_coulomb import dirac_coulomb_outgoing
    # At the inner mesh nodes both stored components underflow. The old
    # sampler entered an infinite step-halving loop for the low-energy case.
    reference_radii = np.array([.05, .1, 1., 10., 50.])
    target = np.unique(np.r_[np.geomspace(1e-7, 50., 1000), reference_radii])
    _, p, q, _ = solve_dirac_continuum_shooting(
        energy, kappa, lambda r: -2/r, r_max=50., n_grid=3000,
        Z_residual=2., interp_grid=target)
    amplitude = np.sqrt((2+energy/C**2)/(np.pi*np.sqrt(energy*(2+energy/C**2))))
    reference = amplitude*np.array([
        dirac_coulomb_outgoing(energy, kappa, 2., r).imag for r in reference_radii]).T
    actual = np.array([p, q])[:, np.searchsorted(target, reference_radii)]
    np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=0.)
    assert np.isfinite(p).all() and np.isfinite(q).all()


def test_zero_middle_interval_is_not_confused_with_regular_origin_boundary():
    from atomx.numerics.shooting.dirac_power_series import dir0
    from atomx.numerics.shooting.dirac_integrator import dir_integrate
    args = (1., -65., 1e-10, 0., 0.)
    p, q, ca, cb, _, _, terms = dir0(*args, 1., 2., -2., 0., 0., 0.)
    assert p == q == 0. and terms < 60
    assert not ca.any() and not cb.any()
    result = dir_integrate(1., -65., 1e-10, 0, 0., 0., 1., 2., -2., 0., 0., 0.)
    assert result[0] == result[1] == 0. and result[4] == 1
    # P(0)=Q(0)=0 selects a nonzero regular Frobenius solution at the origin.
    assert dir0(*args, 0., 1e-7, -2., 0., 0., 0.)[0] != 0.


REFERENCE = json.loads((Path(__file__).parent/'data/radial_screened_continuum_reference.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'])
def test_screened_dense_output_against_external_radial_dfree(case):
    assert C == REFERENCE['speed_of_light']
    samples = np.array(case['samples'])
    target, reference = samples[:, 0], samples[:, 1:].T
    def potential(r):
        return -(case['tail']+(case['z']-case['tail'])*np.exp(-r/case['scale']))/r
    energy = case['energy']
    _, p, q, phase = solve_dirac_continuum_shooting(
        energy, case['kappa'], potential, Z_residual=case['tail'],
        r_max=REFERENCE['r_max'], n_grid=REFERENCE['n_grid'], interp_grid=target)
    amplitude = np.sqrt((2+energy/C**2)/(np.pi*np.sqrt(energy*(2+energy/C**2))))
    actual = np.array([p, q])/amplitude
    sign = 1 if np.vdot(actual, reference) >= 0 else -1
    np.testing.assert_allclose(actual, sign*reference, rtol=0, atol=3e-8)
    assert abs(np.arctan(np.tan(phase-case['reference_phase']))) < 3e-8


@pytest.mark.parametrize('kappa', [-1, 1, -3, 5])
def test_adaptive_origin_sampling_preserves_regular_solution_scale(kappa):
    from atomx.numerics.shooting.dirac_sampling import sample_dirac_solution
    from atomx.numerics.shooting.dirac_power_series import dir0
    # A constant deep potential is an exactly soluble numerical stress case.
    # Its first interval contains many oscillations and cannot use one series.
    energy, depth = 1., 1e7
    r = np.array([0., .001, .002])
    target = np.r_[0., np.geomspace(1e-9, .002, 500)]
    effective = energy+depth
    momentum = np.sqrt(effective*(2+effective/C**2))
    ell = kappa if kappa > 0 else -kappa-1
    small = ell-1 if kappa > 0 else ell+1
    factor = np.sqrt(effective/(effective+2*C*C))*(1 if kappa > 0 else -1)
    def reference(radius):
        x = momentum*radius
        return np.array([x*spherical_jn(ell, x), factor*x*spherical_jn(small, x)])
    p, q = reference(r)
    assert dir0(energy, float(kappa), 1e-10, 0., 0., 0., r[1],
                0., -depth, 0., 0.)[-1] >= 60
    actual = sample_dirac_solution(target, r, p, q, np.zeros(3, dtype=np.int64),
        np.zeros(1), np.array([-depth]), np.zeros(1), np.zeros(1), energy, kappa, 1e-10)
    np.testing.assert_allclose(actual, reference(target), rtol=0, atol=2e-8)
