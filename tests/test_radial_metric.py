import numpy as np
import pytest
from types import SimpleNamespace
from atomx.numerics.radial_metric import RadialMetric
from atomx.numerics.shooting.quadrature import slag6
from atomx.corrections.adapters import _grid_workspace_of
from atomx.results import OrbitalWavefunction, Waves


def test_native_metric_preserves_full_endpoint_stencil():
    r = np.linspace(0., 2., 31)
    jacobian = np.full(len(r), r[1]-r[0])
    f = r**4*np.exp(-r)
    metric = RadialMetric.shooting(r, jacobian)
    expected = slag6(1., f*jacobian, len(r))[-1]
    assert metric.integrate(f) == pytest.approx(expected, abs=2e-15)
    workspace = _grid_workspace_of(SimpleNamespace(_grid=r, _difr=jacobian))
    assert workspace['grw'] @ f[1:] == pytest.approx(expected, abs=2e-15)
    r[1] = 123.
    assert metric.radius[1] != 123.
    with pytest.raises(ValueError):
        metric.weights[1] = 0.


def test_observable_uses_native_quadrature_and_rejects_other_grid():
    r = np.linspace(0., 2., 31)
    metric = RadialMetric.shooting(r, np.full(len(r), r[1]))
    orbital = OrbitalWavefunction(r*np.exp(-r), None, 1, 0, None, 1., -.5)
    Waves(r, {'1s': orbital}, 'grid', metric)
    assert orbital.expectation_r(r) == pytest.approx(metric.integrate(orbital.P**2*r))
    with pytest.raises(ValueError, match='grid differs'):
        orbital.expectation_r(r*2)


def test_result_radial_snapshot_owns_native_spinors():
    from atomx.results import ShootingResult
    from atomx.methods.dirac_fock import Subshell
    from atomx.corrections.adapters import _dirac_orbital_payload
    r = np.linspace(0., 2., 31)
    p = (r*np.exp(-r))[None, :]
    result = ShootingResult(True, 1, -.5, {}, Z=1,
        subshells=[Subshell(1, -1, 1.)], _grid=r,
        _difr=np.full(len(r), r[1]), _pa=p, _qa=np.zeros_like(p))
    snapshot = result.radial_state
    before = snapshot.orbitals[(1, -1)][0].copy()
    p[:] = 999.
    np.testing.assert_array_equal(snapshot.orbitals[(1, -1)][0], before)
    orbitals, workspace = _dirac_orbital_payload(result)
    np.testing.assert_array_equal(orbitals[0][0], before[1:])
    np.testing.assert_array_equal(workspace['grw'], snapshot.metric.weights[1:])
