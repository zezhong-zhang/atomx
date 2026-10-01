"""Physical and public API controls for the full shooting migration."""
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from atomx import Atom, Basis, Corrections, ShootingSpace
from atomx.corrections.adapters import NormalMassShiftCorrection, UehlingVPCorrection
from atomx.corrections.stack import CorrectionStack
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.dirac_green import _scaled_slag6
from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.grid import sgrid


def test_scaled_quadrature_survives_amplitudes_beyond_float_range():
    logs = np.arange(8001)*.1
    integral = _scaled_slag6(np.ones(len(logs)), logs)
    expected = -np.expm1(-logs)/.1
    np.testing.assert_allclose(integral[5:], expected[5:], rtol=2e-8)
    np.testing.assert_allclose(_scaled_slag6(np.ones(len(logs)), logs+10000.),
                               integral, rtol=2e-11)


def test_full_box_response_retains_a_long_range_exchange_source():
    r, dr, n = sgrid(60., 1e-7, .0125, 8000)
    p = r*r*np.exp(-r/4)*(1-r/r[-1])
    dp = np.exp(-r/4)*(2*r-r*r/4-(3*r*r-r**3/4)/r[-1])
    q, dq = -.01*p, -.01*dp
    inv = np.divide(1., r, out=np.zeros(n), where=r > 0)
    e, kappa, z = -200., -1, 10.
    source = np.array([(-z*inv-e)*p+SL*(-dq+kappa*q*inv),
        SL*(dp+kappa*p*inv)+(-z*inv-2*SL**2-e)*q])
    response = DiracResponse(e, kappa, r, dr, np.full(n, -z), boundary='box')
    actual = response.apply(source)
    expected = np.array([p, q])
    error = np.sqrt(np.sum((actual-expected)**2*response.weights)
                    /np.sum(expected**2*response.weights))
    assert error < 2e-6
    # A core's practical infinity is inappropriate for this nonlocal source.
    with pytest.raises(ValueError, match='beyond practical infinity'):
        DiracResponse(e, kappa, r, dr, np.full(n, -z)).apply(source)


@pytest.fixture(scope='module')
def full_atom():
    controls = ShootingSpace(n_grid=2400, bound_roots=2, continuum_roots=64,
                             outer_intervals=1600)
    stack = CorrectionStack(se_backend=None, include_vp=True, include_breit=True,
                            nuclear_mass_me=7294.3)
    atom = Atom(2, config=['1s2', '2s2'], engine='shooting', nuclear='point',
        basis=Basis(r_max=60., shooting_space=controls), corrections=Corrections(stack=stack))
    return atom.solve(tol=2e-5, max_iter=12)


def test_public_shooting_is_full_correlated_gradient(full_atom):
    atom = full_atom
    assert atom.converged
    report = atom.result.orbital_optimization
    assert report['full_correlated_orbital_gradient']
    assert report['orbital_residual'] < 2e-5
    assert report['initializer_converged'] is None
    assert report['electron_space_leakage'] < 1e-10
    assert not report['radial_space_converged']
    assert atom.total_energy < -2.86
    assert atom.result.bsp_data == {}


def test_correlated_corrections_share_the_native_metric(full_atom):
    result = full_atom.result
    nms = NormalMassShiftCorrection(7294.3).evaluate(result).energy_shift
    vp = UehlingVPCorrection().evaluate(result).energy_shift
    assert result.energy_components['nms'] == pytest.approx(nms, abs=1e-13)
    assert result.energy_components['E_VP'] == pytest.approx(vp, abs=1e-13)
    assert result.energy_components['E_Breit'] != 0.


def test_resolved_calculation_record_is_immutable_and_serializable(full_atom):
    spec = full_atom.calculation_spec
    assert spec.engine == 'shooting' and spec.method == 'MCDHF'
    assert spec.physics['electron_projector'] == 'nuclear'
    assert spec.numerics['electron_space']['continuum_roots'] == 64
    assert len(spec.state['csfs']) == 2
    assert spec.corrections['breit_stage'] == 'CI'
    with pytest.raises(FrozenInstanceError):
        spec.engine = 'bspline'
    with pytest.raises(TypeError):
        spec.numerics['electron_space']['continuum_roots'] = 1
    json.dumps(spec.to_dict(), allow_nan=False)


def test_correlated_result_records_actual_operator_stages(full_atom):
    from atomx.corrections.stack import finalize_with_corrections
    result = full_atom.result
    assert dict(result.operator_stages) == {
        'breit': 'ci_hamiltonian', 'uehling_vp': 'post_ci_expectation',
        'nms': 'post_ci_expectation',
    }
    before = result.state_energies.copy()
    duplicate = CorrectionStack(se_backend=None, include_vp=False, include_breit=True)
    with pytest.raises(ValueError, match='breit at ci_hamiltonian'):
        finalize_with_corrections(result, duplicate)
    np.testing.assert_array_equal(result.state_energies, before)
