"""Independent Salvat DCOULF values and Dirac flux invariants."""
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.numerics.shooting.dirac_coulomb import dirac_coulomb_outgoing

REFERENCE = json.loads((Path(__file__).parent/'data/radial_dcoul_reference.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'])
def test_against_live_radial_export(case):
    c = REFERENCE['speed_of_light']
    p, q = dirac_coulomb_outgoing(case['e'], case['k'], case['z'], case['r'], c=c)
    np.testing.assert_allclose([p.imag, q.imag, p.real, q.real], case['reference'],
                               rtol=2e-10, atol=1e-11)
    expected = -np.sqrt(case['e']/(case['e']+2*c*c))
    assert p.imag*q.real-q.imag*p.real == pytest.approx(expected, rel=1e-10)


def test_repulsive_reference_satisfies_dirac_equation():
    # An independent finite-difference residual covers the repulsive sign.
    c = REFERENCE['speed_of_light']
    energy, kappa, charge, radius = 2., -2, -3., 15.
    step = 1e-4
    def at(r):
        return dirac_coulomb_outgoing(energy, kappa, charge, r, c=c)
    derivative = (-at(radius+2*step)+8*at(radius+step)
                  -8*at(radius-step)+at(radius-2*step))/(12*step)
    p, q = at(radius)
    kinetic = energy+charge/radius
    expected = [-kappa*p/radius+(2*c+kinetic/c)*q,
                kappa*q/radius-kinetic/c*p]
    np.testing.assert_allclose(derivative, expected, rtol=1e-9, atol=1e-10)


CONTINUUM = json.loads((Path(__file__).parent/'data/radial_continuum_reference.json').read_text())


@pytest.mark.parametrize('case', CONTINUUM['cases'])
def test_integrated_wave_has_radial_absolute_normalization_and_phase(case):
    from atomx.constants import C
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_shooting

    assert CONTINUUM['speed_of_light'] == C
    energy, charge = case['e'], case['z']
    r, p, q, phase = solve_dirac_continuum_shooting(
        energy, case['k'], lambda r: -charge/r, r_max=case['box'],
        n_grid=CONTINUUM['n_grid'], Z_residual=charge)
    indices = [np.argmin(abs(r-x)) for x in case['r']]
    np.testing.assert_allclose(r[indices], case['r'], rtol=1e-12, atol=1e-14)
    valid = np.array(case['reference_valid'])
    # DCOULF fails or exceeds its error budget at some inner points.
    # They are retained as missing references, never counted as passes.
    reference = np.array(case['reference_regular'], dtype=float)[:, valid]
    momentum = np.sqrt(energy*(2+energy/C**2))
    amplitude = np.sqrt((2+energy/C**2)/(np.pi*momentum))
    actual = np.array([p[indices], q[indices]])[:, valid]/amplitude
    sign = 1 if np.vdot(actual, reference) >= 0 else -1
    # Only the overall real sign is arbitrary. No fitted amplitude/phase.
    np.testing.assert_allclose(actual, sign*reference, rtol=0, atol=2e-9)
    assert abs(np.arctan(np.tan(phase))) < 2e-9
