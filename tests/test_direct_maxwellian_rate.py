"""Direct thresholded thermal integrals against analytic and adaptive oracles."""

import numpy as np
import pytest
from scipy.constants import e, m_e
from scipy.integrate import quad

from atomx.plasma.rates import maxwellian_rate_direct


@pytest.mark.parametrize('threshold,temperature', [(0., 12.), (258.8, 32.), (3., .5)])
def test_constant_cross_section_has_independent_analytic_flux(threshold, temperature):
    sigma = 2.3e-18
    actual = maxwellian_rate_direct(lambda energy: np.full_like(energy, sigma),
                                    threshold, temperature, order=4)
    expected = sigma*100*np.sqrt(8*e*temperature/(np.pi*m_e))*(1+threshold/temperature)*np.exp(-threshold/temperature)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_energy_dependent_model_matches_independent_adaptive_energy_integral():
    threshold, temperature = 17., 20.
    def sigma(energy):
        return 1e-18*(1-threshold/energy)**2/energy
    expected_integral = quad(lambda energy: float(sigma(energy))*energy*np.exp(-energy/temperature),
                             threshold, np.inf, epsabs=1e-28, epsrel=1e-10)[0]
    expected = 100*np.sqrt(8*e/(np.pi*m_e))*temperature**-1.5*expected_integral
    actual = maxwellian_rate_direct(sigma, threshold, temperature, order=64)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize('callback', [lambda e: -np.ones_like(e), lambda e: np.zeros(1),
                                      lambda e: np.full_like(e, np.nan)])
def test_incomplete_or_negative_cross_sections_reject(callback):
    with pytest.raises(ValueError, match='complete finite nonnegative'):
        maxwellian_rate_direct(callback, 1., 2.)
