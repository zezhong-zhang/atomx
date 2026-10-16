"""Energy/exposure/angle units and distinct directional/hemisphere observables."""

import numpy as np
import pytest

from atomx.reference_data.euv_experiment import (
    EUVAbsoluteCalibration,
    calibrated_conversion_efficiency,
)
from atomx.spectrum.slab_transfer import trace_ray


def test_absolute_ce_requires_response_exposure_and_angular_energy_calibration():
    c = EUVAbsoluteCalibration(np.array([13.3, 13.4, 13.5, 13.6, 13.7]),
                               np.full(4, 1000.), 2., 10, "synthetic-counts-per-J", "synthetic-angle")
    r = calibrated_conversion_efficiency(np.full(4, 100.), np.zeros(4), 1., c)
    # 2.7 bins *0.1J accumulated /10 shots *2 angular correction /1J laser.
    assert r["conversion_efficiency"] == pytest.approx(.054, abs=1e-14)
    with pytest.raises(ValueError, match="counts alone"):
        calibrated_conversion_efficiency(np.full(4, 100.), np.zeros(4), 1., None)


def test_signed_background_noise_is_not_clipped_to_a_positive_efficiency():
    c = EUVAbsoluteCalibration(np.array([13., 14.]), np.array([1000.]), 1., 1,
                               "synthetic-calibration", "direct-hemisphere")
    r = calibrated_conversion_efficiency([10.], [11.], 1., c)
    assert r["conversion_efficiency"] < 0


def test_ray_formal_solution_and_directional_units():
    j, alpha, L = 2., 3., .4
    expected = j / alpha * -np.expm1(-alpha * L)
    intensity = trace_ray(np.array([[4*np.pi*j]]), np.array([[alpha]]), [L])
    assert intensity[0] == pytest.approx(expected, rel=1e-14)
    transparent = trace_ray(np.array([[4*np.pi*j]]), np.array([[0.]]), [L])
    assert transparent[0] == pytest.approx(j * L)
    # Splitting the same cell changes no radiance.
    split = trace_ray(np.full((10, 1), 4*np.pi*j), np.full((10, 1), alpha), np.full(10, L/10))
    np.testing.assert_allclose(split, intensity, rtol=2e-15)
