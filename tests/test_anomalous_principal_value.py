"""Analytic and independent Cauchy quadrature anchors for dispersion."""

import numpy as np
import pytest
from scipy.integrate import quad

from atomx.scattering.anomalous import f1_kramers_kronig, f2_from_sigma


def test_constant_absorption_exact_at_and_near_log_grid_nodes():
    x = np.geomspace(0.1, 100, 60)
    energies = np.array([0, 0.01, x[35], x[35] * (1 + 1e-10), 200])
    expected = (
        1.7
        / np.pi
        * np.log(abs((energies**2 - x[0] ** 2) / (energies**2 - x[-1] ** 2)))
    )
    actual = f1_kramers_kronig(x, np.full_like(x, 1.7), energies)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_linear_spectrum_is_grid_independent():
    w = np.array([0, 0.5, 1.5, 3, 10])
    m, c, a, b = 0.13, 0.7, 0.2, 7.0

    def primitive(x):
        return m * (-x + w / 2 * np.log(abs((w + x) / (w - x)))) - c / 2 * np.log(
            abs(w * w - x * x)
        )

    expected = 2 / np.pi * (primitive(b) - primitive(a))
    for x in (np.array([a, b]), np.geomspace(a, b, 199)):
        assert f1_kramers_kronig(x, m * x + c, w) == pytest.approx(expected, abs=1e-12)


def test_piecewise_spectrum_matches_independent_scipy_cauchy_quadrature():
    x, f = np.array([0.1, 0.3, 0.8, 2, 4, 9]), np.array([0, 1.5, 0.7, 2, 0.9, 0.1])
    for w in (0.55, 1.2, 4.2):
        expected = (
            quad(
                lambda t, energy=w: -t * np.interp(t, x, f) / (energy + t),
                x[0],
                x[-1],
                weight="cauchy",
                wvar=w,
                epsabs=1e-12,
                epsrel=1e-12,
                limit=300,
            )[0]
            * 2
            / np.pi
        )
        assert f1_kramers_kronig(x, f, np.array([w]))[0] == pytest.approx(
            expected, abs=2e-9
        )


def test_invalid_spectrum_and_window_endpoints_fail():
    with pytest.raises(ValueError, match="endpoint"):
        f1_kramers_kronig(np.array([1, 2]), np.array([0, 1]), np.array([1]))
    with pytest.raises(ValueError, match="finite"):
        f1_kramers_kronig(np.array([1, 2]), np.array([0, np.nan]), np.array([1.5]))
    with pytest.raises(ValueError):
        f2_from_sigma(np.array([1]), np.array([np.nan]))
