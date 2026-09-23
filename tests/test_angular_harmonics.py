"""Angular normalisation gates for the spherical/spinor harmonics.

Moved out of ``test_viz_plotly3d.py`` (2026-09 S6 audit): those tests sat
*after* a module-level ``pytest.importorskip("plotly.graph_objects")``, so in
any environment without the optional ``viz`` extra they silently skipped —
even though they touch nothing but :mod:`atomx.angular.spinor_harmonics`.
``plotly`` is an optional extra (``pyproject.toml``), so eight genuine
physics gates were invisible in exactly the environment the extras
declaration advertises.

The gates are first-principles normalisations:

* ``Integral |Y_lm|^2 dOmega = 1`` over the sphere (orthonormality of the
  real spherical-harmonic basis);
* ``Integral rho_{kappa,mu} dOmega = 1`` with ``rho >= 0`` for the
  relativistic spinor density (probability density in Omega at fixed r).

Quadrature: Gauss-Legendre in ``u = cos(theta)`` times a periodic trapezoid
in ``phi``.  The integrands are polynomials in ``u`` and trigonometric
polynomials in ``phi``, so this is exact to machine precision — measured
9e-16 — and the gates are set at 1e-12 rather than the 5e-2 the plotly file
used (that file's 200x200 trapezoid grid was itself only good to 5.0e-3:
it returned 0.99498 for ``Integral |Y_00|^2``).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from atomx.angular.spinor_harmonics import (
    real_spherical_harmonic,
    spinor_density,
)

NORM_TOL = 1e-12  # measured 9e-16 (Gauss-Legendre x periodic trapezoid)
_N_U = 40  # exact for the l <= 3 polynomial integrand
_N_PHI = 80  # exact for the |m| <= 3 trigonometric integrand


def _sphere_integral(f):
    """Integral over the sphere of a callable f(theta, phi)."""
    u, weights = leggauss(_N_U)
    theta = np.arccos(u)
    phi = np.linspace(0.0, 2.0 * np.pi, _N_PHI, endpoint=False)
    t_grid, p_grid = np.meshgrid(theta, phi, indexing="ij")
    values = f(t_grid, p_grid)
    inner = values.sum(axis=1) * (2.0 * np.pi / _N_PHI)
    return float(np.sum(weights * inner))


@pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (2, -1), (2, 0), (2, 2), (3, 2)])
def test_real_spherical_harmonic_normalised(l, m):
    """Integral |Y_lm|^2 dOmega = 1 (orthonormal angular basis)."""
    norm = _sphere_integral(lambda t, p: real_spherical_harmonic(l, m, t, p) ** 2)
    assert pytest.approx(1.0, abs=NORM_TOL) == norm


@pytest.mark.parametrize(
    "kappa,mu",
    [(-1, 0.5), (-1, -0.5), (+1, 0.5), (+1, -0.5), (-2, 1.5), (-2, -0.5)],
)
def test_spinor_density_normalised(kappa, mu):
    """rho >= 0 and Integral rho dOmega = 1 for the spinor density."""
    rho = spinor_density(kappa, mu, *_mesh())
    assert (rho >= 0.0).all()
    norm = _sphere_integral(lambda t, p: spinor_density(kappa, mu, t, p))
    assert pytest.approx(1.0, abs=NORM_TOL) == norm


def _mesh():
    u, _ = leggauss(_N_U)
    t_grid, p_grid = np.meshgrid(
        np.arccos(u), np.linspace(0.0, 2.0 * np.pi, _N_PHI, endpoint=False),
        indexing="ij")
    return t_grid, p_grid


def test_real_spherical_harmonic_rejects_out_of_range_m():
    theta, phi = _mesh()
    with pytest.raises(ValueError):
        real_spherical_harmonic(1, 2, theta, phi)


def test_spinor_density_rejects_out_of_range_mu():
    theta, phi = _mesh()
    with pytest.raises(ValueError):
        # j = 1/2 for kappa = -1, so |mu| <= 1/2
        spinor_density(-1, 1.5, theta, phi)
