"""Tests for the 3-D orbital viz and the spinor-harmonic helpers."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.angular.spinor_harmonics import (
    real_spherical_harmonic,
    spinor_density,
)
from atomx.results import OrbitalWavefunction, Waves


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------


def _grid():
    theta = np.linspace(1e-3, np.pi - 1e-3, 200)
    phi = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    return theta, phi, T, P


def _sphere_integral(f, theta, phi, T):
    return float(np.trapezoid(np.trapezoid(f * np.sin(T), theta, axis=0), phi))


class TestSphericalHarmonics:
    @pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (2, -1)])
    def test_real_spherical_harmonic_normalised(self, l, m):
        theta, phi, T, P = _grid()
        Y = real_spherical_harmonic(l, m, T, P)
        norm = _sphere_integral(Y * Y, theta, phi, T)
        # Discretisation error ~1%
        assert norm == pytest.approx(1.0, abs=5e-2)

    def test_real_spherical_harmonic_rejects_out_of_range_m(self):
        theta, phi, T, P = _grid()
        with pytest.raises(ValueError):
            real_spherical_harmonic(1, 2, T, P)


class TestSpinorDensity:
    @pytest.mark.parametrize("kappa,mu", [(-1, 0.5), (-1, -0.5), (+1, 0.5), (-2, 1.5)])
    def test_spinor_density_normalised(self, kappa, mu):
        theta, phi, T, P = _grid()
        rho = spinor_density(kappa, mu, T, P)
        assert (rho >= 0.0).all()
        norm = _sphere_integral(rho, theta, phi, T)
        assert norm == pytest.approx(1.0, abs=5e-2)

    def test_spinor_density_rejects_out_of_range_mu(self):
        theta, phi, T, P = _grid()
        with pytest.raises(ValueError):
            # j = 1/2 for kappa = -1, so |mu| ≤ 1/2
            spinor_density(-1, 1.5, T, P)


# ---------------------------------------------------------------------------
# plot_orbital_3d smoke tests
# ---------------------------------------------------------------------------


plotly = pytest.importorskip("plotly.graph_objects")


def _make_schrodinger_waves():
    r = np.linspace(0.01, 10.0, 401)
    P = 2.0 * r * np.exp(-r)
    orb = OrbitalWavefunction(
        P=P, Q=None, n=1, l=0, kappa=None,
        occupation=1.0, energy=-0.5, label="1s",
    )
    return Waves(grid=r, orbitals={"1s": orb}, basis_type="grid")


def _make_dirac_waves():
    r = np.linspace(0.01, 10.0, 401)
    # Hydrogenic-ish 2p3/2 (j=3/2, kappa=-2)
    P = r ** 2 * np.exp(-r / 2.0)
    Q = -0.01 * r ** 2 * np.exp(-r / 2.0)
    orb = OrbitalWavefunction(
        P=P, Q=Q, n=2, l=1, kappa=-2,
        occupation=4.0, energy=-0.125, label="2p3/2",
    )
    return Waves(grid=r, orbitals={"2p3/2": orb}, basis_type="grid")


class TestPlotOrbital3D:
    def test_schrodinger_returns_two_signed_isosurfaces(self):
        w = _make_schrodinger_waves()
        fig = w.plot3d("1s", n_grid=20)
        assert isinstance(fig, plotly.Figure)
        # Non-relativistic amplitude: positive + negative lobes.
        assert len(fig.data) == 2

    def test_dirac_returns_single_density_isosurface(self):
        w = _make_dirac_waves()
        fig = w.plot3d("2p3/2", n_grid=20)
        assert isinstance(fig, plotly.Figure)
        # Relativistic density: single non-negative surface.
        assert len(fig.data) == 1

    def test_unknown_label_raises(self):
        w = _make_schrodinger_waves()
        with pytest.raises(KeyError, match="not in waves"):
            w.plot3d("nope", n_grid=12)

    def test_rejects_non_plotly_backend(self):
        w = _make_schrodinger_waves()
        with pytest.raises(ValueError, match="plotly"):
            w.plot3d("1s", backend="matplotlib", n_grid=12)

    def test_custom_isosurface_and_box(self):
        w = _make_schrodinger_waves()
        fig = w.plot3d("1s", n_grid=16, box_size=4.0, isosurface=0.05)
        assert isinstance(fig, plotly.Figure)
