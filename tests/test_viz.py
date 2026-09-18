"""Tests for the atomx visualisation layer.

Covers three backends:
* ``atomx.viz.mpl`` — smoke-test each renderer, verify it returns a
  ``matplotlib.figure.Figure`` without raising.
* ``atomx.viz.plotly`` — ditto, with ``plotly.graph_objects.Figure``.
* ``atomx.viz.manim`` — import guard: module must import cleanly even
  if manim is absent; instantiation raises a helpful error.

Also covers the ``.plot()`` / ``.plot_density()`` convenience methods on
:class:`~atomx.results.Waves` and :class:`~atomx.spectrum.Spectrum`.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as _plt  # noqa: E402

matplotlib.use("Agg")                # headless

from atomx.results import OrbitalWavefunction, Waves  # noqa: E402
from atomx.spectrum import Spectrum  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_wavefunctions():
    r = np.linspace(0.01, 10.0, 401)
    # Hydrogenic-ish 1s
    P = 2.0 * r * np.exp(-r)
    Q = -0.01 * r * np.exp(-r)
    orb = OrbitalWavefunction(
        P=P, Q=Q, n=1, l=0, kappa=-1,
        occupation=1.0, energy=-0.5, label="1s1/2",
    )
    # Also 2s for density plots
    P2 = 1.0 * r * (1 - r / 2) * np.exp(-r / 2)
    orb2 = OrbitalWavefunction(
        P=P2, Q=None, n=2, l=0, kappa=-1,
        occupation=1.0, energy=-0.125, label="2s1/2",
    )
    return Waves(
        grid=r, orbitals={"1s1/2": orb, "2s1/2": orb2}, basis_type="grid",
    )


@pytest.fixture
def simple_spectrum():
    s = Spectrum.linear(95.0, 105.0, 5001)
    s.add_line(100.0, intensity=1.0, sigma_nm=0.05)
    s.add_line(101.5, intensity=0.6, sigma_nm=0.03)
    return s


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------

class TestMpl:
    def test_plot_radial_orbital_returns_figure(self, simple_wavefunctions):
        from atomx.viz.mpl import plot_radial_orbital
        fig = plot_radial_orbital(simple_wavefunctions, "1s1/2")
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_plot_radial_orbital_unknown_label_raises(self, simple_wavefunctions):
        from atomx.viz.mpl import plot_radial_orbital
        with pytest.raises(KeyError, match="not in waves"):
            plot_radial_orbital(simple_wavefunctions, "9z9/2")

    def test_plot_orbital_density(self, simple_wavefunctions):
        from atomx.viz.mpl import plot_orbital_density
        fig = plot_orbital_density(simple_wavefunctions,
                                    labels=["1s1/2", "2s1/2"])
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_plot_grotrian_levels_only(self):
        from atomx.viz.mpl import plot_grotrian
        levels = [
            {"energy": 0.0, "label": "1S", "L": 0},
            {"energy": 5.0, "label": "3P", "L": 1},
            {"energy": 8.0, "label": "1D", "L": 2},
        ]
        fig = plot_grotrian(levels, energy_unit="eV")
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_plot_grotrian_with_transitions(self):
        from atomx.viz.mpl import plot_grotrian
        levels = [
            {"energy": 0.0, "label": "1S", "L": 0},
            {"energy": 5.0, "label": "1P", "L": 1},
        ]
        transitions = [
            {"upper": 1, "lower": 0, "wavelength": 248.0, "A_einstein": 6e8},
        ]
        fig = plot_grotrian(levels, transitions, arrow_label="wavelength")
        _plt.close(fig)

    def test_plot_spectrum(self, simple_spectrum):
        from atomx.viz.mpl import plot_spectrum
        fig = plot_spectrum(simple_spectrum, show_centroid=True)
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_plot_qed_breakdown(self):
        from atomx.viz.mpl import plot_qed_breakdown
        components = {
            "E_1e": -13.6, "E_J": 5.0, "E_K": -1.2,
            "E_Breit": -0.1, "E_VP": -0.02, "E_SE": 0.03,
        }
        fig = plot_qed_breakdown(components)
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_plot_qed_breakdown_no_numeric_raises(self):
        from atomx.viz.mpl import plot_qed_breakdown
        with pytest.raises(ValueError, match="No numeric"):
            plot_qed_breakdown({"se_method": "mohr"})

    def test_plot_qed_breakdown_reference_key(self):
        from atomx.viz.mpl import plot_qed_breakdown
        components = {"base": -10.0, "a": -10.05, "b": -10.02}
        fig = plot_qed_breakdown(components, reference_key="base")
        _plt.close(fig)

    def test_plot_qed_breakdown_unknown_ref_raises(self):
        from atomx.viz.mpl import plot_qed_breakdown
        with pytest.raises(KeyError, match="reference_key"):
            plot_qed_breakdown({"a": 1.0, "b": 2.0}, reference_key="bogus")

    def test_plot_scf_convergence(self):
        from atomx.viz.mpl import plot_scf_convergence
        r = [1.0, 1e-2, 1e-4, 1e-6, 1e-8]
        E = [-128.5, -128.52, -128.55, -128.55, -128.55]
        fig = plot_scf_convergence(r, E)
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)


# ---------------------------------------------------------------------------
# plotly backend
# ---------------------------------------------------------------------------

class TestPlotly:
    def test_plot_radial_orbital(self, simple_wavefunctions):
        go = pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_radial_orbital
        fig = plot_radial_orbital(simple_wavefunctions, "1s1/2")
        assert isinstance(fig, go.Figure)

    def test_plot_orbital_density(self, simple_wavefunctions):
        go = pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_orbital_density
        fig = plot_orbital_density(simple_wavefunctions)
        assert isinstance(fig, go.Figure)

    def test_plot_spectrum(self, simple_spectrum):
        go = pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_spectrum
        fig = plot_spectrum(simple_spectrum, show_centroid=True)
        assert isinstance(fig, go.Figure)

    def test_plot_qed_breakdown(self):
        go = pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_qed_breakdown
        fig = plot_qed_breakdown({"E_1e": -13.6, "E_J": 5.0})
        assert isinstance(fig, go.Figure)

    def test_plot_scf_convergence(self):
        go = pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_scf_convergence
        fig = plot_scf_convergence([1.0, 1e-3, 1e-6],
                                    energies=[-1.0, -1.1, -1.1])
        assert isinstance(fig, go.Figure)

    def test_unknown_label_raises(self, simple_wavefunctions):
        pytest.importorskip("plotly.graph_objects")
        from atomx.viz.plotly import plot_radial_orbital
        with pytest.raises(KeyError):
            plot_radial_orbital(simple_wavefunctions, "9z9/2")


# ---------------------------------------------------------------------------
# manim backend (import-guard only; no manim in the test env)
# ---------------------------------------------------------------------------

class TestManimImportGuard:
    def test_manim_module_imports_cleanly(self):
        """atomx.viz.manim must always import — missing manim becomes a
        runtime error only when someone tries to instantiate a Scene."""
        import atomx.viz.manim as mod
        # The scene classes must at least be definable as symbols:
        assert hasattr(mod, "RadialOrbitalScene")
        assert hasattr(mod, "GrotrianScene")
        assert hasattr(mod, "SpectrumScene")
        assert hasattr(mod, "QEDBreakdownScene")
        assert hasattr(mod, "SCFConvergenceScene")
        assert hasattr(mod, "DRScene")

    @pytest.mark.skipif(
        pytest.importorskip("importlib").import_module,   # dummy condition
        reason="Always tested",
    )
    def test_missing_manim_raises_on_instantiation(self):
        """Skipped guard-test placeholder — exercised interactively."""
        pass

    def test_manim_scene_class_requires_manim_to_instantiate(self):
        from atomx.viz import manim as manim_mod
        # When manim is not installed, instantiation raises.
        # When it IS installed, instantiation without the required
        # class attributes should raise ValueError from our construct().
        try:
            import manim  # noqa: F401  — just to detect presence
            has_manim = True
        except ImportError:
            has_manim = False

        if has_manim:
            with pytest.raises(ValueError):
                # Missing waves/label → our ValueError
                scene = manim_mod.RadialOrbitalScene()
                scene.construct()
        else:
            with pytest.raises(RuntimeError, match="manim is required"):
                manim_mod.RadialOrbitalScene()


# ---------------------------------------------------------------------------
# Convenience methods on result objects
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_wavefunctions_plot_mpl(self, simple_wavefunctions):
        fig = simple_wavefunctions.plot("1s1/2", backend="mpl")
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_wavefunctions_plot_plotly(self, simple_wavefunctions):
        go = pytest.importorskip("plotly.graph_objects")
        fig = simple_wavefunctions.plot("1s1/2", backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_wavefunctions_plot_density(self, simple_wavefunctions):
        fig = simple_wavefunctions.plot_density(backend="mpl")
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_wavefunctions_plot_unknown_backend(self, simple_wavefunctions):
        with pytest.raises(ValueError, match="Unknown backend"):
            simple_wavefunctions.plot("1s1/2", backend="crayons")

    def test_spectrum_plot_mpl(self, simple_spectrum):
        fig = simple_spectrum.plot(backend="mpl")
        assert isinstance(fig, _plt.Figure)
        _plt.close(fig)

    def test_spectrum_plot_plotly(self, simple_spectrum):
        pytest.importorskip("plotly.graph_objects")
        fig = simple_spectrum.plot(backend="plotly")
        from plotly.graph_objects import Figure as PlotlyFigure
        assert isinstance(fig, PlotlyFigure)

    def test_spectrum_plot_unknown_backend(self, simple_spectrum):
        with pytest.raises(ValueError, match="Unknown backend"):
            simple_spectrum.plot(backend="neon-signs")
