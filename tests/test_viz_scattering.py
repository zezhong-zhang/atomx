"""Smoke tests for atomx.viz.scattering (Agg backend, no display)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.viz.scattering import (  # noqa: E402
    plot_collision_strength,
    plot_cross_section,
    plot_dr_rate,
    plot_dr_resonances,
    plot_eii_channels,
    plot_spectrum_channels,
)


def test_cross_section_with_reference_and_ratio():
    E = np.geomspace(100.0, 1.0e5, 40)
    s = 1e-18 / E
    ref = (np.array([500.0, 5000.0, 5.0e4]),
           np.array([2.1e-21, 2.0e-22, 2.1e-23]))
    fig = plot_cross_section(E, s, reference=ref, reference_label="FAC",
                             threshold_eV=120.0,
                             extra_series={"BEB": (E, 0.8e-18 / E)})
    assert len(fig.axes) == 2  # main + ratio
    fig2 = plot_cross_section(E, s, ratio_panel=False)
    assert len(fig2.axes) == 1


def test_collision_strength():
    E = np.linspace(1000.0, 9000.0, 30)
    fig = plot_collision_strength(E, 0.01 * np.log(E), 900.0,
                                  reference=([1000, 4000], [0.068, 0.081]))
    assert fig.axes[0].get_xlabel().startswith("$E$")


def test_eii_channels():
    class R:
        energies_eV = np.geomspace(1e3, 1e5, 20)
        sigma_total_cm2 = 1e-20 / energies_eV * 1e3
        sigma_by_orbital = {"1s1/2": sigma_total_cm2 * 0.6,
                            "2s1/2": sigma_total_cm2 * 0.4}
        method = "DW"
    fig = plot_eii_channels(R())
    assert "EII" in fig.axes[0].get_title()


def test_dr_resonances_and_rate():
    from atomx.scattering.autoionization.auger import AutoionizationState
    from atomx.scattering.dielectronic_recombination.solver import dr_rate_coefficient
    res = [AutoionizationState(label=f"r{i}", E_resonance_eV=200.0 + 30 * i,
                               J=0.5 + (i % 3), gamma_auger_per_s=1e13,
                               a_radiative_per_s=2e12) for i in range(9)]
    fig = plot_dr_resonances(res)
    assert fig.axes[0].get_yscale() == "log"
    with pytest.raises(ValueError, match="weight"):
        plot_dr_resonances(res, weight="bogus")
    rc = dr_rate_coefficient(res, np.geomspace(10, 5000, 25),
                             initial_state_g=1.0)
    fig2 = plot_dr_rate(rc, references={
        "FAC clean-room": (rc.T_eV, rc.alpha_cm3_s * 1.1)})
    assert fig2.axes[0].get_xscale() == "log"


def test_spectrum_channels():
    from atomx.spectrum.spectrum import Spectrum
    tot = Spectrum.linear(5.0, 25.0, 200, unit="emissivity")
    tot.values[:] = 1e-3
    tot.add_line(12.0, 5.0, sigma_nm=0.05)
    ff = Spectrum.linear(5.0, 25.0, 200, unit="emissivity")
    ff.values[:] = 8e-4
    fig = plot_spectrum_channels(tot, continuum_spectra={"free-free": ff})
    assert fig.axes[0].get_yscale() == "log"
