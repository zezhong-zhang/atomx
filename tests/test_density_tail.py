"""Regression tests for density tail extrapolation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from atomx.numerics.density_tail import smooth_exponential_tail


def _noisy_exponential_tail():
    r = np.linspace(0.0, 80.0, 161)
    rho = np.exp(-r)
    below = np.flatnonzero(rho <= 1.0e-18)
    start = int(below[0]) + 1
    oscillation = np.array([1.5, 0.35, 1.8, 0.25])
    rho[start:] = (
        1.0e-19 * oscillation[np.arange(r.size - start) % oscillation.size]
    )
    return r, rho, start


def test_smooth_exponential_tail_leaves_clean_tail_unchanged():
    r = np.linspace(0.0, 80.0, 161)
    rho = np.exp(-r)

    smoothed = smooth_exponential_tail(r, rho, noise_floor=1.0e-18)

    np.testing.assert_allclose(smoothed, rho)


def test_smooth_exponential_tail_replaces_below_floor_oscillation():
    r, rho, noisy_start = _noisy_exponential_tail()

    smoothed = smooth_exponential_tail(r, rho, noise_floor=1.0e-18)

    assert np.any(np.diff(rho[noisy_start:]) > 0.0)
    assert np.all(np.diff(smoothed[noisy_start:]) <= 0.0)
    assert np.all(smoothed[noisy_start:] > 0.0)

    slope = np.diff(np.log(smoothed[noisy_start:noisy_start + 8])) / np.diff(
        r[noisy_start:noisy_start + 8]
    )
    assert np.std(slope) < 1.0e-12


def test_density_payload_writes_smoothed_total_tail():
    spec = importlib.util.spec_from_file_location(
        "density_gs",
        Path(__file__).resolve().parents[1] / "scripts" / "density_gs.py",
    )
    density_gs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(density_gs)

    r, rho, noisy_start = _noisy_exponential_tail()
    orb = SimpleNamespace(
        P=np.sqrt(rho),
        Q=np.zeros_like(rho),
        n=1,
        l=0,
        kappa=-1,
        label="1s1/2",
    )

    payload = density_gs.build_orbital_payload(orb, nq=1.0, grid=r)
    density = payload["datasets"]["density"]

    assert density.size == r.size
    assert np.all(np.diff(density[noisy_start:]) <= 0.0)
