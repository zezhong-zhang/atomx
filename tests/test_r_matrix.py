"""Single-channel Wigner R-matrix scaffold regression tests."""

import numpy as np
import pytest

from atomx.scattering.r_matrix import (
    RMatrixResult,
    compute,
    compute_single_channel,
)


def _V_zero(r):
    return np.zeros_like(r)


def _V_attractive_well(r):
    return np.where(r < 5.0, -0.1, 0.0)


def test_returns_RMatrixResult():
    res = compute_single_channel(_V_zero, np.array([1.0]), a_au=10.0,
                                   n_basis=50, l=0)
    assert isinstance(res, RMatrixResult)
    assert res.a_au == 10.0
    assert res.energies_eV.shape == (1,)


def test_inputs_finite_and_real():
    """All outputs are finite real numbers."""
    E = np.geomspace(0.1, 10.0, 10)
    res = compute_single_channel(_V_zero, E, a_au=10.0, n_basis=80, l=0)
    assert np.all(np.isfinite(res.phase_shift_rad))
    assert np.all(np.isfinite(res.cross_section_cm2))
    assert np.all(res.cross_section_cm2 >= 0)


def test_phase_shift_in_range():
    """Phase shifts are wrapped to (-π/2, π/2]."""
    E = np.geomspace(0.1, 10.0, 20)
    res = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                   n_basis=120, l=0)
    assert np.all(res.phase_shift_rad > -np.pi / 2 - 1e-9)
    assert np.all(res.phase_shift_rad <= np.pi / 2 + 1e-9)


def test_partial_wave_cross_section_units():
    """σ_l = (4π/k²)(2l+1) sin²(δ_l) — bounded by the unitarity limit
    σ_max = 4π(2l+1)/k² (in a₀² before cm² conversion)."""
    E = np.array([1.0])
    res = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                   n_basis=120, l=0)
    k_au = np.sqrt(2.0 * 1.0 / 27.211)
    sigma_max_a0 = 4.0 * np.pi / (k_au ** 2)
    sigma_a0 = res.cross_section_cm2[0] / 2.8e-17
    assert sigma_a0 <= sigma_max_a0 * 1.001


def test_higher_l_runs_finite():
    """Higher-l partial waves return finite cross sections."""
    E = np.array([0.1, 1.0, 10.0])
    res_p = compute_single_channel(_V_attractive_well, E, a_au=15.0,
                                     n_basis=120, l=1)
    assert np.all(np.isfinite(res_p.cross_section_cm2))
    assert np.all(res_p.cross_section_cm2 >= 0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), a_au=0.0)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), n_basis=2)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([1.0]), l=-1)
    with pytest.raises(ValueError):
        compute_single_channel(_V_zero, np.array([-1.0]))


def test_eigenvalues_sorted():
    """The interior eigenvalues E_n come out sorted ascending."""
    res = compute_single_channel(_V_attractive_well,
                                   np.array([1.0]),
                                   a_au=15.0, n_basis=80, l=0)
    assert np.all(np.diff(res.eigenvalues_E_n_au) >= 0)


def test_multi_channel_compute_not_implemented():
    with pytest.raises(NotImplementedError):
        compute()
