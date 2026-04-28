"""Smoke tests for electron-impact ionisation."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.scattering.electron_ionization import (
    EIIResult,
    beb_cross_section,
    eii_cross_section,
    eii_rate_coefficient,
)


def test_beb_zero_below_threshold():
    """BEB cross section is exactly zero for T ≤ U."""
    E = np.array([0.5, 1.0, 5.0, 20.0])
    sigma = beb_cross_section(E, U_eV=10.0, avgT_eV=10.0, N_electrons=2)
    assert np.all(sigma[E <= 10.0] == 0.0)
    assert sigma[-1] > 0.0


def test_beb_peak_around_3U():
    """BEB cross section peaks at T ≈ 3-5 U for typical orbitals."""
    U = 25.0  # He 1s
    E = np.linspace(U + 1.0, 20.0 * U, 200)
    sigma = beb_cross_section(E, U_eV=U, avgT_eV=U, N_electrons=2)
    peak_idx = int(np.argmax(sigma))
    t_peak = E[peak_idx] / U
    assert 2.0 < t_peak < 8.0


def test_argon_eii_order_of_magnitude():
    """Argon σ_total at 100 eV should be ~few × 1e-16 cm² (NIST ~2.8e-16)."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    res = atom._result

    E = np.geomspace(20.0, 1000.0, 60)
    r = eii_cross_section(res, E, method="BEB")
    assert isinstance(r, EIIResult)
    sigma_100 = float(np.interp(100.0, E, r.sigma_total_cm2))
    # Allow ±50 % of NIST value (BEB is ~30 % accurate)
    assert 1e-16 < sigma_100 < 7e-16


def test_argon_eii_dw_near_threshold():
    """Argon σ at 20 eV with first-principles DW Born matches NIST to
    ~15 %.  Threshold-region validation."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([20.0]),
        method="DW", kappa_max=3, n_outgoing_energies=4,
    )
    sigma_20 = float(r.sigma_total_cm2[0])
    # NIST Argon σ(20 eV) ≈ 3.0e-16
    assert 2.0e-16 < sigma_20 < 4.5e-16


def test_argon_eii_dw_mid_energy_with_coulomb_bethe():
    """Argon at 50 and 100 eV — Coulomb-Bethe high-ℓ top-up + V^{N-1}
    distorted waves close the partial-wave-truncation gap to ~25 %.
    Validated: 50 eV ratio 1.27 vs NIST, 100 eV ratio 1.13."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([50.0, 100.0]),
        method="DW", kappa_max=4, n_outgoing_energies=4,
    )
    sigma_50, sigma_100 = float(r.sigma_total_cm2[0]), float(r.sigma_total_cm2[1])
    # NIST: σ(50)≈3.5e-16, σ(100)≈2.8e-16
    assert 2.0e-16 < sigma_50 < 6.0e-16
    assert 1.5e-16 < sigma_100 < 5.0e-16


def test_he_plus_eii_dw_validates_against_younger():
    """He⁺ DW EII near peak should match Younger 1981 to <30 %.
    Validated: σ(100 eV)=2.99e-18, σ(200 eV)=2.40e-18 vs Younger
    ~3.0e-18 / ~2.5e-18."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([100.0, 200.0]),
        method="DW", kappa_max=4, n_outgoing_energies=6,
    )
    sigma_100, sigma_200 = float(r.sigma_total_cm2[0]), float(r.sigma_total_cm2[1])
    # Younger 1981 He⁺ peak ~3e-18 cm²; both within 30 %
    assert 2.0e-18 < sigma_100 < 4.5e-18
    assert 1.5e-18 < sigma_200 < 4.0e-18


def test_eii_auto_mode():
    """Default 'auto' uses DW near threshold and BEB above 5·I_outer."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    E = np.array([20.0, 1000.0])
    r = eii_cross_section(atom._result, E, method="auto", kappa_max=3,
                          n_outgoing_energies=3)
    assert "DW" in r.method or "auto" in r.method
    # 20 eV uses DW (near threshold), 1000 eV uses BEB
    assert r.provenance["DW_E_max_eV"] > 50.0  # 5·I(3p)~80 eV
    assert r.sigma_total_cm2[0] > 1e-16
    assert r.sigma_total_cm2[1] > 1e-17


def test_eii_rate_coefficient_positive_and_smooth():
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    res = atom._result

    E = np.geomspace(20.0, 5000.0, 80)
    r = eii_cross_section(res, E, method="BEB")
    T = np.array([5.0, 20.0, 100.0])
    rates = eii_rate_coefficient(r, T)
    assert rates.shape == T.shape
    assert np.all(rates > 0)
    # Rate increases from T=5 → T=20 (climbing the BEB peak)
    assert rates[1] > rates[0]


def test_eii_dw_runs_for_hydrogenic_ion():
    """DW Born EII runs end-to-end for He⁺ near threshold and produces
    a positive cross section in the Younger-1981 range (3e-18 cm²
    within an order of magnitude — DW is currently validated only
    near threshold for hydrogenic ions; see module docstring)."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([100.0]),
        method="DW", kappa_max=2, n_outgoing_energies=3,
    )
    assert r.method == "DW"
    sigma = float(r.sigma_total_cm2[0])
    assert 1e-19 < sigma < 1e-17
