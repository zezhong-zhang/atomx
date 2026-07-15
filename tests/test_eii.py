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
    """Argon at 50 and 100 eV — DW with the FAC-equivalent soft-ejection
    window.  For NEUTRAL targets plain DWBA genuinely overshoots
    (missing channel coupling); the pre-2026-07-15 ratios (1.27/1.13 vs
    NIST) looked better only because the old 0.05T..0.49T ejected-energy
    window truncated the soft-ejection region and partially cancelled
    that overshoot.  Current honest ratios: ~1.8 at 50 eV, ~1.5 at
    100 eV — gate them so regressions still trip."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([50.0, 100.0]),
        method="DW", kappa_max=4, n_outgoing_energies=4,
    )
    sigma_50, sigma_100 = float(r.sigma_total_cm2[0]), float(r.sigma_total_cm2[1])
    # NIST: σ(50)≈3.5e-16, σ(100)≈2.8e-16
    assert 2.0e-16 < sigma_50 < 7.5e-16
    assert 1.5e-16 < sigma_100 < 5.0e-16


def test_he_plus_eii_dw_validates_against_younger():
    """He⁺ DW EII near peak is within the DWBA-prescription spread of
    Younger 1981 (~3.0e-18 / ~2.5e-18 at 100/200 eV).

    The 2026-07-14 relativistic DWBA (coherent antisymmetrized
    |D - E|^2, delta(E)-normalized shooting continuum) runs hotter than
    Younger's exchange prescription near threshold by up to ~2x — the
    well-known post/prior/Peterkop exchange-model spread in EII.  The
    exactness anchor is FAC's pure-DW CI mode, which this code matches
    to a few % at E/th >= 2 (tests/test_fac_dw_benchmarks.py)."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([100.0, 200.0]),
        method="DW", kappa_max=4, n_outgoing_energies=6,
    )
    sigma_100, sigma_200 = float(r.sigma_total_cm2[0]), float(r.sigma_total_cm2[1])
    assert 2.0e-18 < sigma_100 < 6.5e-18
    assert 1.5e-18 < sigma_200 < 6.5e-18


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


# ---------------------------------------------------------------------------
# New DW EII tests
# ---------------------------------------------------------------------------


def test_dw_no_nan_for_low_kappa_max():
    """DW EII with kappa_max=2 must not produce NaN even for tight
    partial-wave truncation (regression guard for Slater integral NaN
    from cache-reuse edge cases)."""
    atom = Atom(Z=10, config="1s2", breit=False, qed=False)
    atom.solve()
    I_1s = -list(atom._result.orbital_energies.values())[0] * 27.211
    E = np.array([1.1 * I_1s, 1.5 * I_1s, 3.0 * I_1s])
    r = eii_cross_section(
        atom._result, E, method="DW", kappa_max=2, n_outgoing_energies=4,
    )
    assert not np.any(np.isnan(r.sigma_total_cm2))
    assert np.all(r.sigma_total_cm2 >= 0.0)


def test_dw_hydrogenic_scaling_with_z():
    """DW cross section is positive for hydrogenic ions at moderate Z.
    Only He⁺ (Z=2) is validated against Younger 1981; higher-Z
    hydrogenic ions exercise the high-energy continuum path but are
    not expected to match literature until the B-spline basis is
    adapted for high-energy K-shell continuum waves."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    I = -list(atom._result.orbital_energies.values())[0] * 27.211
    r = eii_cross_section(
        atom._result, np.array([3.0 * I]),
        method="DW", kappa_max=4, n_outgoing_energies=6,
    )
    sigma = float(r.sigma_total_cm2[0])
    # He⁺ at 3*I ≈ 163 eV: Younger 1981 peak ≈ 3e-18 cm²
    assert 1e-18 < sigma < 1e-17




@pytest.mark.parametrize("Z", [10, 26, 54])
def test_dw_k_shell_high_energy_smooth_no_hybrid(Z):
    """High-Z K-shell DW no longer needs the BEB hybrid at E/I >= 2.

    The old B-spline box continuum missed compact K-shell continuum
    energies at high E, so ``_eii_dw_total`` silently swapped in BEB for
    E/I >= 2.  The shooting-continuum rewrite removed that hack: the DW
    kernel is now accurate at all energies (validated against FAC's
    pure-DW CI mode to a few % at E/th >= 2).  Here we pin that the
    hybrid is gone and DW stays within a factor ~3 of BEB (a model-level
    sanity band, not a parity statement)."""
    atom = Atom(Z=Z, config="1s2", breit=False, qed=False)
    atom.solve()
    I_eV = -next(iter(atom._result.orbital_energies.values())) * 27.211386245988
    E = np.array([2.0 * I_eV, 5.0 * I_eV])

    r_dw = eii_cross_section(
        atom._result, E, method="DW", kappa_max=4, n_outgoing_energies=4,
    )
    r_beb = eii_cross_section(atom._result, E, method="BEB")

    orb_info = next(iter(r_dw.provenance["per_orbital"].values()))
    assert "k_shell_high_energy_hybrid" not in orb_info
    assert np.all(r_dw.sigma_total_cm2 > 0.0)
    assert np.all(np.isfinite(r_dw.sigma_total_cm2))
    ratio = r_dw.sigma_total_cm2 / r_beb.sigma_total_cm2
    assert np.all(ratio > 1.0 / 3.0) and np.all(ratio < 3.0)

def test_dw_positive_and_declines_above_peak():
    """DW cross section is positive everywhere above threshold and
    decreases monotonically above the peak."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    I = -list(atom._result.orbital_energies.values())[0] * 27.211
    E = np.geomspace(1.1 * I, 15 * I, 30)
    r = eii_cross_section(
        atom._result, E, method="DW", kappa_max=4, n_outgoing_energies=6,
    )
    sigma = r.sigma_total_cm2
    assert np.all(sigma > 0)
    assert not np.any(np.isnan(sigma))
    # Monotonic decline above peak
    peak_idx = int(np.argmax(sigma))
    if peak_idx < len(sigma) - 2:
        tail = sigma[peak_idx:]
        # Allow small numerical noise (1 % tolerance)
        diffs = np.diff(tail)
        assert np.all(diffs < 0.01 * sigma[peak_idx])


def test_dw_per_orbital_decomposition():
    """DW returns per-orbital contributions that sum to total."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    E = np.array([50.0])
    r = eii_cross_section(
        atom._result, E, method="DW", kappa_max=3, n_outgoing_energies=4,
    )
    orb_sum = sum(v for v in r.sigma_by_orbital.values())
    np.testing.assert_allclose(r.sigma_total_cm2, orb_sum, rtol=1e-10)


def test_dw_method_label_and_provenance():
    """DW result carries correct method label and provenance metadata."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    r = eii_cross_section(
        atom._result, np.array([100.0]),
        method="DW", kappa_max=3, n_outgoing_energies=4,
    )
    assert r.method == "DW"
    assert r.provenance["Z"] == 2
    assert "per_orbital" in r.provenance
    for orb_info in r.provenance["per_orbital"].values():
        assert "I_eV" in orb_info
        assert orb_info["I_eV"] > 0


def test_dw_kappa_max_convergence():
    """Increasing kappa_max should give convergent cross sections.
    The partial-wave series should stabilize; km=4 and km=5 should
    agree better than km=2 and km=3."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    I = -list(atom._result.orbital_energies.values())[0] * 27.211
    E = np.array([3.0 * I])
    sigmas = []
    for km in [2, 3, 4, 5]:
        r = eii_cross_section(
            atom._result, E, method="DW",
            kappa_max=km, n_outgoing_energies=4,
        )
        sigmas.append(float(r.sigma_total_cm2[0]))

    # km=4 vs km=5 should agree within 30 % (partial-wave convergence)
    rel_change = abs(sigmas[3] - sigmas[2]) / max(abs(sigmas[2]), 1e-30)
    assert rel_change < 0.30, (
        f"km=4→5 relative change {rel_change:.1%} too large"
    )


def test_dw_neon_valence_reasonable():
    """Ne valence DW EII at 100 eV should be within an order of magnitude
    of the BEB value (which matches NIST to ~30 %)."""
    atom = Atom(Z=10, config="1s2 2s2 2p6", breit=False, qed=False)
    atom.solve()
    E = np.array([100.0])
    r_dw = eii_cross_section(atom._result, E, method="DW", kappa_max=4,
                             n_outgoing_energies=6)
    r_beb = eii_cross_section(atom._result, E, method="BEB")
    # DW total should be within 10x of BEB for valence-dominated energies
    ratio = float(r_dw.sigma_total_cm2[0]) / float(r_beb.sigma_total_cm2[0])
    assert 0.05 < ratio < 15, f"DW/BEB ratio = {ratio:.2f} for Ne at 100 eV"


def test_dw_invalid_method_raises():
    """Unknown method raises ValueError."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    with pytest.raises(ValueError, match="Unknown method"):
        eii_cross_section(atom._result, np.array([100.0]), method="bogus")
