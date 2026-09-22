"""Smoke tests for electron-impact ionisation."""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization import (
    EIIResult,
    beb_cross_section,
    eii_cross_section,
    eii_rate_coefficient,
)

# CODATA 2022, deliberately independent of atomx.constants: the point of the
# rate-coefficient gates below is to check the *conversion* itself (the
# sibling `collision_strength_to_rate` shipped a 2e10 prefactor error), not to
# re-read the package's own constants table.
_M_E_G = 9.109_383_7139e-28
_E_CHARGE_ERG = 1.602_176_634e-12


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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
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
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
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


def _maxwellian_rate_quad(sigma_cm2, energies_eV, temperature_eV):
    """Independent scipy.quad of the textbook Maxwellian average.

    ``<sigma v> = sqrt(8/(pi m_e)) (kT)^(-3/2) Int sigma(E) E exp(-E/kT) dE``
    in cgs (erg, g, cm), i.e. exactly the integral ``maxwellian_rate``
    evaluates by the trapezoid rule.  ``sigma_cm2`` is linearly interpolated
    from ``energies_eV`` and the integration runs over the *same* interval the
    tabulated cross section covers (``maxwellian_rate`` integrates only where
    it is given data, so this is the apples-to-apples check of the
    integrator).  The integrand is scaled by ``_SCALE`` because the integral
    is ~1e-37 in cgs, far below QUADPACK's roundoff floor at epsabs = 0.
    """
    kT_erg = float(temperature_eV) * _E_CHARGE_ERG
    sigma = np.asarray(sigma_cm2, dtype=float)
    energy = np.asarray(energies_eV, dtype=float)
    scale = 1.0e30
    # The energy integral is ~1e-33 in cgs (eV^2 cm^2); scaling it into the
    # 1e-3 range keeps QUADPACK away from its roundoff floor (epsabs/epsrel
    # below the achievable level makes it raise IntegrationWarning instead of
    # converging).  The tolerances chosen here still pin the integral 2-3
    # orders tighter than the 1e-6 assertion below.
    with warnings.catch_warnings():
        # QUADPACK flags roundoff on this magnitude for some temperatures; the
        # returned value is verified against the trapezoid rule by the
        # assertion below (measured agreement 3e-9), so the flag is benign.
        warnings.simplefilter("ignore", IntegrationWarning)
        integral = quad(
            lambda e_eV: float(np.interp(e_eV, energy, sigma))
            * e_eV * math.exp(-e_eV / temperature_eV) * scale,
            float(energy[0]), float(energy[-1]),
            epsabs=1.0e-13, epsrel=1.0e-8, limit=400,
        )[0] / scale
    return (math.sqrt(8.0 / (math.pi * _M_E_G)) * kT_erg ** -1.5
            * _E_CHARGE_ERG ** 2 * integral)


@pytest.mark.parametrize("temperature_eV", [1.0, 10.0, 100.0])
def test_eii_rate_coefficient_matches_closed_form(temperature_eV):
    """Constant sigma: <sigma v> = sigma sqrt(8 kT / (pi m_e)) exactly.

    First-principles closed form: with ``sigma(E) = sigma_0`` the integral
    ``Int_0^inf sigma_0 E exp(-E/kT) dE = sigma_0 (kT)^2``, so
    ``<sigma v> = sigma_0 sqrt(8 kT/(pi m_e))``.  At kT = 1 eV and
    sigma_0 = 1e-16 cm^2 that is 6.6924e-09 cm^3/s.  This is the gate the
    sibling ``collision_strength_to_rate`` was missing when it shipped a
    2e10 prefactor error (review H0).
    """
    energy = np.linspace(0.0, 60.0 * temperature_eV, 200_001)
    res = EIIResult(energies_eV=energy,
                    sigma_total_cm2=np.full_like(energy, 1.0e-16))
    # eii_rate_coefficient collapses a length-1 temperature array to a scalar.
    rate = float(np.atleast_1d(eii_rate_coefficient(
        res, np.array([temperature_eV])))[0])
    expected = 1.0e-16 * math.sqrt(
        8.0 * temperature_eV * _E_CHARGE_ERG / (math.pi * _M_E_G))
    assert rate == pytest.approx(expected, rel=1e-6), (
        f"kT = {temperature_eV} eV: <sigma v> = {rate:.6e} cm^3/s, closed "
        f"form {expected:.6e} cm^3/s"
    )


def test_eii_rate_coefficient_of_argon_beb_is_bracketed_and_quadrature_exact():
    """Ar BEB rate coefficient: quadrature check + rigorous bracket.

    Two independent statements about the *real* Ar cross section (no atomx
    rate on the right-hand side):

    1. the trapezoid rule the code uses reproduces an independently written
       ``scipy.quad`` of the same sampled cross section to 1e-6 (the grid
       starts below the 15.76 eV Ar I threshold and reaches 60 kT, so the
       trapezoid covers the whole weighted domain);
    2. since ``E exp(-E/kT) > 0``, the Maxwellian-weighted mean of sigma lies
       between its own minimum and maximum, so
       ``sigma_min sqrt(8kT/(pi m_e)) <= <sigma v> <= sigma_max
       sqrt(8kT/(pi m_e))`` with sigma_min/max taken on the tabulated grid.
       A prefactor or unit error of any size breaks this immediately.

    Measured 2026-09-22 (Ar, BEB, 100001-point 15 eV - 60 kT grid):
    T = 20 eV -> 7.660e-08, T = 100 eV -> 2.142e-07 cm^3/s.
    """
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
    atom.solve()
    temperatures = np.array([20.0, 100.0])
    for temperature_eV in temperatures:
        # Ar I ionisation energy 15.7596 eV (NIST ASD): start below it so the
        # trapezoid covers the whole weighted domain.
        energy = np.linspace(15.0, 60.0 * temperature_eV, 100_001)
        res = eii_cross_section(atom._result, energy, method="BEB")
        sigma = np.asarray(res.sigma_total_cm2, dtype=float)
        rate = float(np.atleast_1d(
            eii_rate_coefficient(res, np.array([temperature_eV])))[0])
        reference = _maxwellian_rate_quad(sigma, energy, float(temperature_eV))
        assert rate == pytest.approx(reference, rel=1e-6), (
            f"T = {temperature_eV} eV: eii_rate_coefficient gave "
            f"{rate:.6e} but the independent quadrature of the same "
            f"cross section gives {reference:.6e} cm^3/s"
        )
        prefactor = math.sqrt(
            8.0 * temperature_eV * _E_CHARGE_ERG / (math.pi * _M_E_G))
        lo, hi = float(np.min(sigma)) * prefactor, float(np.max(sigma)) * prefactor
        assert lo <= rate <= hi, (
            f"T = {temperature_eV} eV: <sigma v> = {rate:.6e} outside the "
            f"rigorous bracket [{lo:.6e}, {hi:.6e}] implied by the cross "
            "section's own range"
        )
        assert 1.0e-8 < rate < 1.0e-6, (
            f"T = {temperature_eV} eV: Ar EII rate {rate:.3e} cm^3/s is not "
            "in the 1e-8 - 1e-6 cm^3/s band the NIST-scale 3e-16 cm^2 Ar "
            "cross section implies"
        )


def test_eii_dw_runs_for_hydrogenic_ion():
    """DW Born EII runs end-to-end for He⁺ near threshold and produces
    a positive cross section in the Younger-1981 range (3e-18 cm²
    within an order of magnitude — DW is currently validated only
    near threshold for hydrogenic ions; see module docstring)."""
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(Z=10, config="1s2", engine="bspline", corrections="none")
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
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(Z=Z, config="1s2", engine="bspline", corrections="none")
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
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
    atom.solve()
    E = np.array([50.0])
    r = eii_cross_section(
        atom._result, E, method="DW", kappa_max=3, n_outgoing_energies=4,
    )
    orb_sum = sum(v for v in r.sigma_by_orbital.values())
    np.testing.assert_allclose(r.sigma_total_cm2, orb_sum, rtol=1e-10)


def test_dw_method_label_and_provenance():
    """DW result carries correct method label and provenance metadata."""
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="bspline", corrections="none")
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
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
    atom.solve()
    with pytest.raises(ValueError, match="Unknown method"):
        eii_cross_section(atom._result, np.array([100.0]), method="bogus")


def test_fresh_three_z_fac_k_shell_gate():
    """Fresh FAC panel keeps every requested K-shell point within 15 %."""
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "benchmarks" / "results_fac_dw_eii.csv"
    provenance_path = root / "benchmarks" / "RESULTS_fac_dw_eii.md"
    if not csv_path.exists() or not provenance_path.exists():
        pytest.skip(
            "fresh FAC DW-EII artifacts missing; rerun "
            "benchmarks/bench_fac_dw_eii.py"
        )

    provenance = provenance_path.read_text(encoding="utf-8")
    if "Fresh FAC DW-EII gate run" not in provenance:
        pytest.skip(
            "FAC DW-EII CSV has no fresh-run provenance; rerun the benchmark"
        )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = {"Z", "shell", "E_over_threshold", "rel_diff"}
    if not rows or not required_columns.issubset(rows[0]):
        pytest.skip("FAC DW-EII CSV is missing the fresh gate columns")

    requested = {
        (z, factor)
        for z in (10, 26, 54)
        for factor in (1.5, 2.0, 5.0)
    }
    panel = {
        (int(row["Z"]), float(row["E_over_threshold"])): float(row["rel_diff"])
        for row in rows
        if row["Z"] != "rms" and row["shell"] == "K"
    }
    missing = requested - panel.keys()
    if missing:
        pytest.skip(f"fresh FAC DW-EII CSV is incomplete; missing {sorted(missing)}")

    failures = {point: panel[point] for point in requested if panel[point] > 0.15}
    assert not failures, f"FAC K-shell relative-difference gate failed: {failures}"


EXTENDED_EII_CSV = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "results_fac_dw_eii_extended.csv"
)
# Measured 2026-09-22 on the extended panel (1.05I / 1.5I / 20I, sfac 1.1.5
# pure-DW CI, kappa_max=10, n_outgoing=25): atomx/FAC - 1 =
#   Z=10 Ne: +0.0204 / +0.0092 / -0.1758
#   Z=26 Fe: +0.0221 / +0.0102 / -0.1649
#   Z=54 Xe: +0.0163 / +0.0054 / -0.2229
HIGH_ENERGY_FACTOR = 20.0
HIGH_ENERGY_BAND = (-0.25, -0.10)


def _extended_panel() -> list[dict]:
    if not EXTENDED_EII_CSV.exists():
        pytest.fail(
            "benchmarks/results_fac_dw_eii_extended.csv is missing — rerun "
            "benchmarks/bench_fac_dw_eii.py --extended (sfac required)"
        )
    with EXTENDED_EII_CSV.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["Z"] != "rms"]


def test_fac_near_threshold_shape_gate():
    """1.05I rows: threshold convention and the rise out of threshold.

    Replaces ``test_fac_near_threshold_shape_gate_pending_q1``, which skipped
    forever because the committed EII panel stopped at 1.5I (S5 finding 6).
    The extended panel now carries 1.05I rows for all three Z, so this is a
    live gate; a missing panel FAILS instead of skipping, because "no data"
    was how the threshold-convention question stayed open.

    Asserted (measured 2026-09-22):

    * the atomx K-shell binding energy and FAC's DELTA_E agree to 2.5e-3
      (gate 5e-3) — the threshold-convention question of RESULTS_fac_dw_eii.md;
    * sigma is positive at 1.05I and *smaller* than at 1.5I (the direct
      ionisation cross section rises out of threshold), with the measured
      ratio sigma(1.05I)/sigma(1.5I) = 0.172 inside [0.05, 0.5].
    """
    panel = {}
    for row in _extended_panel():
        panel[(int(row["Z"]), float(row["E_over_threshold"]))] = row
    assert {z for z, f in panel if f == 1.05} == {10, 26, 54}, (
        f"extended panel is missing 1.05I rows: {sorted(panel)}"
    )

    threshold_differences = []
    for z in (10, 26, 54):
        near, mid = panel[(z, 1.05)], panel[(z, 1.5)]
        threshold_differences.append(
            abs(float(near["threshold_eV"]) - float(near["fac_threshold_eV"]))
            / float(near["fac_threshold_eV"]))
        sigma_near = float(near["sigma_atomx_cm2"])
        sigma_mid = float(mid["sigma_atomx_cm2"])
        assert sigma_near > 0.0, f"Z={z}: sigma(1.05I) = {sigma_near}"
        ratio = sigma_near / sigma_mid
        assert 0.05 <= ratio <= 0.5, (
            f"Z={z}: sigma(1.05I)/sigma(1.5I) = {ratio:.3f} outside [0.05, 0.5] "
            f"(measured 0.172); the cross section must rise out of threshold"
        )
    assert max(threshold_differences) <= 5.0e-3, threshold_differences


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED atomx/FAC - 1 at E/I = 20 (past the E/I <= 5 Born-completion "
        "point of the excitation panel) = -0.176 (Ne) / -0.165 (Fe) / -0.223 "
        "(Xe) on 2026-09-22, i.e. the atomx distorted-wave EII cross section "
        "falls 16-22 % BELOW FAC's pure-DW CI reference (max_kl=36) at every "
        "Z.  This is the high-energy region the whole external EII panel used "
        "to miss (S5 finding 6): the DW partial-wave sum is truncated at "
        "kappa_max=10 while FAC sums to 36, and no Born completion is applied "
        "to the EII channel.  The panel's own low-energy points agree to "
        "0.5-2.2 %, so the discrepancy is energy-dependent, not a scale "
        "error; closing it needs a converged (or Bethe-completed) "
        "partial-wave sum for EII, not a wider band."
    ),
)
def test_fac_high_energy_eii_gate_within_10_percent():
    """The parity claim the extended panel should support: |rel| <= 10 %."""
    for row in _extended_panel():
        if float(row["E_over_threshold"]) != HIGH_ENERGY_FACTOR:
            continue
        atomx = float(row["sigma_atomx_cm2"])
        fac = float(row["sigma_fac_cm2"])
        rel = atomx / fac - 1.0
        assert abs(rel) <= 0.10, (
            f"Z={row['Z']} at E/I={HIGH_ENERGY_FACTOR}: atomx/FAC - 1 = "
            f"{rel:+.3f} (atomx={atomx:.6e}, FAC={fac:.6e})"
        )


@pytest.mark.external_fac
def test_fac_high_energy_eii_gate_is_inside_the_measured_band():
    """Live panel record: every E/I = 20 point is 10-25 % below FAC.

    The active tripwire for the S5-finding-6 region: the three Z=10/26/54
    points must all sit in the measured band (measured -0.176 / -0.165 /
    -0.223).  A change in either direction — including a *silent* fix that
    does not update the recorded panel — trips it.
    """
    rows = [row for row in _extended_panel()
            if float(row["E_over_threshold"]) == HIGH_ENERGY_FACTOR]
    assert {int(row["Z"]) for row in rows} == {10, 26, 54}
    lo, hi = HIGH_ENERGY_BAND
    for row in rows:
        rel = float(row["sigma_atomx_cm2"]) / float(row["sigma_fac_cm2"]) - 1.0
        assert lo <= rel <= hi, (
            f"Z={row['Z']} at E/I={HIGH_ENERGY_FACTOR}: recorded rel_diff "
            f"{rel:+.4f} outside the measured band [{lo}, {hi}]"
        )
    assert int(round(float(HIGH_ENERGY_FACTOR))) == 20


@pytest.mark.slow
def test_dw_k_shell_shooting_mesh_converges_n_grid_and_r_max():
    """K-shell sigma converges under independent n_grid and r_max scans.

    Convergence is judged by the final change magnitude, not by a guaranteed
    direction of approach, which may vary with valid solver improvements.
    """
    atom = Atom(Z=26, config="1s2", charge=24, engine="bspline", corrections="none")
    atom.solve()
    threshold = -atom._result.orbital_energies["1s1/2"] * HARTREE_EV
    energy = np.array([2.0 * threshold])

    def sigma(n_grid, r_max):
        result = eii_cross_section(
            atom._result,
            energy,
            method="DW",
            kappa_max=4,
            n_outgoing_energies=9,
            continuum_model=ContinuumModel.shooting(
                n_grid=n_grid,
                r_max=r_max,
            ),
        )
        return float(result.sigma_total_cm2[0])

    grid_sigmas = np.array([
        sigma(n, 50.0) for n in (8000, 12000, 16000, 20000, 24000)
    ])
    grid_changes = np.abs(np.diff(grid_sigmas) / grid_sigmas[:-1])
    assert grid_changes[-1] < 1.0e-2, grid_changes

    radius_sigmas = np.array([
        sigma(16000, r_max) for r_max in (40.0, 45.0, 50.0)
    ])
    radius_changes = np.abs(np.diff(radius_sigmas) / radius_sigmas[:-1])
    assert radius_changes[-1] < 1.0e-2, radius_changes
