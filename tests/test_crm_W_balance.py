"""Tungsten CRM balance and UTA bridge regression tests."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_fac_crm_W as bench_w  # noqa: E402
from bench_fac_crm_W import (  # noqa: E402
    BALANCE_TEV,
    RESULTS_CSV,
    W_CHARGES,
    FACIonReference,
    FACPanelReference,
    build_w_ion_fixture,
    build_w_panel,
    write_results_csv,
)


def _load_committed_fac_panel_reference(path: Path = RESULTS_CSV) -> FACPanelReference:
    """Load the committed FAC-derived W panel fixture for CI-safe tests."""
    ions: dict[int, FACIonReference] = {}
    ips: dict[int, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            charge = int(row["charge"])
            if charge in ions:
                continue
            ips[charge] = float(row["fac_ip_eV"])
            ions[charge] = FACIonReference(
                charge=charge,
                electrons=bench_w.Z_W - charge,
                open_config=row["fac_open_config"],
                e0_eV=float(row["fac_e0_eV"]),
                n_levels=int(row["fac_levels"]),
                n_transitions=int(row["fac_transitions"]),
                transition_power_eVs=float(row["fac_transition_power_eVs"]),
                elapsed_s=0.0,
            )
    missing = set(W_CHARGES) - set(ions)
    if missing:
        raise RuntimeError(
            f"incomplete FAC W fixture {path}: missing W charges {sorted(missing)}"
        )
    return FACPanelReference(ions=ions, ionization_potentials_eV=ips)


def test_w_panel_has_expected_charge_states_and_configs():
    panel = build_w_panel()
    assert [item.charge for item in panel] == W_CHARGES
    for item in panel:
        assert item.atom is None
        assert len(item.configs) >= 20
        assert item.crm.Z == 74
        assert item.crm.charge == item.charge


def test_w_fractional_abundance_fixture_is_well_formed_and_documents_the_fac_copy(tmp_path):
    """Fixture/CSV structure — NOT an atomx-vs-FAC comparison.

    The atomx fractional-abundance column is a verbatim copy of the FAC
    fixture (``bench_fac_crm_W``: ``abund = dict(fac_abund)``), so any
    atomx-vs-FAC RMS is identically zero.  That copy is asserted here *as
    the documented placeholder it is*: the equality below is what a real
    atomx W coronal balance will break, and breaking it must be accompanied
    by removing the placeholder note in ``benchmarks/bench_fac_crm_W.py``.
    There is no independent physics gate in this file.
    """
    fac_ref = _load_committed_fac_panel_reference()
    out = tmp_path / "w.csv"
    metrics = write_results_csv(out, fac_ref=fac_ref)
    assert metrics["fac_structure_runs"] == float(len(W_CHARGES))
    assert metrics["fac_transition_rows"] > 0.0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(BALANCE_TEV) * len(W_CHARGES)
    for row in rows:
        fraction = float(row["atomx_fraction"])
        assert fraction >= 0.0
        assert float(row["fac_fraction"]) >= 0.0
        # documented placeholder: the "atomx" column IS the FAC column
        assert fraction == float(row["fac_fraction"])
        assert float(row["fraction_rel_error"]) == 0.0
        assert float(row["fac_ip_eV"]) > 0.0
        assert int(row["fac_transitions"]) > 0
        assert float(row["fac_transition_power_eVs"]) > 0.0
        assert row["fac_source"].startswith("sfac Structure/TransitionTable/PrintTable")
        assert row["adas_fraction"] == ""


def test_w_results_writer_defaults_to_live_fac_reference(monkeypatch, tmp_path):
    """The benchmark default stays live-FAC; tests must inject fixtures explicitly."""
    fac_ref = _load_committed_fac_panel_reference()
    calls = 0

    def fake_live_fac_reference() -> FACPanelReference:
        nonlocal calls
        calls += 1
        return fac_ref

    monkeypatch.setattr(bench_w, "run_fac_panel_reference", fake_live_fac_reference)
    metrics = bench_w.write_results_csv(tmp_path / "w.csv")
    assert calls == 1
    assert metrics["fac_transition_rows"] > 0.0


def test_w_mn_shell_spectrum_uses_uta_for_large_arrays():
    from atomx.spectrum import crm_emission_spectrum

    fixture = build_w_ion_fixture(40)
    spec = crm_emission_spectrum(
        fixture.crm,
        n_e_cm3=1e14,
        T_e_eV=120.0,
        lam_min_nm=1.0,
        lam_max_nm=200.0,
        n_points=4001,
        ion_mass_amu=183.84,
        uta_threshold=20,
    )
    assert spec.integrate() > 0.0
    assert spec.metadata["uta_arrays"]
    assert spec.metadata["uta_arrays"][0]["n_lines"] > 20


def _bundled_array_power(fixture, pop, lam_min_nm, lam_max_nm):
    """Sum rule bookkeeping for the one bundled array of the W fixture.

    Reproduces the CRM's own bundling (a config pair with more than 20
    transitions) independently of ``crm_synth`` and returns
    ``(sum_strength_energy, centroid_eV, sigma_nm, n_lines)`` where
    ``sum_strength_energy = sum_i n_u(i) A_ul(i) E_ul(i)`` over the bundled
    transitions (units of the spectrum's arbitrary emissivity).
    """
    from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM

    groups = {}
    for tr in fixture.crm.transitions:
        dE_eV = (
            fixture.crm.levels[tr.upper].energy_eV
            - fixture.crm.levels[tr.lower].energy_eV
        )
        strength = pop[tr.upper] * tr.A_rate
        if dE_eV <= 0.0 or strength <= 0.0:
            continue
        lam_nm = HARTREE_EV / (HARTREE_INV_ANGSTROM * dE_eV) / 10.0
        key = (
            fixture.crm.levels[tr.upper].config,
            fixture.crm.levels[tr.lower].config,
        )
        groups.setdefault(key, []).append((dE_eV, strength, lam_nm))

    bundled = [rows for rows in groups.values() if len(rows) > 20]
    assert len(bundled) == 1  # exactly one UTA array in this fixture
    rows = bundled[0]
    # every bundled line is already inside the window: the residual below is
    # therefore the Gaussian's tails, not lines falling out of band
    assert all(lam_min_nm <= r[2] <= lam_max_nm for r in rows)

    total_strength = sum(r[1] for r in rows)
    centroid = sum(r[1] * r[0] for r in rows) / total_strength
    variance = sum(r[1] * (r[0] - centroid) ** 2 for r in rows) / total_strength
    std_eV = math.sqrt(variance)
    sigma_nm = 1239.841984 * std_eV / centroid ** 2
    return total_strength * centroid, centroid, sigma_nm, len(rows)


def test_uta_integrated_power_matches_explicit_fixture_within_window_tails():
    """UTA vs explicit power: the sum rule + the analytic Gaussian tail loss.

    ``src/atomx/spectrum/uta.py`` renders a whole bundled array as ONE
    Gaussian of integrated intensity ``total_strength * centroid_eV``.
    Because ``centroid_eV`` is the strength-weighted mean transition energy,
    that intensity *is* the sum rule ``sum_i n_u A_ul E_ul`` over the
    bundled lines — measured here against an independent recomputation from
    the CRM's own populations and transitions (relative 0, i.e. the
    rounding of the metadata columns).  The residual against the explicit
    per-line rendering is therefore not unconserved power: it is the part of
    the single broad Gaussian that falls outside the wavelength window.
    For the W(40+) fixture at T_e = 120 eV the array has centroid 50.81 eV
    (24.40 nm) and sigma_lambda = 19.62 nm, so 11.65 % of its normalised
    Gaussian lies outside [1, 200] nm — reproduced below from the erf
    fraction to 2e-8.

    Measured explicit-vs-UTA integrated power: 11.617 % (1-200 nm),
    11.166 % (1-2000 nm), 7.867 % (1-20000 nm).  The old gate accepted
    25 %, i.e. 3x the effect it was bounding.
    """
    from scipy.special import erf

    from atomx.spectrum import crm_emission_spectrum

    lam_min_nm, lam_max_nm, n_points = 1.0, 200.0, 12001
    fixture = build_w_ion_fixture(40)
    common = dict(
        n_e_cm3=1e14,
        T_e_eV=120.0,
        lam_min_nm=lam_min_nm,
        lam_max_nm=lam_max_nm,
        n_points=n_points,
        ion_mass_amu=183.84,
    )
    explicit = crm_emission_spectrum(
        fixture.crm, uta_threshold=10_000, force_uta=False, **common)
    uta = crm_emission_spectrum(
        fixture.crm, uta_threshold=20, force_uta=True, **common)

    rel = abs(uta.integrate() - explicit.integrate()) / explicit.integrate()
    assert math.isfinite(rel)
    assert rel <= 0.12  # measured 0.11617 (was a 0.25 gate)

    pop = fixture.crm.steady_state(1e14, 120.0)
    array_power, centroid_eV, sigma_nm, n_lines = _bundled_array_power(
        fixture, pop, lam_min_nm, lam_max_nm)
    meta = uta.metadata["uta_arrays"][0]
    assert meta["n_lines"] == n_lines
    assert pytest.approx(array_power, rel=1e-12, abs=0.0) == (
        meta["total_strength"] * meta["centroid_eV"])
    assert pytest.approx(centroid_eV, rel=1e-12, abs=0.0) == meta["centroid_eV"]

    # the whole residual is the analytic tail of that Gaussian
    lam0 = 1239.841984 / centroid_eV
    captured = 0.5 * (
        erf((lam_max_nm - lam0) / (np.sqrt(2.0) * sigma_nm))
        - erf((lam_min_nm - lam0) / (np.sqrt(2.0) * sigma_nm))
    )
    assert captured == pytest.approx(0.88346, rel=1e-4, abs=0.0)  # measured
    assert pytest.approx(captured, rel=1e-6, abs=0.0) == (
        uta.integrate() / array_power)
