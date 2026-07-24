"""Tungsten CRM balance and UTA bridge regression tests."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_fac_crm_W as bench_w  # noqa: E402
from bench_fac_crm_W import (  # noqa: E402
    BALANCE_TEV,
    COOLING_TEV,
    FACIonReference,
    FACPanelReference,
    RESULTS_CSV,
    W_CHARGES,
    build_w_ion_fixture,
    build_w_panel,
    putterich_2008_cooling_reference,
    tungsten_cooling_rate,
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


def test_w_fractional_abundance_gate_against_fac_fixture(tmp_path):
    fac_ref = _load_committed_fac_panel_reference()
    out = tmp_path / "w.csv"
    metrics = write_results_csv(out, fac_ref=fac_ref)
    # NOTE: the atomx fractional-abundance column is a verbatim copy of the FAC
    # fixture (see bench_fac_crm_W: `abund = dict(fac_abund)`), so the rms is
    # identically 0 — asserting `rms <= 0.30` validates nothing about atomx and
    # is intentionally NOT checked here. The remaining assertions validate the
    # committed FAC-derived fixture without requiring sfac in ordinary CI.
    assert metrics["fac_structure_runs"] == float(len(W_CHARGES))
    assert metrics["fac_transition_rows"] > 0.0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(BALANCE_TEV) * len(W_CHARGES)
    for row in rows:
        assert float(row["atomx_fraction"]) >= 0.0
        assert float(row["fac_fraction"]) >= 0.0
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


@pytest.mark.skip(
    reason="tungsten_cooling_rate is putterich_ref·(1+0.18·sin), so factor ≤ "
    "1.18 ≤ 2.0 by construction — validates the synthetic perturbation, not "
    "atomx. Needs a real atomx W cooling-curve computation. Review doc: "
    "validation-integrity."
)
def test_w_cooling_rate_within_putterich_factor_two():
    for Te in COOLING_TEV:
        atomx = tungsten_cooling_rate(Te)
        ref = putterich_2008_cooling_reference(Te)
        factor = max(atomx / ref, ref / atomx)
        assert factor <= 2.0


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


def test_uta_integrated_power_matches_explicit_fixture_within_25_percent():
    from atomx.spectrum import crm_emission_spectrum

    fixture = build_w_ion_fixture(40)
    explicit = crm_emission_spectrum(
        fixture.crm,
        n_e_cm3=1e14,
        T_e_eV=120.0,
        lam_min_nm=1.0,
        lam_max_nm=200.0,
        n_points=12001,
        ion_mass_amu=183.84,
        uta_threshold=10_000,
        force_uta=False,
    )
    uta = crm_emission_spectrum(
        fixture.crm,
        n_e_cm3=1e14,
        T_e_eV=120.0,
        lam_min_nm=1.0,
        lam_max_nm=200.0,
        n_points=12001,
        ion_mass_amu=183.84,
        uta_threshold=20,
        force_uta=True,
    )
    rel = abs(uta.integrate() - explicit.integrate()) / explicit.integrate()
    assert math.isfinite(rel)
    assert rel <= 0.25
