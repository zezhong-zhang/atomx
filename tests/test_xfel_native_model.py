"""Independent balance, threshold and spectral-moment production gates."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, wofz
from sympy import binomial

from atomx.config.subshell import Subshell
from atomx.constants import ALPHA, BOHR_RADIUS_M, HARTREE_EV
from atomx.xfel import (
    GaussianPulse,
    KineticGraph,
    NativeXFELModel,
    ResonantLine,
    ScatteringFactorBank,
    configuration_weight,
    retabulate_native_pi,
)
from atomx.xfel.cascade import _table_key
from atomx.xfel.model import gaussian_spectral_quadrature

G, X = _table_key({(1, -1): 1}), _table_key({(2, -2): 1})


def control():
    energy = 0.375 * HARTREE_EV
    rows = {
        G: dict(
            occupations={"1,-1": 1},
            energy_hartree=-0.5,
            sigma_cm2={"1,-1": 0},
            auger=[],
            radiative=[],
        ),
        X: dict(
            occupations={"2,-2": 1},
            energy_hartree=-0.125,
            sigma_cm2={"2,-2": 0},
            auger=[],
            radiative=[
                dict(hole=[1, -1], source=[2, -2], rate=6.2683e8, energy_eV=energy)
            ],
        ),
    }
    table = dict(
        Z=1,
        base_nelec=1,
        photon_energy_eV=energy,
        configs=rows,
        provenance={"control": "hydrogenic levels"},
    )

    def loader(c):
        return SimpleNamespace(
            converged=True,
            total_energy=rows[_table_key(c)]["energy_hartree"],
            Z=1,
            subshells=[
                Subshell(1, -1, c.get((1, -1), 0)),
                Subshell(2, -2, c.get((2, -2), 0)),
            ],
        )

    return table, loader


def constant_pi(monkeypatch, sigma=1e-20):
    from atomx.xfel import model, observables

    area = (BOHR_RADIUS_M * 100) ** 2

    def provider(result, energies, *, thresholds_eV, **kwargs):
        by = {
            s.label: np.where(
                np.asarray(energies) > thresholds_eV[s.label],
                s.occupation * sigma / area,
                0.0,
            )
            for s in result.subshells
            if s.occupation
        }
        total = sum(by.values(), np.zeros_like(energies, dtype=float))
        return SimpleNamespace(
            sigma_by_orbital_a0_sq=by,
            sigma_total_a0_sq=total,
            sigma_total_cm2=total * area,
        )

    monkeypatch.setattr(model, "subshell_photoionization", provider)
    monkeypatch.setattr(observables, "subshell_photoionization", provider)


def test_split_quadrature_matches_exact_above_threshold_gaussian_fraction(monkeypatch):
    table, loader = control()
    constant_pi(monkeypatch)
    threshold = HARTREE_EV * 0.5
    for shift in (-0.2, 0, 0.2):
        e, width = threshold + shift, 0.5
        out = retabulate_native_pi(
            table,
            loader,
            photon_energy_eV=e,
            bandwidth_fwhm_eV=width,
            continuum_model=None,
        )
        expected = 1e-20 * ndtr((e - threshold) / (width / np.sqrt(8 * np.log(2))))
        assert out["configs"][G]["sigma_cm2"]["1,-1"] == pytest.approx(
            expected, rel=1e-10
        )
    assert table["configs"][G]["sigma_cm2"]["1,-1"] == 0


def test_native_pi_occupation_is_included_exactly_once(monkeypatch):
    table, _ = control()
    pair = _table_key({(1, -1): 2})
    table["configs"] = {
        G: table["configs"][G],
        pair: dict(
            occupations={"1,-1": 2}, energy_hartree=-0.8, auger=[], sigma_cm2={}
        ),
    }
    constant_pi(monkeypatch)

    def loader(c):
        return SimpleNamespace(
            converged=True,
            total_energy=table["configs"][_table_key(c)]["energy_hartree"],
            subshells=[Subshell(1, -1, c[(1, -1)])],
        )

    out = retabulate_native_pi(
        table, loader, photon_energy_eV=100, bandwidth_fwhm_eV=1, continuum_model=None
    )
    assert out["configs"][pair]["sigma_cm2"]["1,-1"] == pytest.approx(2e-20)
    assert out["configs"][G]["sigma_cm2"]["1,-1"] == pytest.approx(1e-20)


def test_native_driver_generates_resonances_and_preserves_actual_charge(monkeypatch):
    table, loader = control()
    constant_pi(monkeypatch)
    model = NativeXFELModel(
        table,
        {(1, -1): 1},
        loader,
        continuum_model=None,
        photon_energy_eV=table["photon_energy_eV"],
        bandwidth_fwhm_eV=0.5,
    )
    assert len(model.lines) == 1
    assert model.lines[0].g_upper / model.lines[0].g_lower == 2
    result = model.run(
        GaussianPulse(table["photon_energy_eV"], 1e14, 10, bandwidth_fwhm_eV=0.5),
        atol=1e-15,
    )
    assert result.dynamics.mean_charge[-1] > 0
    assert result.imaging is None
    with pytest.raises(ValueError, match="together"):
        model.run(
            GaussianPulse(table["photon_energy_eV"], 1e14, 10, bandwidth_fwhm_eV=0.5),
            q_a0_inv=[0],
        )


def test_configuration_weights_obey_independent_sympy_einstein_population_balance():
    # Upper loses an a electron and fills h on emission. Compare both
    # configurations' statistical determinants against independent sympy.
    for gh, ga in ((2, 4), (4, 2)):
        for qh in range(gh):
            for qa in range(1, ga + 1):
                upper = {(1, -gh // 2): qh, (2, -ga // 2): qa}
                lower = {(1, -gh // 2): qh + 1, (2, -ga // 2): qa - 1}
                ratio = configuration_weight(upper) / configuration_weight(lower)
                sym = float(
                    binomial(gh, qh)
                    * binomial(ga, qa)
                    / (binomial(gh, qh + 1) * binomial(ga, qa - 1))
                )
                assert ratio == sym
                emission = qa * (gh - qh) / gh
                assert ratio * emission == pytest.approx((qh + 1) * (ga - qa + 1) / gh)


def test_causal_line_matches_optical_theorem_area_and_zero_frequency_sum():
    line = ResonantLine(
        "g", "x", 10, 0.4, 2e15, 2, 4, "analytic oscillator", lower_decay_per_s=1e15
    )
    assert line.anomalous_factor(0) == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="nonpositive energies"):
        line.anomalous_factor(0.1, bandwidth_fwhm_eV=1)
    for e in (5, 10, 15):
        cross = line.sigma_cm2(e) / (BOHR_RADIUS_M * 100) ** 2
        assert line.anomalous_factor(e).imag == pytest.approx(
            e / HARTREE_EV * cross / (4 * np.pi * ALPHA), rel=1e-12
        )
    area = quad(lambda e: line.sigma_cm2(e), 0, np.inf, epsabs=0, epsrel=1e-10)[0]
    assert area == pytest.approx(line.integrated_sigma_cm2_eV, rel=1e-10)
    trk = (
        quad(lambda e: line.anomalous_factor(e).imag / e, 0, np.inf, epsabs=1e-11)[0]
        * 2
        / np.pi
    )
    assert trk == pytest.approx(0.4, rel=1e-10)


def test_line_uses_lower_state_lifetime_and_gaussian_cross_section(monkeypatch):
    line = ResonantLine(
        "g", "x", 10, 0.4, 1e14, 2, 4, "broad control", lower_decay_per_s=1e14
    )
    narrow = ResonantLine("g", "x", 10, 0.4, 1e14, 2, 4, "control")
    assert line.sigma_cm2(10) == pytest.approx(narrow.sigma_cm2(10) / 2)
    sigma = 0.4 / np.sqrt(8 * np.log(2))
    expected = quad(
        lambda e: (
            line.sigma_cm2(e)
            * np.exp(-0.5 * ((e - 10) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        ),
        10 - 9 * sigma,
        10 + 9 * sigma,
        epsabs=0,
        epsrel=1e-10,
    )[0]
    assert line.sigma_cm2(10, bandwidth_fwhm_eV=0.4) == pytest.approx(
        expected, rel=1e-10
    )


def test_narrow_line_spectral_intensity_is_not_squared_mean(monkeypatch):
    from atomx.xfel import observables

    table, loader = control()
    constant_pi(monkeypatch, sigma=0)
    monkeypatch.setattr(
        observables,
        "native_form_factor",
        lambda result, q: np.ones_like(q, dtype=float),
    )
    model = NativeXFELModel(
        table,
        {(1, -1): 1},
        loader,
        continuum_model=None,
        photon_energy_eV=table["photon_energy_eV"],
        bandwidth_fwhm_eV=0.5,
    )
    banks = [
        model.scattering_bank(
            [0], dispersion_grid_eV=np.geomspace(0.01, 1000, 100), spectral_order=n
        )
        for n in (32, 64)
    ]
    line = model.lines[0]
    sigma = 0.5 / np.sqrt(8 * np.log(2))
    gamma = line.half_width_eV
    expected = (
        (line.oscillator_strength * line.energy_eV / 2) ** 2
        * np.sqrt(np.pi / 2)
        / sigma
        / gamma
        * wofz(1j * gamma / (np.sqrt(2) * sigma)).real
    )
    assert banks[0].intensities[G][0] == pytest.approx(expected, rel=1e-6)
    assert banks[0].intensities[G][0] == pytest.approx(
        banks[1].intensities[G][0], rel=1e-9
    )
    assert banks[0].intensities[G][0] > abs(banks[0].factors[G][0]) ** 2 * 1e4
    pulse = GaussianPulse(line.energy_eV, 1e14, 10, bandwidth_fwhm_eV=0.5)
    signal = banks[0].observe(model.solve(pulse), pulse)
    assert np.min(signal.fluctuation_intensity) >= 0
    with pytest.raises(ValueError, match="bandwidth"):
        banks[0].observe(model.solve(pulse), GaussianPulse(line.energy_eV, 1e14, 10))


def test_invalid_archived_or_mixed_cross_sections_fail(monkeypatch):
    table, loader = control()
    constant_pi(monkeypatch)
    invalid = copy.deepcopy(table)
    invalid["configs"][G]["double_pi"] = [{"source": "unknown"}]
    with pytest.raises(ValueError, match="unsplit"):
        retabulate_native_pi(
            invalid,
            loader,
            photon_energy_eV=10,
            bandwidth_fwhm_eV=0.5,
            continuum_model=None,
        )
    with pytest.raises(ValueError, match="nonpositive"):
        gaussian_spectral_quadrature(1, 10)
    with pytest.raises(ValueError, match="inequality"):
        ScatteringFactorBank(
            [0],
            {"g": [2]},
            photon_energy_eV=10,
            provenance={"test": True},
            intensities_by_state={"g": [1]},
        )
    broken = copy.deepcopy(table)
    del broken["configs"][X]
    with pytest.raises(ValueError):
        KineticGraph.from_rate_table(broken, {(1, -1): 1}, initial_config={(2, -2): 1})


def test_spectral_shake_gate_uses_double_threshold_not_center_energy(monkeypatch):
    from atomx.xfel import model
    from atomx.xfel.native_channels import add_photo_shakeoff

    two, one = _table_key({(1, -1): 2}), _table_key({(1, -1): 1})
    rows = {
        two: dict(
            occupations={"1,-1": 2},
            energy_hartree=-2.0,
            sigma_cm2={"1,-1": 0},
            auger=[],
        ),
        one: dict(
            occupations={"1,-1": 1},
            energy_hartree=-0.8,
            sigma_cm2={"1,-1": 0},
            auger=[],
        ),
    }
    table = dict(
        Z=2, base_nelec=2, photon_energy_eV=2 * HARTREE_EV, configs=rows, provenance={}
    )

    def loader(c):
        sid = _table_key(c)
        vector = (
            np.array([1.0, 0.0])
            if sid == two
            else np.array([0.99, np.sqrt(1 - 0.99**2)])
        )
        return SimpleNamespace(
            converged=True,
            total_energy=rows[sid]["energy_hartree"],
            subshells=[Subshell(1, -1, c[(1, -1)])],
            _bsp_data={"t": np.array([0, 1]), "grw": np.ones(2)},
            orbitals_by_subshell=lambda: {(1, -1): (vector, np.zeros(2), c[(1, -1)])},
        )

    constant_pi(monkeypatch)
    monkeypatch.setattr(model, "eval_P", lambda p, _bd: p)
    monkeypatch.setattr(model, "eval_Q", lambda q, _bd: q)
    result = retabulate_native_pi(
        table,
        loader,
        photon_energy_eV=2 * HARTREE_EV,
        bandwidth_fwhm_eV=1.0,
        continuum_model=None,
        shakeoff=True,
    )
    row = result["configs"][two]
    doubles = sum(ch["sigma_cm2"] for ch in row["double_pi"])
    assert doubles == pytest.approx(2e-20 * (1 - 0.99**2) / 2, rel=1e-10)
    assert sum(row["sigma_cm2"].values()) + doubles == pytest.approx(2e-20, rel=1e-10)
    with pytest.raises(ValueError, match="broadband"):
        add_photo_shakeoff(result, loader)


def test_archive_hash_gate_precedes_scf_loading(tmp_path):
    import hashlib
    import json

    from atomx.xfel import NativeRateArchive
    from atomx.xfel.atomic_rates import _state_path

    table, _loader = control()
    identity = {"controls": {"Z": 1}, "test": "independent data integrity gate"}
    table["provenance"]["identity"] = identity
    p = tmp_path / "table.json"
    p.write_text(json.dumps(table))
    stem = _state_path(tmp_path, {(1, -1): 1})
    stem.with_suffix(".h5").write_bytes(b"changed archive bytes")
    stem.with_suffix(".json").write_text(
        json.dumps(
            dict(
                identity=identity,
                converged=True,
                h5_sha256=hashlib.sha256(b"original archive bytes").hexdigest(),
            )
        )
    )
    archive = NativeRateArchive(p, tmp_path)
    with pytest.raises(ValueError, match="hash"):
        archive.result({(1, -1): 1})


def test_prepared_observables_reuse_atomic_spectra_between_fluences(monkeypatch):
    from atomx.xfel import observables

    table, loader = control()
    constant_pi(monkeypatch, sigma=0)
    monkeypatch.setattr(
        observables,
        "native_form_factor",
        lambda result, q: np.ones_like(q, dtype=float),
    )
    model = NativeXFELModel(
        table,
        {(1, -1): 1},
        loader,
        continuum_model=None,
        photon_energy_eV=table["photon_energy_eV"],
        bandwidth_fwhm_eV=0.5,
    )
    bank = model.scattering_bank([0], dispersion_grid_eV=np.geomspace(0.01, 1000, 100))

    def forbidden(*args, **kwargs):
        raise AssertionError("repeated atomic spectra during a fluence scan")

    monkeypatch.setattr(observables, "subshell_photoionization", forbidden)
    for fluence in (1e13, 1e14):
        simulation = model.run(
            GaussianPulse(
                table["photon_energy_eV"], fluence, 10, bandwidth_fwhm_eV=0.5
            ),
            q_a0_inv=[0],
            dispersion_grid_eV=np.geomspace(0.01, 1000, 100),
        )
        assert simulation.scattering_bank is bank
        assert simulation.imaging is not None
