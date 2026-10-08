"""Independent reference-stream, table-closure and analytic XFEL MC gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.xfel import (
    CompiledRateTable,
    Pulse,
    RateProvider,
    TableRateProvider,
    cascade_mc,
)
from atomx.xfel.cascade import build_rate_table, config_label, enumerate_configs

BASE = {(1, -1): 2.0, (2, -1): 2.0, (2, 1): 2.0, (2, -2): 4.0}
ARCHIVE = Path(__file__).parent / "data/ne_1keV_rate_table.json"


class GraphProvider(RateProvider):
    """Positive PI and branching Auger on a complete integer graph.

    These are synthetic mechanics rates, not an atomic reference panel.
    """

    def __init__(self):
        super().__init__(lambda label: None)

    def pi_sigmas_cm2(self, config, photon_energy_eV):
        return {s: q * (2e-20 if s == (1, -1) else 1e-20) for s, q in config.items()}

    def auger_channels(self, config):
        if config.get((1, -1), 0) >= 2:
            return []
        channels = []
        for source, rate in (((2, -1), 1e14), ((2, -2), 3e14)):
            if config.get(source, 0) >= 2:
                channels.append(((1, -1), source, source, rate))
        return channels


@pytest.fixture(scope="module")
def complete_table():
    return build_rate_table(None, BASE, 1000, max_charge=10, provider=GraphProvider())


def assert_reference_equal(reference, actual):
    for name in (
        "charge_distribution",
        "mean_charge",
        "mean_absorbed",
        "mean_emitted",
        "configs_visited",
    ):
        assert getattr(actual, name) == getattr(reference, name), name


@pytest.mark.parametrize("fluence", [0, 1e18, 1e20, 1e23])
@pytest.mark.parametrize("seed", [0, 7, 928])
def test_compiled_walk_matches_independent_python_rng_path(
    complete_table, fluence, seed
):
    pulse = Pulse(1000, fluence)
    reference = cascade_mc(
        None,
        BASE,
        pulse,
        provider=TableRateProvider(complete_table),
        n_trajectories=317,
        n_bins=16,
        seed=seed,
    )
    compiled = CompiledRateTable(complete_table, BASE)
    actual = compiled.run(pulse, n_trajectories=317, n_bins=16, seed=seed)
    split = compiled.run(pulse, n_trajectories=317, n_bins=16, seed=seed, batch_size=13)
    assert_reference_equal(reference, actual)
    assert_reference_equal(reference, split)
    assert not actual.missing_configurations
    assert actual.mean_emitted == pytest.approx(actual.mean_charge)


def test_analytic_single_electron_bin_survival():
    base = {(1, -1): 1}
    table = {
        "photon_energy_eV": 1000,
        "base_nelec": 1,
        "configs": {
            "one": {
                "occupations": {"1,-1": 1},
                "sigma_cm2": {"1,-1": 1e-20},
                "auger": [],
            },
        },
    }
    compiled = CompiledRateTable(table, base)
    result = compiled.run(
        Pulse(1000, 0.7e20), n_trajectories=100000, n_bins=16, seed=17
    )
    survival = (1 - 0.7 / 16) ** 16
    standard_error = np.sqrt(survival * (1 - survival) / result.n_trajectories)
    assert abs(result.charge_distribution[0] - survival) < 5 * standard_error
    assert result.mean_absorbed == result.mean_emitted == result.mean_charge


def test_serial_builder_preserves_distinct_jj_states(complete_table):
    # Independent count of integer occupations, including the bare ion.
    assert len(complete_table["configs"]) == (3 * 3 * 3 * 5) - 1 == 134
    configs = [r["occupations"] for r in complete_table["configs"].values()]
    first = {"1,-1": 2.0, "2,-1": 2.0, "2,1": 1.0, "2,-2": 2.0}
    second = {"1,-1": 2.0, "2,-1": 2.0, "2,1": 2.0, "2,-2": 1.0}
    assert first in configs and second in configs
    assert (
        config_label({(1, -1): 2, (2, -1): 2, (2, 1): 1, (2, -2): 2}) == "1s2 2s2 2p3"
    )


def test_parallel_builder_does_not_overwrite_merged_labels(monkeypatch):
    import concurrent.futures

    # The package exports a function named cascade; load its module explicitly.
    import importlib

    module = importlib.import_module("atomx.xfel.cascade")

    class SerialExecutor:
        def __init__(self, **kwargs):
            pass

        def map(self, function, jobs):
            return map(function, jobs)

        def shutdown(self, **kwargs):
            pass

    def worker(job):
        rows = []
        for occ in job["configs"]:
            config = {tuple(int(x) for x in k.split(",")): q for k, q in occ.items()}
            rows.append(
                {
                    "label": config_label(config),
                    "occupations": occ,
                    "sigma_cm2": {k: 1e-20 for k in occ},
                    "auger": [],
                }
            )
        return {"rows": rows, "solves": 1}

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", SerialExecutor)
    monkeypatch.setattr(module, "_rate_group_worker", worker)
    table = module.build_rate_table_parallel(10, BASE, 1000, n_workers=2, max_charge=10)
    assert len(table["configs"]) == 134
    CompiledRateTable(table, BASE)  # The complete graph passes the closure gate.


def test_archived_ne_table_missing_states_are_explicit():
    table = json.loads(ARCHIVE.read_text())
    with pytest.raises(ValueError, match="48 missing configurations"):
        CompiledRateTable(table, BASE)
    compiled = CompiledRateTable(table, BASE, on_missing="freeze")
    pulse = Pulse(1000, 1000 / 2.8002852e-17)
    result = compiled.run(pulse, n_trajectories=2000, n_bins=16, seed=17)
    reference = cascade_mc(
        None,
        BASE,
        pulse,
        n_trajectories=2000,
        n_bins=16,
        seed=17,
        provider=TableRateProvider(table),
    )
    assert_reference_equal(reference, result)
    assert result.missing_policy == "freeze" and result.missing_configurations


@pytest.mark.parametrize("field,value", [("sigma_cm2", -1), ("sigma_cm2", np.inf)])
def test_invalid_cross_sections_rejected(complete_table, field, value):
    table = copy.deepcopy(complete_table)
    row = next(iter(table["configs"].values()))
    row[field][next(iter(row[field]))] = value
    with pytest.raises(ValueError):
        CompiledRateTable(table, BASE)


@pytest.mark.parametrize(
    "option,value",
    [("n_bins", 0), ("n_trajectories", 0), ("batch_size", -1), ("n_bins", 2.5)],
)
def test_invalid_mc_controls_rejected(complete_table, option, value):
    compiled = CompiledRateTable(complete_table, BASE)
    with pytest.raises(ValueError):
        compiled.run(Pulse(1000, 1e20), **{option: value})


def test_table_identity_and_inputs_stay_frozen(complete_table):
    table = copy.deepcopy(complete_table)
    compiled = CompiledRateTable(table, BASE)
    pulse = Pulse(1000, 1e20)
    before = compiled.run(pulse, n_trajectories=100, seed=3)
    table["configs"].clear()
    after = compiled.run(pulse, n_trajectories=100, seed=3)
    assert before == after
    assert len(compiled.table_sha256) == 64
    assert not compiled.sigmas.flags.writeable
    with pytest.raises(ValueError, match="photon energy"):
        compiled.run(Pulse(2000, 1e20))
    with pytest.raises(ValueError, match="fluence"):
        compiled.run(Pulse(1000, -1))


def test_fractional_jj_occupations_are_not_discrete_mc_states(complete_table):
    base = dict(BASE)
    base[2, 1] = 0.5
    with pytest.raises(ValueError, match="integer jj"):
        CompiledRateTable(complete_table, base)


def test_ne_combinatorial_enumeration_is_jj_resolved():
    assert len(enumerate_configs(BASE, max_charge=10)) == 134


@pytest.mark.parametrize("max_charge", [1, 3, 12])
def test_charge_stop_and_long_pulses_match_reference(complete_table, max_charge):
    pulse = Pulse(1000, 1e21)
    reference = cascade_mc(
        None,
        BASE,
        pulse,
        provider=TableRateProvider(complete_table),
        n_trajectories=251,
        n_bins=64,
        seed=71,
        max_charge=max_charge,
    )
    actual = CompiledRateTable(complete_table, BASE).run(
        pulse,
        n_trajectories=251,
        n_bins=64,
        seed=71,
        max_charge=max_charge,
        batch_size=17,
    )
    assert_reference_equal(reference, actual)


def test_channel_order_is_part_of_seeded_table_identity(complete_table):
    altered = copy.deepcopy(complete_table)
    row = next(iter(altered["configs"].values()))
    row["sigma_cm2"] = dict(reversed(list(row["sigma_cm2"].items())))
    assert (
        CompiledRateTable(complete_table, BASE).table_sha256
        != CompiledRateTable(altered, BASE).table_sha256
    )
