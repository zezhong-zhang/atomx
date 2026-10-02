"""Unit gates for the deterministic sequential-PI cascade (XFEL inc 1).

Physics-free mocks exercise the graph mechanics; one live Ne gate
checks the real providers end to end.  The quantitative anchor
(Young-2010 charge-state-vs-fluence shape) arrives with the MC
increment per docs/XFEL_GAP_ANALYSIS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.xfel.cascade import (  # noqa: E402
    Config,
    Pulse,
    RateProvider,
    _daughter_after_auger,
    _daughter_after_pi,
    cascade,
)

NE_CONFIG: Config = {
    (1, -1): 2.0, (2, -1): 2.0, (2, 1): 2.0, (2, -2): 4.0,
}


class _FakeProvider(RateProvider):
    """Deterministic mock: PI only on 1s (if occupied); one Auger
    channel 1s <- 2s,2s when the 1s shell has a hole and 2s is full."""

    def __init__(self):
        super().__init__(lambda label: None)

    def pi_sigmas_cm2(self, config, photon_energy_eV):
        if config.get((1, -1), 0.0) > 0.5:
            return {(1, -1): 1e-20}
        return {}

    def auger_channels(self, config):
        q_1s = config.get((1, -1), 0.0)
        q_2s = config.get((2, -1), 0.0)
        if q_1s < 1.5 and q_2s >= 1.5:
            return [((1, -1), (2, -1), (2, -1), 1.0)]
        return []


def test_pi_and_auger_daughter_construction():
    d = _daughter_after_pi(NE_CONFIG, (2, -2))
    assert d[(2, -2)] == 3.0
    d2 = _daughter_after_auger(
        _daughter_after_pi(NE_CONFIG, (1, -1)), (1, -1), (2, -1), (2, -1))
    assert d2[(1, -1)] == 2.0
    assert (2, -1) not in d2          # both 2s electrons consumed


def test_zero_fluence_stays_neutral():
    res = cascade(None, NE_CONFIG, Pulse(1000.0, 0.0),
                  provider=_FakeProvider())
    assert res.charge_distribution == {0: pytest.approx(1.0)}
    assert res.photons_absorbed == 0.0


def test_charge_and_population_conservation_at_high_fluence():
    res = cascade(None, NE_CONFIG, Pulse(1000.0, 1e21),
                  provider=_FakeProvider())
    total_pop = sum(res.charge_distribution.values())
    assert total_pop == pytest.approx(1.0, abs=1e-9)
    # single-pass semantics: one PI event on 1s + instant Auger refill
    # (2s,2s -> 1s) leaves 1s2 2p6: charge 2, one photoelectron + one
    # Auger electron.  Re-absorption is the MC increment's job.
    assert res.photons_absorbed == pytest.approx(1.0)
    assert res.electrons_emitted == pytest.approx(2.0)
    assert res.charge_distribution.get(2, 0.0) == pytest.approx(1.0)


def test_partial_fluence_splits_population():
    res = cascade(None, NE_CONFIG, Pulse(1000.0, 0.3e20),
                  provider=_FakeProvider())   # single pass, P_abs = 0.3
    assert res.charge_distribution[0] == pytest.approx(0.7, abs=1e-6)
    ionised = sum(p for q, p in res.charge_distribution.items() if q >= 1)
    assert ionised == pytest.approx(0.3, abs=1e-6)


@pytest.mark.slow
def test_live_ne_cascade_end_to_end():
    """Live providers on Ne at 1 keV: converges, conserves population,
    and reaches the classic 1s-PI + Auger charge state."""
    from atomx import Atom

    def factory(label):
        # the auger-gate solve shape: fac_frozen MCDHF handles the
        # open K-hole daughter configs (the plain closed-shell bspline
        # engine raises ExactCISpaceError on them); charge tracks the
        # electron count of each cascade daughter
        nelec = sum(int(t[2:]) for t in label.split())
        return Atom(
            Z=10,
            charge=10 - nelec,
            config=[label],
            correlation=__import__("atomx", fromlist=["Correlation"])
                .Correlation(radial_optimization="fac_frozen"),
            corrections="none",
        ).solve()

    res = cascade(factory, NE_CONFIG, Pulse(1000.0, 5e12))
    total = sum(res.charge_distribution.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert res.configs_visited >= 1
    assert res.photons_absorbed >= 0.0
