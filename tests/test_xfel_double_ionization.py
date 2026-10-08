"""Source and energy gates for additional direct-double absorption.

The small table is an analytic independent-electron control, not a fit.
"""

import itertools
import json

import pytest

from atomx.xfel import (
    GaussianPulse,
    KineticGraph,
    add_direct_double_ionization,
)
from atomx.xfel.cascade import _table_key

BASE = {(1, -1): 2, (2, -1): 2}


def table():
    rows = {}
    for q1, q2 in itertools.product(range(3), repeat=2):
        c = {s: q for s, q in zip(BASE, (q1, q2), strict=True) if q}
        if not c:
            continue
        rows[_table_key(c)] = {
            "occupations": {f"{s[0]},{s[1]}": q for s, q in c.items()},
            "energy_hartree": -0.9 * q1 - 0.4 * q2,
            "sigma_cm2": {f"{s[0]},{s[1]}": q * 1e-20 for s, q in c.items()},
            "auger": [],
        }
    return dict(
        Z=5,
        base_nelec=4,
        photon_energy_eV=100.0,
        configs=rows,
        provenance={"control": "independent electron"},
    )


def data(channels=None):
    return dict(
        source="analytic cross-section control; no experimental claim",
        units="cm2",
        channels=channels
        or [
            dict(
                initial_state=_table_key(BASE),
                removed=[list(s), list(s)],
                energies_eV=[80, 100, 120],
                sigma_cm2=[1e-21, 2e-21, 3e-21],
            )
            for s in BASE
        ],
    )


def ingest(tmp_path, t, d):
    path = tmp_path / "direct.json"
    path.write_text(json.dumps(d))
    return add_direct_double_ionization(t, path)


def test_additional_double_channels_preserve_primary_and_charge_accounting(tmp_path):
    original = table()
    augmented = ingest(tmp_path, original, data())
    row = augmented["configs"][_table_key(BASE)]
    assert row["sigma_cm2"] == original["configs"][_table_key(BASE)]["sigma_cm2"]
    assert sum(c["sigma_cm2"] for c in row["double_pi"]) == pytest.approx(4e-21)
    assert "double_pi" not in original["configs"][_table_key(BASE)]
    graph = KineticGraph.from_rate_table(augmented, BASE)
    assert graph.charges[graph.initial_state] == 1
    assert graph.charges[""] == 5
    result = graph.solve(GaussianPulse(100, 1e18, 10))
    assert result.mean_charge[-1] - 1 == pytest.approx(
        result.electrons_emitted[-1], abs=1e-10
    )
    assert result.electrons_emitted[-1] > result.photons_absorbed[-1]


@pytest.mark.parametrize(
    "change,match",
    [
        ("source", "source"),
        ("units", "units"),
        ("extrapolation", "cover"),
        ("threshold", "threshold"),
        ("duplicate", "duplicate"),
        ("mixed", "existing"),
    ],
)
def test_missing_invalid_or_double_counted_data_fails(tmp_path, change, match):
    t, d = table(), data()
    if change == "source":
        del d["source"]
    elif change == "units":
        d["units"] = "Mb"
    elif change == "extrapolation":
        t["photon_energy_eV"] = 150
    elif change == "threshold":
        t["photon_energy_eV"] = 40
        for ch in d["channels"]:
            ch["energies_eV"] = [30, 40, 50]
    elif change == "duplicate":
        d["channels"].append(d["channels"][0])
    elif change == "mixed":
        t["configs"][_table_key(BASE)]["double_pi"] = [{"source": "shakeoff"}]
    with pytest.raises(ValueError, match=match):
        ingest(tmp_path, t, d)


def test_table_requires_Z_and_matching_spectral_bandwidth():
    t = table()
    graph = KineticGraph.from_rate_table(t, BASE)
    with pytest.raises(ValueError, match="continuum"):
        graph.solve(GaussianPulse(100, 1e18, 10, bandwidth_fwhm_eV=1))
    with pytest.raises(ValueError, match="sum to one"):
        graph.solve(
            GaussianPulse(100, 1e18, 10),
            initial_populations={graph.initial_state: 0.999999},
        )
    del t["Z"]
    with pytest.raises(ValueError, match="nuclear charge"):
        KineticGraph.from_rate_table(t, BASE)
