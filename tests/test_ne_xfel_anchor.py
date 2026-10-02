"""Ne XFEL anchor fixture gates (Young-2010-style charge-vs-fluence).

tests/data/ne_xfel_charge_vs_fluence.json: the vendored XATOM
on-the-fly MC (xatom_mc_fly, HFS default) at 1000 eV, 19 fluence
points 1..1e12 a.u., pulse-end charge distributions pop(0..10) with
mean charge 0.02 -> 8.09 (Ne^8+ saturation plateau — the frustrated-
absorption signature of Young et al., Nature 466, 56 (2010)).

This module gates the FIXTURE (structure, normalisation, monotone
saturation); the live atomx cascade_mc comparison lands with the
performance-gated live increment per the XFEL gap analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = Path(__file__).parent / "data" / "ne_xfel_charge_vs_fluence.json"


@pytest.fixture(scope="module")
def anchor():
    if not FIXTURE.exists():
        pytest.skip(f"anchor fixture missing: {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def test_fixture_structure_and_normalisation(anchor):
    rows = anchor["rows"]
    assert anchor["photon_energy_eV"] == 1000.0
    assert len(rows) >= 15
    fluences = [r["fluence_au"] for r in rows]
    assert fluences == sorted(fluences)
    for r in rows:
        assert len(r["pop_end"]) == 11
        total = sum(r["pop_end"])
        assert total == pytest.approx(1.0, abs=5e-3), (
            f"F={r['fluence_au']}: population sums to {total}"
        )
        assert all(p >= 0.0 for p in r["pop_end"])


def test_mean_charge_monotone_and_saturates(anchor):
    rows = anchor["rows"]
    means = [r["mean_charge"] for r in rows]
    # monotone in the fluence scan up to MC noise
    for lo, hi in zip(means, means[1:], strict=True):
        assert hi >= lo - 0.15
    # saturation plateau: the highest decade stays on Ne^8+
    top = [r for r in rows if r["fluence_au"] >= 1e5]
    assert all(7.9 <= r["mean_charge"] <= 8.3 for r in top), (
        "the frustrated-absorption Ne^8+ plateau must hold at high "
        "fluence (1 keV photons cannot strip the last two L electrons "
        "once the K shell is empty)"
    )
    # single-photon regime: mean charge grows from ~0
    assert rows[0]["mean_charge"] < 0.1
    # one K vacancy -> Auger -> charge 2 dominates the first ionised bin
    low = next(r for r in rows if 1 <= r["fluence_au"] <= 3)
    assert max(range(1, 11), key=lambda q: low["pop_end"][q]) == 2, (
        "the first ionised product of 1s PI + KLL Auger is Ne^2+"
    )
