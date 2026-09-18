"""Live coverage for ``atomx.scattering.dielectronic_recombination.fac_reference``.

``fac_reference.py`` (212 LOC) had no reachable reference anywhere in ``src/``,
``tests/``, ``scripts/`` or ``benchmarks/`` (review §H4), so it could rot
silently.  The adapter's own docstring claims it "lets AtomX's Burgess kernel
reproduce a FAC DR spectrum to high accuracy by using FAC's own
level-resolved Auger and radiative widths as a reference correction".

This module exercises the whole path:
:func:`make_fac_reference_widths` -> ``auger_rate_fn`` / ``radiative_rate_fn``
/ ``energy_fn`` -> :func:`atomx.scattering.autoionization.dr_calculate`, and
compares the resulting alpha_DR(T) against the FAC reference in
``tests/data/fe17_dr_fac.json`` (AITable+TRTable combined through the Burgess
kernel; provenance inside that file).

Measured 2026-09-17: adapter/kernel alpha_DR ratio 0.964 / 0.887 / 0.872 /
0.860 at 1 / 5 / 10 / 50 MK (4-14 % low, the residual being the adapter's
per-configuration width scaling), so the gate is [0.80, 1.05] and the
docstring's "high accuracy" claim is recorded as ~14 %, not exact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from atomx.scattering.autoionization import dr_calculate  # noqa: E402
from atomx.scattering.dielectronic_recombination.fac_reference import (  # noqa: E402
    _parse_fac_configuration,
    make_fac_reference_widths,
)

FIXTURE = Path(__file__).parent / "data" / "fe17_dr_fac.json"

# Atom-like input for the enumeration only (same lightweight stub used by
# tests/test_dr_vs_fac.py); the widths and energies come from the adapter.
class _FeXVIIModel:
    Z = 26
    charge = 16
    converged = True
    _config_strings = ["1s2 2s2 2p6"]
    result = None
    orbital_energies = {
        "1s1/2": -260.0,
        "2s1/2": -47.0,
        "2p1/2": -45.0,
        "2p3/2": -44.6,
        "3s1/2": -18.0,
        "3p1/2": -16.0,
        "3p3/2": -15.8,
        "3d3/2": -14.4,
        "3d5/2": -14.4,
        "4s1/2": -10.0,
        "4p1/2": -9.5,
        "4p3/2": -9.4,
        "4d3/2": -9.0,
        "4d5/2": -9.0,
    }


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"FAC DR fixture missing: {FIXTURE}")
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def width_fns(fixture_data):
    return make_fac_reference_widths(
        fixture_data,
        _FeXVIIModel(),
        "1s2 2s2 2p6",
        n_max=4,
        l_max=2,
    )


def test_parse_fac_configuration_known_answers(fixture_data):
    """FAC label -> (core hole, excited pair) for the fixture's own labels."""
    labels = [row["label"] for row in fixture_data["resonances"]]
    parsed = [_parse_fac_configuration(lab) for lab in labels]
    assert all(p is not None for p in parsed), "a fixture label failed to parse"
    # The first fixture levels are 2p5.3s1.3p1 -> 2p hole + 3s/3p pair.
    assert parsed[0] == ("2p", ("3p", "3s"))
    # Every parsed hole is an n=2 subshell below capacity.
    assert {p[0] for p in parsed} <= {"2s", "2p"}
    # 193/194 labels carry the hole + two captured n=3 electrons (dr33/dr34).
    # The one exception is the 2s-hole label "2s1 2s+1(1)1" (ILEV 135), whose
    # token list contains no captured-electron subshell: the parser returns an
    # empty pair.  Recorded here so the limitation is visible rather than
    # silently changing the grouping.
    pair_sizes = [len(p[1]) for p in parsed]
    assert pair_sizes.count(2) == 193
    assert [lab for lab, p in zip(labels, parsed, strict=False) if len(p[1]) != 2] == [
        "FAC ILEV 135: 2*7.3*2 2s1 2s+1(1)1"
    ]
    assert parsed[labels.index("FAC ILEV 135: 2*7.3*2 2s1 2s+1(1)1")] == ("2s", ())
    assert _parse_fac_configuration("not a FAC label") is None
    assert _parse_fac_configuration("FAC ILEV 1: 2*7.3*2 2p6.3s1") is None


def test_callbacks_are_finite_and_physically_bounded(fixture_data, width_fns):
    """The three callbacks return finite, non-negative, bounded values."""
    from atomx.scattering.dielectronic_recombination.enumerate import (
        enumerate_dr_resonances,
    )

    specs = enumerate_dr_resonances(
        _FeXVIIModel(), "1s2 2s2 2p6", n_max=4, l_max=2
    )
    assert len(specs) > 100
    ga = [width_fns.auger_rate_fn(s, 0.0) for s in specs]
    ar = [width_fns.radiative_rate_fn(s, 0.0) for s in specs]
    en = [width_fns.energy_fn(s) for s in specs]
    assert all(np.isfinite(x) and x >= 0.0 for x in ga + ar + en)
    assert any(x > 0.0 for x in ga) and any(x > 0.0 for x in ar)
    # Energies are FAC level averages, inside the fixture's resonance window.
    e_min = min(float(r["E_resonance_eV"]) for r in fixture_data["resonances"])
    e_max = max(float(r["E_resonance_eV"]) for r in fixture_data["resonances"])
    positive = [x for x in en if x > 0.0]
    assert positive
    assert e_min - 1.0 <= min(positive) and max(positive) <= e_max + 1.0


@pytest.mark.external_fac
def test_adapter_reproduces_fac_alpha_dr(fixture_data, width_fns):
    """End-to-end KAT: adapter + Burgess kernel vs the FAC alpha_DR(T).

    Measured 2026-09-17: ratios 0.964 / 0.887 / 0.872 / 0.860 — the adapter
    reproduces the FAC spectrum to 14 %, not exactly (its per-configuration
    width scaling is a grouped approximation), so the gate is [0.80, 1.05].
    """
    T_eV = np.asarray(fixture_data["T_eV"], dtype=float)
    rc = dr_calculate(
        _FeXVIIModel(),
        "1s2 2s2 2p6",
        n_max=4,
        l_max=2,
        T_eV=T_eV,
        auger_rate_fn=width_fns.auger_rate_fn,
        radiative_rate_fn=width_fns.radiative_rate_fn,
        energy_fn=width_fns.energy_fn,
    )
    assert len(rc.resonances) > 100
    expected = np.asarray(fixture_data["alpha_fac_cm3_s"], dtype=float)
    ratios = np.asarray(rc.alpha_cm3_s, dtype=float) / expected
    for T, ratio in zip(fixture_data["T_MK"], ratios, strict=False):
        assert 0.80 <= ratio <= 1.05, (
            f"T={T} MK: adapter alpha_DR / FAC = {ratio:.3f} outside [0.80, 1.05]"
        )
