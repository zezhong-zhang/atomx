"""Fe XVII KLL Auger widths: LIVE atomx vs the FAC AITable panel.

History / why this file was rewritten (2026-09-17, validation-integrity pass):
the previous version held a ``REFERENCE_PANEL`` dict of five FAC numbers and
compared it against the *same* numbers read from
``tests/data/fe17_dr_fac.json`` — all five were bit-identical (rel err 0.0),
atomx was never called, and the failure message printed the FAC value under
the label ``atomx=``.  It was a tautology, and the README's per-level Auger
accuracy row rested on it.

What the live panel is
----------------------
``tests/data/fe17_kll_auger_fac.json`` records, for the five strongest KLL
resonances, both (a) the FAC AITable Auger width and (b) the *atomx channel
decomposition* (hole + two captured electrons) needed to call
``auger_rate_csf_average``.  This module now:

1. cross-checks each panel FAC width against the independent AITable record in
   ``tests/data/fe17_dr_fac.json`` (fixture integrity — still not atomx);
2. computes the per-resonance atomx width **live** with
   ``atomx.scattering.autoionization.auger_rate_csf_average`` from one Fe XVII
   MCDHF solve (the working pattern of ``test_auger_kll_vs_fac.py``);
3. gates the live values on magnitude and on an order-of-magnitude band
   against FAC;
4. keeps the aspirational strict 10 % per-resonance gate as
   ``xfail(strict=False)`` — measured live/FAC ratios on 2026-09-17 are
   1.13 / 1.38 / 5.96 / 7.11 / 14.82, so the 10 % claim is NOT met by the
   configuration-average entry point (the J-resolved ``jj_level`` path is the
   closer route: see ``tests/test_jj_level_auger.py``).

Recorded-but-not-live-computable levels
---------------------------------------
``tests/data/fe17_dr_fac.json`` also carries ILEV 24 / 62 / 135 / 150, which
the old ``REFERENCE_PANEL`` pinned.  That fixture stores only FAC labels, no
atomx channel decomposition, and ILEV 135 is a 2s-hole state whose decay is
outside the KLL 3l3l' -> 2p^6 channel this API models; there is therefore no
honest live atomx computation for them.  They are retained as *recorded*
external data (asserted for presence only) and carry no parity gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.scattering.autoionization import auger_rate_csf_average  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "data"
KLL_FIXTURE = FIXTURE_DIR / "fe17_kll_auger_fac.json"
AITABLE_FIXTURE = FIXTURE_DIR / "fe17_dr_fac.json"

# Aspirational per-resonance parity; not met — see module docstring.
STRICT_TOLERANCE = 0.10
# Magnitude gate: pre-HYP-530 atomx returned ~1e5 s^-1 here (eight orders
# low); every live panel value must sit in the physical Auger band.
MAGNITUDE_FLOOR = 1.0e13
MAGNITUDE_CEIL = 1.0e17
# Honest current-accuracy band vs FAC (measured 1.13-14.82, i.e. atomx
# over-counts the configuration-average angular factor).  This is deliberately
# NOT a parity claim; it only catches order-of-magnitude regressions.
RATIO_FLOOR = 0.1
RATIO_CEIL = 30.0

# ILEVs present in the AITable fixture with no honest live computation.
RECORDED_ONLY_ILEV = (24, 62, 135, 150)


def _ilev(label: str) -> int:
    prefix, _ = label.split(":", 1)
    return int(prefix.removeprefix("FAC ILEV "))


@pytest.fixture(scope="module")
def kll_panel() -> dict:
    if not KLL_FIXTURE.exists():
        pytest.skip(f"KLL panel fixture missing: {KLL_FIXTURE}")
    return json.loads(KLL_FIXTURE.read_text())


@pytest.fixture(scope="module")
def aitable() -> dict[int, dict]:
    if not AITABLE_FIXTURE.exists():
        pytest.skip(f"FAC AITable fixture missing: {AITABLE_FIXTURE}")
    data = json.loads(AITABLE_FIXTURE.read_text())
    return {_ilev(row["label"]): row for row in data["resonances"]}


@pytest.fixture(scope="module")
def fe17_result():
    """Solve Fe XVII once with the ground + 2p^5 3l CSFs (radials only).

    ``auger_rate_csf_average`` needs only the radial orbitals: the 3l virtuals
    are well defined even though their ground-state occupation is ~0.  Same
    setup as ``test_auger_kll_vs_fac.py``.
    """
    atom = Atom(
        Z=26,
        charge=16,
        configs=[
            "1s2 2s2 2p6",
            "1s2 2s2 2p5 3s1",
            "1s2 2s2 2p5 3p1",
            "1s2 2s2 2p5 3d1",
        ],
        j_resolved_csfs=False,
        breit=False,
        qed=False,
    )
    atom.solve()
    return atom._result


@pytest.fixture(scope="module")
def live_rates(kll_panel, fe17_result) -> list[dict]:
    """Live atomx per-resonance KLL Auger widths, paired with their FAC value."""
    out = []
    for row in kll_panel["resonances"]:
        result = auger_rate_csf_average(
            fe17_result,
            hole_label=row["atomx_hole_label"],
            source_label=row["atomx_source_label"],
            source2_label=row["atomx_source2_label"],
            resonance_energy_eV=row["E_resonance_eV"],
        )
        atomx = float(result["Gamma_per_s"])
        fac = float(row["gamma_auger_per_s"])
        out.append({
            "ilev": int(row["ilev"]),
            "channel": (
                f"{row['atomx_source_label']}+{row['atomx_source2_label']}"
                f" -> {row['atomx_hole_label']}+eps"
            ),
            "J": float(row["J"]),
            "atomx_per_s": atomx,
            "fac_per_s": fac,
            "ratio": atomx / fac,
        })
    return out


def _report(rates: list[dict]) -> str:
    return "\n".join(
        f"  ILEV {r['ilev']} J={r['J']} {r['channel']}: "
        f"atomx={r['atomx_per_s']:.3e} s^-1  FAC={r['fac_per_s']:.3e} s^-1  "
        f"atomx/FAC={r['ratio']:.2f}"
        for r in rates
    )


def test_panel_fac_values_are_consistent_with_the_aitable_fixture(kll_panel, aitable):
    """Fixture integrity: the panel's FAC widths equal the AITable record.

    This asserts only that the two stored *external* FAC files agree (5-sig-fig
    rounding), so a stale/edited fixture is caught.  It is NOT an atomx gate.
    """
    for row in kll_panel["resonances"]:
        ilev = int(row["ilev"])
        assert ilev in aitable, f"ILEV {ilev} missing from the AITable fixture"
        fac_panel = float(row["gamma_auger_per_s"])
        fac_aitable = float(aitable[ilev]["gamma_auger_per_s"])
        rel = abs(fac_panel - fac_aitable) / fac_aitable
        assert rel <= 1.0e-4, (
            f"ILEV {ilev}: KLL panel FAC={fac_panel:.4e} disagrees with the "
            f"AITable record FAC={fac_aitable:.4e} (rel={rel:.2e})"
        )
    # The recorded-only levels stay present but carry no gate.
    for ilev in RECORDED_ONLY_ILEV:
        assert ilev in aitable


@pytest.mark.external_fac
def test_kll_auger_magnitude_gate(live_rates):
    """Live atomx KLL widths must be in [1e13, 1e17] s^-1.

    Pre-HYP-530 the configuration-average entry point returned ~1e5 s^-1 for
    every KLL channel (eight orders low); this gate is the regression tripwire
    for that class of failure.  Measured 2026-09-17: 4.62e14 - 4.57e15 s^-1.
    """
    assert live_rates, "panel produced no live rates"
    for r in live_rates:
        assert r["atomx_per_s"] > 0.0 and r["atomx_per_s"] == r["atomx_per_s"]
        assert MAGNITUDE_FLOOR <= r["atomx_per_s"] <= MAGNITUDE_CEIL, (
            f"ILEV {r['ilev']}: atomx={r['atomx_per_s']:.3e} s^-1 outside "
            f"[{MAGNITUDE_FLOOR:.0e}, {MAGNITUDE_CEIL:.0e}]; live panel:\n"
            + _report(live_rates)
        )


@pytest.mark.external_fac
def test_kll_auger_within_order_of_magnitude_of_fac(live_rates):
    """Honest current accuracy: live atomx/FAC in [0.1, 30] per resonance.

    Measured 2026-09-17: 1.13 / 1.38 / 5.96 / 7.11 / 14.82.  This documents
    the real state ("right order of magnitude, over-counted by the
    average-of-configuration angular factor"), it is NOT the README's
    per-level accuracy claim.
    """
    for r in live_rates:
        assert RATIO_FLOOR <= r["ratio"] <= RATIO_CEIL, (
            f"ILEV {r['ilev']}: atomx/FAC = {r['ratio']:.2f} outside "
            f"[{RATIO_FLOOR}, {RATIO_CEIL}]; live panel:\n" + _report(live_rates)
        )


@pytest.mark.external_fac
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Aspirational 10% per-resonance KLL parity.  The configuration-average "
        "angular factor over-counts magnetic substates for a J-resolved "
        "resonance: measured live atomx/FAC = 1.13 / 1.38 / 5.96 / 7.11 / 14.82 "
        "on 2026-09-17.  The J-resolved jj_level path reaches 0.18-3.9x on the "
        "same panel and 1.02x for ILEV 71 "
        "(tests/test_jj_level_auger.py); closing the remaining gap needs "
        "j-resolved CSF eigenvectors wired into the jj_level amplitudes."
    ),
)
def test_kll_auger_per_resonance_10pct_vs_fac(live_rates):
    """Strict |atomx/FAC - 1| <= 10 % per resonance — aspirational, xfailed."""
    worst = max(abs(r["ratio"] - 1.0) for r in live_rates)
    assert worst <= STRICT_TOLERANCE, (
        f"worst |atomx/FAC - 1| = {worst:.2f} exceeds {STRICT_TOLERANCE:.0%}; "
        f"live panel:\n" + _report(live_rates)
    )
