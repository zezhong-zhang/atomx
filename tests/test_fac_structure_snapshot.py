"""Staleness gate for the committed FAC structure benchmark.

``benchmarks/results_fac_structure.csv`` is quoted in ``README.md`` and
``docs/VERIFICATION.md`` as the atomx-vs-FAC structure record, but nothing
recomputed it: the atomx column sat unchanged for five months of physics
changes (last touched 2026-04-20, HEAD 2026-09-18) and the "7 keV / 1.6 % at
Z=82" limitation that both documents carried came from that stale record plus
an unmatched ``qed=False``.  With the flags matched to FAC's default the same
row used mismatched Breit flags. The repaired driver enables the static
Breit term that is enabled by default in FAC.

This test recomputes the atomx column live with the driver's own function and
compares it against the committed values, so drift is detected instead of
being quoted.  The FAC column is an external record and is **not** recomputed
(that needs ``sfac``).

See ``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding H1.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

from bench_fac_structure import run_atomx  # noqa: E402

from atomx.constants import HARTREE_EV  # noqa: E402

CSV = ROOT / "benchmarks" / "results_fac_structure.csv"

#: The atomx column is a *self* snapshot: this gate catches drift, not error.
#: 2e-4 relative is ~4x tighter than the worst measured scatter of a
#: re-run (5e-5) and far looser than any real physics change.
SNAPSHOT_REL = 2e-4


def _rows():
    with CSV.open() as f:
        return list(csv.DictReader(f))


def test_csv_exists_and_covers_the_panel():
    rows = _rows()
    assert rows, f"{CSV} is empty"
    Zs = sorted({int(r["Z"]) for r in rows})
    assert Zs[0] == 10 and Zs[-1] == 82, Zs
    assert all(float(r["E_fac_eV"]) < 0.0 for r in rows)
    assert "flags" in rows[0], (
        "the CSV must record which physics flags produced its atomx column"
    )


@pytest.mark.parametrize("row", _rows(), ids=lambda r: f"Z{r['Z']}-{r['iso']}")
def test_atomx_column_is_not_stale(row):
    """Recompute atomx the way the driver does and compare to the record."""
    Z, iso = int(row["Z"]), row["iso"]
    e_ha, _t, _conv = run_atomx(Z, iso, breit=True, qed=True)
    recorded = float(row["E_atomx_Ha"])
    rel = abs(e_ha - recorded) / abs(recorded)
    assert rel < SNAPSHOT_REL, (
        f"Z={Z} {iso}: live atomx {e_ha:.6f} Ha vs recorded {recorded:.6f} "
        f"(rel={rel:.2e}).  If the physics changed on purpose, regenerate the "
        f"atomx column with `python benchmarks/bench_fac_structure.py`; do NOT "
        f"regenerate the FAC column without sfac."
    )
    # The recorded diff/rel_diff must be consistent with the FAC column.
    if row["E_fac_eV"]:
        fac = float(row["E_fac_eV"])
        assert float(row["E_atomx_eV"]) == pytest.approx(e_ha * HARTREE_EV, rel=1e-6)
        assert float(row["diff_eV"]) == pytest.approx(
            e_ha * HARTREE_EV - fac, abs=5e-3
        )


def test_panel_agrees_with_fac_at_the_documented_level():
    """The claim README makes: ≤0.1 % across the whole panel.

    Keep the established accuracy thresholds; flags are now matched explicitly.
    """
    worst = max(abs(float(r["rel_diff"])) for r in _rows() if r["rel_diff"])
    assert worst < 1.0e-3, (
        f"worst |atomx-FAC| relative deviation is {worst:.2e}; the README "
        f"claims <= 1e-3.  Re-measure and update the documented figure."
    )
    # And the absolute worst row, which the README quotes separately.
    worst_abs = max(abs(float(r["diff_eV"])) for r in _rows() if r["diff_eV"])
    assert worst_abs < 500.0, f"worst absolute deviation {worst_abs:.0f} eV"
