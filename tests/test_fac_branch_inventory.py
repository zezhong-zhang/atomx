"""Inventory snapshots of the FAC angular-coefficient coverage *gaps*.

Every assertion here is a frozen count taken from ``scripts/
fac_branch_inventory.py`` (D-class): 210 of the 330 Fe XVII rows still have
no production FAC angular coefficient, 192 branch-9 two-electron rows are
"pending", 12 one-diff rows use a closed-spectator fallback and 6 use a
plain unit-coefficient fallback.  These are a **coverage budget for
unimplemented physics**, not validation of the FAC port — the ported
branches are gated end-to-end by ``tests/test_fac_dfs_shooting.py``
(atomx vs the FAC calibration panel) and by the FAC angular-coefficient
fixtures elsewhere in the suite.  The whole module is marked ``snapshot``
so ``pytest -m "not snapshot"`` does not count it as physics coverage.

Missing gate the audit asked for (deferred, needs a FAC run or a frozen
``benchmarks/fac_dumps`` level table): an end-to-end Fe XVII level energy
for the ``fac-angular`` branches against live ``sfac``.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fac_branch_inventory import (  # noqa: E402
    FE17_CONFIGS,
    BranchRow,
    inventory_panel,
    print_summary,
)

pytestmark = pytest.mark.snapshot


def test_belike_inventory_marks_supported_branch8_terms():
    rows = inventory_panel("Be-like odd", ("1s2 2s1 2p1",), include_diagonal=False)
    counts = Counter((row.interaction, row.branch, row.support) for row in rows)

    assert counts[("one-diff", 8, "fac-angular")] == 4
    assert counts[("one-diff", 8, "closed-spectator-fallback")] == 0


def test_fe17_inventory_tracks_branch5_gap_and_two_diff_branch9_pending():
    rows = inventory_panel("Fe17 L-shell", FE17_CONFIGS, include_diagonal=False)

    assert any(
        row.interaction == "one-diff"
        and row.branch == 5
        and row.support == "fallback"
        for row in rows
    )
    assert any(
        row.interaction == "one-diff"
        and row.branch == 8
        and row.support == "fac-angular"
        and "3:-3 3:2 2:-1 2:-1" in row.shells
        for row in rows
    )
    assert any(
        row.interaction == "one-diff"
        and row.branch == 5
        and row.support == "fallback"
        for row in rows
    )
    assert any(
        row.interaction == "two-diff-exchange"
        and row.branch == 9
        and row.support == "fac-branch9-pending"
        for row in rows
    )
    assert any(
        row.interaction == "two-diff-direct"
        and row.branch == 9
        and row.support == "fac-branch9-pending"
        for row in rows
    )
    assert not any(
        row.interaction.startswith("two-diff")
        and row.branch == 9
        and row.support == "fallback"
        for row in rows
    )
    assert sum(
        row.interaction.startswith("two-diff")
        and row.branch == 9
        and row.support == "fac-branch9-pending"
        for row in rows
    ) == 192


def test_fe17_inventory_counts_remaining_one_diff_gaps():
    rows = inventory_panel("Fe17 L-shell", FE17_CONFIGS, include_diagonal=False)
    counts = Counter((row.interaction, row.branch, row.support) for row in rows)

    assert counts[("one-diff", 8, "fac-angular")] == 106
    assert counts[("one-diff", 8, "closed-spectator-fallback")] == 12
    assert counts[("one-diff", 7, "fac-angular")] == 14
    assert counts[("one-diff", 5, "fallback")] == 6
    assert counts[("two-diff-direct", 9, "fac-branch9-pending")] == 96
    assert counts[("two-diff-exchange", 9, "fac-branch9-pending")] == 96
    assert sum(row.support != "fac-angular" for row in rows) == 210


def test_fe17_summary_reports_closed_inventory(capsys):
    rows = inventory_panel("Fe17 L-shell", FE17_CONFIGS, include_diagonal=False)

    print_summary(rows)

    output = capsys.readouterr().out
    assert "Rows without production FAC angular coefficient: 210 / 330" in output
    assert "Unsupported fallback unit-coefficient weight:" in output
    assert "one-diff          branch=8  fac-angular                106" in output
    assert "one-diff          branch=8  closed-spectator-fallback  12" in output
    assert "one-diff          branch=5  fallback                   6" in output
    assert "two-diff-direct   branch=9  fac-branch9-pending        96" in output
    assert "two-diff-exchange branch=9  fac-branch9-pending        96" in output
    assert "one-diff          branch=7  fac-angular                14" in output


def test_all_builtin_panels_report_known_fac_angular_gaps():
    rows = []
    for name, configs in {
        "Be-like odd": ("1s2 2s1 2p1",),
        "Ne-like hole": (
            "1s2 2s2 2p6",
            "1s2 2s2 2p5 3s1",
            "1s2 2s2 2p5 3p1",
            "1s2 2s2 2p5 3d1",
        ),
        "Fe17 L-shell": FE17_CONFIGS,
    }.items():
        rows.extend(inventory_panel(name, configs, include_diagonal=True))

    assert len(rows) == 1480
    assert sum(row.support != "fac-angular" for row in rows) == 278


def test_belike_diagonal_branch6_is_production_fac_angular():
    rows = inventory_panel("Be-like odd", ("1s2 2s1 2p1",), include_diagonal=True)
    diag_branch6 = [
        row for row in rows
        if row.interaction == "diagonal" and row.branch == 6
    ]

    assert len(diag_branch6) == 12
    assert {row.support for row in diag_branch6} == {"fac-angular"}
    assert any("2:-2 2:-1 2:-2 2:-1" in row.shells for row in diag_branch6)
    assert any("2:-1 2:1 2:-1 2:1" in row.shells for row in diag_branch6)


def test_inventory_csv_schema_uses_exported_row_keys(tmp_path):
    row = BranchRow(
        panel="p",
        interaction="one-diff",
        support="fac-angular",
        branch=8,
        order="2 3 1 0",
        phase=5,
        j2=2,
        parity=-1,
        bra="a",
        ket="b",
        shells="2:-2 2:1 2:-1 2:-1",
        ranks="1",
        fallback_unit_coeff=0.0,
    )
    out = tmp_path / "inventory.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.as_dict()))
        writer.writeheader()
        writer.writerow(row.as_dict())

    assert "J2" in out.read_text().splitlines()[0]
