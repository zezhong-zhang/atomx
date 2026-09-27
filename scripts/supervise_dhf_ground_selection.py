#!/usr/bin/env python3
"""Reconcile extrapolated ionic ground candidates as DHF jobs finish."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from repair_dhf_ground_selection import ARCHIVE, OUTPUT, repair

ROOT = Path(__file__).resolve().parents[1]
INITIAL_REPAIR = ROOT / "output/nist_spectra/dhf_isoelectronic_ground_repair.json"
NEARBY_CANDIDATES = ROOT / "output/nist_spectra/dhf_nearby_ionic_candidates.json"
REPORT = ROOT / "output/nist_spectra/dhf_ground_selection_watch.json"
EVENTS = ROOT / "output/nist_spectra/dhf_ground_selection_events.jsonl"


def candidate_counts(ledger: Path, initial_report: Path) -> dict:
    changes = json.loads(initial_report.read_text())["repaired"]
    with sqlite3.connect(ledger, timeout=60) as db:
        rows = {(z, charge, config): (status, energy)
                for z, charge, config, status, energy in db.execute(
                    """SELECT z,charge,configuration,status,energy_ha
                       FROM config_jobs WHERE z>=104""")}
    counts = {"pairs": len(changes), "both_converged": 0,
              "ionic_analogue_lower": 0, "neutral_fallback_lower": 0,
              "unresolved": 0}
    for item in changes:
        key = (item["z"], item["charge"])
        old = rows.get((*key, item["old"]))
        new = rows.get((*key, item["new"]))
        if (old is None or new is None or old[0] != "done"
                or new[0] != "done"):
            counts["unresolved"] += 1
            continue
        counts["both_converged"] += 1
        counts["ionic_analogue_lower" if new[1] <= old[1]
               else "neutral_fallback_lower"] += 1
    return counts


def nearby_candidate_counts(ledger: Path,
                            nearby_report: Path = NEARBY_CANDIDATES) -> dict:
    if not nearby_report.is_file():
        return {"candidates": 0, "converged": 0,
                "selected": 0, "lower_than_selected": 0}
    records = json.loads(nearby_report.read_text())["candidates"]
    counts = {"candidates": len(records), "converged": 0,
              "selected": 0, "lower_than_selected": 0}
    with sqlite3.connect(ledger, timeout=60) as db:
        for row in records:
            alt = db.execute("""SELECT status,energy_ha,is_ground
                FROM config_jobs WHERE z=? AND charge=? AND configuration=?""",
                (row["z"], row["charge"], row["configuration"])).fetchone()
            selected = db.execute("""SELECT status,energy_ha FROM config_jobs
                WHERE z=? AND charge=? AND is_ground=1""",
                (row["z"], row["charge"])).fetchone()
            if alt is None or alt[0] != "done":
                continue
            counts["converged"] += 1
            counts["selected"] += bool(alt[2])
            if (selected is not None and selected[0] == "done"
                    and alt[1] < selected[1] - 1e-6):
                counts["lower_than_selected"] += 1
    return counts


def sweep(archive: Path = ARCHIVE, ledger: Path = OUTPUT,
          initial_report: Path = INITIAL_REPAIR, report: Path = REPORT,
          events: Path = EVENTS) -> dict:
    result = repair(archive, ledger)
    counts = candidate_counts(ledger, initial_report)
    with sqlite3.connect(ledger, timeout=60) as db:
        active = db.execute("""SELECT COUNT(*) FROM config_jobs
            WHERE status IN ('pending','running','retry_pending',
                             'repair_pending','candidate_pending')""").fetchone()[0]
    summary = {"checked_utc": datetime.now(timezone.utc).isoformat(),
               "active_jobs": active, "selection_changes": result["repaired"],
               "selection_sources": result["selection_sources"],
               "candidate_checks": counts,
               "nearby_candidate_checks": nearby_candidate_counts(ledger)}
    report.parent.mkdir(parents=True, exist_ok=True)
    part = report.with_suffix(report.suffix + ".part")
    part.write_text(json.dumps(summary, indent=2) + "\n")
    part.replace(report)
    if result["repaired"]:
        with events.open("a") as handle:
            for item in result["repaired"]:
                handle.write(json.dumps({"checked_utc": summary["checked_utc"],
                                         **item}) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    while True:
        result = sweep()
        print(f"active={result['active_jobs']} "
              f"candidate_checks={result['candidate_checks']} "
              f"selection_changes={len(result['selection_changes'])}",
              flush=True)
        if args.once or not result["active_jobs"]:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
