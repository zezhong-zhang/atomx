#!/usr/bin/env python3
"""Verify the exploratory heavy-ion core-p probe against the live ledger."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomx.config.parsing import parse_nr_shells  # noqa: E402
from scripts.audit_dhf_core_p_roots import candidate_jobs  # noqa: E402
from scripts.generate_full_dhf_nist import OUTPUT  # noqa: E402

PROBE = ROOT / "output/nist_spectra/dhf_heavy_core_p_root_probe.jsonl"
SUMMARY = ROOT / "output/nist_spectra/dhf_heavy_core_p_root_probe_summary.json"


def has_low_p_hole(configuration: str) -> bool:
    return any(n <= 3 and ell == 1 and occupation < 6
               for n, ell, occupation in parse_nr_shells(configuration))


def summarize(ledger: Path = OUTPUT, probe: Path = PROBE) -> dict:
    rows = [json.loads(line) for line in probe.read_text().splitlines()] if probe.exists() else []
    keys = {(row["z"], row["charge"], row["configuration"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate heavy core-p probe job key")
    with sqlite3.connect(ledger) as db:
        current = {(z, q, cfg): (status, start, energy)
                   for z, q, cfg, status, start, energy in db.execute("""
                       SELECT z,charge,configuration,status,starting_potential,
                              energy_ha FROM config_jobs WHERE z BETWEEN 41 AND 118""")}
    manifest_path = ledger.parent / "dhf_core_p_branch_repairs_summary.json"
    repaired_keys = ({(row["z"], row["charge"], row["configuration"])
                      for row in json.loads(manifest_path.read_text())["rows"]}
                     if manifest_path.exists() else set())
    invalid = [row for row in rows if row["status"] == "invalid_core_p_nodes"]
    repaired = 0
    open_invalid = []
    unverified_changes = []
    for row in invalid:
        key = (row["z"], row["charge"], row["configuration"])
        state = current.get(key)
        if state and state[:2] == ("done", "gsz"):
            open_invalid.append(key)
        elif state and state[0] == "done" and key in repaired_keys:
            repaired += 1
        else:
            unverified_changes.append(key)
    candidates = [row for row in candidate_jobs(ledger, 41, 74, float("inf"))
                  if has_low_p_hole(row[2])]
    unaudited = sum(row[:3] not in keys for row in candidates)
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "exploratory stored-GSZ core-hole sample, Z=41..118",
            "tested_atomic_numbers": sorted({row["z"] for row in rows}),
            "audited_jobs": len(rows),
            "audit_statuses": dict(Counter(row["status"] for row in rows)),
            "invalid_repaired": repaired,
            "invalid_still_in_ledger": len(open_invalid),
            "invalid_changed_without_repair_report": len(unverified_changes),
            "current_gsz_low_p_hole_jobs_z41_74": len(candidates),
            "current_gsz_low_p_hole_not_yet_audited_z41_74": unaudited,
            "audit_log": str(probe)}


if __name__ == "__main__":
    result = summarize()
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
