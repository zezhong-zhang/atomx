#!/usr/bin/env python3
"""Summarize the resumable DHF core-p audit and unresolved historical jobs."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_dhf_core_p_roots import REPORT, candidate_jobs
from scripts.generate_full_dhf_nist import OUTPUT

SUMMARY = ROOT / 'output/nist_spectra/dhf_core_p_root_audit_summary.json'


def summarize(ledger: Path = OUTPUT, report: Path = REPORT) -> dict:
    rows = [json.loads(line) for line in report.read_text().splitlines()] if report.exists() else []
    keys = {(row['z'], row['charge'], row['configuration']) for row in rows}
    if len(keys) != len(rows):
        raise ValueError('duplicate root-audit job key')
    with sqlite3.connect(ledger) as db:
        statuses = {(z, q, cfg): (status, start)
                    for z, q, cfg, status, start in db.execute('''
                        SELECT z,charge,configuration,status,starting_potential
                        FROM config_jobs WHERE z BETWEEN 21 AND 40''')}
    invalid = [row for row in rows if row['status'] == 'invalid_core_p_nodes']
    manifest_path = ledger.parent / 'dhf_core_p_branch_repairs_summary.json'
    repair_keys = ({(row['z'], row['charge'], row['configuration'])
                    for row in json.loads(manifest_path.read_text())['rows']}
                   if manifest_path.exists() else set())
    changed = [(row['z'], row['charge'], row['configuration'])
               for row in invalid if
               statuses.get((row['z'], row['charge'], row['configuration']),
                            ('', ''))[0] == 'done' and
               statuses[(row['z'], row['charge'], row['configuration'])][1]
               != 'gsz']
    repaired = sum(key in repair_keys for key in changed)
    changed_unverified = len(changed) - repaired
    open_invalid = sum(statuses.get((row['z'], row['charge'], row['configuration'])) ==
                       ('done', 'gsz') for row in invalid)
    current_candidates = candidate_jobs(ledger, 21, 40, float('inf'))
    unaudited = sum((z, q, cfg) not in keys
                    for z, q, cfg, *_rest in current_candidates)
    return {'checked_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'completed GSZ core-hole jobs with n<=3 p orbitals, Z=21..40',
            'audited_jobs': len(rows),
            'audit_statuses': dict(Counter(row['status'] for row in rows)),
            'invalid_repaired': repaired,
            'invalid_changed_without_repair_report': changed_unverified,
            'invalid_still_in_ledger': open_invalid,
            'current_gsz_candidates': len(current_candidates),
            'current_gsz_not_yet_audited': unaudited,
            'audit_log': str(report)}


if __name__ == '__main__':
    result = summarize()
    SUMMARY.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'audited_jobs', 'audit_statuses', 'invalid_repaired',
        'invalid_still_in_ledger', 'current_gsz_not_yet_audited')}))
