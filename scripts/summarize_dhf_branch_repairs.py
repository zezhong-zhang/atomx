#!/usr/bin/env python3
"""Verify the retained core-p branch repairs against the live DHF ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/nist_spectra'


def summarize(out: Path = OUTPUT) -> dict:
    rows = []
    with sqlite3.connect(out / 'full_dhf.sqlite') as db:
        for path in sorted(out.glob('*branch_repair.json')):
            record = json.loads(path.read_text())
            z, charge, configuration = (record['z'], record['charge'],
                                        record['configuration'])
            current = db.execute('''SELECT status,energy_ha,starting_potential
                FROM config_jobs WHERE z=? AND charge=? AND configuration=?''',
                (z, charge, configuration)).fetchone()
            selected = record['selected_energy_ha']
            if (current is None or current[0] != 'done' or
                    abs(current[1] - selected) > 1e-8 or
                    current[2] != record['selected_start']):
                raise ValueError(f'ledger disagrees with {path.name}')
            original = record.get('initial_ledger_energy_ha',
                                  record['old_energy_ha'])
            if abs(original - selected) <= 1e-5:
                raise ValueError(f'no verified branch change in {path.name}')
            gsz = next((branch for branch in record['branches']
                        if branch['start'] == 'gsz'), None)
            if (gsz is None or gsz.get('correct_core_p_nodes') is not False
                    or abs(gsz['energy_ha'] - original) > 1e-4):
                raise ValueError(f'GSZ node failure absent in {path.name}')
            rows.append({'z': z, 'charge': charge,
                         'configuration': configuration,
                         'old_energy_ha': original,
                         'new_energy_ha': selected,
                         'energy_change_ha': selected - original,
                         'improvement_ha': original - selected,
                         'selected_start': record['selected_start'],
                         'source_report': str(path.relative_to(out))})
    return {'checked_utc': datetime.now(timezone.utc).isoformat(),
            'verified_repairs': len(rows),
            'total_energy_improvement_ha': sum(
                max(0, row['improvement_ha']) for row in rows),
            'total_energy_increase_ha': sum(
                max(0, row['energy_change_ha']) for row in rows),
            'net_energy_change_ha': sum(row['energy_change_ha'] for row in rows),
            'rows': rows}


if __name__ == '__main__':
    result = summarize()
    path = OUTPUT / 'dhf_core_p_branch_repairs_summary.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verified_repairs': result['verified_repairs'],
                      'total_energy_improvement_ha':
                          result['total_energy_improvement_ha']}))
