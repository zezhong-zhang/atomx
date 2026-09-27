#!/usr/bin/env python3
"""Replace a reproduced wrong-root DHF job with a node-valid branch."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import (
    Atom, DiracFock, OUTPUT, basis_for_job, orbital_node_valid,
    orbital_order_valid, update_lines,
)


def repair(ledger: Path, z: int, charge: int, configuration: str,
           report_path: Path) -> dict:
    with sqlite3.connect(ledger, timeout=120) as db:
        db.execute('PRAGMA busy_timeout=120000')
        old = db.execute('''SELECT status,energy_ha,nuclear_mass_me,starting_potential
                            FROM config_jobs WHERE z=? AND charge=?
                            AND configuration=?''', (z, charge, configuration)).fetchone()
        if old is None or old[0] != 'done' or old[1] is None:
            raise ValueError('repair requires an existing completed configuration')
        old_status, old_energy, mass, old_start = old
        if old_start != 'gsz':
            raise ValueError('this repair requires a stored GSZ branch')
        basis, radius, hmax = basis_for_job(z, charge, configuration)
        branches = []
        best = None
        for starting, damping in [('gsz', None), ('tfda', None),
                                  ('tfda', 0.35), ('tfda', 0.70)]:
            label = starting + (f'+damping={damping:g}' if damping is not None else '')
            started = time.monotonic()
            try:
                result = Atom(Z=z, config=configuration, engine='bspline',
                              basis=basis).solve(
                    DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                              starting_potential=starting, scf_damping=damping),
                    max_iter=220)
                ordered = bool(orbital_order_valid(result))
                nodes = bool(orbital_node_valid(result, z, configuration))
                valid = bool(result.converged and math.isfinite(result.total_energy)
                             and ordered and nodes)
                row = {'start': label, 'converged': bool(result.converged),
                       'energy_ha': float(result.total_energy),
                       'normal_mass_ha': float(result.energy_components.get('nms', 0.0)),
                       'residual': float(result.residual),
                       'ordered_orbitals': ordered, 'correct_core_p_nodes': nodes,
                       'valid': valid, 'elapsed_seconds': time.monotonic() - started}
                if valid and (best is None or result.total_energy < best[0].total_energy):
                    best = (result, label, row['elapsed_seconds'])
            except Exception as exc:
                row = {'start': label, 'valid': False,
                       'error': f'{type(exc).__name__}: {exc}',
                       'elapsed_seconds': time.monotonic() - started}
            branches.append(row)
        if best is None:
            raise RuntimeError('no converged, ordered, node-valid branch')
        result, label, seconds = best
        gsz = next(branch for branch in branches if branch['start'] == 'gsz')
        source_reproduced = bool(
            gsz.get('converged') and
            gsz.get('energy_ha') is not None and
            abs(gsz['energy_ha'] - old_energy) <= 1e-4)
        source_wrong_root = bool(source_reproduced and
                                 gsz.get('correct_core_p_nodes') is False)
        changed = bool(source_wrong_root and
                       abs(float(result.total_energy) - old_energy) > 1e-5)
        report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'z': z, 'charge': charge, 'configuration': configuration,
                  'nuclear_mass_me': mass, 'r_max_bohr': radius,
                  'grid_hmax_bohr': hmax, 'breit': True, 'qed': True,
                  'old_energy_ha': old_energy,
                  'old_start': old_start,
                  'source_branch_reproduced': source_reproduced,
                  'source_wrong_root': source_wrong_root,
                  'selected_energy_ha': float(result.total_energy),
                  'selected_start': label, 'energy_improvement_ha':
                      old_energy - float(result.total_energy),
                  'ledger_updated': False, 'branches': branches}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if changed:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('''SELECT status,energy_ha FROM config_jobs
                                    WHERE z=? AND charge=? AND configuration=?''',
                                 (z, charge, configuration)).fetchone()
            if current != (old_status, old_energy):
                db.rollback()
                raise RuntimeError('job changed while branch audit ran')
            db.execute('''UPDATE config_jobs SET energy_ha=?,nms_ha=?,
                          elapsed_seconds=?,converged=1,error=NULL,
                          r_max_bohr=?,grid_hmax_bohr=?,starting_potential=?
                          WHERE z=? AND charge=? AND configuration=?''',
                       (float(result.total_energy),
                        float(result.energy_components.get('nms', 0.0)),
                        seconds, radius, hmax, label,
                        z, charge, configuration))
            update_lines(db, z, charge)
            db.commit()
    report['ledger_updated'] = changed
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=OUTPUT)
    parser.add_argument('--z', type=int, required=True)
    parser.add_argument('--charge', type=int, required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = repair(args.ledger, args.z, args.charge,
                    args.configuration, args.report)
    print(json.dumps({key: report[key] for key in
                      ('old_energy_ha', 'selected_energy_ha',
                       'selected_start', 'ledger_updated')}))


if __name__ == '__main__':
    main()
