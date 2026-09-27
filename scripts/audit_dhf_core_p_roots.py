#!/usr/bin/env python3
"""Resume a same-grid radial-node audit of completed core-hole DHF jobs."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import (  # noqa: E402
    Atom, DiracFock, OUTPUT, basis_for_job, has_core_hole,
    orbital_order_valid,
)
from atomx.config.parsing import parse_nr_shells  # noqa: E402

REPORT = ROOT / 'output/nist_spectra/dhf_core_p_root_audit.jsonl'


def node_profile(result) -> list[dict]:
    profiles = []
    orbitals = result.waves.orbitals
    for shell in result.subshells:
        if shell.l != 1 or shell.n > 3:
            continue
        wave = orbitals.get(shell.label)
        if wave is None:
            profiles.append({'label': shell.label, 'expected': shell.n - 2,
                             'observed': None})
            continue
        values = np.asarray(wave.P).ravel()
        if not values.size or not np.all(np.isfinite(values)):
            observed = None
        else:
            peak = float(np.max(np.abs(values)))
            significant = values[np.abs(values) > 0.01 * peak] if peak else []
            observed = (int(np.count_nonzero(np.signbit(significant[:-1]) !=
                                             np.signbit(significant[1:])))
                        if peak else None)
        profiles.append({'label': shell.label, 'expected': shell.n - 2,
                         'observed': observed})
    return profiles


def candidate_jobs(ledger: Path, min_z: int, max_z: int,
                   max_original_seconds: float) -> list[tuple]:
    with sqlite3.connect(ledger) as db:
        rows = db.execute('''SELECT z,charge,configuration,energy_ha,
            nuclear_mass_me,r_max_bohr,grid_hmax_bohr,elapsed_seconds
            FROM config_jobs WHERE z BETWEEN ? AND ? AND status='done'
            AND starting_potential='gsz' ORDER BY z,elapsed_seconds,charge''',
            (min_z, max_z)).fetchall()
    return [row for row in rows if (row[7] is not None and
            row[7] <= max_original_seconds and has_core_hole(row[2]) and
            any(n <= 3 and ell == 1 for n, ell, _q in parse_nr_shells(row[2])))]


def audit_one(row: tuple) -> dict:
    z, charge, configuration, stored_energy, mass, stored_radius, stored_hmax, _ = row
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'z': z, 'charge': charge, 'configuration': configuration,
              'stored_energy_ha': stored_energy,
              'nuclear_mass_me': mass}
    started = time.monotonic()
    try:
        basis, radius, hmax = basis_for_job(z, charge, configuration)
        if (abs(radius - stored_radius) > 1e-8 or
                abs(hmax - stored_hmax) > 1e-8):
            result.update(status='grid_changed', current_r_max_bohr=radius,
                          current_grid_hmax_bohr=hmax,
                          stored_r_max_bohr=stored_radius,
                          stored_grid_hmax_bohr=stored_hmax)
            return result
        solved = Atom(Z=z, config=configuration, engine='bspline',
                      basis=basis).solve(
            DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                      starting_potential='gsz'), max_iter=200)
        energy = float(solved.total_energy)
        profile = node_profile(solved)
        nodes_valid = bool(profile and all(
            item['observed'] == item['expected'] for item in profile))
        reproduced = (bool(solved.converged) and math.isfinite(energy)
                      and abs(energy - stored_energy) <= 1e-4)
        status = ('not_converged' if not solved.converged else
                  'energy_mismatch' if not reproduced else
                  'invalid_core_p_nodes' if not nodes_valid else
                  'validated_core_p_nodes')
        result.update(status=status, repeat_energy_ha=energy,
                      energy_difference_ha=energy - stored_energy,
                      converged=bool(solved.converged),
                      residual=float(solved.residual),
                      orbital_order_valid=bool(orbital_order_valid(solved)),
                      core_p_nodes_valid=nodes_valid, node_profile=profile)
    except Exception as exc:
        result.update(status='audit_error',
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--min-z', type=int, default=21)
    parser.add_argument('--max-z', type=int, default=40)
    parser.add_argument('--max-jobs', type=int, default=20)
    parser.add_argument('--max-per-element', type=int, default=2)
    parser.add_argument('--max-original-seconds', type=float, default=60)
    args = parser.parse_args()
    if not 21 <= args.min_z <= args.max_z <= 40:
        parser.error('the validated node check covers Z=21..40')
    seen = set()
    if args.report.exists():
        for line in args.report.read_text().splitlines():
            item = json.loads(line)
            seen.add((item['z'], item['charge'], item['configuration']))
    candidates = candidate_jobs(args.ledger, args.min_z, args.max_z,
                                args.max_original_seconds)
    by_z = defaultdict(list)
    for row in candidates:
        if (row[0], row[1], row[2]) not in seen:
            by_z[row[0]].append(row)
    selected = []
    for index in range(args.max_per_element):
        for z in sorted(by_z):
            if index < len(by_z[z]):
                selected.append(by_z[z][index])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.report.open('a') as stream:
        for row in selected[:args.max_jobs]:
            result = audit_one(row)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            counts[result['status']] += 1
            print(result['z'], result['charge'],
                  result['configuration'].split()[-1], result['status'],
                  f"{result['elapsed_seconds']:.1f}s", flush=True)
    print(json.dumps({'new_rows': sum(counts.values()), 'statuses': dict(counts),
                      'eligible_jobs': len(candidates)}), flush=True)


if __name__ == '__main__':
    main()
