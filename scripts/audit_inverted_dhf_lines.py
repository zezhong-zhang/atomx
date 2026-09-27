"""Explain DHF configuration-order reversals using evaluated NIST centroids."""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def collect(ledger: Path, centroids: list[dict], max_z: int = 118) -> tuple[list[dict], dict]:
    by_config = {(row['z'], row['charge'], row['configuration']): row
                 for row in centroids}
    groups: dict[tuple[int, int, str, str], list[int]] = defaultdict(list)
    with sqlite3.connect(ledger) as db:
        for z, charge, index, lower, upper in db.execute('''
                SELECT z,charge,row_index,lower_configuration,upper_configuration
                FROM line_jobs WHERE status='inverted_configuration_order'
                AND z<=? ORDER BY z,charge,row_index''', (max_z,)):
            groups[(z, charge, lower, upper)].append(index)
    rows = []
    counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for (z, charge, lower, upper), indices in sorted(groups.items()):
        low = by_config.get((z, charge, lower))
        high = by_config.get((z, charge, upper))
        evaluated = bool(low and high and
                         low['nist_manifold_status'] == 'evaluated_complete' and
                         high['nist_manifold_status'] == 'evaluated_complete')
        nist_gap = (high['nist_centroid_cm1'] - low['nist_centroid_cm1']
                    if evaluated else None)
        theory_gap = (high['theory_excitation_cm1'] - low['theory_excitation_cm1']
                      if low and high and
                      low['theory_excitation_cm1'] is not None and
                      high['theory_excitation_cm1'] is not None else None)
        if nist_gap is None:
            status = 'no_complete_evaluated_centroid_pair'
        elif nist_gap < 0:
            status = 'NIST_centroid_agrees_with_DHF_order'
        elif nist_gap > 0:
            status = 'NIST_centroid_disagrees_with_DHF_order'
        else:
            status = 'equal_NIST_centroids'
        counts[status] += len(indices)
        pair_counts[status] += 1
        rows.append({
            'z': z, 'charge': charge,
            'spectrum': low['spectrum'] if low else high['spectrum'] if high else f'Z={z} q={charge}',
            'lower_configuration': lower, 'upper_configuration': upper,
            'nist_line_rows': len(indices),
            'nist_row_indices': ','.join(map(str, indices)),
            'nist_lower_manifold_status': low['nist_manifold_status'] if low else None,
            'nist_upper_manifold_status': high['nist_manifold_status'] if high else None,
            'nist_centroid_gap_cm1': nist_gap,
            'dhf_configuration_gap_cm1': theory_gap,
            'status': status,
        })
    return rows, {'pairs': len(rows), 'line_rows': sum(counts.values()),
                  'statuses': dict(counts), 'pair_statuses': dict(pair_counts),
                  'note': 'The NIST line lower/upper labels refer to individual levels. A configuration centroid can have the opposite order while that specific line still has a positive photon energy.'}


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
