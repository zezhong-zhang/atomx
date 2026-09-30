"""Individual line ordering must not be confused with centroid ordering."""

import sqlite3

from scripts.audit_inverted_dhf_lines import collect


def test_inverted_line_groups_keep_nist_centroid_interpretation(tmp_path):
    path = tmp_path / 'jobs.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE line_jobs(
            z INTEGER,charge INTEGER,row_index INTEGER,
            lower_configuration TEXT,upper_configuration TEXT,status TEXT);
            INSERT INTO line_jobs VALUES(18,1,1,'lower','upper','inverted_configuration_order');
            INSERT INTO line_jobs VALUES(18,1,2,'lower','upper','inverted_configuration_order');
            INSERT INTO line_jobs VALUES(18,1,3,'other','upper','inverted_configuration_order');''')
    centroids = [
        {'z': 18, 'charge': 1, 'configuration': 'lower', 'spectrum': 'Ar II',
         'nist_manifold_status': 'evaluated_complete',
         'nist_centroid_cm1': 20000, 'theory_excitation_cm1': 21000},
        {'z': 18, 'charge': 1, 'configuration': 'upper', 'spectrum': 'Ar II',
         'nist_manifold_status': 'evaluated_complete',
         'nist_centroid_cm1': 10000, 'theory_excitation_cm1': 12000},
        {'z': 18, 'charge': 1, 'configuration': 'other', 'spectrum': 'Ar II',
         'nist_manifold_status': 'incomplete_manifold',
         'nist_centroid_cm1': None, 'theory_excitation_cm1': 30000},
    ]
    rows, summary = collect(path, centroids)
    assert summary['line_rows'] == 3
    assert summary['statuses']['NIST_centroid_agrees_with_DHF_order'] == 2
    assert summary['statuses']['no_complete_evaluated_centroid_pair'] == 1
    agreeing = next(row for row in rows if row['lower_configuration'] == 'lower')
    assert agreeing['nist_line_rows'] == 2
    assert agreeing['nist_row_indices'] == '1,2'
    assert agreeing['nist_centroid_gap_cm1'] == -10000
    assert agreeing['dhf_configuration_gap_cm1'] == -9000
