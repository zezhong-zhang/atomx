"""Isotope records keep their own mass and observed/Ritz status."""

import json
import sqlite3

import pytest

from scripts.compare_isotope_records_nist import collect, collect_levels, shifted_energy


def test_normal_mass_transfer_uses_named_isotope():
    job = ('done', -0.5001, 0.0001, 1836.0)
    assert shifted_energy(job, 3672.0) == pytest.approx(-0.50015)
    assert shifted_energy(('running', None, None, 1836.0), 3672.0) is None


def test_every_isotope_row_is_audited_without_scoring_ritz_only(tmp_path):
    archive = tmp_path / 'source.sqlite'
    ledger = tmp_path / 'jobs.sqlite'
    masses = tmp_path / 'masses.json'
    masses.write_text(json.dumps({'all_isotopes': {'1:2': {'nuclear_mass_me': 3672.0}}}))
    with sqlite3.connect(archive) as db:
        db.executescript('''CREATE TABLE nist_isotope_sources(
            kind TEXT,spectrum TEXT,z INTEGER,mass_number INTEGER,charge INTEGER,
            url TEXT,sha256 TEXT,status TEXT);
            CREATE TABLE nist_isotope_records(
            kind TEXT,spectrum TEXT,row_index INTEGER,record_json TEXT);''')
        db.execute("INSERT INTO nist_isotope_sources VALUES ('lines','D I',1,2,0,'https://physics.nist.gov/test','abc','imported')")
        db.execute("INSERT INTO nist_isotope_sources VALUES ('levels','D I',1,2,0,'https://physics.nist.gov/levels','def','imported')")
        observed = {'obs_wl_vac(nm)': '121.567', 'unc_obs_wl': '0.001',
                    'ritz_wl_vac(nm)': '121.566', 'conf_i': '1s',
                    'conf_k': '2p', 'J_i': '1/2', 'J_k': '3/2'}
        ritz = {**observed, 'obs_wl_vac(nm)': '', 'unc_obs_wl': ''}
        db.executemany('INSERT INTO nist_isotope_records VALUES (?,?,?,?)',
                       [('lines', 'D I', 0, json.dumps(observed)),
                        ('lines', 'D I', 1, json.dumps(ritz)),
                        ('levels', 'D I', 0, json.dumps({
                            'Configuration': '1s', 'J': '1/2',
                            'Level (cm-1)': '0', 'Level ID': 'ground'})),
                        ('levels', 'D I', 1, json.dumps({
                            'Configuration': '2p', 'J': '3/2',
                            'Level (cm-1)': '82281.4', 'Level ID': 'excited',
                            'Prefix': '[', 'Suffix': ']'}))])
    with sqlite3.connect(ledger) as db:
        db.execute('''CREATE TABLE config_jobs(z INTEGER,charge INTEGER,
            configuration TEXT,status TEXT,energy_ha REAL,nms_ha REAL,
            nuclear_mass_me REAL)''')
        db.executemany('INSERT INTO config_jobs VALUES (?,?,?,?,?,?,?)',
                       [(1, 0, '1s1', 'done', -0.5001, 0.0001, 1836.0),
                        (1, 0, '2p1', 'done', -0.1251, 0.00003, 1836.0)])
    rows, summary = collect(archive, ledger, masses)
    assert summary['line_rows'] == 2
    assert summary['observed_rows'] == 1
    assert rows[0]['status'] == 'dhf_configuration_average'
    assert rows[0]['dhf_mass_transferred_nm'] is not None
    assert rows[0]['dirac_mass_nm'] is not None
    assert rows[1]['status'] == 'ritz_only'
    assert rows[1]['dhf_energy_error_percent'] is None
    assert rows[0]['source_sha256'] == 'abc'
    levels, level_summary = collect_levels(archive, ledger, masses)
    assert level_summary['level_rows'] == 2
    assert levels[0]['status'] == 'ground_reference'
    assert levels[1]['reference_kind'] == 'semiempirical'
    assert levels[1]['status'] == 'semiempirical_reference'
    assert levels[1]['dhf_mass_transferred_cm1'] is not None
