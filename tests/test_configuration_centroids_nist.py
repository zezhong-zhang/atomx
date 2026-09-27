"""Physics and source-quality checks for configuration centroid comparisons."""

import sqlite3

import pytest

from scripts.compare_configuration_centroids_nist import collect, level_centroids


def test_centroid_uses_complete_j_weights_and_evaluated_energies():
    source = sqlite3.connect(":memory:")
    source.execute("""CREATE TABLE levels (
        z INTEGER, charge INTEGER, configuration TEXT, j_text TEXT,
        energy_cm1 REAL, energy_kind TEXT, statistical_weight_raw TEXT)""")
    source.executemany("INSERT INTO levels VALUES (3,0,?,?,?,?,?)", [
        ("1s2 2s1", "1/2", 0.0, "evaluated", "2"),
        ("1s2 2p1", "1/2", 14900.0, "evaluated", "3"),
        ("1s2 2p1", "3/2", 14903.0, "evaluated", "4"),
    ])
    key = (3, 0, "1s2 2p1")
    jobs = {(3, 0, "1s2 2s1"), key}
    result = level_centroids(source, jobs, 3)[key]
    assert result["status"] == "evaluated_complete"
    assert result["expected_weight"] == result["listed_weight"] == 6
    assert result["centroid_cm1"] == pytest.approx((2 * 14900 + 4 * 14903) / 6)
    assert result["nist_g_disagreements"] == 1

    source.execute("""UPDATE levels SET energy_kind='theoretical'
        WHERE configuration='1s2 2p1' AND j_text='3/2'""")
    assert level_centroids(source, jobs, 3)[key]["status"] == "nonexperimental_complete"

    source.execute("""DELETE FROM levels
        WHERE configuration='1s2 2p1' AND j_text='3/2'""")
    incomplete = level_centroids(source, jobs, 3)[key]
    assert incomplete["status"] == "incomplete_manifold"
    assert incomplete["centroid_cm1"] is None
    source.close()


def test_candidate_queue_is_waiting_in_centroid_audit(tmp_path):
    archive, ledger = tmp_path / "archive.sqlite", tmp_path / "jobs.sqlite"
    with sqlite3.connect(archive) as source:
        source.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER,label TEXT);
            INSERT INTO species VALUES(3,0,'Li I');
            CREATE TABLE levels(z INTEGER,charge INTEGER,configuration TEXT,
                j_text TEXT,energy_cm1 REAL,energy_kind TEXT,
                statistical_weight_raw TEXT);
            INSERT INTO levels VALUES(3,0,'1s2 2s1','1/2',0,'evaluated','2');
            INSERT INTO levels VALUES(3,0,'1s2 2p1','1/2',14900,'evaluated','2');
            INSERT INTO levels VALUES(3,0,'1s2 2p1','3/2',14903,'evaluated','4');""")
    with sqlite3.connect(ledger) as jobs:
        jobs.executescript("""CREATE TABLE config_jobs(
            z INTEGER,charge INTEGER,configuration TEXT,status TEXT,
            energy_ha REAL,is_ground INTEGER,isotope_a INTEGER);
            INSERT INTO config_jobs VALUES(3,0,'1s2 2s1','done',-7.5,1,7);
            INSERT INTO config_jobs VALUES(3,0,'1s2 2p1','candidate_pending',NULL,0,7);""")
    rows, summary = collect(archive, ledger, 3)
    excited = next(row for row in rows if row["configuration"] == "1s2 2p1")
    assert excited["comparison_status"] == "waiting_for_theory"
    assert summary["statuses"]["waiting_for_theory"] == 1
