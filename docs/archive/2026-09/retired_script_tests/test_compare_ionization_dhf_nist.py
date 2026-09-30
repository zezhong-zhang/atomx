"""Ground-state energy differences use the same charge and isotope chain."""

import sqlite3

import pytest

from atomx.constants import HARTREE_EV
from scripts.compare_ionization_dhf_nist import (collect, ground_centroids,
                                                reference_kind)
from scripts.generate_full_dhf_nist import SCHEMA
from atomx.reference_data.matching import expand_nist_config


def test_ionization_uses_adjacent_ground_total_energies(tmp_path):
    archive = tmp_path / "nist.sqlite"
    ledger = tmp_path / "jobs.sqlite"
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
                configuration TEXT,energy_cm1 REAL,j_text TEXT,
                energy_kind TEXT,statistical_weight_raw TEXT);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
                spectrum TEXT,ionization_eV REAL,ionization_raw TEXT,
                ionization_cm1 REAL,uncertainty_eV REAL,
                uncertainty_cm1 REAL,binding_cm1 REAL,binding_kind TEXT,
                reference TEXT,ground_shells TEXT);
            INSERT INTO species VALUES (1,0),(2,0),(2,1),(8,0),(8,1),(8,7);
            INSERT INTO levels VALUES
                (1,0,0,'1s1',0,'1/2','evaluated','2'),
                (2,0,0,'1s2',0,'0','evaluated','1'),
                (2,1,0,'1s1',0,'1/2','evaluated','2'),
                (8,0,0,'2s2 2p4',0,'2','evaluated','5'),
                (8,1,0,'2s2 2p3',0,'3/2','evaluated','4'),
                (8,7,0,'1s1',0,'1/2','evaluated','2');
            INSERT INTO nist_ionization VALUES
                (1,0,'H I',13.5984346,'13.5984346',109678.77,0.000001,0.01,NULL,NULL,'ref',NULL),
                (2,0,'He I',24.5874,'[24.5874]',NULL,0.0001,NULL,NULL,NULL,'ref',NULL),
                (2,1,'He II',54.4178,'(54.4178)',NULL,0.0001,NULL,NULL,NULL,'ref',NULL),
                (8,0,'O I',13.6181,'13.6181',NULL,0.0001,NULL,NULL,NULL,'ref',NULL);
        """)
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,energy_ha,isotope_a,
             nuclear_mass_me,is_ground)
            VALUES (?,?,?,'done',?,?,7000,1)""",
            [(1,0,'1s1',-0.5,1), (2,0,'1s2',-2.86,4),
             (2,1,'1s1',-2.0,4), (8,0,'1s2 2s2 2p4',-74.7,16),
             (8,1,'1s2 2s2 2p3',-74.2,16),
             (8,7,'1s1',-32.0,16)])
    rows = collect(archive, ledger, max_z=8)
    assert len(rows) == 6
    assert all(row["status"] == "compared" for row in rows[:4])
    assert rows[0]["calculated_ionization_eV"] == pytest.approx(0.5 * HARTREE_EV)
    assert rows[1]["calculated_ionization_eV"] == pytest.approx(0.86 * HARTREE_EV)
    assert rows[2]["calculated_ionization_eV"] == pytest.approx(2.0 * HARTREE_EV)
    assert [row["reference_kind"] for row in rows] == [
        "evaluated", "semiempirical", "theoretical", "evaluated",
        "missing", "missing"]
    assert [row["comparison_basis"] for row in rows[:4]] == [
        "unique_ground_level", "unique_ground_level",
        "unique_ground_level", "configuration_average"]
    assert [row["ground_configuration_source"] for row in rows] == [
        "nist_levels"] * 6
    assert rows[0]["ionized_ground_configuration_source"] == "bare_nucleus"
    assert rows[0]["centroid_status"] == "evaluated_complete"
    assert rows[3]["centroid_status"] == "incomplete_level_manifold"
    assert rows[4]["status"] == "ground_configuration_unavailable"
    assert rows[5]["status"] == "no_nist_ionization_energy"
    assert rows[5]["reference_kind"] == "missing"
    assert rows[5]["calculated_ionization_eV"] == pytest.approx(32 * HARTREE_EV)


def test_ionization_uses_selected_ground_not_obsolete_neutral_fallback(tmp_path):
    archive = tmp_path / "nist.sqlite"
    ledger = tmp_path / "jobs.sqlite"
    old = expand_nist_config("4f13 6s2", 69)
    ground = expand_nist_config("[Xe].4f14.5d", 69)
    ionized = expand_nist_config("[Xe].4f14", 68)
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE levels(z INTEGER,charge INTEGER,
            row_index INTEGER,
            configuration TEXT,j_text TEXT,energy_cm1 REAL,
            energy_kind TEXT,statistical_weight_raw TEXT);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
            spectrum TEXT,ionization_eV REAL,ionization_raw TEXT,
            ionization_cm1 REAL,uncertainty_eV REAL,
            uncertainty_cm1 REAL,binding_cm1 REAL,binding_kind TEXT,
            reference TEXT,ground_shells TEXT);
            INSERT INTO levels VALUES (76,7,0,'5d1','3/2',0,'evaluated','4');
            INSERT INTO levels VALUES (76,8,0,'4f14','0',0,'evaluated','1');
            INSERT INTO nist_ionization VALUES
                (76,7,'Os VIII',102.02,'[102.02]',NULL,0.07,NULL,NULL,NULL,'ref',
                 '[Xe].4f14.5d');""")
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,energy_ha,isotope_a,
             nuclear_mass_me,is_ground)
            VALUES (?,?,?,?,?,192,350000,?)""", [
                (76,7,old,'done',-17218.821,0),
                (76,7,ground,'pending',None,1),
                (76,8,ionized,'done',-17218.843,1)])
    row = collect(archive, ledger, max_z=76)[0]
    assert row["ground_configuration"] == ground
    assert row["ground_configuration_source"] == "nist_ground_shells"
    assert row["status"] == "waiting_for_ground_energies"
    assert row["calculated_ionization_eV"] is None


def test_nist_ground_energy_provenance():
    assert reference_kind("[10.2]") == "semiempirical"
    assert reference_kind("(10.2)") == "theoretical"
    assert reference_kind("10.2") == "evaluated"
    assert reference_kind("") == "missing"


def test_complete_oxygen_centroids_derive_weights_from_j():
    with sqlite3.connect(":memory:") as db:
        db.execute("""CREATE TABLE levels(z INTEGER,charge INTEGER,
            configuration TEXT,j_text TEXT,energy_cm1 REAL,
            energy_kind TEXT,statistical_weight_raw TEXT)""")
        db.executemany("INSERT INTO levels VALUES (?,?,?,?,?,'evaluated',?)", [
            (8, 0, "2s2 2p4", "2", 0.0, "4"),  # source g conflicts with 2J+1
            (8, 0, "2s2 2p4", "1", 158.265, "3"),
            (8, 0, "2s2 2p4", "0", 226.977, "1"),
            (8, 0, "2s2 2p4", "2", 15867.862, "5"),
            (8, 0, "2s2 2p4", "0", 33792.583, "1"),
            (8, 1, "2s2 2p3", "3/2", 0.0, "4"),
            (8, 1, "2s2 2p3", "5/2", 26810.55, "6"),
            (8, 1, "2s2 2p3", "3/2", 26830.57, "4"),
            (8, 1, "2s2 2p3", "3/2", 40468.01, "4"),
            (8, 1, "2s2 2p3", "1/2", 40470.0, "2"),
        ])
        centroids = ground_centroids(db, {
            (8, 0): "1s2 2s2 2p4", (8, 1): "1s2 2s2 2p3"}, 8)
    assert centroids[8, 0]["status"] == "evaluated_complete"
    assert centroids[8, 1]["status"] == "evaluated_complete"
    assert centroids[8, 0]["centroid_cm1"] == pytest.approx(7588.911)
    assert centroids[8, 1]["centroid_cm1"] == pytest.approx(25549.881)
    assert centroids[8, 0]["nist_g_disagreements"] == 1
