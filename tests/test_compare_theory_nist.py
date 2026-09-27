"""Checks for classified, medium-safe theory/reference pairing."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_theory_nist.py"
spec = importlib.util.spec_from_file_location("compare_theory_nist", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_hydrogenic_j_identification():
    assert module.kappa_of("2p1", "1/2") == (2, 1)
    assert module.kappa_of("2p1", "3/2") == (2, -2)
    assert module.kappa_of("2p1", "5/2") is None
    assert module.calculated_nm(1, "1s1", "1/2", "2p1", "3/2") == pytest.approx(
        121.5, rel=0.002)
    assert module.calculated_nm(1, "1s1", "1/2", "2s1", "1/2") is None


def test_same_n_dirac_interval_is_outside_line_score():
    assert module.calculated_any_nm(16, "2s1", "1/2", "2p1", "3/2") is None
    assert module.calculated_any_nm(16, "1s1", "1/2", "2p1", "3/2") is not None


def test_hydrogen_calibration_uses_the_same_observed_line(tmp_path):
    reference = tmp_path / "reference.sqlite"
    cache = tmp_path / "cache.json"
    with sqlite3.connect(reference) as db:
        db.executescript("""CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
          observed_nm REAL,medium TEXT,lower_configuration TEXT,lower_j TEXT,
          upper_configuration TEXT,upper_j TEXT);
          INSERT INTO lines VALUES(1,0,74,121.56699,'vacuum','1s1','1/2','2p1','3/2');
          INSERT INTO lines VALUES(1,0,75,121.56699,'air','1s1','1/2','2p1','1/2');""")
    cache.write_text(json.dumps({"scf|1|1s1|1|1": -.5000020838408783,
                                 "scf|1|2p1|1|1": -.12500072686659716}))
    rows = module.hydrogen_dhf_calibration(reference, cache)
    assert len(rows) == 1
    assert rows[0]["row_index"] == 74
    assert rows[0]["dhf_bqed_nm"] == pytest.approx(121.5018337447)
    assert rows[0]["dirac_nm"] == pytest.approx(121.5002514631)


def test_observed_and_ritz_remain_distinct(tmp_path):
    path = tmp_path / "minimal.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE species(z INTEGER, charge INTEGER, label TEXT);
        CREATE TABLE lines(z INTEGER, charge INTEGER, row_index INTEGER,
          observed_nm REAL, observed_unc_nm REAL, ritz_nm REAL, ritz_unc_nm REAL,
          medium TEXT,
          lower_configuration TEXT, lower_term TEXT, lower_j TEXT,
          upper_configuration TEXT, upper_term TEXT, upper_j TEXT);
        INSERT INTO species VALUES(1,0,'H I'),(2,1,'He II');
        INSERT INTO lines VALUES(1,0,1,121.567,.001,121.566,.001,'vacuum','1s1','2S','1/2','2p1','2P*','3/2');
        INSERT INTO lines VALUES(2,1,2,NULL,NULL,30.378,.001,'vacuum','1s1','2S','1/2','2p1','2P*','3/2');
        INSERT INTO lines VALUES(1,0,3,121.5,.1,121.5,.1,'air','1s1','2S','1/2','2p1','2P*','1/2');
        INSERT INTO lines VALUES(1,0,4,100.0,.1,100.0,.1,'vacuum','1s1','2S','1/2','2s1','2S','1/2');
    """)
    db.close()
    rows = module.load_lines(path)
    assert len(rows) == 2
    assert [r["reference_kind"] for r in rows] == ["observed", "Ritz"]
    assert rows[0]["reference_nm"] == 121.567
    assert rows[1]["reference_nm"] == 30.378
    assert rows[0]["energy_error_percent"] == pytest.approx(
        100 * (121.567 / rows[0]["calculated_nm"] - 1))


def test_rms_uses_all_matched_lines():
    rows = [{"energy_error_percent": -1.0}, {"energy_error_percent": 3.0}]
    result = module.metrics(rows)
    assert result["rms_energy_error_percent"] == pytest.approx(5 ** .5)
    assert result["median_abs_energy_error_percent"] == 2.0


def test_hydrogenic_overview_excludes_multielectron_rows(tmp_path):
    reference = tmp_path / "reference.sqlite"
    audit_path = tmp_path / "audit.sqlite"
    with sqlite3.connect(reference) as db:
        db.executescript("""
          CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
            observed_nm REAL,observed_unc_nm REAL,ritz_nm REAL,
            lower_configuration TEXT,lower_term TEXT,lower_j TEXT,
            upper_configuration TEXT,upper_term TEXT,upper_j TEXT);
          INSERT INTO lines VALUES(1,0,0,121.5,.1,121.6,'1s1','2S','1/2','2p1','2P*','3/2');
          INSERT INTO lines VALUES(3,0,0,670.9,.1,671.0,'1s2 2s1','2S','1/2','1s2 2p1','2P*','1/2');
        """)
    with sqlite3.connect(audit_path) as db:
        db.executescript("""
          CREATE TABLE line_audit(z INTEGER,charge INTEGER,row_index INTEGER,
            status TEXT,calculated_nm REAL,observed_energy_error_percent REAL,model TEXT);
          INSERT INTO line_audit VALUES(1,0,0,'compared_observed',121.4,.08,'point-nucleus Dirac-Coulomb');
          INSERT INTO line_audit VALUES(3,0,0,'compared_observed',660.0,1.65,'MCDHF+B+QED');
        """)
    rows = module.load_all_observed(reference, audit_path)
    assert [(row["z"], row["nist_row_index"]) for row in rows] == [(1, 0)]


def test_every_reference_row_gets_an_audit_status(tmp_path):
    reference = tmp_path / "reference.sqlite"
    db = sqlite3.connect(reference)
    db.executescript("""
        CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
          observed_nm REAL,ritz_nm REAL,medium TEXT,lower_configuration TEXT,
          lower_j TEXT,upper_configuration TEXT,upper_j TEXT);
        CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
          configuration TEXT,term TEXT,j_text TEXT,energy_cm1 REAL);
        INSERT INTO lines VALUES(1,0,0,121.567,121.566,'vacuum','1s1','1/2','2p1','3/2');
        INSERT INTO lines VALUES(2,0,0,50.0,49.9,'vacuum','1s2','0','1s1 2p1','1');
        INSERT INTO lines VALUES(1,0,1,NULL,102.5,'vacuum','1s1','1/2','3p1','3/2');
        INSERT INTO levels VALUES(1,0,0,'1s1','2S','1/2',0);
        INSERT INTO levels VALUES(1,0,1,'','Limit','---',109678.77);
        INSERT INTO levels VALUES(2,0,0,'1s2','1S','0',0);
    """)
    db.close()
    audit_path = tmp_path / "audit.sqlite"
    counts = module.build_record_audit(reference, audit_path)
    assert sum(counts["lines"].values()) == 3
    assert sum(counts["levels"].values()) == 3
    audit = sqlite3.connect(audit_path)
    assert audit.execute("SELECT status FROM line_audit WHERE z=1 AND row_index=0").fetchone()[0] == "compared_observed"
    assert audit.execute("SELECT status FROM line_audit WHERE z=2").fetchone()[0] == "no_atomx_line_calculation"
    assert audit.execute("SELECT status FROM level_audit WHERE z=1 AND row_index=1").fetchone()[0] == "compared_ionization_limit"
    audit.close()


def test_full_dhf_attempt_status_is_visible_without_replacing_scored_line(tmp_path):
    audit_path, ledger_path = (tmp_path / name for name in
                               ("audit.sqlite", "ledger.sqlite"))
    with sqlite3.connect(audit_path) as db:
        db.executescript("""CREATE TABLE line_audit(z INTEGER,charge INTEGER,
            row_index INTEGER,status TEXT,calculated_nm REAL,
            observed_energy_error_percent REAL,model TEXT);
            INSERT INTO line_audit VALUES
              (1,0,1,'compared_observed',121.5,0.1,'point Dirac'),
              (12,0,2,'no_atomx_line_calculation',NULL,NULL,NULL),
              (12,0,3,'no_atomx_line_calculation',NULL,NULL,NULL),
              (74,0,4,'no_atomx_line_calculation',NULL,NULL,NULL);""")
    with sqlite3.connect(ledger_path) as db:
        db.executescript("""CREATE TABLE line_jobs(z INTEGER,charge INTEGER,
            row_index INTEGER,status TEXT,calculated_nm REAL,
            energy_error_percent REAL,lower_configuration TEXT,
            upper_configuration TEXT);
            INSERT INTO line_jobs VALUES
              (1,0,1,'configuration_average_only',121.6,0.2,'1s1','2p1'),
              (12,0,2,'radial_box_unresolved',7379.27,66.98,
               '1s2 2s2 2p6 3s2','1s2 2s2 2p6 3s1 7i1'),
              (12,0,3,'compared',162.5,0.1,
               '1s2 2s2 2p6 3s2','1s2 2s2 2p6 3s1 44p1'),
              (74,0,4,'waiting_for_levels',NULL,NULL,'6s2','6s1 7p1');""")
    module.add_full_dhf_statuses(audit_path, ledger_path)
    with sqlite3.connect(audit_path) as db:
        assert db.execute("SELECT model FROM line_audit WHERE z=1").fetchone()[0] == "point Dirac"
        row = db.execute("SELECT status,calculated_nm FROM line_audit WHERE z=12").fetchone()
        assert row == ("dhf_radial_box_unresolved", 7379.27)
        assert db.execute("SELECT status FROM line_audit WHERE z=12 AND row_index=3").fetchone()[0] == (
            "dhf_configuration_average_only")
        assert db.execute("SELECT status FROM line_audit WHERE z=74").fetchone()[0] == (
            "dhf_waiting_for_levels")
