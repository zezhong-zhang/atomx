"""Complete-record and ionization-summary checks for the atlas API."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/view_nist_spectra.py"
spec = importlib.util.spec_from_file_location("view_nist_spectra", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ion_detail_and_complete_pagination(tmp_path):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE species(z INTEGER, charge INTEGER, symbol TEXT, label TEXT);
        CREATE TABLE source_files(z INTEGER, charge INTEGER, kind TEXT, url TEXT,
          bytes INTEGER, sha256 TEXT, declared_rows INTEGER, error TEXT);
        CREATE TABLE levels(z INTEGER, charge INTEGER, row_index INTEGER,
          configuration TEXT, term TEXT, energy_cm1 REAL, uncertainty_cm1 REAL);
        CREATE TABLE lines(z INTEGER, charge INTEGER, row_index INTEGER,
          lower_configuration TEXT, upper_configuration TEXT, lower_term TEXT,
          upper_term TEXT, transition_type TEXT, line_reference TEXT,
          tp_reference TEXT, observed_nm REAL, observed_unc_nm REAL,
          ritz_nm REAL, ritz_unc_nm REAL, medium TEXT);
        CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
          ionization_eV REAL,uncertainty_eV REAL,ground_configuration TEXT);
        CREATE TABLE nist_term_order_level_sources(z INTEGER,charge INTEGER,
          status TEXT,newly_imported_rows INTEGER);
        CREATE TABLE nist_line_only_level_sources(z INTEGER,charge INTEGER,
          level_id TEXT);
        INSERT INTO species VALUES(1,0,'H','H I');
        INSERT INTO levels VALUES(1,0,0,'1s1','2S',0,0.01);
        INSERT INTO levels VALUES(1,0,1,'','Limit',109678.77,0.1);
        INSERT INTO source_files VALUES(1,0,'lines','https://example.test/lines',100,'abc',2,NULL);
        INSERT INTO nist_ionization VALUES(1,0,13.5984346,0.000000001,'1s');
        INSERT INTO nist_term_order_level_sources VALUES(1,0,'imported',2);
        INSERT INTO nist_line_only_level_sources VALUES(1,0,'line-only');
        INSERT INTO lines VALUES(1,0,0,'1s1','2p1','2S','2P*','E1','ref1','ref2',121.567,0.001,121.566,0.0001,'vacuum');
        INSERT INTO lines VALUES(1,0,1,'1s1','3p1','2S','2P*','E1','ref3','ref4',NULL,NULL,102.57,0.001,'vacuum');
    """)
    detail = module.ion_detail(db, 1, 0)
    assert detail["ground"]["configuration"] == "1s1"
    assert detail["limits"][0]["ionization_cm1"] == 109678.77
    assert detail["limits"][0]["ionization_eV"] == pytest.approx(13.598, rel=1e-4)
    assert detail["limits"][0]["ionization_unc_eV"] > 0
    assert detail["official_ionization"]["ionization_eV"] == pytest.approx(13.5984346)
    assert detail["official_ionization"]["ground_configuration"] == "1s"
    assert detail["term_order_level_source"]["newly_imported_rows"] == 2
    assert detail["line_only_level_count"] == 1
    assert detail["line_only_level_sources"][0]["level_id"] == "line-only"
    first = module.record_page(db, "lines", 1, 0, 1, 1)
    second = module.record_page(db, "lines", 1, 0, 2, 1)
    assert first["total"] == second["total"] == 2
    assert first["rows"][0]["observed_unc_nm"] == 0.001
    assert second["rows"][0]["observed_nm"] is None
    assert module.record_page(db, "lines", 1, 0, 1, 100, "3p1")["total"] == 1
    with pytest.raises(ValueError):
        module.record_page(db, "bad", 1, 0, 1, 100)
    csv_text = ("lower_configuration,upper_configuration,lower_j,upper_j,wavelength_nm\n"
                "1s1,2p1,1/2,3/2,121.5\n"
                "1s1,3p1,1/2,3/2,102.5\n")
    db.execute("ALTER TABLE lines ADD COLUMN lower_j TEXT")
    db.execute("ALTER TABLE lines ADD COLUMN upper_j TEXT")
    db.execute("UPDATE lines SET lower_j='1/2', upper_j='3/2'")
    observed = module.compare_uploaded_lines(db, 1, 0, "observed", csv_text)
    assert observed["n_matched"] == 1
    assert observed["unmatched"][0]["reason"] == "no vacuum wavelength in selected NIST basis"
    assert observed["matches"][0]["reference_unc_nm"] == 0.001
    ritz = module.compare_uploaded_lines(db, 1, 0, "ritz", csv_text)
    assert ritz["n_matched"] == 2
    db.execute("ALTER TABLE levels ADD COLUMN level_id TEXT")
    db.execute("ALTER TABLE lines ADD COLUMN lower_level_id TEXT")
    db.execute("ALTER TABLE lines ADD COLUMN upper_level_id TEXT")
    db.execute("UPDATE levels SET level_id='ground' WHERE row_index=0")
    db.execute("""INSERT INTO levels(z,charge,row_index,configuration,term,
        energy_cm1,uncertainty_cm1,level_id)
        VALUES (1,0,2,'2p1','2P*',82259,0.01,'upper')""")
    db.execute("""UPDATE lines SET upper_configuration='',
        lower_level_id='ground',upper_level_id='upper' WHERE row_index=0""")
    by_ids = module.compare_uploaded_lines(db, 1, 0, "observed",
        "lower_level_id,upper_level_id,wavelength_nm\n"
        "ground,upper,121.5\n")
    assert by_ids["n_matched"] == 1
    assert by_ids["matches"][0]["upper"] == "2p1"
    assert by_ids["matches"][0]["match_basis"] == "nist_level_ids"
    db.execute("""INSERT INTO lines(z,charge,row_index,lower_configuration,
        upper_configuration,observed_nm,medium,lower_j,upper_j,
        lower_level_id,upper_level_id)
        VALUES (1,0,2,'1s1','',121.568,'vacuum','1/2','3/2',
                'ground','upper')""")
    ambiguous = module.compare_uploaded_lines(db, 1, 0, "observed",
        "lower_level_id,upper_level_id,wavelength_nm\n"
        "ground,upper,121.5\n")
    assert ambiguous["n_matched"] == 0
    exact = module.compare_uploaded_lines(db, 1, 0, "observed",
        "lower_level_id,upper_level_id,nist_row_index,wavelength_nm\n"
        "ground,upper,0,121.5\n")
    assert exact["n_matched"] == 1
    assert exact["matches"][0]["nist_row_index"] == 0
    audit_path = tmp_path / "audit.sqlite"
    audit = sqlite3.connect(audit_path)
    audit.executescript("""
        CREATE TABLE line_audit(z INTEGER,charge INTEGER,row_index INTEGER,status TEXT,
          calculated_nm REAL,observed_energy_error_percent REAL,
          ritz_energy_error_percent REAL,model TEXT);
        CREATE TABLE level_audit(z INTEGER,charge INTEGER,row_index INTEGER,status TEXT,
          calculated_energy_cm1 REAL,residual_cm1 REAL,model TEXT);
        INSERT INTO line_audit VALUES(1,0,0,'compared_observed',121.5,0.05,0.04,'Dirac');
        INSERT INTO level_audit VALUES(1,0,1,'compared_ionization_limit',109700,21.23,'Dirac');
    """)
    audit.close()
    decorated = module.add_record_audit(first, audit_path)
    assert decorated["rows"][0]["status"] == "compared_observed"
    ledger_path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger_path) as ledger:
        ledger.executescript("""CREATE TABLE line_jobs(z INTEGER,charge INTEGER,
            row_index INTEGER,lower_configuration TEXT,upper_configuration TEXT,
            status TEXT,classification_source TEXT);
            INSERT INTO line_jobs VALUES
            (1,0,0,'1s1','2p1','configuration_average_only','nist_level_id');""")
    with_ledger = module.add_record_audit(first, audit_path, ledger_path)
    assert with_ledger["rows"][0]["classification_source"] == "nist_level_id"
    assert with_ledger["rows"][0]["dhf_upper_configuration"] == "2p1"
    assert with_ledger["rows"][0]["status"] == "compared_observed"
    level_page = module.record_page(db, "levels", 1, 0, 2, 1)
    decorated_level = module.add_record_audit(level_page, audit_path)
    assert decorated_level["rows"][0]["status"] == "compared_ionization_limit"
    assert module.audit_ion_summary(audit_path, 1, 0)["compared_observed"] == 1
    db.close()
