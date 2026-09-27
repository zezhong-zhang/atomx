"""The offline source audit detects changed retained NIST bytes."""

import sqlite3

from scripts.audit_nist_archive_integrity import audit


def test_source_hash_mismatch_is_reported(tmp_path):
    archive = tmp_path / "spectra.sqlite"
    raw = tmp_path / "raw/lines"
    raw.mkdir(parents=True)
    (raw / "asd.001000").write_bytes(b"changed source")
    with sqlite3.connect(archive) as db:
        db.executescript("""
            CREATE TABLE source_files(kind TEXT,z INTEGER,charge INTEGER,
                bytes INTEGER,sha256 TEXT,declared_rows INTEGER);
            CREATE TABLE species(line_listed INTEGER,level_listed INTEGER);
            CREATE TABLE lines(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER);
            CREATE TABLE nist_line_only_level_sources(z INTEGER,charge INTEGER);
            CREATE TABLE nist_line_detail_sources(z INTEGER,charge INTEGER,
                sha256 TEXT,source_rows INTEGER,matched_rows INTEGER,status TEXT);
            CREATE TABLE nist_level_detail_sources AS
                SELECT * FROM nist_line_detail_sources WHERE 0;
            CREATE TABLE nist_weighted_a_sources AS
                SELECT * FROM nist_line_detail_sources WHERE 0;
            CREATE TABLE nist_term_order_level_sources(z INTEGER,charge INTEGER,
                sha256 TEXT,declared_rows INTEGER,status TEXT);
            CREATE TABLE nist_isotope_sources(kind TEXT,spectrum TEXT,
                sha256 TEXT,holdings_rows INTEGER,parsed_rows INTEGER,status TEXT);
            CREATE TABLE nist_isotope_records(kind TEXT,spectrum TEXT);
            CREATE TABLE source_count_discrepancies(kind TEXT);
            INSERT INTO source_files VALUES ('lines',1,0,14,'bad-hash',1);
            INSERT INTO species VALUES (1,0);
            INSERT INTO lines VALUES (1,0);
        """)
    result = audit(tmp_path)
    assert result["source_integrity_verified"] is False
    assert any(row["issue"] == "sha256_mismatch" for row in result["issues"])
