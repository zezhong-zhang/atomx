"""Supplementary NIST level fields require a verified archived row identity."""

import csv
import io
import sqlite3

from scripts.import_nist_level_details import import_one, prepare_database


def source_row(*, energy="82258.9191133") -> bytes:
    row = {
        "Configuration": "2p", "Term": "2P*", "J": "1/2",
        "g": "2", "Prefix": "", "Level (cm-1)": energy,
        "Suffix": "", "Uncertainty (cm-1)": "0.0000003",
        "Splitting": "82258.9191133", "Level ID": "001001.001.000002",
    }
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row), delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue().encode()


def archive_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE levels(z INTEGER, charge INTEGER,
        row_index INTEGER, configuration_nist TEXT, j_text TEXT,
        energy_cm1 REAL)""")
    db.execute("INSERT INTO levels VALUES (1,0,0,'2p','1/2',82258.9191133)")
    prepare_database(db)
    return db


def test_extended_level_import_preserves_id_and_splitting():
    db = archive_db()
    okay, message = import_one(db, 1, 0, "https://nist.example/levels", source_row())
    assert okay, message
    assert db.execute("SELECT level_id,splitting_raw FROM levels").fetchone() == (
        "001001.001.000002", "82258.9191133")


def test_extended_level_import_rejects_changed_energy():
    db = archive_db()
    okay, message = import_one(
        db, 1, 0, "https://nist.example/levels", source_row(energy="82259"))
    assert not okay and "energy" in message
    assert db.execute("SELECT level_id FROM levels").fetchone()[0] is None
    assert db.execute("SELECT status FROM nist_level_detail_sources").fetchone()[0] == (
        "mismatch")


def test_extended_level_reimport_ignores_reconciled_supplement():
    db = archive_db()
    db.execute("ALTER TABLE levels ADD COLUMN source_variant TEXT DEFAULT 'energy_order'")
    db.execute("""INSERT INTO levels(z,charge,row_index,configuration_nist,
        j_text,energy_cm1,level_id,source_variant)
        VALUES (1,0,1,'3p','1/2',97000,'supplement','term_order')""")
    okay, message = import_one(db, 1, 0, "https://nist.example/levels",
                               source_row())
    assert okay, message
    assert db.execute("SELECT COUNT(*) FROM levels").fetchone()[0] == 2
    assert db.execute("""SELECT level_id FROM levels
        WHERE source_variant='term_order'""").fetchone()[0] == "supplement"
