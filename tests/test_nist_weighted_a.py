"""Direct NIST weighted A values are attached only to matching line rows."""

import csv
import io
import sqlite3

from scripts.import_nist_weighted_a import import_one, prepare_database


def source_row(*, ritz="91.2323660") -> bytes:
    row = {
        "obs_wl_vac(nm)": "", "ritz_wl_vac(nm)": ritz,
        "Ei(cm-1)": "0.0000000000", "Ek(cm-1)": "[109610.2232]",
        "gA(s^-1)": "3.9226e+05",
    }
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row), delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue().encode()


def archive_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
        observed_nm REAL,ritz_nm REAL,lower_energy_cm1 REAL,
        upper_energy_cm1 REAL)""")
    db.execute("INSERT INTO lines VALUES (1,0,0,NULL,91.2323660,0,109610.2232)")
    prepare_database(db)
    return db


def test_weighted_a_import_matches_archived_line():
    db = archive_db()
    okay, message = import_one(db, 1, 0, "https://nist.example/weighted-a", source_row())
    assert okay, message
    assert db.execute("SELECT weighted_aki_raw FROM lines").fetchone()[0] == "3.9226e+05"


def test_weighted_a_import_rejects_changed_line():
    db = archive_db()
    okay, message = import_one(
        db, 1, 0, "https://nist.example/weighted-a", source_row(ritz="91.3"))
    assert not okay and "ritz_nm" in message
    assert db.execute("SELECT weighted_aki_raw FROM lines").fetchone()[0] is None
