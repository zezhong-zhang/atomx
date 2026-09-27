"""Supplementary NIST line fields require a verified archived row identity."""

import csv
import io
import sqlite3

from scripts.import_nist_line_details import (
    EXTRA_COLUMNS, VERIFY_COLUMNS, import_one, prepare_database,
)


def source_row(*, ritz="18.2743", omit_g=False) -> bytes:
    data = {name: "" for name in
            list(VERIFY_COLUMNS.values()) + list(EXTRA_COLUMNS.values())}
    data.update({
        "ritz_wl_vac(nm)": ritz,
        "Ei(cm-1)": "50300.9+x",
        "Ek(cm-1)": "597518+x",
        "fik": "2.4474e-05",
        "ID_i": "021007.001.000001",
    })
    if omit_g:
        del data["g_i"]
        del data["g_k"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(data), delimiter="\t")
    writer.writeheader()
    writer.writerow(data)
    return buffer.getvalue().encode()


def archive_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE lines(z INTEGER, charge INTEGER, row_index INTEGER,
        observed_nm REAL, ritz_nm REAL, aki_s1 REAL,
        lower_energy_cm1 REAL, upper_energy_cm1 REAL)""")
    db.execute("INSERT INTO lines VALUES (21,6,0,NULL,18.2743,NULL,50300.9,597518)")
    prepare_database(db)
    return db


def test_extended_line_import_preserves_optional_fields_after_identity_check():
    db = archive_db()
    okay, message = import_one(db, 21, 6, "https://nist.example/lines", source_row())
    assert okay, message
    assert db.execute("SELECT oscillator_strength_raw,lower_level_id FROM lines").fetchone() == (
        "2.4474e-05", "021007.001.000001")
    assert db.execute("SELECT status,matched_rows FROM nist_line_detail_sources").fetchone() == (
        "imported", 1)


def test_extended_line_import_rejects_changed_nist_row():
    db = archive_db()
    okay, message = import_one(
        db, 21, 6, "https://nist.example/lines", source_row(ritz="18.3"))
    assert not okay and "ritz_nm" in message
    assert db.execute("SELECT oscillator_strength_raw FROM lines").fetchone()[0] is None
    assert db.execute("SELECT status FROM nist_line_detail_sources").fetchone()[0] == (
        "mismatch")


def test_nist_may_omit_an_entire_optional_column():
    db = archive_db()
    okay, message = import_one(
        db, 21, 6, "https://nist.example/lines", source_row(omit_g=True))
    assert okay, message
    assert db.execute("SELECT lower_g_raw,oscillator_strength_raw FROM lines").fetchone() == (
        None, "2.4474e-05")
